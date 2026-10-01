import os
import gc
import json
import itertools
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# 1. Environment & Base Configuration
PERSISTENT_BASE = "/home/morg/NLP_2526b/shairotman/NLP_final"
os.environ["HF_HOME"] = os.path.join(PERSISTENT_BASE, "hf_cache")
DATASET_PATH = os.path.join(PERSISTENT_BASE, "music_abc_splits_augmented")

# 2. Slice Datasets (100 Train / 50 Validation)
print("--- Loading Datasets ---")
splits = load_from_disk(DATASET_PATH)
train_dataset = splits["train"].select(range(100))
eval_dataset = splits["validation"].select(range(50))

def prepare_sample(example):
    return {
        "text": f"{example['prompt']}\n{example['completion']}<|end_of_text|>"
    }

train_dataset = train_dataset.map(prepare_sample, remove_columns=train_dataset.column_names)
eval_dataset = eval_dataset.map(prepare_sample, remove_columns=eval_dataset.column_names)

# 3. Load Base Model Once in 4-bit Quantization
MODEL_ID = "meta-llama/Meta-Llama-3-8B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Loading base model to GPU...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)
base_model = prepare_model_for_kbit_training(base_model)
base_model.config.use_cache = False

# 4. Hyperparameter Search Space (12 Total Combinations)
hyperparameter_grid = {
    "learning_rate": [1e-4, 2e-4, 3e-4],
    "num_train_epochs": [3, 5],
    "effective_batch_size": [24, 32],
    "lora_r": [16]
}

keys, values = zip(*hyperparameter_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
print(f"Total experiment configurations to evaluate: {len(combinations)}")

experiment_results = []

# 5. Run Grid Search Loop
for exp_id, params in enumerate(combinations, start=1):
    lr = params["learning_rate"]
    epochs = params["num_train_epochs"]
    eff_bs = params["effective_batch_size"]
    r = params["lora_r"]
    alpha = 2 * r
    grad_accum = eff_bs // 2  # per_device_batch_size=2

    print("\n" + "=" * 60)
    print(f"Running Experiment {exp_id}/{len(combinations)}")
    print(f"Parameters: LR={lr}, Epochs={epochs}, Effective Batch={eff_bs}, LoRA r={r}")
    print("=" * 60)

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    current_model = get_peft_model(base_model, lora_config)
    output_dir = f"./tuning_exp_{exp_id}"

    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=lr,
        num_train_epochs=epochs,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=1,
        bf16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        dataset_text_field="text",
        max_length=1536,  # matches MAX_SEQ_LENGTH in prepare_dataset.py
    )

    trainer = SFTTrainer(
        model=current_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        args=training_args,
    )

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    final_eval_loss = eval_metrics.get("eval_loss", float("inf"))
    print(f">> Exp {exp_id} Finished | Validation Loss: {final_eval_loss:.4f}")

    experiment_results.append({
        "exp_id": exp_id,
        "learning_rate": lr,
        "num_train_epochs": epochs,
        "effective_batch_size": eff_bs,
        "lora_r": r,
        "eval_loss": final_eval_loss,
        "train_loss": train_result.training_loss
    })

    base_model = current_model.unload()
    del trainer
    del current_model
    gc.collect()
    torch.cuda.empty_cache()

# 6. Sort and Print Summary Leaderboard
experiment_results.sort(key=lambda x: x["eval_loss"])

print("\n" + "=" * 80)
print(f"{'RANK':<5} | {'EXP ID':<8} | {'LR':<8} | {'EPOCHS':<8} | {'BATCH':<8} | {'EVAL LOSS':<12} | {'TRAIN LOSS':<12}")
print("=" * 80)

for rank, res in enumerate(experiment_results, start=1):
    print(f"{rank:<5} | {res['exp_id']:<8} | {res['learning_rate']:<8} | {res['num_train_epochs']:<8} | {res['effective_batch_size']:<8} | {res['eval_loss']:<12.4f} | {res['train_loss']:<12.4f}")

best_exp = experiment_results[0]
print("=" * 80)
print(f"BEST CONFIGURATION: Experiment {best_exp['exp_id']} with Eval Loss = {best_exp['eval_loss']:.4f}")
print(f"Optimal Hyperparameters: LR={best_exp['learning_rate']}, Epochs={best_exp['num_train_epochs']}, Effective Batch Size={best_exp['effective_batch_size']}")

# Save results to JSON
with open("grid_search_results.json", "w") as f:
    json.dump(experiment_results, f, indent=4)