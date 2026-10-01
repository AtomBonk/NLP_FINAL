"""Sanity overfit on the current soprano->alto data.

Re-runs the 8-example overfitting check (originally commit 3d6c257) on the
current training split. Settings match the original run except:
  - data: first 8 rows of music_abc_splits["train"] (not a test-set dump)
  - max_length 1536 (the current examples exceed the old 512 cap)
Writes only new files: sanity_current_results.json and sanity_current_output/.
Saves no model weights. Runs offline from the local model cache.
"""
import os
import sys
import json

PERSISTENT_BASE = "/home/morg/NLP_2526b/shairotman/NLP_final"
# Prefer the node-local copy left by the training run (fast local disk);
# fall back to the shared cache on network storage.
LOCAL_CACHE = "/tmp/amit_hf_cache/hub"
CACHE_DIR = LOCAL_CACHE if os.path.isdir(LOCAL_CACHE) else os.path.join(PERSISTENT_BASE, ".cache")
RESULTS_PATH = os.path.join(PERSISTENT_BASE, "sanity_current_results.json")
OUTPUT_DIR = os.path.join(PERSISTENT_BASE, "sanity_current_output")

if os.path.exists(RESULTS_PATH):
    sys.exit(f"{RESULTS_PATH} already exists; refusing to overwrite.")

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

MODEL_ID = "meta-llama/Meta-Llama-3-8B"
SANITY_SAMPLE_SIZE = 8
MAX_SEQ_LENGTH = 1536

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    cache_dir=CACHE_DIR,
    local_files_only=True,
)
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

train = load_from_disk(os.path.join(PERSISTENT_BASE, "music_abc_splits"))["train"]
sanity_dataset = train.select(range(SANITY_SAMPLE_SIZE))
sanity_dataset = sanity_dataset.map(
    lambda ex: {"text": f"{ex['prompt']}\n{ex['completion']}<|end_of_text|>"},
    remove_columns=sanity_dataset.column_names,
)
lengths = [len(tokenizer(t)["input_ids"]) for t in sanity_dataset["text"]]
print(f"Token lengths of the {SANITY_SAMPLE_SIZE} examples: {lengths}")

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    num_train_epochs=30,
    logging_steps=2,
    save_strategy="no",
    bf16=True,
    optim="paged_adamw_8bit",
    report_to="none",
    dataset_text_field="text",
    max_length=MAX_SEQ_LENGTH,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=sanity_dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
    args=training_args,
)

print("Starting sanity overfit on current training data...")
trainer.train()

logs = [h for h in trainer.state.log_history if "loss" in h]
summary = {
    "data": "music_abc_splits train[0:8]",
    "token_lengths": lengths,
    "max_length": MAX_SEQ_LENGTH,
    "learning_rate": 2e-4,
    "epochs": 30,
    "first": logs[0],
    "last": logs[-1],
    "log_history": trainer.state.log_history,
}
with open(RESULTS_PATH, "w") as f:
    json.dump(summary, f, indent=2)
print(f"First logged: loss={logs[0]['loss']}, acc={logs[0].get('mean_token_accuracy')}")
print(f"Last logged:  loss={logs[-1]['loss']}, acc={logs[-1].get('mean_token_accuracy')}")
print(f"Summary written to {RESULTS_PATH}")
