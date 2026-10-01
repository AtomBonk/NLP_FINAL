import os
# חובה בשורות הראשונות של הקובץ!
os.makedirs("/tmp/amit_hf_cache", exist_ok=True)
os.environ["HF_HOME"] = "/tmp/amit_hf_cache"

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"])

# 1. Environment & Base Configuration
PERSISTENT_BASE = "/home/morg/NLP_2526b/shairotman/NLP_final"

# הפניית המטמון לכונן הזמני של השרת כדי לעקוף את מגבלת הדיסק


user_name = os.environ.get("USER", "my_user")
cache_dir = f"/tmp/{user_name}_hf_cache"

os.makedirs(cache_dir, exist_ok=True)
os.environ["HF_HOME"] = cache_dir

DATASET_PATH = os.path.join(PERSISTENT_BASE, "music_abc_splits_augmented")

# Must match MAX_SEQ_LENGTH in prepare_dataset.py. Examples longer than this
# were already dropped when the splits were built, so a lower cap here would
# silently truncate targets the dataset had deliberately kept.
MAX_SEQ_LENGTH = 1536


# 2. Slice Dataset (7500 Train, No Validation)
print("--- Loading Dataset ---")
splits = load_from_disk(DATASET_PATH)
train_dataset = splits["train"]
print(f"Train dataset size: {len(train_dataset)} samples")

def prepare_sample(example):
    return {
        "text": f"{example['prompt']}\n{example['completion']}<|end_of_text|>"
    }

train_dataset = train_dataset.map(prepare_sample, remove_columns=train_dataset.column_names)

# 3. Load Base Model Once in 4-bit Quantization
MODEL_ID = "meta-llama/Meta-Llama-3-8B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# הטוקנייזר חייב גישה לרשת כדי להוריד את חוקי החלוקה (שוקל בקושי 2MB)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Loading base model to GPU...")
# --- תיקון: הוספת local_files_only=True כדי לחסום הורדה מחדש של 15GB ---
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    #################################local_files_only=True 
)
base_model = prepare_model_for_kbit_training(base_model)
base_model.config.use_cache = False

# 4. LoRA Configuration
print("Configuring LoRA...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

current_model = get_peft_model(base_model, lora_config)
output_dir = "./music_finetuned_model"

# 5. Setup Training Arguments
training_args = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32, # Effective batch size = 32
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    learning_rate=3e-4,
    num_train_epochs=3,             # 3 Epochs
    
    # Checkpointing Settings
    save_strategy="steps",          # שמירה לפי צעדים
    save_steps=250,                 # צ'קפוינט כל 250 צעדים
    save_total_limit=3,             # שומר את 3 הצ'קפוינטים האחרונים כדי לא לסתום את הזיכרון בדיסק
    
    logging_steps=10,
    bf16=True,
    optim="paged_adamw_8bit",
    report_to="none",               # בטל דיווח ל-wandb
    dataset_text_field="text",
    max_length=MAX_SEQ_LENGTH,
)

# 6. Initialize Trainer
trainer = SFTTrainer(
    model=current_model,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    args=training_args,
)

# 7. Start Training
print("Starting Fine-Tuning...")
# resume_from_checkpoint=True גורם למודל לחפש אוטומטית צ'קפוינטים בתיקיית היעד במקרה של קריסה
trainer.train(resume_from_checkpoint=True if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0 else None)

# 8. Save Final Model
print("Training complete! Saving final model...")
trainer.save_model(f"{output_dir}/final")
tokenizer.save_pretrained(f"{output_dir}/final")
print(f"Final model successfully saved to {output_dir}/final")