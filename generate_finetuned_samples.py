import os
import json
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import login
from tqdm import tqdm

# ==========================================
# 0. הגדרת Cache והתחברות
# ==========================================
# הגדרת נתיב הקאש לפני טעינת הספריות הכבדות של HuggingFace
PERSISTENT_BASE = "/home/morg/NLP_2526b/shairotman/NLP_final"
CACHE_DIR = os.path.join(PERSISTENT_BASE, ".cache")
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

login(token=os.environ["HF_TOKEN"])

# ==========================================
# הגדרות הרצה
# ==========================================
DATASET_PATH = os.path.join(PERSISTENT_BASE, "music_abc_splits_augmented")
FINETUNED_MODEL_PATH = os.path.join(PERSISTENT_BASE, "music_finetuned_model/final")
BASE_MODEL_ID = "meta-llama/Meta-Llama-3-8B"
OUTPUT_PATH = "finetuned_model_train_results.json"

BATCH_SIZE = 8
MAX_NEW_TOKENS = 2000  # כמות הטוקנים המקסימלית ליצירה

# ==========================================
# 1. טעינת הדאטה (test)
# ==========================================
splits = load_from_disk(DATASET_PATH)
test_dataset = splits["test"]  # שימוש ב-Test לצורך יצירת דוגמאות (לפי בקשתך)
NUM_SAMPLES = len(test_dataset)

# ==========================================
# 2. טעינת התקדמות (Checkpointing)
# ==========================================
results = []
if os.path.exists(OUTPUT_PATH):
    try:
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"Found existing checkpoint with {len(results)} samples. Resuming from there...")
    except json.JSONDecodeError:
        print("Existing JSON file is corrupted or empty. Starting from scratch.")
        results = []

start_idx = len(results)

if start_idx >= NUM_SAMPLES:
    print("All samples are already processed. Exiting.")
    exit()

# ==========================================
# 3. הגדרת Tokenizer
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, cache_dir=CACHE_DIR)
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"
# שימוש בטוקן נייטרלי ל-Padding במקום eos_token
tokenizer.pad_token = "<|reserved_special_token_0|>"

# ==========================================
# 4. טעינת המודל (Base + LoRA)
# ==========================================
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto",
    cache_dir=CACHE_DIR
)
base_model.config.pad_token_id = tokenizer.pad_token_id

print(f"Loading LoRA adapters from {FINETUNED_MODEL_PATH}...")
model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL_PATH)
model.eval()
print("Model loaded successfully.")

# ==========================================
# 5. לולאת היצירה באצוות (Batch Generation)
# ==========================================
for i in tqdm(range(start_idx, NUM_SAMPLES, BATCH_SIZE), desc="Generating in batches"):
    # חיתוך הבאץ' הנוכחי
    batch_indices = range(i, min(i + BATCH_SIZE, NUM_SAMPLES))
    batch = test_dataset.select(batch_indices)
    
    # הוספת שורת רווח ל-Prompt כמו באימון
    prompts = [p + "\n" for p in batch["prompt"]]
    ground_truths = batch["completion"]

    inputs = tokenizer(
        prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=2048 # משאיר מספיק מקום לקונטקסט (במידת הצורך אפשר להגדיל)
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS, 
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id 
        )

    # חילוץ רק של הטוקנים החדשים שנוצרו
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[:, input_length:]
    
    # פיענוח לטקסט
    generated_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    # יצירת מילון התוצאות ל-JSON
    for prompt_text, gt, gen_text in zip(batch["prompt"], ground_truths, generated_texts):
        results.append({
            "prompt": prompt_text,
            "ground_truth_completion": gt,
            "finetuned_generation": gen_text.rstrip() # מותאם בדיוק למפתח שביקשת
        })

    # שמירת צ'קפוינט אחרי כל Batch
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    # ניקוי זיכרון CUDA כדי למנוע קריסות (OOM) בגלל הבאץ' והטוקנים הארוכים
    torch.cuda.empty_cache()

print(f"Fine-tuned generation complete! All {len(results)} samples saved to {OUTPUT_PATH}")