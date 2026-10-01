import os
import json
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import os
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"])
# שמירה וקריאה של המודל מהדיסק המקומי הזמני והמהיר
os.environ["HF_HOME"] = "/tmp/hf_cache"
NUM_SAMPLES = len(load_from_disk("/home/morg/NLP_2526b/shairotman/NLP_final/music_abc_splits_augmented")["test"])  # הגבלת מספר הדוגמאות ל-1500 או פחות אם יש פחות
BATCH_SIZE = 8
OUTPUT_PATH = "zero_shot_baseline_results_NEW.json"

# ==========================================
# 0. Load existing progress (Checkpointing)
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
# 1. Load the dataset
# ==========================================
dataset_path = "/home/morg/NLP_2526b/shairotman/NLP_final/music_abc_splits_augmented"
dataset_dict = load_from_disk(dataset_path)

split_name = "test"
ds = dataset_dict[split_name].select(range(NUM_SAMPLES))

# ==========================================
# 2. Load Tokenizer
# ==========================================
model_id = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = "left"

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ==========================================
# 3. Configure Model
# ==========================================
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="cuda",
    # attn_implementation="flash_attention_2" # שים בהערה אם לא הצלחת להתקין את Flash Attention
)
print("Model loaded successfully.")

# ==========================================
# 4. Batched Generation Loop with Checkpoints
# ==========================================
# מתחילים מהאינדקס שבו עצרנו (start_idx)
for i in tqdm(range(start_idx, len(ds), BATCH_SIZE), desc="Generating in batches"):
    batch = ds[i : i + BATCH_SIZE]
    prompts = batch["prompt"]
    ground_truths = batch["completion"]

    inputs = tokenizer(
        prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=1100
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1100,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[:, input_length:]
    generated_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    # הוספת התוצאות החדשות לרשימה
    for prompt, gt, gen_text in zip(prompts, ground_truths, generated_texts):
        results.append({
            "prompt": prompt,
            "ground_truth_completion": gt,
            "zero_shot_generated": gen_text.strip()
        })

    # שמירת צ'קפוינט אחרי כל Batch
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

print(f"Zero-shot baseline complete! All {len(results)} samples saved to {OUTPUT_PATH}")