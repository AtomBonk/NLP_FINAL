import os
import json
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
from peft import PeftModel
from huggingface_hub import login

# Fix caches & Login
os.makedirs("/home/morg/NLP_2526b/shairotman/.cache/huggingface", exist_ok=True)
os.environ["HF_HOME"] = "/home/morg/NLP_2526b/shairotman/.cache/huggingface"
login(token=os.environ["HF_TOKEN"])

# קיבוע האקראיות לייצור תוצאות הדירות
set_seed(42)

MODEL_ID = "meta-llama/Meta-Llama-3-8B"
ADAPTER_DIR = "./full_finetune_output"
DATASET_PATH = "/home/morg/NLP_2526b/shairotman/processed_wikimusictext"
NUM_TEST_SAMPLES = 10  # כמות הדגימות לבדיקה מה-Test (אפשר להגדיל)

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Quantization Config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

# Load Base Model & Attach LoRA Adapters
print("Loading Base Model...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)
print("Attaching Fine-Tuned Adapters...")
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()

# Load exclusively the TEST split (No Data Leakage)
dataset_dict = load_from_disk(DATASET_PATH)
test_ds = dataset_dict["test"].select(range(NUM_TEST_SAMPLES))

results = []
print(f"Generating completions for {NUM_TEST_SAMPLES} test samples...")

for i, example in enumerate(test_ds):
    prompt_text = example.get("prompt", "")
    ground_truth = example.get("completion", "")

    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,  # Decoding scheme parameter fixed
            pad_token_id=tokenizer.pad_token_id
        )

    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # שימוש במפתח "zero_shot_generated" כדי שתהיה תאימות מלאה לסקריפט ההערכה של שלב 3
    results.append({
        "prompt": prompt_text,
        "ground_truth_completion": ground_truth,
        "zero_shot_generated": generated_text
    })
    
    print(f"Processed sample {i+1}/{NUM_TEST_SAMPLES}")

# Save Fine-Tuned Results
output_path = "fine_tuned_results.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"Inference complete. Saved to {output_path}")