import os
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer
from huggingface_hub import login


# 1. Environment Configuration
PERSISTENT_BASE = "/home/morg/NLP_2526b/shairotman/NLP_final"
os.environ["HF_HOME"] = os.path.join(PERSISTENT_BASE, "hf_cache")
DATASET_PATH = os.path.join(PERSISTENT_BASE, "music_abc_splits_augmented")
MODEL_ID = "meta-llama/Meta-Llama-3-8B"
os.environ["HF_HOME"] = "/home/morg/NLP_2526b/shairotman/.cache/huggingface"
login(token=os.environ["HF_TOKEN"])

print("Loading dataset and tokenizer...")
splits = load_from_disk(DATASET_PATH)
train_dataset = splits["test"]  # Using the test split for analysis

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# 2. Formatting & Tokenizing
def prepare_sample(example):
    return {
        "text": f"{example['prompt']}\n{example['completion']}<|end_of_text|>"
    }

train_dataset = train_dataset.map(prepare_sample, remove_columns=train_dataset.column_names)

def calculate_token_length(example):
    tokens = tokenizer(example["text"], truncation=False)["input_ids"]
    return {"token_length": len(tokens)}

print("Calculating token lengths...")
lengths_dataset = train_dataset.map(calculate_token_length, num_proc=4)
lengths = np.array(lengths_dataset["token_length"])

# 3. Terminal Histogram Generation
# הגדרת הטווחים (Bins) לבדיקת ההתפלגות
bins = [0, 512, 1024, 1536, 2048, 2560, 3072, 4000, 5000]
hist, bin_edges = np.histogram(lengths, bins=bins)

print("\n" + "=" * 65)
print(f"{'TOKEN RANGE':<15} | {'COUNT':<6} | {'PERCENT':<7} | {'HISTOGRAM'}")
print("=" * 65)

total_samples = len(lengths)
for i in range(len(hist)):
    count = hist[i]
    percentage = (count / total_samples) * 100
    
    # בניית מד התקדמות ויזואלי (כל 'בלוק' מייצג 2 אחוזים)
    bar = "█" * int(percentage / 2) 
    
    range_str = f"{int(bin_edges[i])}-{int(bin_edges[i+1])}"
    print(f"{range_str:<15} | {count:<6} | {percentage:>6.2f}% | {bar}")

print("=" * 65)