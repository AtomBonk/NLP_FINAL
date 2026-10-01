import os
import re
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
import json
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"])

# Global state tracker
failed_samples_log = []

# פותר שגיאות אזהרה של Rust כשמריצים טוקניזציה בכמה תהליכים במקביל
os.environ["TOKENIZERS_PARALLELISM"] = "false"

PERSISTENT_BASE = "/home/morg/NLP_2526b/shairotman/NLP_final"
HF_CACHE_DIR = os.path.join(PERSISTENT_BASE, ".cache/huggingface")
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_CACHE_DIR, "datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_CACHE_DIR, "hub")

print("--- Step 1: Loading ABC Dataset from HuggingFace ---")
DATASET_NAME = "Seeker38/music_abc_notation"
dataset = load_dataset(DATASET_NAME, cache_dir=HF_CACHE_DIR)


def extract_header_and_tune(abc_text: str):
    lines = abc_text.strip().split('\n')
    header_lines = []
    body_lines = []
    header_pattern = re.compile(r'^[XTMKLQCRV]:')
    
    for line in lines:
        l = line.strip()
        if not l or l.startswith('%'):
            continue
        if header_pattern.match(l):
            header_lines.append(l)
        else:
            body_lines.append(l)
            
    return "\n".join(header_lines), " ".join(body_lines)


def process_native_abc_voices(abc_raw: str):
    # 1. Fast fail: Rejects the single-track noise efficiently
    raw_nospace = abc_raw.replace(" ", "")
    if "V:1" not in raw_nospace or "V:2" not in raw_nospace:
        return None, None
        
    cleaned_abc = re.sub(r'^Assistant:\s*', '', abc_raw.strip())
    cleaned_abc = re.sub(r'</s>\s*$', '', cleaned_abc).strip()

    # Clean out formatting artifacts like I:linebreak $
    cleaned_abc = cleaned_abc.replace("I:linebreak $", "")
    
    # 2. Sanitize unclosed structural brackets BEFORE parity check 
    cleaned_abc = re.sub(r'\[([12])', r'\1.', cleaned_abc)  
    cleaned_abc = re.sub(r'\[\|', r'|', cleaned_abc)        
    cleaned_abc = re.sub(r'\|([12])', r'|\1.', cleaned_abc) 

    lines = cleaned_abc.split('\n')
    headers = []
    v1_raw = []
    v2_raw = []
    
    # current_voice: 1 (V:1), 2 (V:2), 0 (Ignore extra voices like V:3/V:4)
    current_voice = 1 
    header_pattern = re.compile(r'^([XTMKLQCRV]):(.*)')
    
    for line in lines:
        # Split the line at the first '%' and keep only the musical notation on the left.
        # This safely deletes BOTH header noise (%%score) and inline markers (%7).
        l = line.split('%')[0].strip()
        
        if not l:
            continue
            
        match = header_pattern.match(l)
        if match:
            key, val = match.groups()
            if key in ['X', 'T', 'M', 'L', 'K', 'Q', 'C', 'R']:
                headers.append(l)
                continue
            elif key == 'V':
                if '1' in val: current_voice = 1
                elif '2' in val: current_voice = 2
                else: current_voice = 0  # CRITICAL: Ignores V:3, V:4, etc.
                continue
                
        # Handle inline voice markers defensively
        if '[V:' in l:
            chunks = re.split(r'(\[V:\s*\d+[^\]]*\])', l)
            for chunk in chunks:
                chunk = chunk.strip()
                if not chunk: continue
                if re.match(r'\[V:\s*1', chunk): current_voice = 1
                elif re.match(r'\[V:\s*2', chunk): current_voice = 2
                elif re.match(r'\[V:', chunk): current_voice = 0
                elif not chunk.startswith('[V:'):
                    if current_voice == 1: v1_raw.append(chunk)
                    elif current_voice == 2: v2_raw.append(chunk)
        else:
            if current_voice == 1: v1_raw.append(l)
            elif current_voice == 2: v2_raw.append(l)
            # If current_voice == 0, the line is safely discarded
            
    if not v1_raw or not v2_raw:
        return None, None

    def tag_bars(voice_string, voice_id):
        tokens = re.split(r'(\|:|\:\||::|\|\]|\|\||\|)', voice_string)
        tagged_tokens = []
        bar_count = 1
        
        for token in tokens:
            token = token.strip()
            if not token: continue
            if re.match(r'^(\|:|\:\||::|\|\]|\|\||\|)$', token):
                tagged_tokens.append(token)
            else:
                tagged_tokens.append(f"[V:{voice_id} B{bar_count}] {token}")
                bar_count += 1
        return " ".join(tagged_tokens)

    v1_tagged = tag_bars(" ".join(v1_raw), 1)
    v2_tagged = tag_bars(" ".join(v2_raw), 2)
    
    prompt_melody = "\n".join(headers) + "\n" + v1_tagged
    completion_accomp = v2_tagged
    
    return prompt_melody, completion_accomp


def create_instruction_pair(example):
    raw_output = example.get("output", "")
    if not raw_output or not isinstance(raw_output, str):
        return {"prompt": "", "completion": "", "full_input": "", "is_valid": False}

    prompt_melody, completion_accomp = process_native_abc_voices(raw_output)
    
    if not prompt_melody or not completion_accomp:
        return {"prompt": "", "completion": "", "full_input": "", "is_valid": False}

    # FIX: Temporarily strip the ABC end-barline '|]' to calculate mathematical bracket parity
    mel_check = prompt_melody.replace('|]', '')
    acc_check = completion_accomp.replace('|]', '')

    if mel_check.count('[') != mel_check.count(']') or acc_check.count('[') != acc_check.count(']'):
        return {"prompt": "", "completion": "", "full_input": "", "is_valid": False}

    caption = "Generate a complementary harmonic accompaniment track for the following melody."
    prompt_text = f"### Instruction:\n{caption}\n\n### Primary Melody (SMT-ABC):\n{prompt_melody}"
    completion_text = f"### Accompaniment Track (SMT-ABC):\n{completion_accomp}"

    return {
        "prompt": prompt_text,
        "completion": completion_text,
        "full_input": f"{prompt_text}\n\n{completion_text}",
        "is_valid": True
    }


print("\n--- Step 2: Processing Dataset to SMT-ABC Pairs & Calculating Token Lengths ---")
processed_dataset = dataset.map(
    create_instruction_pair,
    remove_columns=dataset["train"].column_names,
    num_proc=8, # commented out because it was disabling the logging of failed samples
    #load_from_cache_file=False, # needed to be disabled because it was preventing actual processing of the dataset and logging of failed samples
    desc="Converting ABC to SMT-ABC Pairs"
)

MODEL_ID = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
MAX_SEQ_LENGTH = 1536

def add_token_length(example):
    if not example["is_valid"] or len(example["prompt"]) == 0:
        return {"token_length": 0}
    
    tokens = tokenizer(example["full_input"], truncation=False)
    return {"token_length": len(tokens["input_ids"])}

processed_dataset = processed_dataset.map(
    add_token_length,
    num_proc=8,
    desc="Calculating token lengths"
)
valid_count = sum(processed_dataset["train"]["is_valid"])
print(f"Total valid samples before token length filtering: {valid_count}")


filtered_dataset = processed_dataset.filter(
    lambda x: x["is_valid"] and 0 < x["token_length"] <= MAX_SEQ_LENGTH,
    num_proc=8,
    desc=f"Filtering valid sequences <= {MAX_SEQ_LENGTH} tokens"
)

full_ds = filtered_dataset["train"]
print(f"Total valid samples after token length filtering: {len(full_ds)}")

print("\n--- Step 3: Deduplicating on Prompt to Prevent Data Leakage ---")
seen_prompts = set()
unique_indices = []

for idx, prompt in enumerate(full_ds["prompt"]):
    if prompt not in seen_prompts:
        seen_prompts.add(prompt)
        unique_indices.append(idx)

dedup_ds = full_ds.select(unique_indices)
print(f"Total unique samples after deduplication: {len(dedup_ds)}")

print("\n--- Step 4: Splitting into Train (90%), Validation (10%), Test (10%) ---")
train_testvalid = dedup_ds.train_test_split(test_size=0.2, seed=42)
test_valid = train_testvalid["test"].train_test_split(test_size=0.5, seed=42)

split_dict = {
    "train": train_testvalid["train"],
    "validation": test_valid["train"],
    "test": test_valid["test"]
}

print("\n--- Step 5: Saving Leak-Free Splits to Disk ---")
OUTPUT_PATH = os.path.join(PERSISTENT_BASE, "music_abc_splits")
os.makedirs(OUTPUT_PATH, exist_ok=True)
final_splits = DatasetDict(split_dict)
final_splits.save_to_disk(OUTPUT_PATH)
print(f"Done! Clean splits saved to: {OUTPUT_PATH}")

print("\n" + "="*50)
print("FINAL SPLITS STATISTICS")
print("="*50)

for split_name in final_splits.keys():
    split_data = final_splits[split_name]
    num_samples = len(split_data)
    
    if num_samples > 0:
        avg_length = sum(split_data["token_length"]) / num_samples
        print(f"Split: '{split_name}'")
        print(f"  - Total samples: {num_samples:,}")
        print(f"  - Average token length: {avg_length:.2f}")
    else:
        print(f"Split: '{split_name}'")
        print(f"  - Total samples: 0")
    print("-" * 50)


with open("parser_trips.json", "w") as f:
    json.dump(failed_samples_log, f, indent=4)
print("\n[!] Debug log saved to parser_trips.json")