import os
import subprocess
import tempfile
import json
from datasets import load_from_disk

PERSISTENT_BASE = "/home/morg/NLP_2526b/shairotman/NLP_final"
SPLITS_PATH = os.path.join(PERSISTENT_BASE, "music_abc_splits")

dataset = load_from_disk(SPLITS_PATH)
train_ds = dataset["train"].select(range(2))

def transpose_abc_clean(abc_string, shift_value):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.abc', delete=False) as f:
        f.write(abc_string)
        temp_name = f.name
        
    res = subprocess.run(["abc2abc", temp_name, "-t", str(shift_value)], capture_output=True, text=True)
    os.remove(temp_name)
    
    if res.returncode != 0 or not res.stdout:
        return None, f"ERROR: {res.stderr}"
        
    # Strip injected %Error comments
    lines = res.stdout.strip().split('\n')
    clean_lines = []
    for line in lines:
        l = line.split('%')[0].strip()
        if l:
            clean_lines.append(l)
            
    return "\n".join(clean_lines), None

test_results = []

for i, row in enumerate(train_ds):
    prompt_melody = row["prompt"].split("### Primary Melody (SMT-ABC):\n")[-1].strip()
    completion_accomp = row["completion"].split("### Accompaniment Track (SMT-ABC):\n")[-1].strip()
    full_abc = f"{prompt_melody}\n{completion_accomp}"
    
    transposed_text, err = transpose_abc_clean(full_abc, 2)
    
    parsed_mel = ""
    parsed_acc = ""
    
    if transposed_text:
        # Safely split at the exact newline where the second voice track begins
        parts = transposed_text.split('\n[V:2 ')
        if len(parts) == 2:
            parsed_mel = parts[0].strip()
            parsed_acc = f"[V:2 {parts[1].strip()}"
        else:
            parsed_mel = "ERROR: Split failed. Parts count: " + str(len(parts))
            
    test_results.append({
        "sample_id": i,
        "original_smt_abc": full_abc,
        "transposed_raw_clean": transposed_text,
        "final_prompt_melody": parsed_mel,
        "final_completion_accomp": parsed_acc
    })

with open("augmentation_logic_test.json", "w") as f:
    json.dump(test_results, f, indent=4)

print("Saved augmentation_logic_test.json. Check if the tracks separated cleanly and the %Error comments are gone.")