import os
import tempfile
import subprocess
from datasets import load_from_disk, Dataset, DatasetDict

PERSISTENT_BASE = "/home/morg/NLP_2526b/shairotman/NLP_final"
SPLITS_PATH = os.path.join(PERSISTENT_BASE, "music_abc_splits")
AUGMENTED_PATH = os.path.join(PERSISTENT_BASE, "music_abc_splits_augmented")

print("--- Loading Clean Splits ---")
dataset = load_from_disk(SPLITS_PATH)

def transpose_abc(abc_string, shift_value):
    """Transposes the ABC string and safely strips injected compiler comments."""
    # Clean out the inline linebreak renderer before compiling as a failsafe
    abc_string = abc_string.replace("I:linebreak $", "")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.abc', delete=False) as f:
        f.write(abc_string)
        temp_name = f.name
        
    res = subprocess.run(["abc2abc", temp_name, "-t", str(shift_value)], capture_output=True, text=True)
    os.remove(temp_name)
    
    if res.returncode != 0 or not res.stdout:
        return None
        
    lines = res.stdout.strip().split('\n')
    clean_lines = []
    for line in lines:
        # Strip the injected %Error comments so abc2midi doesn't mute the track
        l = line.split('%')[0].strip()
        if l:
            clean_lines.append(l)
            
    return "\n".join(clean_lines)

def augment_dataset_split(split_name, split_data):
    print(f"\n--- Augmenting {split_name.capitalize()} Set (-5 to +6 semitones) ---")
    augmented_rows = []

    for idx, row in enumerate(split_data):
        # 1. Always keep the original, unshifted 0-shift sample
        augmented_rows.append(row)
        
        # Extract the raw ABC from the structured prompt
        prompt_melody = row["prompt"].split("### Primary Melody (SMT-ABC):\n")[-1].strip()
        completion_accomp = row["completion"].split("### Accompaniment Track (SMT-ABC):\n")[-1].strip()
        full_abc = f"{prompt_melody}\n{completion_accomp}"
        
        # 2. Transpose into the other 11 keys
        for shift in range(-5, 7):
            if shift == 0: continue
            
            transposed_text = transpose_abc(full_abc, shift)
            if not transposed_text: continue
                
            # Re-split strictly at the exact newline where the second voice track begins
            parts = transposed_text.split('\n[V:2 ')
            if len(parts) != 2: continue
            
            mel = parts[0].strip()
            acc = f"[V:2 {parts[1].strip()}"
            
            caption = "Generate a complementary harmonic accompaniment track for the following melody."
            prompt_text = f"### Instruction:\n{caption}\n\n### Primary Melody (SMT-ABC):\n{mel}"
            completion_text = f"### Accompaniment Track (SMT-ABC):\n{acc}"
            
            augmented_rows.append({
                "prompt": prompt_text,
                "completion": completion_text,
                "full_input": f"{prompt_text}\n\n{completion_text}",
                "is_valid": True,
                "token_length": row["token_length"]
            })
            
    augmented_ds = Dataset.from_list(augmented_rows)
    print(f"Augmented {split_name.capitalize()} Set Size: {len(augmented_ds)} (Original: {len(split_data)})")
    return augmented_ds

# Process Train, Validation, and Test sets through the 12-key loop
augmented_splits = {}
for split_name in dataset.keys():
    augmented_splits[split_name] = augment_dataset_split(split_name, dataset[split_name])

print("\n--- Saving Final Augmented Dataset ---")
final_splits = DatasetDict(augmented_splits)
final_splits.save_to_disk(AUGMENTED_PATH)
print(f"Done! Ready for LLaMA-3 fine-tuning at: {AUGMENTED_PATH}")