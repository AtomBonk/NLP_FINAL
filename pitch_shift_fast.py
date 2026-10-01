import os
import subprocess
import tempfile
import json
from datasets import load_from_disk

PERSISTENT_BASE = "/home/morg/NLP_2526b/shairotman/NLP_final"
SPLITS_PATH = os.path.join(PERSISTENT_BASE, "music_abc_splits")

dataset = load_from_disk(SPLITS_PATH)
train_ds = dataset["train"].select(range(3))

test_results = []

for i, row in enumerate(train_ds):
    # Reconstruct the SMT-ABC string just as it appears in your dataset
    prompt_melody = row["prompt"].split("### Primary Melody (SMT-ABC):\n")[-1].strip()
    completion_accomp = row["completion"].split("### Accompaniment Track (SMT-ABC):\n")[-1].strip()
    
    full_abc = f"{prompt_melody}\n{completion_accomp}"
    
    # Write to a temporary file for abc2abc
    with tempfile.NamedTemporaryFile(mode='w', suffix='.abc', delete=False) as f:
        f.write(full_abc)
        temp_name = f.name
        
    # Transpose up 2 semitones
    res = subprocess.run(["abc2abc", temp_name, "-t", "2"], capture_output=True, text=True)
    os.remove(temp_name)
    
    test_results.append({
        "sample_id": i,
        "original_smt_abc": full_abc,
        "transposed_output": res.stdout if res.returncode == 0 else f"ERROR: {res.stderr}"
    })

# Dump to JSON for visual inspection
with open("transpose_test_output.json", "w") as f:
    json.dump(test_results, f, indent=4)

print("Saved transpose_test_output.json. Please inspect the B# tags in the transposed_output.")