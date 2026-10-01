from datasets import load_from_disk
import os

# 1. Load the Arrow dataset from disk
PERSISTENT_BASE = "/home/morg/NLP_2526b/shairotman/NLP_final"
OUTPUT_PATH = os.path.join(PERSISTENT_BASE, "music_abc_splits_augmented")
dataset = load_from_disk(OUTPUT_PATH)

# 2. Export the splits to human-readable JSON Lines files
dataset["train"].to_json("train_split.json")
dataset["validation"].to_json("validation_split.json")
dataset["test"].to_json("test_split.json")
print("JSON exports complete.")

# 3. Generate the individual .abc files for your syntax checker
train_data = dataset["train"]
os.makedirs("debug_abc_files", exist_ok=True)

# Generate debug files for the first 20 samples
for i, sample in enumerate(train_data.select(range(20))):
    # Note: Using "completion" key as defined in our pipeline, 
    # replacing your old "ground_truth_completion" key
    prompt_abc = sample["prompt"].split("### Primary Melody (SMT-ABC):")[-1].strip()
    gt_abc = sample["completion"].split("### Accompaniment Track (SMT-ABC):")[-1].strip()
    full_abc = f"{prompt_abc}\n{gt_abc}"
    
    filename = os.path.join("debug_abc_files", f"debug_gt_sample_{i}.abc")
    with open(filename, "w") as f:
        f.write(full_abc)

print("Generated 20 individual .abc debug files in /debug_abc_files")