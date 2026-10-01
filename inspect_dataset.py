import os
import re
from datasets import load_from_disk

# 1. Path to the newly processed dataset
DATASET_PATH = "/home/morg/NLP_2526b/hilaetziony/processed_music_abc"

print("=== Checking Processed Dataset ===")

# 2. Verify Dataset Path Exists
if not os.path.exists(DATASET_PATH):
    print(f"Error: Path '{DATASET_PATH}' does not exist!")
    exit(1)

# 3. Load Dataset from Disk
dataset = load_from_disk(DATASET_PATH)
print("Dataset successfully loaded!")
print(f"Dataset structure: {dataset}")
train_split = "train" if "train" in dataset else list(dataset.keys())[0]
total_samples = len(dataset[train_split])
print(f"Total training samples: {total_samples}\n")

if total_samples == 0:
    print("Error: Dataset contains 0 samples.")
    exit(1)

# 4. Inspect Sample 0
sample = dataset[train_split][0]

print("=" * 60)
print("SAMPLE 0 - FULL PROMPT:")
print("=" * 60)
print(sample["prompt"])

print("\n" + "=" * 60)
print("SAMPLE 0 - FULL COMPLETION:")
print("=" * 60)
print(sample["completion"])


# 5. Dedicated Syntax and Integrity Checks
def check_bracket_balance(text: str) -> bool:
    """Verifies that square brackets '[' and ']' are strictly balanced."""
    open_sq = text.count('[')
    close_sq = text.count(']')
    return open_sq == close_sq


def count_smt_bars(text: str, voice_tag: str) -> int:
    """Counts how many structured bar markers exist for a specific voice (e.g., [V:1 B#])."""
    pattern = rf'\[{voice_tag}\s+B\d+\]'
    return len(re.findall(pattern, text))


# Run checks across a batch of 1000 samples
NUM_TO_TEST = min(1000, total_samples)
bracket_errors = 0
bar_mismatch_errors = 0

for i in range(NUM_TO_TEST):
    item = dataset[train_split][i]
    prompt_str = item["prompt"]
    comp_str = item["completion"]

    # Check balanced brackets (Addressing the syntax issue in the image)
    if not check_bracket_balance(prompt_str) or not check_bracket_balance(comp_str):
        bracket_errors += 1

    # Check 1-to-1 measure synchronization between Melody (V:1) and Accompaniment (V:2)
    v1_bars = count_smt_bars(prompt_str, "V:1")
    v2_bars = count_smt_bars(comp_str, "V:2")
    if v1_bars == 0 or v1_bars != v2_bars:
        bar_mismatch_errors += 1

print("\n" + "=" * 60)
print(f"AUTOMATED INTEGRITY AUDIT (First {NUM_TO_TEST} samples):")
print("=" * 60)

v1_sample_bars = count_smt_bars(sample["prompt"], "V:1")
v2_sample_bars = count_smt_bars(sample["completion"], "V:2")

print(f"Sample 0 Melody Bars ([V:1]): {v1_sample_bars}")
print(f"Sample 0 Accompaniment Bars ([V:2]): {v2_sample_bars}")
print(f"Unbalanced Bracket Errors: {bracket_errors} / {NUM_TO_TEST}")
print(f"Bar Synchronization Mismatch Errors: {bar_mismatch_errors} / {NUM_TO_TEST}")

if bracket_errors == 0 and bar_mismatch_errors == 0 and v1_sample_bars > 1:
    print("\nSTATUS: PASSED - Dataset is clean, synchronized, and syntactically valid!")
else:
    print("\nSTATUS: FAILED - Formatting issues detected.")