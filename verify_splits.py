import os
import re
from datasets import load_from_disk

SPLITS_PATH = "/home/morg/NLP_2526b/hilaetziony/music_abc_splits_augmented"

print("=== 1. Loading and Verifying Splits Structure ===")
splits = load_from_disk(SPLITS_PATH)
print(splits)

train_len = len(splits["train"])
val_len = len(splits["validation"])
test_len = len(splits["test"])
total_len = train_len + val_len + test_len

print(f"\nCounts Summary:")
print(f"Train:      {train_len:,} ({train_len/total_len*100:.1f}%)")
print(f"Validation: {val_len:,} ({val_len/total_len*100:.1f}%)")
print(f"Test:       {test_len:,} ({test_len/total_len*100:.1f}%)")
print(f"Total:      {total_len:,}")

assert total_len == 49534, f"Error: Total sample count {total_len} does not match expected 49,534 unique samples!"

print("\n=== 2. Checking for Data Leakage (Disjoint Sets) ===")
train_prompts = set(splits["train"]["prompt"])
val_prompts = set(splits["validation"]["prompt"])
test_prompts = set(splits["test"]["prompt"])

train_val_overlap = train_prompts.intersection(val_prompts)
train_test_overlap = train_prompts.intersection(test_prompts)
val_test_overlap = val_prompts.intersection(test_prompts)

print(f"Train vs Validation Overlap: {len(train_val_overlap)}")
print(f"Train vs Test Overlap:       {len(train_test_overlap)}")
print(f"Validation vs Test Overlap: {len(val_test_overlap)}")

if len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0:
    print("STATUS: PASSED - Strict zero data leakage confirmed across all splits!")
else:
    print("WARNING: Data overlap detected between splits.")


print("\n=== 3. Quality Audit on Test and Validation Splits ===")
def audit_split(split_name, split_data):
    bracket_errors = 0
    sync_errors = 0
    
    for item in split_data:
        p = item["prompt"]
        c = item["completion"]
        
        # Check balanced brackets
        if p.count('[') != p.count(']') or c.count('[') != c.count(']'):
            bracket_errors += 1
            
        # Check SMT Bar synchronization
        v1_bars = len(re.findall(r'\[V:1\s+B\d+\]', p))
        v2_bars = len(re.findall(r'\[V:2\s+B\d+\]', c))
        if v1_bars == 0 or v1_bars != v2_bars:
            sync_errors += 1
            
    print(f"[{split_name.upper()}] Bracket Errors: {bracket_errors} | Sync Errors: {sync_errors}")
    return bracket_errors == 0 and sync_errors == 0

val_ok = audit_split("validation", splits["validation"])
test_ok = audit_split("test", splits["test"])

if val_ok and test_ok and len(train_val_overlap) == 0:
    print("\nOVERALL STATUS: ALL SPLITS CLEAN, SYNCHRONIZED, AND READY FOR BASELINE & TRAINING!")