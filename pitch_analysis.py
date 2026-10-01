from datasets import load_from_disk
import re
from collections import Counter
import os

PERSISTENT_BASE = "/home/morg/NLP_2526b/shairotman/NLP_final"
SPLITS_PATH = os.path.join(PERSISTENT_BASE, "music_abc_splits")

print("--- Analyzing Key Signatures for Pitch-Shifted Clones ---")
dataset = load_from_disk(SPLITS_PATH)

all_keys = []
for split in dataset.keys():
    for row in dataset[split]:
        # Extract the K: header from the prompt
        match = re.search(r'\nK:(.*)', row["prompt"])
        if match:
            all_keys.append(match.group(1).strip())

key_counts = Counter(all_keys)
print(f"Total samples analyzed: {len(all_keys)}")
print("\nKey Signature Distribution:")
for k, v in key_counts.most_common():
    print(f"Key K:{k:<5} -> {v} samples")