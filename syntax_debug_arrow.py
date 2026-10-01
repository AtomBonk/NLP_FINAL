import os
import shutil
import subprocess
import pretty_midi
from datasets import load_from_disk

ABC2MIDI_PATH = shutil.which("abc2midi")
OUTPUT_DIR = "syntax_debug_arrow_output"
DATASET_PATH = "/home/morg/NLP_2526b/shairotman/NLP_final/music_abc_splits_augmented"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"--- Loading Dataset from {DATASET_PATH} ---")
dataset = load_from_disk(DATASET_PATH)

# Change this to "train" or "validation" to test other splits
split_to_test = "train"
data = dataset[split_to_test]

print(f"\n=== 1. SYNTAX FAILURE LOGS ({split_to_test.upper()} SET) ===")
errors_found = 0
last_mid_file = None

for i, sample in enumerate(data):
    # Extract clean ABC strings directly from the Arrow dataset columns
    prompt_abc = sample["prompt"].split("### Primary Melody (SMT-ABC):\n")[-1].strip()
    gt_abc = sample["completion"].split("### Accompaniment Track (SMT-ABC):\n")[-1].strip()
    full_abc = f"{prompt_abc}\n{gt_abc}"
    
    abc_file = os.path.join(OUTPUT_DIR, f"debug_arrow_sample_{i}.abc")
    mid_file = os.path.join(OUTPUT_DIR, f"debug_arrow_sample_{i}.mid")
    last_mid_file = mid_file
    
    with open(abc_file, "w") as f:
        f.write(full_abc)
        
    res = subprocess.run([ABC2MIDI_PATH, abc_file, "-o", mid_file], capture_output=True, text=True)
    log = res.stderr + "\n" + res.stdout
    
    if "Error in line" in log or "Fatal error" in log:
        print(f"\n[!] SAMPLE {i} FAILED.")
        print(f"--- EXACT ABC2MIDI ERROR ---")
        print(log.strip())
        errors_found += 1
        
        if errors_found >= 5:
            print("\nReached 5 errors. Stopping early.")
            break

if errors_found == 0:
    print(f"All {len(data)} samples in the {split_to_test} set compiled successfully with 0 syntax errors.")

print("\n=== 2. MIDI TRACK ANALYSIS (FAD SUSPICION) ===")
if last_mid_file and os.path.exists(last_mid_file):
    try:
        pm = pretty_midi.PrettyMIDI(last_mid_file)
        print(f"File analyzed: {os.path.basename(last_mid_file)}")
        print(f"Total Instruments in MIDI: {len(pm.instruments)}")
        for idx, inst in enumerate(pm.instruments):
            print(f"  Track {idx} | Name: {inst.name} | Note Count: {len(inst.notes)}")
    except Exception as e:
        print("Could not parse MIDI:", e)
else:
    print("No MIDI files were generated to analyze.")