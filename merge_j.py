import json
import sys

def merge_json_files(file1_path, file2_path, output_path):
    """
    Merges two JSON files containing model results based on the 'prompt' key.
    Combines fields from both files into a single unified record per prompt.
    """
    try:
        with open(file1_path, 'r', encoding='utf-8') as f:
            data1 = json.load(f)
    except Exception as e:
        print(f"Error reading {file1_path}: {e}")
        return

    try:
        with open(file2_path, 'r', encoding='utf-8') as f:
            data2 = json.load(f)
    except Exception as e:
        print(f"Error reading {file2_path}: {e}")
        return

    if isinstance(data1, dict):
        data1 = [data1]
    if isinstance(data2, dict):
        data2 = [data2]

    # Map file2 items by prompt for fast lookup
    file2_dict = {}
    for item in data2:
        prompt = item.get("prompt")
        if prompt:
            file2_dict[prompt] = item

    merged_data = []
    matched_count = 0

    for item1 in data1:
        prompt = item1.get("prompt")
        if not prompt:
            continue

        # Create a combined record starting with item1's data
        combined_item = item1.copy()

        if prompt in file2_dict:
            matched_count += 1
            item2 = file2_dict[prompt]
            # Merge all keys from item2 that aren't already present or are generation outputs
            for key, value in item2.items():
                if key != "prompt":
                    combined_item[key] = value

        merged_data.append(combined_item)

    # Include any items from file2 that weren't in file1
    file1_prompts = {item.get("prompt") for item in data1}
    for item2 in data2:
        if item2.get("prompt") not in file1_prompts:
            merged_data.append(item2)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=4, ensure_ascii=False)

    print(f"Successfully merged {len(merged_data)} records ({matched_count} matched by prompt).")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    file1 = sys.argv[1] if len(sys.argv) > 1 else "zero_shot_baseline_results_NEW.json"
    file2 = sys.argv[2] if len(sys.argv) > 2 else "finetuned_model_train_results.json"
    output = sys.argv[3] if len(sys.argv) > 3 else "merged_results.json"

    merge_json_files(file1, file2, output)