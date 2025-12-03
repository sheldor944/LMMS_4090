import json
import random
from collections import defaultdict

# ==== CONFIG: CHANGE THESE TO MATCH YOUR DATA ====
INPUT_FILE = "./converted_selected_longvideobench_16_8_optimized/2_4_7_10.json"  # Single input JSON file
OUTPUT_FILE = "./converted_selected_longvideobench_16_8_optimized/processed/LV_2_4_7_10.json"  # Single output JSON file

CATEGORY_KEY = "duration_group"  # For LongVideoBench (use "duration" for VideoMME)

# Categories (change based on dataset)
CATEGORY_VALUES = [15, 60, 600, 3600]  # For LongVideoBench
# CATEGORY_VALUES = ["short", "medium", "long"]  # For VideoMME

# Hardcoded distribution (total = 300)
TARGET_PER_CAT = {
    15: 10,      
    60: 10,      
    600: 30,    
    3600: 150    
}
# For VideoMME:
# TARGET_PER_CAT = {
#     "short": 100,
#     "medium": 100,
#     "long": 100
# }

SEED = 1337
# =================================================

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def main():
    random.seed(SEED)
    
    print(f"{'='*60}")
    print(f"Processing Single File: {INPUT_FILE}")
    print(f"{'='*60}")
    
    # Load data
    try:
        data = load_json(INPUT_FILE)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {INPUT_FILE}")
        return
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON: {e}")
        return
    
    if not isinstance(data, list):
        print(f"[ERROR] JSON must contain a list of items")
        return
    
    print(f"\nTotal items in file: {len(data)}")
    
    # Group by category
    buckets = defaultdict(list)
    for item in data:
        cat = item.get(CATEGORY_KEY)
        if cat in CATEGORY_VALUES:
            buckets[cat].append(item)
        else:
            print(f"  [WARN] Item with unknown category: {cat}")
    
    # Print category statistics
    print(f"\nCategory distribution:")
    total_target = 0
    for cat in CATEGORY_VALUES:
        n = len(buckets[cat])
        target = TARGET_PER_CAT[cat]
        total_target += target
        print(f"  {cat}: found {n} items, target {target}")
        if n < target:
            print(f"    [WARN] Only {n} available, will sample {n} instead of {target}")
    
    print(f"\nTotal target: {total_target}")
    
    # Sample from each category
    subset = []
    for cat in CATEGORY_VALUES:
        items = buckets[cat]
        target = TARGET_PER_CAT[cat]
        k = min(target, len(items))
        if k > 0:
            sampled = random.sample(items, k)
            subset.extend(sampled)
            print(f"  Sampled {k} items from category {cat}")
    
    # Shuffle the final subset
    random.shuffle(subset)
    
    # Save output
    save_json(OUTPUT_FILE, subset)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"✓ SUCCESS")
    print(f"{'='*60}")
    print(f"Saved {len(subset)} examples to: {OUTPUT_FILE}")
    print(f"\nFinal distribution:")
    for cat in CATEGORY_VALUES:
        count_cat = sum(1 for x in subset if x.get(CATEGORY_KEY) == cat)
        print(f"  {cat}: {count_cat}")

if __name__ == "__main__":
    main()