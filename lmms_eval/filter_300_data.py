import json
import random
from collections import defaultdict

# ==== CONFIG: CHANGE THESE TO MATCH YOUR DATA ====
INPUT_JSON  = "./datasets/custom_video_qa/selected_frames_dbfp_20_alpha_0.75_score_diff_power_law_power_2.0.json"

OUTPUT_JSON = "./datasets/custom_video_qa/selected_frames_dbfp_20_alpha_0.75_score_diff_power_law_power_2.0.json"

CATEGORY_KEY = "duration"  # e.g. "length_cat", "bucket", "type"
CATEGORY_VALUES = ["short", "medium", "long"]
TARGET_PER_CAT = 100
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

    data = load_json(INPUT_JSON)

    buckets = defaultdict(list)
    for item in data:
        cat = item.get(CATEGORY_KEY)
        if cat in CATEGORY_VALUES:
            buckets[cat].append(item)

    for cat in CATEGORY_VALUES:
        n = len(buckets[cat])
        print(f"{cat}: found {n} items")
        if n < TARGET_PER_CAT:
            print(f"  [WARN] Only {n} available, will sample {n} instead of {TARGET_PER_CAT}")

    subset = []
    for cat in CATEGORY_VALUES:
        items = buckets[cat]
        k = min(TARGET_PER_CAT, len(items))
        subset.extend(random.sample(items, k))

    random.shuffle(subset)
    save_json(OUTPUT_JSON, subset)

    print(f"\nSaved {len(subset)} examples to {OUTPUT_JSON}")
    for cat in CATEGORY_VALUES:
        count_cat = sum(1 for x in subset if x.get(CATEGORY_KEY) == cat)
        print(f"  {cat}: {count_cat}")

if __name__ == "__main__":
    main()
