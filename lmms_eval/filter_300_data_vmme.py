import json
import random
import os
import glob
from pathlib import Path
from collections import defaultdict

# ==== CONFIG: CHANGE THESE TO MATCH YOUR DATA ====
INPUT_DIR = "./converted_frames_Videomme_16_optimized_for_parameter_tuning/"  # Directory containing input JSON files
OUTPUT_DIR = "./converted_frames_Videomme_16_optimized_for_parameter_tuning/processed/"  # Directory for output files

# Optional: file pattern to match (e.g., "*.json" or "selected_*.json")
FILE_PATTERN = "*.json"

CATEGORY_KEY = "duration"  # Key for duration category in VideoMME

# Categories for VideoMME: short, medium, long
CATEGORY_VALUES = ["short", "medium", "long"]

# Hardcoded distribution - adjust these numbers as needed
TARGET_PER_CAT = {
    "short": 100,
    "medium": 100,
    "long": 100
}
# Total: 300 (adjust as needed)

SEED = 73321541
# =================================================

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_json(path, data):
    # Ensure the directory exists before saving
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def process_file(input_path, output_path):
    """Process a single JSON file - uses GLOBAL seed so all files get same random selection"""
    print(f"\n{'='*60}")
    print(f"Processing: {input_path}")
    print(f"{'='*60}")
    
    # Reset to the SAME seed for every file - this ensures identical random selection
    random.seed(SEED)
    
    try:
        data = load_json(input_path)
        
        # Handle case where data might not be a list
        if not isinstance(data, list):
            print(f"  [SKIP] File does not contain a list. Skipping.")
            return False

        # Group items by duration category
        buckets = defaultdict(list)
        for item in data:
            cat = item.get(CATEGORY_KEY)
            if cat in CATEGORY_VALUES:
                buckets[cat].append(item)
            else:
                # Handle items without valid duration or with unexpected values
                if cat is not None:
                    print(f"  [WARN] Found unexpected duration value: {cat}")

        # Print category statistics
        print(f"\nCategory distribution in source:")
        total_target = 0
        for cat in CATEGORY_VALUES:
            n = len(buckets[cat])
            target = TARGET_PER_CAT[cat]
            total_target += target
            print(f"  {cat}: found {n} items, target {target}")
            if n < target:
                print(f"    [WARN] Only {n} available, will sample {n} instead of {target}")

        print(f"\nTotal target: {total_target}")

        # Sample from each category (same random selection for all files due to seed reset)
        subset = []
        for cat in CATEGORY_VALUES:
            items = buckets[cat]
            target = TARGET_PER_CAT[cat]
            k = min(target, len(items))
            if k > 0:
                sampled = random.sample(items, k)
                subset.extend(sampled)
                print(f"  Sampled {k} items from '{cat}'")

        # Shuffle the final subset (same shuffle for all files due to seed)
        random.shuffle(subset)
        
        # Save output
        save_json(output_path, subset)

        # Print results
        print(f"\n✓ Saved {len(subset)} examples to {output_path}")
        print(f"Final distribution:")
        for cat in CATEGORY_VALUES:
            count_cat = sum(1 for x in subset if x.get(CATEGORY_KEY) == cat)
            print(f"  {cat}: {count_cat}")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"  [ERROR] Invalid JSON in file: {e}")
        return False
    except Exception as e:
        print(f"  [ERROR] Failed to process file: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # Set global seed
    random.seed(SEED)
    
    # Check if input directory exists
    if not os.path.exists(INPUT_DIR):
        print(f"[ERROR] Input directory does not exist: {INPUT_DIR}")
        return
    
    # Create output directory if it doesn't exist
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        if os.path.exists(OUTPUT_DIR):
            print(f"✓ Output directory ready: {OUTPUT_DIR}")
        else:
            print(f"✓ Created output directory: {OUTPUT_DIR}")
    except Exception as e:
        print(f"[ERROR] Failed to create output directory: {e}")
        return
    
    # Find all JSON files matching the pattern
    search_path = os.path.join(INPUT_DIR, FILE_PATTERN)
    json_files = glob.glob(search_path)
    
    if not json_files:
        print(f"No JSON files found matching pattern: {search_path}")
        return
    
    # Sort files for consistent processing order
    json_files.sort()
    
    print(f"\nFound {len(json_files)} JSON file(s) to process")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"File pattern: {FILE_PATTERN}")
    print(f"\nTarget distribution per file:")
    total_target = sum(TARGET_PER_CAT.values())
    for cat, target in TARGET_PER_CAT.items():
        print(f"  {cat}: {target} samples")
    print(f"Total per file: {total_target}")
    
    # Process each file
    success_count = 0
    failed_count = 0
    
    for input_path in json_files:
        # Get the filename
        filename = os.path.basename(input_path)
        
        # Create output path - keep same name
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        # Process the file (seed is reset inside process_file for consistency)
        if process_file(input_path, output_path):
            success_count += 1
        else:
            failed_count += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total files: {len(json_files)}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"\nAll output files saved to: {OUTPUT_DIR}")
    print(f"\n⚠️  IMPORTANT: All files contain IDENTICAL samples (same random selection)")
    print(f"This ensures fair hyperparameter comparison across different test runs.")
    print(f"Seed used: {SEED}")

if __name__ == "__main__":
    main()