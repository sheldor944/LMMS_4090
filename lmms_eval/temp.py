import json

# ==== CONFIG ====
JSON_FILE_PATH = "./datasets/custom_video_qa/selected_dbfp_dense_videomme_blip_k16_alpha0.5_adaptive_short_2.0_med_3.0_long_5.0_temporal_iter2.json"  # Change this to your file path
# ================
# import json
from collections import defaultdict

# ==== CONFIG ====
# JSON_FILE_PATH = "./videomme_data/your_file.json"  # Change this to your file path
# ================

def main():
    try:
        # Load the JSON file
        with open(JSON_FILE_PATH, 'r') as f:
            data = json.load(f)
        
        # Check if data is a list
        if not isinstance(data, list):
            print("[ERROR] JSON file does not contain a list")
            return
        
        print(f"Found {len(data)} items in the file\n")
        
        # Group by duration
        duration_groups = defaultdict(list)
        for item in data:
            duration = item.get('duration', 'unknown')
            video_id = item.get('video_id')
            if video_id:
                duration_groups[duration].append(video_id)
        
        # Sort video IDs within each duration group
        for duration in duration_groups:
            duration_groups[duration].sort(key=lambda x: int(x) if str(x).isdigit() else x)
        
        # Define order for durations
        duration_order = ['short', 'medium', 'long', 'unknown']
        
        # Print grouped and sorted
        print("=" * 60)
        print("VIDEO IDs SORTED BY DURATION")
        print("=" * 60)
        
        total_count = 0
        for duration in duration_order:
            if duration in duration_groups:
                video_ids = duration_groups[duration]
                total_count += len(video_ids)
                
                print(f"\n{duration.upper()}: ({len(video_ids)} videos)")
                print("-" * 60)
                
                # Print in rows of 10 for readability
                for i in range(0, len(video_ids), 10):
                    chunk = video_ids[i:i+10]
                    print("  " + ", ".join(map(str, chunk)))
        
        print("\n" + "=" * 60)
        print(f"TOTAL: {total_count} videos")
        print("=" * 60)
        
        # Print unique video IDs count
        all_video_ids = [item.get('video_id') for item in data if item.get('video_id')]
        unique_video_ids = set(all_video_ids)
        print(f"\nUnique video IDs: {len(unique_video_ids)}")
        print(f"Total entries: {len(data)}")
        
        if len(all_video_ids) != len(unique_video_ids):
            print(f"Note: Some video IDs appear multiple times (likely different questions)")
        
    except FileNotFoundError:
        print(f"[ERROR] File not found: {JSON_FILE_PATH}")
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON format: {e}")
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()