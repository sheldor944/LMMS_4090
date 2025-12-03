import re
import os
import glob
from pathlib import Path

# ==== CONFIG ====
INPUT_DIR = "../results/full_logs/full_fixed_radius/"  # Directory containing your result/log files
OUTPUT_FILE = "./extracted_results_longvideobench.csv"  # Output CSV file
FILE_PATTERN = "*.log"
# ================

def extract_filename_info(filename):
    """Extract setting name from filename, removing timestamp."""
    name = os.path.splitext(filename)[0]
    name = re.sub(r'_results$', '', name)
    name = re.sub(r'_\d{8}_\d{6}$', '', name)
    return name

def extract_overall_accuracy(content):
    """
    Extract overall accuracy from the table.
    Looking for: |longvideobench_custom|...|↑  | 0.59|±  |   N/A|
    Returns value multiplied by 100
    """
    patterns = [
        r'\|longvideobench_custom\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|\s*↑\s*\|\s*(\d+\.?\d*)\s*\|',
        r'lvb_custom_acc\|↑\s*\|\s*(\d+\.?\d*)\s*\|',
        r'\|\s*↑\s*\|\s*(\d+\.?\d*)\s*\|\s*±',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            value = float(match.group(1))
            return value * 100  # Convert to percentage
    return None
def extract_duration_accuracies(content):
    """
    Extract accuracies by duration group using simple string search.
    Looking for lines like: "15s                           : 65.00% (40 samples)"
    """
    results = {}
    
    # Split content into lines
    lines = content.split('\n')
    
    # Different patterns for different durations (they have different spacing!)
    duration_patterns = {
        '15s': '15s                           : ',
        '60s': '60s                           : ',
        '600s': '600s                          : ',  # One less space
        '3600s': '3600s                         : '  # Two less spaces
    }
    
    for line in lines:
        for duration, search_pattern in duration_patterns.items():
            if search_pattern in line:
                # Find position after the pattern
                idx = line.find(search_pattern) + len(search_pattern)
                rest = line[idx:]
                
                # Extract percentage (everything before %)
                percent_idx = rest.find('%')
                if percent_idx > 0:
                    percentage = rest[:percent_idx].strip()
                    
                    results[duration] = {
                        'percentage': percentage
                    }
    
    return results

def process_file(filepath):
    """Process a single result file and extract all information."""
    filename = os.path.basename(filepath)
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        setting_name = extract_filename_info(filename)
        overall_accuracy = extract_overall_accuracy(content)
        duration_accuracies = extract_duration_accuracies(content)
        
        return {
            'setting_name': setting_name,
            'overall_accuracy': overall_accuracy,
            'duration_accuracies': duration_accuracies,
            'filename': filename
        }
    
    except Exception as e:
        print(f"  [ERROR] {filename}: {e}")
        return None

def format_results(results_list):
    """Format results into CSV rows."""
    header = "Setting Name,Overall Accuracy,15s %,60s %,600s %,3600s %\n"
    
    rows = [header]
    
    for result in results_list:
        if result is None:
            continue
        
        setting = result['setting_name']
        overall = f"{result['overall_accuracy']:.2f}" if result['overall_accuracy'] else 'N/A'
        
        duration_acc = result['duration_accuracies']
        
        # Extract percentages
        pct_15s = duration_acc.get('15s', {}).get('percentage', 'N/A')
        pct_60s = duration_acc.get('60s', {}).get('percentage', 'N/A')
        pct_600s = duration_acc.get('600s', {}).get('percentage', 'N/A')
        pct_3600s = duration_acc.get('3600s', {}).get('percentage', 'N/A')
        
        row = f"{setting},{overall},{pct_15s},{pct_60s},{pct_600s},{pct_3600s}\n"
        rows.append(row)
    
    return ''.join(rows)

def main():
    print(f"Searching for files in: {INPUT_DIR}")
    print(f"Pattern: {FILE_PATTERN}\n")
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    search_path = os.path.join(INPUT_DIR, FILE_PATTERN)
    result_files = glob.glob(search_path)
    
    if not result_files:
        print(f"No files found matching pattern: {search_path}")
        return
    
    print(f"Found {len(result_files)} file(s) to process\n")
    
    all_results = []
    success_count = 0
    
    for filepath in sorted(result_files):
        result = process_file(filepath)
        
        if result and (result['overall_accuracy'] or result['duration_accuracies']):
            all_results.append(result)
            success_count += 1
    
    if all_results:
        csv_content = format_results(all_results)
        
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write(csv_content)
        
        print(f"\n{'='*60}")
        print(f"Results saved to: {OUTPUT_FILE}")
        print(f"Successfully extracted: {success_count}/{len(result_files)}")
        print(f"{'='*60}\n")
        
        lines = csv_content.split('\n')
        print("Preview (first 10 rows):")
        for line in lines[:11]:
            print(line)
        
        if len(lines) > 11:
            print(f"... and {len(lines)-11} more rows")
    else:
        print("No results extracted!")

if __name__ == "__main__":
    main()