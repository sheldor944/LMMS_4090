import re
import os
import glob
from pathlib import Path

# ==== CONFIG ====
INPUT_DIR = "../results/full_logs/300_runs/"  # Directory containing your result/log files
OUTPUT_FILE = "./extracted_results_videomme_300.csv"  # Output CSV file
FILE_PATTERN = "*.log"
# ================

def extract_filename_info(filename):
    """Extract setting name from filename, removing timestamp."""
    name = os.path.splitext(filename)[0]
    name = re.sub(r'_results$', '', name)
    name = re.sub(r'_\d{8}_\d{6}$', '', name)
    return name

def extract_overall_accuracy(content):
    """Extract overall accuracy from the table."""
    patterns = [
        r'\|custom_video_qa\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|\s*↑\s*\|\s*(\d+(?:\.\d+)?)\s*\|',
        r'custom_video_qa_score\|↑\s*\|\s*(\d+(?:\.\d+)?)\s*\|',
        r'\|\s*↑\s*\|\s*(\d+(?:\.\d+)?)\s*\|\s*±',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            return match.group(1)
    return None

def extract_length_accuracies(content):
    """
    Extract accuracies by exact string search - SIMPLE VERSION
    """
    results = {}
    
    # Split content into lines
    lines = content.split('\n')
    
    for line in lines:
        # Look for exact string "short                         : "
        if 'short                         : ' in line:
            idx = line.find('short                         : ') + len('short                         : ')
            # Just take next characters until we hit '%'
            rest = line[idx:]
            percent_idx = rest.find('%')
            if percent_idx > 0:
                percentage = rest[:percent_idx].strip()
                results['short'] = {
                    'percentage': percentage,
                    'correct': 'N/A',
                    'total': 'N/A'
                }
        
        # Look for exact string "medium                        : "
        if 'medium                        : ' in line:
            idx = line.find('medium                        : ') + len('medium                        : ')
            rest = line[idx:]
            percent_idx = rest.find('%')
            if percent_idx > 0:
                percentage = rest[:percent_idx].strip()
                results['medium'] = {
                    'percentage': percentage,
                    'correct': 'N/A',
                    'total': 'N/A'
                }
        
        # Look for exact string "long                          : "
        if 'long                          : ' in line:
            idx = line.find('long                          : ') + len('long                          : ')
            rest = line[idx:]
            percent_idx = rest.find('%')
            if percent_idx > 0:
                percentage = rest[:percent_idx].strip()
                results['long'] = {
                    'percentage': percentage,
                    'correct': 'N/A',
                    'total': 'N/A'
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
        length_accuracies = extract_length_accuracies(content)
        
        return {
            'setting_name': setting_name,
            'overall_accuracy': overall_accuracy,
            'length_accuracies': length_accuracies,
            'filename': filename
        }
    
    except Exception as e:
        print(f"  [ERROR] {filename}: {e}")
        return None

def format_results(results_list):
    """Format results into CSV rows."""
    header = "Setting Name,Overall Accuracy,Short %,Medium %,Long %,Short (correct/total),Medium (correct/total),Long (correct/total)\n"
    
    rows = [header]
    
    for result in results_list:
        if result is None:
            continue
        
        setting = result['setting_name']
        overall = result['overall_accuracy'] if result['overall_accuracy'] else 'N/A'
        
        length_acc = result['length_accuracies']
        
        short_pct = length_acc.get('short', {}).get('percentage', 'N/A')
        medium_pct = length_acc.get('medium', {}).get('percentage', 'N/A')
        long_pct = length_acc.get('long', {}).get('percentage', 'N/A')
        
        short_frac = f"{length_acc.get('short', {}).get('correct', 'N/A')}/{length_acc.get('short', {}).get('total', 'N/A')}"
        medium_frac = f"{length_acc.get('medium', {}).get('correct', 'N/A')}/{length_acc.get('medium', {}).get('total', 'N/A')}"
        long_frac = f"{length_acc.get('long', {}).get('correct', 'N/A')}/{length_acc.get('long', {}).get('total', 'N/A')}"
        
        row = f"{setting},{overall},{short_pct},{medium_pct},{long_pct}\n"
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
        
        if result and (result['overall_accuracy'] or result['length_accuracies']):
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