import json

def load_jsonl(file_path):
    """Load JSONL file (one JSON object per line)"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))
    return data

def load_json(file_path):
    """Load regular JSON file (single JSON object or array)"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_accuracy(data, file_name="File"):
    """Calculate accuracy for a dataset"""
    total = len(data)
    correct = 0
    
    for entry in data:
        answer = entry['lvb_custom_acc'].get('answer')
        parsed_pred = entry['lvb_custom_acc'].get('parsed_pred')
        
        if parsed_pred == answer:
            correct += 1
    
    accuracy = (correct / total * 100) if total > 0 else 0
    
    print(f"\n{file_name} Statistics:")
    print(f"  Total entries: {total}")
    print(f"  Correct predictions: {correct}")
    print(f"  Incorrect predictions: {total - correct}")
    print(f"  Accuracy: {accuracy:.2f}%")
    
    return {
        'total': total,
        'correct': correct,
        'incorrect': total - correct,
        'accuracy': accuracy
    }

def compare_files(file1_path, file2_path):
    """
    Compare two JSON files and find entities that are:
    - Correct in File 1 (parsed_pred == answer)
    - Incorrect in File 2 (parsed_pred != answer)
    """
    
    # Try loading as JSONL first, then as regular JSON
    try:
        data1 = load_jsonl(file1_path)
        data2 = load_jsonl(file2_path)
    except:
        try:
            data1 = load_json(file1_path)
            data2 = load_json(file2_path)
            # If it's a single object, wrap it in a list
            if isinstance(data1, dict):
                data1 = [data1]
            if isinstance(data2, dict):
                data2 = [data2]
        except Exception as e:
            print(f"Error loading files: {e}")
            return None, None, None
    
    # Calculate accuracy for both files
    print("=" * 80)
    print("ACCURACY CALCULATION")
    print("=" * 80)
    
    acc1 = calculate_accuracy(data1, "File 1")
    acc2 = calculate_accuracy(data2, "File 2")
    
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    # Create dictionaries for easy lookup using question ID
    file1_dict = {}
    file2_dict = {}
    
    for entry in data1:
        question_id = list(entry['lvb_custom_acc'].keys())[0] if isinstance(entry['lvb_custom_acc'], dict) else entry['lvb_custom_acc']['id']
        file1_dict[question_id] = entry
    
    for entry in data2:
        question_id = list(entry['lvb_custom_acc'].keys())[0] if isinstance(entry['lvb_custom_acc'], dict) else entry['lvb_custom_acc']['id']
        file2_dict[question_id] = entry
    
    # Find entities correct in File 1 but incorrect in File 2
    correct_in_file1_incorrect_in_file2 = []
    
    # Also track other combinations
    both_correct = 0
    both_incorrect = 0
    correct_in_file2_incorrect_in_file1 = 0
    
    for question_id, entry1 in file1_dict.items():
        # Check if this entry exists in File 2
        if question_id not in file2_dict:
            continue
        
        entry2 = file2_dict[question_id]
        
        # Get parsed_pred and answer from both files
        answer1 = entry1['lvb_custom_acc'].get('answer')
        parsed_pred1 = entry1['lvb_custom_acc'].get('parsed_pred')
        
        answer2 = entry2['lvb_custom_acc'].get('answer')
        parsed_pred2 = entry2['lvb_custom_acc'].get('parsed_pred')
        
        file1_correct = (parsed_pred1 == answer1)
        file2_correct = (parsed_pred2 == answer2)
        
        # Track all combinations
        if file1_correct and file2_correct:
            both_correct += 1
        elif not file1_correct and not file2_correct:
            both_incorrect += 1
        elif file1_correct and not file2_correct:
            correct_in_file1_incorrect_in_file2.append({
                'question_id': question_id,
                'doc_id': entry1.get('doc_id'),
                'file1_answer': answer1,
                'file1_parsed_pred': parsed_pred1,
                'file2_answer': answer2,
                'file2_parsed_pred': parsed_pred2,
                'input': entry1.get('input', '')[:200] + '...'  # First 200 chars of question
            })
        elif not file1_correct and file2_correct:
            correct_in_file2_incorrect_in_file1 += 1
    
    # Print comparison statistics
    total_compared = len(file1_dict) if len(file1_dict) <= len(file2_dict) else len(file2_dict)
    print(f"\nComparison Statistics (matched entries):")
    print(f"  Total matched entries: {total_compared}")
    print(f"  Both correct: {both_correct}")
    print(f"  Both incorrect: {both_incorrect}")
    print(f"  File 1 correct, File 2 incorrect: {len(correct_in_file1_incorrect_in_file2)}")
    print(f"  File 2 correct, File 1 incorrect: {correct_in_file2_incorrect_in_file1}")
    
    comparison_stats = {
        'total_matched': total_compared,
        'both_correct': both_correct,
        'both_incorrect': both_incorrect,
        'file1_correct_file2_incorrect': len(correct_in_file1_incorrect_in_file2),
        'file2_correct_file1_incorrect': correct_in_file2_incorrect_in_file1
    }
    
    return correct_in_file1_incorrect_in_file2, acc1, acc2, comparison_stats

def main():
    # Specify your file paths here
    file1_path = './results/full_logs/Fixed_radius/selected_dbfp_dense_longvideobench_blip_k16_alpha0.85_adaptive_r15_2.0_r60_3.0_r600_5.0_r3600_8.0_temporal_iter1_20251129_202600_results/..__LLaVA-NeXT-Video-7B-Qwen2/20251130_042606_samples_longvideobench_custom.jsonl'
    file2_path = './results/full_logs/Fixed_radius/selected_dbfp_dense_longvideobench_blip_k16_alpha0.85_adaptive_r15_2.0_r60_3.0_r600_5.0_r3600_8.0_score_diff_iter1_20251129_190916_results/..__LLaVA-NeXT-Video-7B-Qwen2/20251130_030922_samples_longvideobench_custom.jsonl'
    
    print("Comparing files...")
    print(f"File 1: {file1_path}")
    print(f"File 2: {file2_path}")
    print("-" * 80)
    
    results, acc1, acc2, comp_stats = compare_files(file1_path, file2_path)
    
    if results is None:
        return
    
    print("\n" + "=" * 80)
    print("DETAILED DIFFERENCES (File 1 Correct, File 2 Incorrect)")
    print("=" * 80)
    
    if results:
        print(f"\nFound {len(results)} entities that are correct in File 1 but incorrect in File 2:\n")
        
        for idx, result in enumerate(results, 1):
            print(f"{idx}. Question ID: {result['question_id']}")
            print(f"   Doc ID: {result['doc_id']}")
            print(f"   File 1 - Answer: {result['file1_answer']}, Predicted: {result['file1_parsed_pred']} ✓")
            print(f"   File 2 - Answer: {result['file2_answer']}, Predicted: {result['file2_parsed_pred']} ✗")
            print(f"   Question: {result['input']}")
            print("-" * 80)
        
        # Save results to a file
        output_file = 'comparison_results.json'
        summary = {
            'file1_accuracy': acc1,
            'file2_accuracy': acc2,
            'comparison_statistics': comp_stats,
            'differences': results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_file}")
        
    else:
        print("\nNo entities found that are correct in File 1 but incorrect in File 2.")

if __name__ == "__main__":
    main()