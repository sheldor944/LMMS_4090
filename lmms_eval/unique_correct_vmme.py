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
        answer = entry['custom_video_qa_score'].get('answer')
        pred_answer = entry['custom_video_qa_score'].get('pred_answer')
        
        if pred_answer == answer:
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

def get_question_id(entry):
    """Extract question ID from entry"""
    # The ID is stored in custom_video_qa_score['question_id']
    return entry['custom_video_qa_score'].get('question_id')

def compare_files(file1_path, file2_path):
    """
    Compare two JSON files and find entities based on correctness
    """
    
    # Try loading as JSONL first, then as regular JSON
    try:
        data1 = load_jsonl(file1_path)
        print(f"Loaded File 1 as JSONL: {len(data1)} entries")
    except Exception as e:
        print(f"Failed to load File 1 as JSONL: {e}")
        try:
            data1 = load_json(file1_path)
            if isinstance(data1, dict):
                data1 = [data1]
            print(f"Loaded File 1 as JSON: {len(data1)} entries")
        except Exception as e:
            print(f"Error loading File 1: {e}")
            return None, None, None, None
    
    try:
        data2 = load_jsonl(file2_path)
        print(f"Loaded File 2 as JSONL: {len(data2)} entries")
    except Exception as e:
        print(f"Failed to load File 2 as JSONL: {e}")
        try:
            data2 = load_json(file2_path)
            if isinstance(data2, dict):
                data2 = [data2]
            print(f"Loaded File 2 as JSON: {len(data2)} entries")
        except Exception as e:
            print(f"Error loading File 2: {e}")
            return None, None, None, None
    
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
        question_id = get_question_id(entry)
        if question_id:
            file1_dict[question_id] = entry
        else:
            # Fallback to doc_id if no question id
            file1_dict[entry.get('doc_id')] = entry
    
    for entry in data2:
        question_id = get_question_id(entry)
        if question_id:
            file2_dict[question_id] = entry
        else:
            file2_dict[entry.get('doc_id')] = entry
    
    print(f"\nFile 1 unique IDs: {len(file1_dict)}")
    print(f"File 2 unique IDs: {len(file2_dict)}")
    
    # Find all common IDs
    common_ids = set(file1_dict.keys()) & set(file2_dict.keys())
    print(f"Common IDs: {len(common_ids)}")
    
    # Track all combinations
    both_correct = []
    both_incorrect = []
    correct_in_file1_incorrect_in_file2 = []
    correct_in_file2_incorrect_in_file1 = []
    
    for question_id in common_ids:
        entry1 = file1_dict[question_id]
        entry2 = file2_dict[question_id]
        
        # Get pred_answer and answer from both files
        answer1 = entry1['custom_video_qa_score'].get('answer')
        pred_answer1 = entry1['custom_video_qa_score'].get('pred_answer')
        
        answer2 = entry2['custom_video_qa_score'].get('answer')
        pred_answer2 = entry2['custom_video_qa_score'].get('pred_answer')
        
        file1_correct = (pred_answer1 == answer1)
        file2_correct = (pred_answer2 == answer2)
        
        # Get the full input (question with options)
        full_input = entry1.get('input', '')
        
        result_entry = {
            'question_id': question_id,
            'doc_id': entry1.get('doc_id'),
            'video_id': entry1['custom_video_qa_score'].get('video_id'),
            'duration': entry1['custom_video_qa_score'].get('duration'),
            'domain': entry1['custom_video_qa_score'].get('domain'),
            'sub_category': entry1['custom_video_qa_score'].get('sub_category'),
            'task_type': entry1['custom_video_qa_score'].get('task_type'),
            'file1_answer': answer1,
            'file1_pred_answer': pred_answer1,
            'file1_pred_full': entry1['custom_video_qa_score'].get('pred_full'),
            'file1_frame_idx': entry1['custom_video_qa_score'].get('frame_idx'),
            'file1_frame_num': entry1['custom_video_qa_score'].get('frame_num'),
            'file1_correct': file1_correct,
            'file2_answer': answer2,
            'file2_pred_answer': pred_answer2,
            'file2_pred_full': entry2['custom_video_qa_score'].get('pred_full'),
            'file2_frame_idx': entry2['custom_video_qa_score'].get('frame_idx'),
            'file2_frame_num': entry2['custom_video_qa_score'].get('frame_num'),
            'file2_correct': file2_correct,
            'full_question': full_input  # Full question with all options
        }
        
        # Track all combinations
        if file1_correct and file2_correct:
            both_correct.append(result_entry)
        elif not file1_correct and not file2_correct:
            both_incorrect.append(result_entry)
        elif file1_correct and not file2_correct:
            correct_in_file1_incorrect_in_file2.append(result_entry)
        elif not file1_correct and file2_correct:
            correct_in_file2_incorrect_in_file1.append(result_entry)
    
    # Print comparison statistics
    print(f"\nComparison Statistics (matched entries):")
    print(f"  Total matched entries: {len(common_ids)}")
    print(f"  Both correct: {len(both_correct)}")
    print(f"  Both incorrect: {len(both_incorrect)}")
    print(f"  File 1 correct, File 2 incorrect: {len(correct_in_file1_incorrect_in_file2)}")
    print(f"  File 2 correct, File 1 incorrect: {len(correct_in_file2_incorrect_in_file1)}")
    
    comparison_stats = {
        'total_matched': len(common_ids),
        'both_correct': len(both_correct),
        'both_incorrect': len(both_incorrect),
        'file1_correct_file2_incorrect': len(correct_in_file1_incorrect_in_file2),
        'file2_correct_file1_incorrect': len(correct_in_file2_incorrect_in_file1)
    }
    
    all_results = {
        'both_correct': both_correct,
        'both_incorrect': both_incorrect,
        'file1_correct_file2_incorrect': correct_in_file1_incorrect_in_file2,
        'file2_correct_file1_incorrect': correct_in_file2_incorrect_in_file1
    }
    
    return all_results, acc1, acc2, comparison_stats

def print_detailed_results(results, category, limit=10):
    """Print detailed results for a category"""
    items = results.get(category, [])
    print(f"\n{'=' * 80}")
    print(f"{category.upper().replace('_', ' ')} ({len(items)} items)")
    print("=" * 80)
    
    if not items:
        print("No items in this category.")
        return
    
    for idx, result in enumerate(items[:limit], 1):
        print(f"\n{idx}. Question ID: {result['question_id']}")
        print(f"   Doc ID: {result['doc_id']}")
        print(f"   Video ID: {result['video_id']}")
        print(f"   Duration: {result.get('duration', 'N/A')}")
        print(f"   Domain: {result.get('domain', 'N/A')}")
        print(f"   Sub-category: {result.get('sub_category', 'N/A')}")
        print(f"   Task Type: {result.get('task_type', 'N/A')}")
        print(f"   File 1 - Answer: {result['file1_answer']}, Predicted: {result['file1_pred_answer']} {'✓' if result['file1_correct'] else '✗'}")
        print(f"            Full Prediction: {result.get('file1_pred_full', 'N/A')}")
        print(f"            Frame Num: {result.get('file1_frame_num', 'N/A')}")
        print(f"   File 2 - Answer: {result['file2_answer']}, Predicted: {result['file2_pred_answer']} {'✓' if result['file2_correct'] else '✗'}")
        print(f"            Full Prediction: {result.get('file2_pred_full', 'N/A')}")
        print(f"            Frame Num: {result.get('file2_frame_num', 'N/A')}")
        # Print truncated question for console
        full_q = result.get('full_question', '')
        truncated_q = full_q[:200] + '...' if len(full_q) > 200 else full_q
        print(f"   Question (truncated): {truncated_q}")
        print("-" * 80)
    
    if len(items) > limit:
        print(f"\n... and {len(items) - limit} more items")

def main():
    # Specify your file paths here
    file1_path = './results/full_logs/FINAL_VMME/selected_tmas_videomme_blip_k32_auto_half_life_curvseco_normmax_clip95_hybrid_cov1.73_opt_20260110_203342_results/..__LLaVA-NeXT-Video-7B-Qwen2/20260111_043349_samples_custom_video_qa.jsonl'
    file2_path = './results/full_logs/FINAL_VMME/selected_videomme_blip_aks_k32_ratio1_20260121_112302_results/..__LLaVA-NeXT-Video-7B-Qwen2/20260121_192309_samples_custom_video_qa.jsonl'
    
    print("Comparing files...")
    print(f"File 1: {file1_path}")
    print(f"File 2: {file2_path}")
    print("-" * 80)
    
    results, acc1, acc2, comp_stats = compare_files(file1_path, file2_path)
    
    if results is None:
        return
    
    # Print detailed results for each category
    print_detailed_results(results, 'file1_correct_file2_incorrect', limit=10)
    print_detailed_results(results, 'file2_correct_file1_incorrect', limit=10)
    print_detailed_results(results, 'both_correct', limit=5)
    print_detailed_results(results, 'both_incorrect', limit=5)
    
    # Save results to a file with FULL questions
    output_file = 'comparison_AKS_TMAS_half_life_32_blip_VideoMME_THESIS.json'
    summary = {
        'file1_path': file1_path,
        'file2_path': file2_path,
        'file1_accuracy': acc1,
        'file2_accuracy': acc2,
        'comparison_statistics': comp_stats,
        'file1_correct_file2_incorrect': results['file1_correct_file2_incorrect'],
        'file2_correct_file1_incorrect': results['file2_correct_file1_incorrect'],
        'both_correct': results['both_correct'],
        'both_incorrect': results['both_incorrect']
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n\nFull results saved to: {output_file}")
    
    # Print summary of what's in the output file
    print(f"\nOutput file contains:")
    print(f"  - File paths")
    print(f"  - Accuracy statistics for both files")
    print(f"  - Comparison statistics")
    print(f"  - {len(results['file1_correct_file2_incorrect'])} entries where File 1 correct, File 2 incorrect (with FULL questions)")
    print(f"  - {len(results['file2_correct_file1_incorrect'])} entries where File 2 correct, File 1 incorrect (with FULL questions)")
    print(f"  - {len(results['both_correct'])} entries where both correct (with FULL questions)")
    print(f"  - {len(results['both_incorrect'])} entries where both incorrect (with FULL questions)")

if __name__ == "__main__":
    main()