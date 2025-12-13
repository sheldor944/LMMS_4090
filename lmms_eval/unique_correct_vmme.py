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

def compare_files(file1_path, file2_path):
    """
    Compare two JSON files and find entities that are:
    - Correct in File 1 (pred_answer == answer)
    - Incorrect in File 2 (pred_answer != answer)
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
            return None, None, None, None, None
    
    # Calculate accuracy for both files
    print("=" * 80)
    print("ACCURACY CALCULATION")
    print("=" * 80)
    
    acc1 = calculate_accuracy(data1, "File 1")
    acc2 = calculate_accuracy(data2, "File 2")
    
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    # Create dictionaries for easy lookup using question_id
    file1_dict = {}
    file2_dict = {}
    
    for entry in data1:
        question_id = entry['custom_video_qa_score']['question_id']
        file1_dict[question_id] = entry
    
    for entry in data2:
        question_id = entry['custom_video_qa_score']['question_id']
        file2_dict[question_id] = entry
    
    # Find entities correct in File 1 but incorrect in File 2
    correct_in_file1_incorrect_in_file2 = []
    
    # Find entities correct in File 2 but incorrect in File 1
    correct_in_file2_incorrect_in_file1 = []
    
    # Also track other combinations
    both_correct = 0
    both_incorrect = 0
    
    for question_id, entry1 in file1_dict.items():
        # Check if this entry exists in File 2
        if question_id not in file2_dict:
            continue
        
        entry2 = file2_dict[question_id]
        
        # Get pred_answer and answer from both files
        answer1 = entry1['custom_video_qa_score'].get('answer')
        pred_answer1 = entry1['custom_video_qa_score'].get('pred_answer')
        
        answer2 = entry2['custom_video_qa_score'].get('answer')
        pred_answer2 = entry2['custom_video_qa_score'].get('pred_answer')
        
        file1_correct = (pred_answer1 == answer1)
        file2_correct = (pred_answer2 == answer2)
        
        # Track all combinations
        if file1_correct and file2_correct:
            both_correct += 1
        elif not file1_correct and not file2_correct:
            both_incorrect += 1
        elif file1_correct and not file2_correct:
            correct_in_file1_incorrect_in_file2.append({
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
                'file2_answer': answer2,
                'file2_pred_answer': pred_answer2,
                'file2_pred_full': entry2['custom_video_qa_score'].get('pred_full'),
                'file2_frame_idx': entry2['custom_video_qa_score'].get('frame_idx'),
                'file2_frame_num': entry2['custom_video_qa_score'].get('frame_num'),
                'input': entry1.get('input', '')[:300] + '...'  # First 300 chars of question
            })
        elif not file1_correct and file2_correct:
            correct_in_file2_incorrect_in_file1.append({
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
                'file2_answer': answer2,
                'file2_pred_answer': pred_answer2,
                'file2_pred_full': entry2['custom_video_qa_score'].get('pred_full'),
                'file2_frame_idx': entry2['custom_video_qa_score'].get('frame_idx'),
                'file2_frame_num': entry2['custom_video_qa_score'].get('frame_num'),
                'input': entry1.get('input', '')[:300] + '...'  # First 300 chars of question
            })
    
    # Print comparison statistics
    total_compared = len(file1_dict) if len(file1_dict) <= len(file2_dict) else len(file2_dict)
    print(f"\nComparison Statistics (matched entries):")
    print(f"  Total matched entries: {total_compared}")
    print(f"  Both correct: {both_correct}")
    print(f"  Both incorrect: {both_incorrect}")
    print(f"  File 1 correct, File 2 incorrect: {len(correct_in_file1_incorrect_in_file2)}")
    print(f"  File 2 correct, File 1 incorrect: {len(correct_in_file2_incorrect_in_file1)}")
    
    comparison_stats = {
        'total_matched': total_compared,
        'both_correct': both_correct,
        'both_incorrect': both_incorrect,
        'file1_correct_file2_incorrect': len(correct_in_file1_incorrect_in_file2),
        'file2_correct_file1_incorrect': len(correct_in_file2_incorrect_in_file1)
    }
    
    return correct_in_file1_incorrect_in_file2, correct_in_file2_incorrect_in_file1, acc1, acc2, comparison_stats

def main():
    # Specify your file paths here
    file1_path = './results/full_logs/32_updated_prompt/selected_videomme_clip_frames_k32_aks_20251206_143147_results/..__LLaVA-NeXT-Video-7B-Qwen2/20251206_223152_samples_custom_video_qa.jsonl'
    file2_path = './results/full_logs/32_updated_prompt/selected_frames_clip_videomme_ada_dq_k32_beta0.5_delta3.0_20251206_092025_results/..__LLaVA-NeXT-Video-7B-Qwen2/20251206_172038_samples_custom_video_qa.jsonl'
   
    
    print("Comparing files...")
    print(f"File 1: {file1_path}")
    print(f"File 2: {file2_path}")
    print("-" * 80)
    
    results_f1_correct, results_f2_correct, acc1, acc2, comp_stats = compare_files(file1_path, file2_path)
    
    if results_f1_correct is None:
        return
    
    # Print File 1 correct, File 2 incorrect
    print("\n" + "=" * 80)
    print("DETAILED DIFFERENCES (File 1 Correct ✓, File 2 Incorrect ✗)")
    print("=" * 80)
    
    if results_f1_correct:
        print(f"\nFound {len(results_f1_correct)} entities that are correct in File 1 but incorrect in File 2:\n")
        
        for idx, result in enumerate(results_f1_correct, 1):
            print(f"{idx}. Question ID: {result['question_id']}")
            print(f"   Doc ID: {result['doc_id']}")
            print(f"   Video ID: {result['video_id']}")
            print(f"   Duration: {result['duration']}")
            print(f"   Domain: {result['domain']}")
            print(f"   Sub-category: {result['sub_category']}")
            print(f"   Task Type: {result['task_type']}")
            print(f"   File 1 - Answer: {result['file1_answer']}, Predicted: {result['file1_pred_answer']} ✓")
            print(f"            Full Prediction: {result['file1_pred_full']}")
            print(f"            Frame Indices: {result['file1_frame_idx']}")
            print(f"            Frame Num: {result['file1_frame_num']}")
            print(f"   File 2 - Answer: {result['file2_answer']}, Predicted: {result['file2_pred_answer']} ✗")
            print(f"            Full Prediction: {result['file2_pred_full']}")
            print(f"            Frame Indices: {result['file2_frame_idx']}")
            print(f"            Frame Num: {result['file2_frame_num']}")
            print(f"   Question: {result['input']}")
            print("-" * 80)
    else:
        print("\nNo entities found that are correct in File 1 but incorrect in File 2.")
    
    # Print File 2 correct, File 1 incorrect
    print("\n" + "=" * 80)
    print("DETAILED DIFFERENCES (File 2 Correct ✓, File 1 Incorrect ✗)")
    print("=" * 80)
    
    if results_f2_correct:
        print(f"\nFound {len(results_f2_correct)} entities that are correct in File 2 but incorrect in File 1:\n")
        
        for idx, result in enumerate(results_f2_correct, 1):
            print(f"{idx}. Question ID: {result['question_id']}")
            print(f"   Doc ID: {result['doc_id']}")
            print(f"   Video ID: {result['video_id']}")
            print(f"   Duration: {result['duration']}")
            print(f"   Domain: {result['domain']}")
            print(f"   Sub-category: {result['sub_category']}")
            print(f"   Task Type: {result['task_type']}")
            print(f"   File 1 - Answer: {result['file1_answer']}, Predicted: {result['file1_pred_answer']} ✗")
            print(f"            Full Prediction: {result['file1_pred_full']}")
            print(f"            Frame Indices: {result['file1_frame_idx']}")
            print(f"            Frame Num: {result['file1_frame_num']}")
            print(f"   File 2 - Answer: {result['file2_answer']}, Predicted: {result['file2_pred_answer']} ✓")
            print(f"            Full Prediction: {result['file2_pred_full']}")
            print(f"            Frame Indices: {result['file2_frame_idx']}")
            print(f"            Frame Num: {result['file2_frame_num']}")
            print(f"   Question: {result['input']}")
            print("-" * 80)
    else:
        print("\nNo entities found that are correct in File 2 but incorrect in File 1.")
    
    # Save results to a file
    output_file = 'comparison_results.json'
    summary = {
        'file1_accuracy': acc1,
        'file2_accuracy': acc2,
        'comparison_statistics': comp_stats,
        'file1_correct_file2_incorrect': results_f1_correct,
        'file2_correct_file1_incorrect': results_f2_correct
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n" + "=" * 80)
    print(f"Results saved to: {output_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()