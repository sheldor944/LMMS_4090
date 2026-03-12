import json
import os


def load_jsonl(filepath):
    """Load a JSONL file (one JSON object per line)."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_json(filepath):
    """Load a standard JSON file (single array or object)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_filtered_resps(results_data):
    """
    Extract filtered_resps keyed by question_id.
    Each entry in results_data should have a 'custom_video_qa_score' dict with 'question_id',
    and a 'filtered_resps' list.
    """
    resps_map = {}
    for entry in results_data:
        # Get question_id from custom_video_qa_score
        score_info = entry.get("custom_video_qa_score", {})
        question_id = score_info.get("question_id")
        if question_id is None:
            # Try doc_id as fallback
            question_id = entry.get("doc_id")
        
        # Get filtered_resps — it's a list, join into a single string
        filtered_resps = entry.get("filtered_resps", [])
        if isinstance(filtered_resps, list):
            resps_text = "\n".join(filtered_resps)
        else:
            resps_text = str(filtered_resps)
        
        resps_map[question_id] = resps_text
    return resps_map


def extract_questions(questions_data):
    """
    Extract question info keyed by question_id.
    Returns dict: question_id -> {question, answer_letter, answer_full, options}
    """
    questions_map = {}
    for entry in questions_data:
        question_id = entry.get("question_id")
        question_text = entry.get("question", "")
        answer_letter = entry.get("answer", "")  # e.g., "C"
        options = entry.get("options", [])
        
        # Convert answer letter to full text
        # Options are like ["A. Apples.", "B. Candles.", "C. Berries.", ...]
        answer_full = answer_letter
        letter_to_index = {"A": 0, "B": 1, "C": 2, "D": 3}
        idx = letter_to_index.get(answer_letter)
        if idx is not None and idx < len(options):
            answer_full = options[idx]
        
        questions_map[question_id] = {
            "question": question_text,
            "answer_letter": answer_letter,
            "answer_full": answer_full,
            "options": options,
        }
    return questions_map


def build_judge_prompt(question_text, answer_full, description_1, description_2):
    """Build the judge prompt string."""
#     prompt = f"""You are a strict, impartial judge evaluating two AI models' outputs for a video understanding benchmark.

# You will be given:
# - The original multiple-choice question
# - The ground-truth correct answer (A, B, C, or D)
# - Description 1 (Part 1 + Part 2 from Model 1)
# - Description 2 (Part 1 + Part 2 from Model 2)

# The descriptions were generated using this exact instruction:
# "generate a detailed, standalone two-part description... 
# Part 1 – Direct Frame Evidence (question-specific): Be exhaustive — list every relevant visual detail...
# Part 2 – Full Domain Coverage: Cover every single aspect, sub-topic, variation, step, component... so that any future question on any part of this domain can be answered from this description alone."

# YOUR ONLY JOB: Decide which description is BETTER overall.

# Evaluation criteria (equal weight):
# 1. Part 1 quality: How exhaustive, precise, and detailed the direct frame evidence is (objects, actions, sequence, timeline, text overlays, labels, diagrams, animations, people, environments, tools, etc.). It must feel like precise field notes that let someone reconstruct the exact scene.
# 2. Part 2 quality: How completely exhaustive the full domain coverage is. It must include every single related aspect, sub-topic, variation, step, component, object, concept, sequence, or idea visible in the frames — even those not mentioned in the question — so that any future question on the entire domain can be answered from it alone.

# Rules:
# - Judge ONLY on depth, detail level, exhaustiveness, and coverage — NOT on whether the description "solves" the question.
# - Use the correct answer only as a reference to check relevance of the details.
# - Be extremely critical: more exhaustive detail in both parts wins. No ties allowed.
# - Do not consider anything details which is completely irrelevant to the question.

# Response format (exactly like this):

# Comparison:
# Part 1 analysis: [2-3 sentences comparing exhaustiveness and precision]
# Part 2 analysis: [2-3 sentences comparing full-domain coverage and future-question readiness]

# Winner: [1 or 2]

# Now judge:

# Question: 
# {question_text}
# Correct Answer: {answer_full}

# Description 1:
# {description_1}

# Description 2:
# {description_2}

# Judge now:"""
    prompt=f"""
You are a strict, impartial judge evaluating two already-generated descriptions from a video understanding benchmark.

You will be given:

- The original multiple-choice question

- The ground-truth correct answer (A, B, C, or D) — used ONLY to define the scene boundary

- Description 1 (Part 1 + Part 2 from Model 1)

- Description 2 (Part 1 + Part 2 from Model 2)

These descriptions were produced using a frame-grounded generation prompt. You MUST assume every statement in both descriptions is already strictly limited to visible frames. Your job is NOT to check grounding again.

YOUR ONLY JOB: Decide which description is BETTER overall, using exactly these two rubrics (equal weight):

Rubric 1 – Question-Specific Details (Part 1)

Evaluate how precise, observable visual details are provided that directly relate to answering the given question. Score higher for:

- More exhaustive listing of objects, actions, people, tools, text, labels, spatial relationships, motion, and timeline elements that are relevant to the question.

- Higher precision and reconstructability of the exact scene for this specific question.

Rubric 2 – Related/Future-Question Coverage (Part 2)

Evaluate how additional visible elements from the same scene/domain(Question topic or related topic) are included that could support future questions on the same topic. Score higher for:

- More exhaustive coverage of other objects, steps, contextual elements, variations, or supporting details inside the exact same scene (even if not needed for the current question).

- Better readiness for follow-up questions without rewatching the video.

Rules:

- Judge ONLY on the two rubrics above — nothing else.

- Use the correct answer only to confirm the scene boundary; 

- ignore anything that is not related to the question at all.

- Be extremely critical: the description with clearly more details on BOTH rubrics wins. No ties allowed.

- Penalise any detail that drifts outside the current question’s scene (as defined by the correct answer).

Response format (exactly like this):

Comparison:

Rubric 1 analysis (question-specific details): [2-3 sentences comparing quantity and precision of details directly tied to the question]

Rubric 2 analysis (related/future-question coverage): [2-3 sentences comparing quantity and usefulness of additional scene elements for future questions]

Winner: [1 or 2]

Now judge:

Question: 

{question_text}

Correct Answer: {answer_full}

Description 1:

{description_1}

Description 2:

{description_2}

Judge now:    
"""
    return prompt


def main():
    # ====== CONFIGURE THESE PATHS ======
    # File containing the results with filtered_resps (JSONL format - one JSON per line)
    model1_results_file = "results/VMME_Description/selected_tmas_videomme_blip_k32_auto_budget_based_curvgrad_normmax_clip95_hybrid_cov1.73_opt_results/..__LLaVA-NeXT-Video-7B-Qwen2/20260311_010103_samples_custom_video_qa.jsonl"
    model2_results_file = "results/VMME_Description/selected_videomme_blip_aks_k32_ratio1_results/..__LLaVA-NeXT-Video-7B-Qwen2/20260311_074722_samples_custom_video_qa.jsonl"
    
    # File containing the questions (standard JSON array)
    questions_file = "datasets/custom_video_qa/selected_tmas_videomme_blip_k32_auto_budget_based_curvgrad_normmax_clip95_hybrid_cov1.73_opt.json"
    
    # Output file for the judge prompts
    output_file = "results/VMME_Description/judge_prompts_updated.jsonl"
    # ====================================

    # Load data
    print(f"Loading Model 1 results from: {model1_results_file}")
    model1_data = load_jsonl(model1_results_file)
    
    print(f"Loading Model 2 results from: {model2_results_file}")
    model2_data = load_jsonl(model2_results_file)
    
    print(f"Loading questions from: {questions_file}")
    questions_data = load_json(questions_file)
    
    # Extract filtered_resps from both models
    model1_resps = extract_filtered_resps(model1_data)
    model2_resps = extract_filtered_resps(model2_data)
    
    # Extract questions
    questions_map = extract_questions(questions_data)
    
    print(f"\nModel 1 has {len(model1_resps)} responses")
    print(f"Model 2 has {len(model2_resps)} responses")
    print(f"Questions file has {len(questions_map)} questions")
    
    # Find common question_ids across all three
    common_ids = set(model1_resps.keys()) & set(model2_resps.keys()) & set(questions_map.keys())
    print(f"Common question IDs: {len(common_ids)}")
    
    # Build judge prompts
    results = []
    for qid in sorted(common_ids):
        q_info = questions_map[qid]
        desc1 = model1_resps[qid]
        desc2 = model2_resps[qid]
        
        prompt = build_judge_prompt(
            question_text=q_info["question"],
            answer_full=q_info["answer_full"],
            description_1=desc1,
            description_2=desc2,
        )
        
        results.append({
            "question_id": qid,
            "question": q_info["question"],
            "answer_letter": q_info["answer_letter"],
            "answer_full": q_info["answer_full"],
            "judge_prompt": prompt,
        })
        
        # Also print to console for verification
        print(f"\n{'='*80}")
        print(f"Question ID: {qid}")
        print(f"{'='*80}")
        print(prompt)
        print(f"{'='*80}\n")
    
    # Save to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"\nSaved {len(results)} judge prompts to: {output_file}")


if __name__ == "__main__":
    main()