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
    Tries multiple strategies to find the question_id.
    """
    resps_map = {}
    for idx, entry in enumerate(results_data):
        question_id = None

        # Strategy 1: Check custom_video_qa_score -> question_id
        score_info = entry.get("custom_video_qa_score", {})
        if isinstance(score_info, dict):
            question_id = score_info.get("question_id")

        # Strategy 2: Check doc -> id (matches the "id" field in questions JSON like "86CxyhFV9MI_0")
        if question_id is None:
            doc = entry.get("doc", {})
            if isinstance(doc, dict):
                question_id = doc.get("id")

        # Strategy 3: Check top-level doc_id
        if question_id is None:
            question_id = entry.get("doc_id")

        # Strategy 4: Check top-level question_id
        if question_id is None:
            question_id = entry.get("question_id")

        # Strategy 5: Check top-level id
        if question_id is None:
            question_id = entry.get("id")

        # Get filtered_resps
        filtered_resps = entry.get("filtered_resps", [])
        if isinstance(filtered_resps, list):
            resps_text = "\n".join(str(r) for r in filtered_resps)
        else:
            resps_text = str(filtered_resps)

        if question_id is not None:
            resps_map[question_id] = resps_text
        else:
            # Last resort: use the line index
            resps_map[idx] = resps_text

    return resps_map


def extract_questions(questions_data):
    """
    Extract question info keyed by question id.
    Returns dict: question_id -> {question, answer_index, answer_full, candidates, video_id, ...}
    """
    questions_map = {}
    for idx, entry in enumerate(questions_data):
        question_id = entry.get("id")
        question_text = entry.get("question", "")
        correct_choice = entry.get("correct_choice")
        candidates = entry.get("candidates", [])
        video_id = entry.get("video_id", "")
        video_path = entry.get("video_path", "")
        subtitle_path = entry.get("subtitle_path", "")
        question_category = entry.get("question_category", "")
        level = entry.get("level", "")
        topic_category = entry.get("topic_category", "")
        position = entry.get("position", [])
        frame_idx = entry.get("frame_idx", [])
        duration = entry.get("duration", 0)

        index_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}
        answer_letter = index_to_letter.get(correct_choice, "")

        answer_full = ""
        if correct_choice is not None and correct_choice < len(candidates):
            answer_full = candidates[correct_choice]

        info = {
            "question": question_text,
            "answer_letter": answer_letter,
            "answer_index": correct_choice,
            "answer_full": answer_full,
            "candidates": candidates,
            "video_id": video_id,
            "video_path": video_path,
            "subtitle_path": subtitle_path,
            "question_category": question_category,
            "level": level,
            "topic_category": topic_category,
            "position": position,
            "frame_idx": frame_idx,
            "duration": duration,
        }

        # Store under the string id
        if question_id is not None:
            questions_map[question_id] = info
        # Also store under the numeric index for fallback matching
        questions_map[idx] = info

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


def debug_print_keys(model1_resps, model2_resps, questions_map):
    """Print sample keys from each source for debugging."""
    m1_keys = list(model1_resps.keys())[:5]
    m2_keys = list(model2_resps.keys())[:5]
    q_keys = list(questions_map.keys())[:10]

    print(f"\n--- DEBUG: Key samples ---")
    print(f"Model 1 keys (first 5): {m1_keys}")
    print(f"  types: {[type(k).__name__ for k in m1_keys]}")
    print(f"Model 2 keys (first 5): {m2_keys}")
    print(f"  types: {[type(k).__name__ for k in m2_keys]}")
    print(f"Questions keys (first 10): {q_keys}")
    print(f"  types: {[type(k).__name__ for k in q_keys]}")
    print(f"--- END DEBUG ---\n")


def match_by_index(model1_data, model2_data, questions_data, questions_map):
    """
    Fallback: match model results to questions by line order (index).
    """
    print("Using index-based (order) matching as fallback...")
    matched = []

    min_len = min(len(model1_data), len(model2_data), len(questions_data))
    print(f"Will match up to {min_len} entries by order.")

    for idx in range(min_len):
        # Extract response from model 1
        entry1 = model1_data[idx]
        filtered_resps1 = entry1.get("filtered_resps", [])
        if isinstance(filtered_resps1, list):
            desc1 = "\n".join(str(r) for r in filtered_resps1)
        else:
            desc1 = str(filtered_resps1)

        # Extract response from model 2
        entry2 = model2_data[idx]
        filtered_resps2 = entry2.get("filtered_resps", [])
        if isinstance(filtered_resps2, list):
            desc2 = "\n".join(str(r) for r in filtered_resps2)
        else:
            desc2 = str(filtered_resps2)

        # Get question info by index
        q_entry = questions_data[idx]
        q_id = q_entry.get("id", str(idx))
        q_info = questions_map.get(q_id, questions_map.get(idx))

        if q_info is None:
            print(f"  WARNING: No question info found for index {idx}, skipping.")
            continue

        matched.append({
            "question_id": q_id,
            "q_info": q_info,
            "desc1": desc1,
            "desc2": desc2,
        })

    return matched


def main():
    # ====== CONFIGURE THESE PATHS ======
    model1_results_file = "results/LV_Description/selected_tmas_longvideobench_blip_k32_auto_budget_based_curvgrad_normmax_clip95_hybrid_cov1.73_opt_results/..__LLaVA-NeXT-Video-7B-Qwen2/20260312_164756_samples_longvideobench_custom.jsonl"
    model2_results_file = "results/LV_Description/selected_longvideobench_blip_aks_k32_ratio1_results/..__LLaVA-NeXT-Video-7B-Qwen2/20260312_072348_samples_longvideobench_custom.jsonl"
    questions_file = "datasets/longvideobench/selected_tmas_longvideobench_blip_k32_auto_budget_based_curvlapl_normmax_clip95_hybrid_cov1.73_opt.json"
    output_file = "results/LV_Description/judge_prompts_updated.jsonl"
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
    print(f"Questions file has {len(questions_data)} questions")

    # Debug: show sample keys
    debug_print_keys(model1_resps, model2_resps, questions_map)

    # Find common question_ids across all three
    common_ids = set(model1_resps.keys()) & set(model2_resps.keys()) & set(questions_map.keys())
    print(f"Common question IDs (key-based match): {len(common_ids)}")

    # Decide matching strategy
    if len(common_ids) >= len(questions_data) * 0.5:
        # Key-based matching works
        print(f"Using key-based matching ({len(common_ids)} matches).")
        matched = []
        for qid in sorted(common_ids, key=str):
            q_info = questions_map[qid]
            matched.append({
                "question_id": qid,
                "q_info": q_info,
                "desc1": model1_resps[qid],
                "desc2": model2_resps[qid],
            })
    else:
        # Fallback to index-based matching
        print(f"Key-based matching found only {len(common_ids)} matches (too few).")
        matched = match_by_index(model1_data, model2_data, questions_data, questions_map)

    # Build judge prompts
    results = []
    for item in matched:
        qid = item["question_id"]
        q_info = item["q_info"]
        desc1 = item["desc1"]
        desc2 = item["desc2"]

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

    # Save to output file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(results)} judge prompts to: {output_file}")

    # Print first few for verification
    for item in results[:3]:
        print(f"\n{'='*80}")
        print(f"Question ID: {item['question_id']}")
        print(f"{'='*80}")
        print(item['judge_prompt'][:500] + "...")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()