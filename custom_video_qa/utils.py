"""
Custom Video QA Task Utilities

This module provides processing functions for evaluating video understanding
with multiple-choice questions.
"""

import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml
from loguru import logger as eval_logger

# Load cache configuration
hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)

with open(Path(__file__).parent / "custom_video_qa.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = [line for line in raw_data if "!function" not in line]
    config = yaml.safe_load("".join(safe_data))
    cache_name = config["dataset_kwargs"]["cache_dir"]


def custom_video_qa_doc_to_visual(doc):
    """
    Convert document to video path.

    This function handles both local video files and YouTube downloads.
    It looks for videos in the cache directory using the videoID field.

    Args:
        doc: Dataset document containing videoID or video_id field

    Returns:
        List containing the path to the video file

    Raises:
        SystemExit: If the video file cannot be found in any supported format
    """
    cache_dir = os.path.join(base_cache_dir, cache_name)

    # Get video ID from document (supports both videoID and video_id)
    video_id = doc.get("videoID") or doc.get("video_id")

    if not video_id:
        eval_logger.error(f"No videoID or video_id found in document: {doc}")
        sys.exit("Missing video ID in document")

    # Construct base video path
    video_path = os.path.join(cache_dir, "data", f"{video_id}.mp4")

    # Check for different video formats
    video_extensions = [".mp4", ".MP4", ".mkv", ".webm", ".avi"]

    for ext in video_extensions:
        test_path = video_path.replace(".mp4", ext)
        if os.path.exists(test_path):
            return [test_path]

    # If no video found, provide helpful error message
    eval_logger.error(
        f"Video not found: {video_path}\n"
        f"Checked extensions: {video_extensions}\n"
        f"Cache directory: {cache_dir}\n"
        f"Video ID: {video_id}"
    )

    # If YouTube download is enabled, provide instructions
    if doc.get("url"):
        eval_logger.info(
            f"YouTube URL available: {doc['url']}\n"
            f"Enable 'From_YouTube: True' in YAML to auto-download"
        )

    sys.exit(f"Video path: {video_path} does not exist")


def custom_video_qa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """
    Format the question and options for the model.

    Creates a prompt with the question and multiple-choice options,
    formatted for the model to answer.

    Args:
        doc: Dataset document containing question and options
        lmms_eval_specific_kwargs: Additional configuration from YAML

    Returns:
        Formatted prompt string
    """
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    question = doc["question"]

    # Format options
    # Options can be a list like ["A. Apples.", "B. Candles.", ...]
    # or just ["Apples", "Candles", ...]
    options = doc.get("options", [])

    if not options:
        eval_logger.warning(f"No options found for question: {question}")
        option_text = ""
    else:
        # If options already include letter prefix (e.g., "A. Apples"), use as-is
        # Otherwise, add letter prefixes
        if all(opt.startswith(f"{chr(65+i)}.") for i, opt in enumerate(options)):
            option_text = "\n".join(options)
        else:
            option_text = "\n".join(
                [f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)]
            )

    # Get prompts from config
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get(
        "post_prompt", "\nAnswer with the option's letter from the given choices directly."
    )

    # Build full prompt
    full_prompt = f"{pre_prompt}Question: {question}\n{option_text}{post_prompt}"

    return full_prompt


def extract_answer(pred_text):
    """
    Extract the answer letter (A, B, C, or D) from model prediction.

    Handles various response formats:
    - "A" or "A."
    - "The answer is A"
    - Full option text like "A. Apples"

    Args:
        pred_text: Model's prediction text

    Returns:
        Extracted answer letter (A-D) or empty string if not found
    """
    if not pred_text:
        return ""

    # Remove whitespace and convert to uppercase
    pred_clean = pred_text.strip().upper()

    # Try to find a single letter answer (A, B, C, or D)
    # Look for the first occurrence of A, B, C, or D
    match = re.search(r"\b[ABCD]\b", pred_clean)
    if match:
        return match.group(0)

    # Try to find pattern like "The answer is A" or "Answer: A"
    match = re.search(r"(?:ANSWER|OPTION)[\s:]*([ABCD])", pred_clean)
    if match:
        return match.group(1)

    # If still not found, take the first A/B/C/D character
    match = re.search(r"[ABCD]", pred_clean)
    if match:
        return match.group(0)

    # No valid answer found
    eval_logger.warning(f"Could not extract answer from: {pred_text[:100]}")
    return ""


def custom_video_qa_process_results(doc, results):
    """
    Process model results for a single example.

    Extracts the predicted answer and packages it with metadata
    for aggregation and analysis.

    Args:
        doc: Dataset document
        results: Model predictions (list with one element)

    Returns:
        Dictionary with metric name and result data
    """
    pred = results[0] if results else ""

    # Extract predicted answer
    pred_ans = extract_answer(pred)

    # Get correct answer (handle both "A" and full option text)
    correct_answer = doc["answer"]
    if len(correct_answer) > 1 and correct_answer[0] in "ABCD":
        correct_answer = correct_answer[0]

    # Package all relevant data
    data_dict = {
        "question_id": doc.get("question_id", "unknown"),
        "video_id": doc.get("videoID") or doc.get("video_id", "unknown"),
        "duration": doc.get("duration", "unknown"),
        "domain": doc.get("domain", "unknown"),
        "sub_category": doc.get("sub_category", "unknown"),
        "task_type": doc.get("task_type", "unknown"),
        "pred_answer": pred_ans,
        "answer": correct_answer,
        "pred_full": pred,
        "is_correct": pred_ans == correct_answer,
        "frame_idx": doc.get("frame_idx", []),
        "frame_num": doc.get("frame_num", 0),
        "use_topk": doc.get("use_topk", False),
    }

    return {"custom_video_qa_score": data_dict}


def custom_video_qa_aggregate_results(results):
    """
    Aggregate results across all examples and compute metrics.

    Computes overall accuracy and breaks down performance by:
    - Task type
    - Domain
    - Sub-category
    - Duration (short/medium/long)

    Args:
        results: List of result dictionaries from process_results

    Returns:
        Overall accuracy score (0-100)
    """
    if not results:
        eval_logger.warning("No results to aggregate")
        return 0.0

    # Track overall statistics
    total_correct = 0
    total_answered = 0

    # Track by different categories
    by_task_type = defaultdict(lambda: {"correct": 0, "total": 0})
    by_domain = defaultdict(lambda: {"correct": 0, "total": 0})
    by_sub_category = defaultdict(lambda: {"correct": 0, "total": 0})
    by_duration = defaultdict(lambda: {"correct": 0, "total": 0})

    # Process each result
    for result in results:
        total_answered += 1
        is_correct = result["is_correct"]

        if is_correct:
            total_correct += 1

        # Aggregate by task type
        task_type = result.get("task_type", "unknown")
        by_task_type[task_type]["total"] += 1
        if is_correct:
            by_task_type[task_type]["correct"] += 1

        # Aggregate by domain
        domain = result.get("domain", "unknown")
        by_domain[domain]["total"] += 1
        if is_correct:
            by_domain[domain]["correct"] += 1

        # Aggregate by sub-category
        sub_cat = result.get("sub_category", "unknown")
        by_sub_category[sub_cat]["total"] += 1
        if is_correct:
            by_sub_category[sub_cat]["correct"] += 1

        # Aggregate by duration
        duration = result.get("duration", "unknown")
        by_duration[duration]["total"] += 1
        if is_correct:
            by_duration[duration]["correct"] += 1

    # Calculate and log overall accuracy
    overall_acc = 100 * total_correct / total_answered if total_answered > 0 else 0.0
    eval_logger.info("=" * 80)
    eval_logger.info(f"Overall Accuracy: {overall_acc:.2f}% ({total_correct}/{total_answered})")
    eval_logger.info("=" * 80)

    # Log breakdown by task type
    if by_task_type:
        eval_logger.info("\nAccuracy by Task Type:")
        eval_logger.info("-" * 80)
        for task_type in sorted(by_task_type.keys()):
            stats = by_task_type[task_type]
            acc = 100 * stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            eval_logger.info(
                f"  {task_type:30s}: {acc:5.2f}% ({stats['correct']}/{stats['total']})"
            )

    # Log breakdown by domain
    if by_domain:
        eval_logger.info("\nAccuracy by Domain:")
        eval_logger.info("-" * 80)
        for domain in sorted(by_domain.keys()):
            stats = by_domain[domain]
            acc = 100 * stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            eval_logger.info(
                f"  {domain:30s}: {acc:5.2f}% ({stats['correct']}/{stats['total']})"
            )

    # Log breakdown by sub-category
    if by_sub_category:
        eval_logger.info("\nAccuracy by Sub-Category:")
        eval_logger.info("-" * 80)
        for sub_cat in sorted(by_sub_category.keys()):
            stats = by_sub_category[sub_cat]
            acc = 100 * stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            eval_logger.info(
                f"  {sub_cat:30s}: {acc:5.2f}% ({stats['correct']}/{stats['total']})"
            )

    # Log breakdown by duration
    if by_duration:
        eval_logger.info("\nAccuracy by Duration:")
        eval_logger.info("-" * 80)
        for duration in sorted(by_duration.keys()):
            stats = by_duration[duration]
            acc = 100 * stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            eval_logger.info(
                f"  {duration:30s}: {acc:5.2f}% ({stats['correct']}/{stats['total']})"
            )

    eval_logger.info("=" * 80)

    return overall_acc
