"""
LongVideoBench Task Utilities
Including custom frame selection support
"""

import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import decord
import numpy as np
import torch
import yaml
from decord import VideoReader, cpu
from loguru import logger as eval_logger
from PIL import Image

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file


# ==================== UTILITY FUNCTIONS ====================

def timestamp_to_seconds(timestamp):
    """Convert timestamp string (HH:MM:SS) to seconds"""
    h, m, s = timestamp.split(":")
    total_seconds = int(h) * 3600 + int(m) * 60 + float(s)
    return total_seconds


def load_video(video_file, duration, max_num_frames=16):
    """
    Load video frames uniformly sampled.
    Used by original LongVideoBench tasks.
    """
    from decord import VideoReader

    vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
    fps = vr.get_avg_fps()
    total_valid_frames = int(duration * fps)
    num_frames = min(max_num_frames, int(duration))

    frame_indices = [int(total_valid_frames / num_frames) * i for i in range(num_frames)]

    frames = vr.get_batch(frame_indices)
    if isinstance(frames, torch.Tensor):
        frames = frames.numpy()
    else:
        frames = frames.asnumpy()
    frame_timestamps = [frame_index / fps for frame_index in frame_indices]

    return [Image.fromarray(fr).convert("RGB") for fr in frames]


def compute_frame_timestamps(duration, max_num_frames=16):
    """Compute frame timestamps for uniform sampling"""
    if duration > max_num_frames:
        return [duration / max_num_frames * i for i in range(max_num_frames)]
    else:
        return [i for i in range(int(duration))]


def insert_subtitles_into_frames(frame_timestamps, subtitles, starting_timestamp_for_subtitles, duration):
    """
    Insert subtitles between frame tokens for interleaved mode.
    Used by original LongVideoBench _i tasks.
    """
    interleaved_list = []
    cur_i = 0

    for subtitle in subtitles:
        if "timestamp" in subtitle:
            start, end = subtitle["timestamp"]

            if not isinstance(end, float):
                end = duration

            start -= starting_timestamp_for_subtitles
            end -= starting_timestamp_for_subtitles

            subtitle_timestamp = (start + end) / 2
            subtitle_text = subtitle["text"]
        else:
            start, end = subtitle["start"], subtitle["end"]
            start = timestamp_to_seconds(start)
            end = timestamp_to_seconds(end)
            start -= starting_timestamp_for_subtitles
            end -= starting_timestamp_for_subtitles

            subtitle_timestamp = (start + end) / 2
            subtitle_text = subtitle["line"]

        for i, frame_timestamp in enumerate(frame_timestamps[cur_i:]):
            if frame_timestamp <= subtitle_timestamp:
                interleaved_list.append("<image>")
                cur_i += 1
            else:
                break

        if end - start < 1:
            end = subtitle_timestamp + 0.5
            start = subtitle_timestamp - 0.5

        covering_frames = False
        for frame_timestamp in frame_timestamps:
            if frame_timestamp < end and frame_timestamp > start:
                covering_frames = True
                break

        if covering_frames:
            interleaved_list.append(subtitle_text)

    for i, frame_timestamp in enumerate(frame_timestamps[cur_i:]):
        interleaved_list.append("<image>")

    return "\n".join(interleaved_list)


def get_multi_choice_info(options):
    """
    Given the list of options for multiple choice question
    Return the index2ans and all_choices
    """
    start_chr = "A"
    all_choices = []
    index2ans = {}
    for i, option in enumerate(options):
        index2ans[chr(ord(start_chr) + i)] = option
        all_choices.append(chr(ord(start_chr) + i))

    return index2ans, all_choices


def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    Following the golden utils pattern exactly.
    """
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []

    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f"{choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A. B. C. D.
            if f"{choice}." in response:
                candidates.append(choice)

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)  # -1 will be ignored anyway
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else:  # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index


def eval_multi_choice(gold_i, pred_i):
    """Evaluate if prediction matches gold answer"""
    correct = False
    if isinstance(gold_i, list):
        for answer in gold_i:
            if answer == pred_i:
                correct = True
                break
    else:
        if gold_i == pred_i:
            correct = True
    return correct


def evaluate_longvideobench(samples):
    """Evaluate a list of samples and compute accuracy"""
    pred_correct = 0
    judge_dict = dict()
    for sample in samples:
        gold_i = sample["answer"]
        pred_i = sample["parsed_pred"]
        correct = eval_multi_choice(gold_i, pred_i)

        if correct:
            judge_dict[sample["id"]] = "Correct"
            pred_correct += 1
        else:
            judge_dict[sample["id"]] = "Wrong"

    if len(samples) == 0:
        return {"acc": 0}
    return judge_dict, {"acc": pred_correct / len(samples)}


def calculate_ins_level_acc(results):
    """Calculate instruction-level accuracy across all categories"""
    acc = 0
    ins_num = 0
    for cat_results in results.values():
        acc += cat_results["acc"] * cat_results["num_example"]
        ins_num += cat_results["num_example"]
    if ins_num == 0:
        return 0
    return acc / ins_num


# ==================== CACHE CONFIGURATION ====================

hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)


# ==================== ORIGINAL LONGVIDEOBENCH FUNCTIONS ====================

def longvideobench_doc_to_text(doc, lmms_eval_specific_kwargs):
    """
    Format question text for original LongVideoBench tasks.
    Supports both plain and interleaved (with subtitles) modes.
    """
    candidates = []

    for i in range(5):
        candidate = doc.get(f"option{i}")
        if candidate != "N/A":
            candidates.append(candidate)

    question = doc["question"] + "\n" + "\n".join([". ".join([chr(ord("A") + i), candidate]) for i, candidate in enumerate(candidates)])
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]

    if lmms_eval_specific_kwargs.get("insert_interleave_subtitles", False):
        with open(Path(__file__).parent / "longvideobench_val_i.yaml", "r") as f:
            raw_data = f.readlines()
            safe_data = []
            for i, line in enumerate(raw_data):
                if "!function" not in line:
                    safe_data.append(line)
        cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]
        subtitle_subdir_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"].get("subtitle_subdir", "subtitles")
        cache_dir = os.path.join(base_cache_dir, cache_name, subtitle_subdir_name)
        with open(os.path.join(cache_dir, doc["subtitle_path"])) as f:
            subtitles = json.load(f)

        max_num_frames = yaml.safe_load("".join(safe_data))["dataset_kwargs"].get("max_num_frames", 16)

        frame_timestamps = compute_frame_timestamps(doc["duration"], max_num_frames)
        interleaved_prefix = insert_subtitles_into_frames(frame_timestamps, subtitles, doc["starting_timestamp_for_subtitles"], doc["duration"])
        return f"{pre_prompt}{interleaved_prefix}\n{question}\n{post_prompt}"
    else:
        return f"{pre_prompt}{question}\n{post_prompt}"


def longvideobench_doc_to_visual_v(doc):
    """
    Original LongVideoBench video mode (_v).
    Returns video path only.
    """
    with open(Path(__file__).parent / "longvideobench_val_v.yaml", "r") as f:
        raw_data = f.readlines()
        safe_data = []
        for i, line in enumerate(raw_data):
            if "!function" not in line:
                safe_data.append(line)
    cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]
    vid_subdir_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"].get("video_subdir", "videos/")
    cache_dir = os.path.join(base_cache_dir, cache_name, vid_subdir_name)
    video_path = doc["video_path"]
    video_path = os.path.join(cache_dir, video_path)
    return [video_path]


def longvideobench_doc_to_visual_i(doc):
    """
    Original LongVideoBench interleaved mode (_i).
    Returns list of PIL Images.
    """
    with open(Path(__file__).parent / "longvideobench_val_i.yaml", "r") as f:
        raw_data = f.readlines()
        safe_data = []
        for i, line in enumerate(raw_data):
            if "!function" not in line:
                safe_data.append(line)
    cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]
    vid_subdir_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"].get("video_subdir", "videos/")
    cache_dir = os.path.join(base_cache_dir, cache_name, vid_subdir_name)
    video_path = doc["video_path"]
    video_path = os.path.join(cache_dir, video_path)
    max_num_frames = yaml.safe_load("".join(safe_data))["dataset_kwargs"].get("max_num_frames", 16)
    return load_video(video_path, doc["duration"], max_num_frames)


def longvideobench_process_results(doc, results):
    """
    Process results for original LongVideoBench tasks.
    """
    pred = results[0]
    all_choices = []
    index2ans = {}
    for i in range(5):
        option = doc.get(f"option{i}")
        if option == "N/A":
            break
        index2ans[chr(ord("A") + i)] = option
        all_choices.append(chr(ord("A") + i))

    parsed_pred = parse_multi_choice_response(pred, all_choices, index2ans)
    id = doc["video_id"]
    lvb_acc = {
        "id": id,
        "duration_group": doc["duration_group"],
        "question_category": doc["question_category"],
        "answer": chr(ord("A") + doc["correct_choice"]),
        "parsed_pred": parsed_pred
    }
    return {
        "lvb_acc": lvb_acc,
        "submission": {
            id: pred,
        },
    }


def longvideobench_aggregate_results(results):
    """
    Aggregate results for original LongVideoBench tasks.
    """
    evaluation_result = {}
    subset_to_eval_samples = defaultdict(list)
    for result in results:
        subset_to_eval_samples[result["duration_group"]].append(result)
        subset_to_eval_samples[result["question_category"]].append(result)
    for subset, sub_eval_samples in subset_to_eval_samples.items():
        judge_dict, metric_dict = evaluate_longvideobench(sub_eval_samples)
        metric_dict.update({"num_example": len(sub_eval_samples)})
        evaluation_result[subset] = metric_dict
    printable_results = {}

    for cat_name, cat_results in evaluation_result.items():
        printable_results[cat_name] = {
            "num": int(cat_results["num_example"]),
            "acc": round(cat_results["acc"], 5),
        }
    all_ins_acc = calculate_ins_level_acc(evaluation_result)
    printable_results["Overall"] = {
        "num": sum([cat_results["num_example"] for cat_results in evaluation_result.values()]),
        "acc": round(all_ins_acc, 5),
    }
    eval_logger.info(printable_results)
    return printable_results["Overall"]["acc"]


def longvideobench_aggregate_results_for_submission(results, args):
    """
    Generate submission file for LongVideoBench test set.
    """
    path = generate_submission_file("longvideobench_test_for_submission.json", args)
    results_dict = {list(item.keys())[0]: list(item.values())[0] for item in results}
    with open(path, "w") as f:
        json.dump(results_dict, f)
    eval_logger.info(f"Results saved to {path}.")


# ==================== CUSTOM LONGVIDEOBENCH WITH FRAME SELECTION ====================

def longvideobench_custom_doc_to_visual(doc):
    """
    Custom LongVideoBench visual loader with frame selection.
    Returns video metadata with selected frame indices (same pattern as VideoMME custom task).
    
    Returns:
        List containing a dictionary with:
        - video_path: Full path to video file
        - frame_indices: List of frame indices to extract
        - use_custom_frames: Boolean flag for model to use custom frames
    """
    # DEBUG: Log base_cache_dir
    eval_logger.info(f"DEBUG: base_cache_dir = {base_cache_dir}")
    
    # Load config
    with open(Path(__file__).parent / "longvideobench_custom.yaml", "r") as f:
        raw_data = f.readlines()
        safe_data = [line for line in raw_data if "!function" not in line]
        config = yaml.safe_load("".join(safe_data))
    
    cache_name = config["dataset_kwargs"]["cache_dir"]
    eval_logger.info(f"DEBUG: cache_name from YAML = {cache_name}")
    
    # Build cache_dir: use base_cache_dir if cache_name is relative
    if os.path.isabs(cache_name):
        cache_dir = cache_name
    else:
        cache_dir = os.path.join(base_cache_dir, cache_name)
    
    eval_logger.info(f"DEBUG: final cache_dir = {cache_dir}")
    
    # Get video path from doc
    video_filename = doc.get("video_path")
    eval_logger.info(f"DEBUG: video_filename from doc = {video_filename}")
    
    if not video_filename:
        # Fallback to id if video_path not present
        video_id = doc.get("id", doc.get("video_id"))
        if not video_id:
            eval_logger.error(f"No video_path or id found in document: {doc.keys()}")
            raise ValueError("Missing video identifier")
        video_filename = f"{video_id}.mp4"
    
    # Construct full path
    if video_filename.startswith("videos/"):
        video_path = os.path.join(cache_dir, video_filename)
    else:
        video_path = os.path.join(cache_dir, "videos", video_filename)
    
    eval_logger.info(f"DEBUG: constructed video_path = {video_path}")
    eval_logger.info(f"DEBUG: video_path exists? {os.path.exists(video_path)}")
    
    # Try different extensions if exact path doesn't exist
    if not os.path.exists(video_path):
        base_path = os.path.splitext(video_path)[0]
        exts = [".mp4", ".MP4", ".mkv", ".webm", ".avi"]
        found = False
        
        for ext in exts:
            test_path = f"{base_path}{ext}"
            eval_logger.info(f"DEBUG: trying {test_path} - exists: {os.path.exists(test_path)}")
            if os.path.exists(test_path):
                video_path = test_path
                found = True
                eval_logger.info(f"DEBUG: Found video at {test_path}")
                break
        
        if not found:
            eval_logger.error(
                f"Video not found: {video_filename}\n"
                f"Checked path: {video_path}\n"
                f"Cache dir: {cache_dir}\n"
                f"base_cache_dir: {base_cache_dir}"
            )
            raise FileNotFoundError(f"Video {video_filename} not found")
    
    # Get selected frame indices from your JSON
    selected_frame_indices = doc.get("frame_idx", [])
    
    if selected_frame_indices:
        selected_frame_indices = [int(i) for i in selected_frame_indices]
    else:
        eval_logger.warning(
            f"No frame_idx found for video {video_filename}. "
            f"Will use default uniform sampling."
        )
        selected_frame_indices = []
    
    eval_logger.info(
        f"[longvideobench_custom] Video: {video_filename} | "
        f"Selected {len(selected_frame_indices)} frames"
    )
    
    # Return same format as VideoMME custom task
    return [{
        "video_path": video_path,
        "frame_indices": selected_frame_indices,
        "use_custom_frames": True
    }]

def longvideobench_custom_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """
    Format question text for custom LongVideoBench.
    Handles your JSON format with 'candidates' field.
    """
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    
    # Your JSON has 'candidates' field directly
    candidates = doc.get("candidates", [])
    
    # Filter out empty or N/A candidates
    candidates = [c for c in candidates if c and c != "N/A"]
    
    # Format question with options
    # Your JSON: "question": "In the video, which subtitles appear..."
    question = doc["question"] + "\n" + "\n".join(
        [". ".join([chr(ord("A") + i), candidate]) for i, candidate in enumerate(candidates)]
    )
    
    # Get prompts
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get(
        "post_prompt", 
        "Answer with the option's letter from the given choices directly."
    )
    
    return f"{pre_prompt}{question}\n{post_prompt}"


def longvideobench_custom_process_results(doc, results):
    """
    Process results for custom LongVideoBench.
    Matches your JSON format exactly.
    """
    pred = results[0] if results else ""
    
    # Build choices from candidates
    candidates = doc.get("candidates", [])
    candidates = [c for c in candidates if c and c != "N/A"]
    
    all_choices = []
    index2ans = {}
    for i, candidate in enumerate(candidates):
        index2ans[chr(ord("A") + i)] = candidate
        all_choices.append(chr(ord("A") + i))
    
    # Parse prediction using golden utils method
    parsed_pred = parse_multi_choice_response(pred, all_choices, index2ans)
    
    # Get correct answer
    correct_choice = doc.get("correct_choice")
    if isinstance(correct_choice, int):
        correct_answer = chr(ord("A") + correct_choice)
    else:
        correct_answer = str(correct_choice)
    
    # Get ID
    question_id = doc.get("id", doc.get("video_id", "unknown"))
    
    # DEBUG: Log the comparison
    eval_logger.info(f"DEBUG [ID: {question_id}]:")
    eval_logger.info(f"  Raw prediction: '{pred}'")
    eval_logger.info(f"  Parsed prediction: '{parsed_pred}'")
    eval_logger.info(f"  Correct answer: '{correct_answer}'")
    eval_logger.info(f"  Match: {parsed_pred == correct_answer}")
    
    # Get duration_group and question_category
    duration_group = doc.get("duration_group", "unknown")
    if isinstance(duration_group, (int, float)):
        duration_group = f"{duration_group}s"
    
    question_category = doc.get("question_category", "unknown")
    
    # Package results following EXACT golden utils structure
    lvb_custom_acc = {
        "id": question_id,
        "duration_group": duration_group,
        "question_category": question_category,
        "answer": correct_answer,
        "parsed_pred": parsed_pred,
    }
    
    return {
        "lvb_custom_acc": lvb_custom_acc,
        "submission": {
            question_id: pred,
        },
    }

def longvideobench_custom_aggregate_results(results):
    """
    Aggregate results for custom LongVideoBench.
    Follows golden utils aggregation logic exactly.
    """
    if not results:
        eval_logger.warning("No results to aggregate")
        return 0.0
    
    # Follow golden utils pattern: aggregate by BOTH duration_group AND question_category
    evaluation_result = {}
    subset_to_eval_samples = defaultdict(list)
    
    for result in results:
        # Each result is added to BOTH its duration_group and question_category
        subset_to_eval_samples[result["duration_group"]].append(result)
        subset_to_eval_samples[result["question_category"]].append(result)
    
    # Evaluate each subset separately
    for subset, sub_eval_samples in subset_to_eval_samples.items():
        judge_dict, metric_dict = evaluate_longvideobench(sub_eval_samples)
        metric_dict.update({"num_example": len(sub_eval_samples)})
        evaluation_result[subset] = metric_dict
    
    # Create printable results
    printable_results = {}
    
    for cat_name, cat_results in evaluation_result.items():
        printable_results[cat_name] = {
            "num": int(cat_results["num_example"]),
            "acc": round(cat_results["acc"], 5),
        }
    
    # Calculate overall accuracy using golden utils method
    all_ins_acc = calculate_ins_level_acc(evaluation_result)
    
    # FIXED: Use actual unique sample count instead of sum
    actual_sample_count = len(results)
    
    printable_results["Overall"] = {
        "num": actual_sample_count,  # Correct: actual number of samples
        "acc": round(all_ins_acc, 5),
    }
    
    # Log results
    eval_logger.info("=" * 80)
    eval_logger.info("LONGVIDEOBENCH CUSTOM RESULTS (with Frame Selection)")
    eval_logger.info("=" * 80)
    
    # Sort categories for better readability
    duration_groups = sorted([k for k in printable_results.keys() if k.endswith('s') or k == 'unknown'], 
                            key=lambda x: float(x.replace('s', '')) if x.replace('s', '').replace('.','').isdigit() else 0)
    question_categories = sorted([k for k in printable_results.keys() if k not in duration_groups and k != 'Overall'])
    
    if duration_groups:
        eval_logger.info("\nAccuracy by Duration Group:")
        eval_logger.info("-" * 80)
        for cat_name in duration_groups:
            cat_result = printable_results[cat_name]
            eval_logger.info(
                f"  {cat_name:30s}: {cat_result['acc']*100:5.2f}% ({cat_result['num']} samples)"
            )
    
    if question_categories:
        eval_logger.info("\nAccuracy by Question Category:")
        eval_logger.info("-" * 80)
        for cat_name in question_categories:
            cat_result = printable_results[cat_name]
            eval_logger.info(
                f"  {cat_name:30s}: {cat_result['acc']*100:5.2f}% ({cat_result['num']} samples)"
            )
    
    eval_logger.info("\n" + "=" * 80)
    eval_logger.info(f"OVERALL ACCURACY: {printable_results['Overall']['acc']*100:.2f}% ({printable_results['Overall']['num']} samples)")
    eval_logger.info("=" * 80)
    
    # Return overall accuracy (matching golden utils)
    return printable_results["Overall"]["acc"]

def longvideobench_custom_aggregate_results_for_submission(results, args):
    """
    Generate submission file for custom LongVideoBench.
    """
    path = generate_submission_file("longvideobench_custom_for_submission.json", args)
    results_dict = {list(item.keys())[0]: list(item.values())[0] for item in results}
    with open(path, "w") as f:
        json.dump(results_dict, f)
    eval_logger.info(f"Results saved to {path}.")