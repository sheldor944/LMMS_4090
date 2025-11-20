# import json
# import os
# import random
# import re
# import sys
# from collections import Counter, defaultdict
# from pathlib import Path
# from typing import Dict, List, Optional, Union

# import decord
# import numpy as np
# import torch
# import yaml
# from decord import VideoReader, cpu
# from loguru import logger as eval_logger
# from PIL import Image

# from lmms_eval.tasks._task_utils.file_utils import generate_submission_file


# def timestamp_to_seconds(timestamp):
#     # Split the timestamp into hours, minutes, and seconds
#     h, m, s = timestamp.split(":")
#     # Convert hours, minutes, and total seconds (including fractions) to float and compute total seconds
#     total_seconds = int(h) * 3600 + int(m) * 60 + float(s)
#     return total_seconds


# def load_video(video_file, duration, max_num_frames=16):
#     from decord import VideoReader

#     vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
#     fps = vr.get_avg_fps()
#     total_valid_frames = int(duration * fps)
#     num_frames = min(max_num_frames, int(duration))

#     frame_indices = [int(total_valid_frames / num_frames) * i for i in range(num_frames)]

#     frames = vr.get_batch(frame_indices)
#     if isinstance(frames, torch.Tensor):
#         frames = frames.numpy()
#     else:
#         frames = frames.asnumpy()
#     frame_timestamps = [frame_index / fps for frame_index in frame_indices]

#     return [Image.fromarray(fr).convert("RGB") for fr in frames]


# def compute_frame_timestamps(duration, max_num_frames=16):
#     if duration > max_num_frames:
#         return [duration / max_num_frames * i for i in range(max_num_frames)]
#     else:
#         return [i for i in range(int(duration))]


# def insert_subtitles_into_frames(frame_timestamps, subtitles, starting_timestamp_for_subtitles, duration):
#     interleaved_list = []
#     cur_i = 0

#     for subtitle in subtitles:
#         if "timestamp" in subtitle:
#             start, end = subtitle["timestamp"]

#             if not isinstance(end, float):
#                 end = duration

#             start -= starting_timestamp_for_subtitles
#             end -= starting_timestamp_for_subtitles

#             subtitle_timestamp = (start + end) / 2
#             subtitle_text = subtitle["text"]
#         else:
#             start, end = subtitle["start"], subtitle["end"]
#             start = timestamp_to_seconds(start)
#             end = timestamp_to_seconds(end)
#             start -= starting_timestamp_for_subtitles
#             end -= starting_timestamp_for_subtitles

#             subtitle_timestamp = (start + end) / 2
#             subtitle_text = subtitle["line"]

#         for i, frame_timestamp in enumerate(frame_timestamps[cur_i:]):
#             if frame_timestamp <= subtitle_timestamp:
#                 # print("frame:", frame_timestamp)
#                 interleaved_list.append("<image>")
#                 cur_i += 1
#             else:
#                 break

#         if end - start < 1:
#             end = subtitle_timestamp + 0.5
#             start = subtitle_timestamp - 0.5

#         covering_frames = False
#         for frame_timestamp in frame_timestamps:
#             if frame_timestamp < end and frame_timestamp > start:
#                 covering_frames = True
#                 break

#         if covering_frames:
#             # print("subtitle:", subtitle_timestamp, start, end)
#             interleaved_list.append(subtitle_text)
#         else:
#             pass
#             # print("leaving out subtitle:", start, end)

#     for i, frame_timestamp in enumerate(frame_timestamps[cur_i:]):
#         # print(frame_timestamp)
#         interleaved_list.append("<image>")

#     return "\n".join(interleaved_list)


# def longvideobench_doc_to_text(doc, lmms_eval_specific_kwargs):
#     candidates = []

#     for i in range(5):
#         candidate = doc.get(f"option{i}")
#         if candidate != "N/A":
#             candidates.append(candidate)

#     question = doc["question"] + "\n" + "\n".join([". ".join([chr(ord("A") + i), candidate]) for i, candidate in enumerate(candidates)])
#     pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
#     post_prompt = lmms_eval_specific_kwargs["post_prompt"]

#     if lmms_eval_specific_kwargs.get("insert_interleave_subtitles", False):
#         with open(Path(__file__).parent / "longvideobench_val_i.yaml", "r") as f:
#             raw_data = f.readlines()
#             safe_data = []
#             for i, line in enumerate(raw_data):
#                 # remove function definition since yaml load cannot handle it
#                 if "!function" not in line:
#                     safe_data.append(line)
#         cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]
#         subtitle_subdir_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"].get("subtitle_subdir", "subtitles")
#         cache_dir = os.path.join(base_cache_dir, cache_name, subtitle_subdir_name)
#         with open(os.path.join(cache_dir, doc["subtitle_path"])) as f:
#             subtitles = json.load(f)

#         max_num_frames = yaml.safe_load("".join(safe_data))["dataset_kwargs"].get("max_num_frames", 16)

#         frame_timestamps = compute_frame_timestamps(doc["duration"], max_num_frames)
#         interleaved_prefix = insert_subtitles_into_frames(frame_timestamps, subtitles, doc["starting_timestamp_for_subtitles"], doc["duration"])
#         return f"{pre_prompt}{interleaved_prefix}\n{question}\n{post_prompt}"
#     else:
#         return f"{pre_prompt}{question}\n{post_prompt}"


# hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
# base_cache_dir = os.path.expanduser(hf_home)


# def longvideobench_doc_to_visual_v(doc):
#     with open(Path(__file__).parent / "longvideobench_val_v.yaml", "r") as f:
#         raw_data = f.readlines()
#         safe_data = []
#         for i, line in enumerate(raw_data):
#             # remove function definition since yaml load cannot handle it
#             if "!function" not in line:
#                 safe_data.append(line)
#     cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]
#     vid_subdir_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"].get("video_subdir", "videos/")
#     cache_dir = os.path.join(base_cache_dir, cache_name, vid_subdir_name)
#     video_path = doc["video_path"]
#     video_path = os.path.join(cache_dir, video_path)
#     return [video_path]


# def longvideobench_doc_to_visual_i(doc):
#     with open(Path(__file__).parent / "longvideobench_val_i.yaml", "r") as f:
#         raw_data = f.readlines()
#         safe_data = []
#         for i, line in enumerate(raw_data):
#             # remove function definition since yaml load cannot handle it
#             if "!function" not in line:
#                 safe_data.append(line)
#     cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]
#     vid_subdir_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"].get("video_subdir", "videos/")
#     cache_dir = os.path.join(base_cache_dir, cache_name, vid_subdir_name)
#     video_path = doc["video_path"]
#     video_path = os.path.join(cache_dir, video_path)
#     max_num_frames = yaml.safe_load("".join(safe_data))["dataset_kwargs"].get("max_num_frames", 16)
#     return load_video(video_path, doc["duration"], max_num_frames)


# def get_multi_choice_info(options):
#     """
#     Given the list of options for multiple choice question
#     Return the index2ans and all_choices
#     https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/data_utils.py#L54
#     """

#     start_chr = "A"
#     all_choices = []
#     index2ans = {}
#     for i, option in enumerate(options):
#         index2ans[chr(ord(start_chr) + i)] = option
#         all_choices.append(chr(ord(start_chr) + i))

#     return index2ans, all_choices


# def parse_multi_choice_response(response, all_choices, index2ans):
#     """
#     Changed from MMMU-style complex parsing into simple parsing.
#     Fixed to avoid 'D. A book' be parsed as A.
#     Same as original LongVideoBench paper (from author Haoning Wu), if parsing failed, it will assign a random choice to model.
#     """
#     s = response.strip()
#     answer_prefixes = [
#         "The best answer is",
#         "The correct answer is",
#         "The answer is",
#         "The answer",
#         "The best option is",
#         "The correct option is",
#         "Best answer:",
#         "Best option:",
#     ]
#     for answer_prefix in answer_prefixes:
#         s = s.replace(answer_prefix, "")

#     if len(s.split()) > 10 and not re.search("[ABCDE]", s):
#         return random.choice(all_choices)

#     matches = re.search(r"[ABCDE]", s)
#     if matches is None:
#         return random.choice(all_choices)
#     return matches[0]


# def evaluate_longvideobench(samples):
#     pred_correct = 0
#     judge_dict = dict()
#     for sample in samples:
#         gold_i = sample["answer"]
#         pred_i = sample["parsed_pred"]
#         correct = eval_multi_choice(gold_i, pred_i)

#         if correct:
#             judge_dict[sample["id"]] = "Correct"
#             pred_correct += 1
#         else:
#             judge_dict[sample["id"]] = "Wrong"

#     if len(samples) == 0:
#         return {"acc": 0}
#     return judge_dict, {"acc": pred_correct / len(samples)}


# def eval_multi_choice(gold_i, pred_i):
#     correct = False
#     # only they are exactly the same, we consider it as correct
#     if isinstance(gold_i, list):
#         for answer in gold_i:
#             if answer == pred_i:
#                 correct = True
#                 break
#     else:  # gold_i is a string
#         if gold_i == pred_i:
#             correct = True
#     return correct


# def calculate_ins_level_acc(results):
#     """Calculate the instruction level accuracy for given Subject results
#     https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L246
#     """
#     acc = 0
#     ins_num = 0
#     for cat_results in results.values():
#         acc += cat_results["acc"] * cat_results["num_example"]
#         ins_num += cat_results["num_example"]
#     if ins_num == 0:
#         return 0
#     return acc / ins_num


# def longvideobench_process_results(doc, results):
#     pred = results[0]
#     all_choices = []
#     index2ans = {}
#     for i in range(5):
#         option = doc.get(f"option{i}")
#         if option == "N/A":
#             break
#         index2ans[chr(ord("A") + i)] = option
#         all_choices.append(chr(ord("A") + i))

#     parsed_pred = parse_multi_choice_response(pred, all_choices, index2ans)
#     id = doc["id"]
#     lvb_acc = {"id": id, "duration_group": doc["duration_group"], "question_category": doc["question_category"], "answer": chr(ord("A") + doc["correct_choice"]), "parsed_pred": parsed_pred}
#     return {
#         "lvb_acc": lvb_acc,
#         "submission": {
#             id: pred,
#         },
#     }


# def longvideobench_aggregate_results(results):
#     evaluation_result = {}
#     subset_to_eval_samples = defaultdict(list)
#     for result in results:
#         subset_to_eval_samples[result["duration_group"]].append(result)
#         subset_to_eval_samples[result["question_category"]].append(result)
#     for subset, sub_eval_samples in subset_to_eval_samples.items():
#         judge_dict, metric_dict = evaluate_longvideobench(sub_eval_samples)
#         metric_dict.update({"num_example": len(sub_eval_samples)})
#         evaluation_result[subset] = metric_dict
#     printable_results = {}

#     for cat_name, cat_results in evaluation_result.items():
#         printable_results[cat_name] = {
#             "num": int(cat_results["num_example"]),
#             "acc": round(cat_results["acc"], 5),
#         }
#     all_ins_acc = calculate_ins_level_acc(evaluation_result)
#     printable_results["Overall"] = {
#         "num": sum([cat_results["num_example"] for cat_results in evaluation_result.values()]),
#         "acc": round(all_ins_acc, 5),
#     }
#     eval_logger.info(printable_results)
#     return printable_results["Overall"]["acc"]


# def longvideobench_aggregate_results_for_submission(results, args):
#     path = generate_submission_file("longvideobench_test_for_submission.json", args)
#     results_dict = {list(item.keys())[0]: list(item.values())[0] for item in results}
#     with open(path, "w") as f:
#         json.dump(results_dict, f)
#     eval_logger.info(f"Results saved to {path}.")



# below one is working but wrong measurement accuracy

# """
# LongVideoBench Task Utilities
# Including custom frame selection support
# """

# import json
# import os
# import random
# import re
# import sys
# from collections import Counter, defaultdict
# from pathlib import Path
# from typing import Dict, List, Optional, Union

# import decord
# import numpy as np
# import torch
# import yaml
# from decord import VideoReader, cpu
# from loguru import logger as eval_logger
# from PIL import Image

# from lmms_eval.tasks._task_utils.file_utils import generate_submission_file


# # ==================== UTILITY FUNCTIONS ====================

# def timestamp_to_seconds(timestamp):
#     """Convert timestamp string (HH:MM:SS) to seconds"""
#     h, m, s = timestamp.split(":")
#     total_seconds = int(h) * 3600 + int(m) * 60 + float(s)
#     return total_seconds


# def load_video(video_file, duration, max_num_frames=16):
#     """
#     Load video frames uniformly sampled.
#     Used by original LongVideoBench tasks.
#     """
#     from decord import VideoReader

#     vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
#     fps = vr.get_avg_fps()
#     total_valid_frames = int(duration * fps)
#     num_frames = min(max_num_frames, int(duration))

#     frame_indices = [int(total_valid_frames / num_frames) * i for i in range(num_frames)]

#     frames = vr.get_batch(frame_indices)
#     if isinstance(frames, torch.Tensor):
#         frames = frames.numpy()
#     else:
#         frames = frames.asnumpy()
#     frame_timestamps = [frame_index / fps for frame_index in frame_indices]

#     return [Image.fromarray(fr).convert("RGB") for fr in frames]


# def compute_frame_timestamps(duration, max_num_frames=16):
#     """Compute frame timestamps for uniform sampling"""
#     if duration > max_num_frames:
#         return [duration / max_num_frames * i for i in range(max_num_frames)]
#     else:
#         return [i for i in range(int(duration))]


# def insert_subtitles_into_frames(frame_timestamps, subtitles, starting_timestamp_for_subtitles, duration):
#     """
#     Insert subtitles between frame tokens for interleaved mode.
#     Used by original LongVideoBench _i tasks.
#     """
#     interleaved_list = []
#     cur_i = 0

#     for subtitle in subtitles:
#         if "timestamp" in subtitle:
#             start, end = subtitle["timestamp"]

#             if not isinstance(end, float):
#                 end = duration

#             start -= starting_timestamp_for_subtitles
#             end -= starting_timestamp_for_subtitles

#             subtitle_timestamp = (start + end) / 2
#             subtitle_text = subtitle["text"]
#         else:
#             start, end = subtitle["start"], subtitle["end"]
#             start = timestamp_to_seconds(start)
#             end = timestamp_to_seconds(end)
#             start -= starting_timestamp_for_subtitles
#             end -= starting_timestamp_for_subtitles

#             subtitle_timestamp = (start + end) / 2
#             subtitle_text = subtitle["line"]

#         for i, frame_timestamp in enumerate(frame_timestamps[cur_i:]):
#             if frame_timestamp <= subtitle_timestamp:
#                 interleaved_list.append("<image>")
#                 cur_i += 1
#             else:
#                 break

#         if end - start < 1:
#             end = subtitle_timestamp + 0.5
#             start = subtitle_timestamp - 0.5

#         covering_frames = False
#         for frame_timestamp in frame_timestamps:
#             if frame_timestamp < end and frame_timestamp > start:
#                 covering_frames = True
#                 break

#         if covering_frames:
#             interleaved_list.append(subtitle_text)

#     for i, frame_timestamp in enumerate(frame_timestamps[cur_i:]):
#         interleaved_list.append("<image>")

#     return "\n".join(interleaved_list)


# def get_multi_choice_info(options):
#     """
#     Given the list of options for multiple choice question
#     Return the index2ans and all_choices
#     """
#     start_chr = "A"
#     all_choices = []
#     index2ans = {}
#     for i, option in enumerate(options):
#         index2ans[chr(ord(start_chr) + i)] = option
#         all_choices.append(chr(ord(start_chr) + i))

#     return index2ans, all_choices


# def parse_multi_choice_response(response, all_choices, index2ans):
#     """
#     Parse model response to extract answer choice.
#     If parsing fails, assigns a random choice (following original LongVideoBench).
#     """
#     s = response.strip()
#     answer_prefixes = [
#         "The best answer is",
#         "The correct answer is",
#         "The answer is",
#         "The answer",
#         "The best option is",
#         "The correct option is",
#         "Best answer:",
#         "Best option:",
#     ]
#     for answer_prefix in answer_prefixes:
#         s = s.replace(answer_prefix, "")

#     if len(s.split()) > 10 and not re.search("[ABCDE]", s):
#         return random.choice(all_choices)

#     matches = re.search(r"[ABCDE]", s)
#     if matches is None:
#         return random.choice(all_choices)
#     return matches[0]


# def eval_multi_choice(gold_i, pred_i):
#     """Evaluate if prediction matches gold answer"""
#     correct = False
#     if isinstance(gold_i, list):
#         for answer in gold_i:
#             if answer == pred_i:
#                 correct = True
#                 break
#     else:
#         if gold_i == pred_i:
#             correct = True
#     return correct


# def evaluate_longvideobench(samples):
#     """Evaluate a list of samples and compute accuracy"""
#     pred_correct = 0
#     judge_dict = dict()
#     for sample in samples:
#         gold_i = sample["answer"]
#         pred_i = sample["parsed_pred"]
#         correct = eval_multi_choice(gold_i, pred_i)

#         if correct:
#             judge_dict[sample["id"]] = "Correct"
#             pred_correct += 1
#         else:
#             judge_dict[sample["id"]] = "Wrong"

#     if len(samples) == 0:
#         return {"acc": 0}
#     return judge_dict, {"acc": pred_correct / len(samples)}


# def calculate_ins_level_acc(results):
#     """Calculate instruction-level accuracy across all categories"""
#     acc = 0
#     ins_num = 0
#     for cat_results in results.values():
#         acc += cat_results["acc"] * cat_results["num_example"]
#         ins_num += cat_results["num_example"]
#     if ins_num == 0:
#         return 0
#     return acc / ins_num


# # ==================== CACHE CONFIGURATION ====================

# hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
# base_cache_dir = os.path.expanduser(hf_home)


# # ==================== ORIGINAL LONGVIDEOBENCH FUNCTIONS ====================

# def longvideobench_doc_to_text(doc, lmms_eval_specific_kwargs):
#     """
#     Format question text for original LongVideoBench tasks.
#     Supports both plain and interleaved (with subtitles) modes.
#     """
#     candidates = []

#     for i in range(5):
#         candidate = doc.get(f"option{i}")
#         if candidate != "N/A":
#             candidates.append(candidate)

#     question = doc["question"] + "\n" + "\n".join([". ".join([chr(ord("A") + i), candidate]) for i, candidate in enumerate(candidates)])
#     pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
#     post_prompt = lmms_eval_specific_kwargs["post_prompt"]

#     if lmms_eval_specific_kwargs.get("insert_interleave_subtitles", False):
#         with open(Path(__file__).parent / "longvideobench_val_i.yaml", "r") as f:
#             raw_data = f.readlines()
#             safe_data = []
#             for i, line in enumerate(raw_data):
#                 if "!function" not in line:
#                     safe_data.append(line)
#         cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]
#         subtitle_subdir_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"].get("subtitle_subdir", "subtitles")
#         cache_dir = os.path.join(base_cache_dir, cache_name, subtitle_subdir_name)
#         with open(os.path.join(cache_dir, doc["subtitle_path"])) as f:
#             subtitles = json.load(f)

#         max_num_frames = yaml.safe_load("".join(safe_data))["dataset_kwargs"].get("max_num_frames", 16)

#         frame_timestamps = compute_frame_timestamps(doc["duration"], max_num_frames)
#         interleaved_prefix = insert_subtitles_into_frames(frame_timestamps, subtitles, doc["starting_timestamp_for_subtitles"], doc["duration"])
#         return f"{pre_prompt}{interleaved_prefix}\n{question}\n{post_prompt}"
#     else:
#         return f"{pre_prompt}{question}\n{post_prompt}"


# def longvideobench_doc_to_visual_v(doc):
#     """
#     Original LongVideoBench video mode (_v).
#     Returns video path only.
#     """
#     with open(Path(__file__).parent / "longvideobench_val_v.yaml", "r") as f:
#         raw_data = f.readlines()
#         safe_data = []
#         for i, line in enumerate(raw_data):
#             if "!function" not in line:
#                 safe_data.append(line)
#     cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]
#     vid_subdir_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"].get("video_subdir", "videos/")
#     cache_dir = os.path.join(base_cache_dir, cache_name, vid_subdir_name)
#     video_path = doc["video_path"]
#     video_path = os.path.join(cache_dir, video_path)
#     return [video_path]


# def longvideobench_doc_to_visual_i(doc):
#     """
#     Original LongVideoBench interleaved mode (_i).
#     Returns list of PIL Images.
#     """
#     with open(Path(__file__).parent / "longvideobench_val_i.yaml", "r") as f:
#         raw_data = f.readlines()
#         safe_data = []
#         for i, line in enumerate(raw_data):
#             if "!function" not in line:
#                 safe_data.append(line)
#     cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]
#     vid_subdir_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"].get("video_subdir", "videos/")
#     cache_dir = os.path.join(base_cache_dir, cache_name, vid_subdir_name)
#     video_path = doc["video_path"]
#     video_path = os.path.join(cache_dir, video_path)
#     max_num_frames = yaml.safe_load("".join(safe_data))["dataset_kwargs"].get("max_num_frames", 16)
#     return load_video(video_path, doc["duration"], max_num_frames)


# def longvideobench_process_results(doc, results):
#     """
#     Process results for original LongVideoBench tasks.
#     """
#     pred = results[0]
#     all_choices = []
#     index2ans = {}
#     for i in range(5):
#         option = doc.get(f"option{i}")
#         if option == "N/A":
#             break
#         index2ans[chr(ord("A") + i)] = option
#         all_choices.append(chr(ord("A") + i))

#     parsed_pred = parse_multi_choice_response(pred, all_choices, index2ans)
#     id = doc["id"]
#     lvb_acc = {
#         "id": id,
#         "duration_group": doc["duration_group"],
#         "question_category": doc["question_category"],
#         "answer": chr(ord("A") + doc["correct_choice"]),
#         "parsed_pred": parsed_pred
#     }
#     return {
#         "lvb_acc": lvb_acc,
#         "submission": {
#             id: pred,
#         },
#     }


# def longvideobench_aggregate_results(results):
#     """
#     Aggregate results for original LongVideoBench tasks.
#     """
#     evaluation_result = {}
#     subset_to_eval_samples = defaultdict(list)
#     for result in results:
#         subset_to_eval_samples[result["duration_group"]].append(result)
#         subset_to_eval_samples[result["question_category"]].append(result)
#     for subset, sub_eval_samples in subset_to_eval_samples.items():
#         judge_dict, metric_dict = evaluate_longvideobench(sub_eval_samples)
#         metric_dict.update({"num_example": len(sub_eval_samples)})
#         evaluation_result[subset] = metric_dict
#     printable_results = {}

#     for cat_name, cat_results in evaluation_result.items():
#         printable_results[cat_name] = {
#             "num": int(cat_results["num_example"]),
#             "acc": round(cat_results["acc"], 5),
#         }
#     all_ins_acc = calculate_ins_level_acc(evaluation_result)
#     printable_results["Overall"] = {
#         "num": sum([cat_results["num_example"] for cat_results in evaluation_result.values()]),
#         "acc": round(all_ins_acc, 5),
#     }
#     eval_logger.info(printable_results)
#     return printable_results["Overall"]["acc"]


# def longvideobench_aggregate_results_for_submission(results, args):
#     """
#     Generate submission file for LongVideoBench test set.
#     """
#     path = generate_submission_file("longvideobench_test_for_submission.json", args)
#     results_dict = {list(item.keys())[0]: list(item.values())[0] for item in results}
#     with open(path, "w") as f:
#         json.dump(results_dict, f)
#     eval_logger.info(f"Results saved to {path}.")


# # ==================== CUSTOM LONGVIDEOBENCH WITH FRAME SELECTION ====================

# def longvideobench_custom_doc_to_visual(doc):
#     """
#     Custom LongVideoBench visual loader with frame selection.
#     Returns video metadata with selected frame indices (same pattern as VideoMME custom task).
    
#     Returns:
#         List containing a dictionary with:
#         - video_path: Full path to video file
#         - frame_indices: List of frame indices to extract
#         - use_custom_frames: Boolean flag for model to use custom frames
#     """
#     # Load config
#     with open(Path(__file__).parent / "longvideobench_custom.yaml", "r") as f:
#         raw_data = f.readlines()
#         safe_data = [line for line in raw_data if "!function" not in line]
#         config = yaml.safe_load("".join(safe_data))
    
#     cache_name = config["dataset_kwargs"]["cache_dir"]
#     cache_dir = os.path.join(base_cache_dir, cache_name)
    
#     # Get video path from doc
#     video_filename = doc.get("video_path")
    
#     if not video_filename:
#         # Fallback to id if video_path not present
#         video_id = doc.get("id", doc.get("video_id"))
#         if not video_id:
#             eval_logger.error(f"No video_path or id found in document: {doc.keys()}")
#             raise ValueError("Missing video identifier")
#         video_filename = f"{video_id}.mp4"
    
#     # Construct full path
#     if video_filename.startswith("videos/"):
#         video_path = os.path.join(cache_dir, video_filename)
#     else:
#         video_path = os.path.join(cache_dir, "videos", video_filename)
    
#     # Try different extensions if exact path doesn't exist
#     if not os.path.exists(video_path):
#         base_path = os.path.splitext(video_path)[0]
#         exts = [".mp4", ".MP4", ".mkv", ".webm", ".avi"]
#         found = False
        
#         for ext in exts:
#             test_path = f"{base_path}{ext}"
#             if os.path.exists(test_path):
#                 video_path = test_path
#                 found = True
#                 break
        
#         if not found:
#             eval_logger.error(
#                 f"Video not found: {video_filename}\n"
#                 f"Checked path: {video_path}\n"
#                 f"Cache dir: {cache_dir}"
#             )
#             raise FileNotFoundError(f"Video {video_filename} not found")
    
#     # Get selected frame indices from your JSON
#     selected_frame_indices = [int(i) for i in doc.get("frame_idx", [])]
    
#     if not selected_frame_indices:
#         eval_logger.warning(
#             f"No frame_idx found for video {video_filename}. "
#             f"Will use default uniform sampling."
#         )
    
#     eval_logger.debug(
#         f"[longvideobench_custom] Video: {video_filename} | "
#         f"Selected {len(selected_frame_indices)} frames"
#     )
    
#     # Return same format as VideoMME custom task
#     return [{
#         "video_path": video_path,
#         "frame_indices": selected_frame_indices,
#         "use_custom_frames": True
#     }]


# def longvideobench_custom_doc_to_text(doc, lmms_eval_specific_kwargs=None):
#     """
#     Format question text for custom LongVideoBench.
#     """
#     if lmms_eval_specific_kwargs is None:
#         lmms_eval_specific_kwargs = {}
    
#     # Build candidate options
#     candidates = []
#     for i in range(5):
#         candidate = doc.get(f"option{i}")
#         if candidate and candidate != "N/A":
#             candidates.append(candidate)
    
#     # Format question with options
#     question = doc["question"] + "\n" + "\n".join(
#         [". ".join([chr(ord("A") + i), candidate]) for i, candidate in enumerate(candidates)]
#     )
    
#     # Get prompts
#     pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
#     post_prompt = lmms_eval_specific_kwargs.get(
#         "post_prompt", 
#         "Answer with the option's letter from the given choices directly.\n"
#     )
    
#     return f"{pre_prompt}{question}\n{post_prompt}"


# # def longvideobench_custom_process_results(doc, results):
# #     """
# #     Process results for custom LongVideoBench.
# #     """
# #     pred = results[0] if results else ""
    
# #     # Build choices
# #     all_choices = []
# #     index2ans = {}
# #     for i in range(5):
# #         option = doc.get(f"option{i}")
# #         if option and option != "N/A":
# #             index2ans[chr(ord("A") + i)] = option
# #             all_choices.append(chr(ord("A") + i))
    
# #     # Parse prediction
# #     parsed_pred = parse_multi_choice_response(pred, all_choices, index2ans)
    
# #     # Get correct answer
# #     correct_choice = doc.get("correct_choice")
# #     if isinstance(correct_choice, int):
# #         correct_answer = chr(ord("A") + correct_choice)
# #     else:
# #         correct_answer = str(correct_choice)
    
# #     # Get ID
# #     question_id = doc.get("id", doc.get("question_id", "unknown"))
    
# #     # Package results with metadata
# #     lvb_custom_acc = {
# #         "id": question_id,
# #         "duration_group": doc.get("duration_group", "unknown"),
# #         "question_category": doc.get("question_category", "unknown"),
# #         "answer": correct_answer,
# #         "parsed_pred": parsed_pred,
# #         "pred_full": pred,
# #         "is_correct": parsed_pred == correct_answer,
# #         # Frame selection metadata
# #         "frame_idx": doc.get("frame_idx", []),
# #         "num_frames_selected": len(doc.get("frame_idx", [])),
# #         "duration": doc.get("duration", "unknown"),
# #         "video_path": doc.get("video_path", "unknown"),
# #     }
    
# #     return {
# #         "lvb_custom_acc": lvb_custom_acc,
# #     }

# def longvideobench_custom_process_results(doc, results):
#     """
#     Process results for custom LongVideoBench.
#     """
#     pred = results[0] if results else ""
    
#     # Build choices - handle both formats
#     all_choices = []
#     index2ans = {}
    
#     # Check if using 'candidates' format (your JSON format)
#     if "candidates" in doc:
#         candidates = doc["candidates"]
#         for i, candidate in enumerate(candidates):
#             index2ans[chr(ord("A") + i)] = candidate
#             all_choices.append(chr(ord("A") + i))
#     else:
#         # Original format with option0, option1, etc.
#         for i in range(5):
#             option = doc.get(f"option{i}")
#             if option and option != "N/A":
#                 index2ans[chr(ord("A") + i)] = option
#                 all_choices.append(chr(ord("A") + i))
    
#     # Parse prediction
#     parsed_pred = parse_multi_choice_response(pred, all_choices, index2ans)
    
#     # Get correct answer
#     correct_choice = doc.get("correct_choice")
#     if isinstance(correct_choice, int):
#         correct_answer = chr(ord("A") + correct_choice)
#     else:
#         correct_answer = str(correct_choice)
    
#     # Get ID
#     question_id = doc.get("id", doc.get("question_id", "unknown"))
    
#     # Package results with metadata
#     lvb_custom_acc = {
#         "id": question_id,
#         "duration_group": doc.get("duration_group", "unknown"),
#         "question_category": doc.get("question_category", "unknown"),
#         "answer": correct_answer,
#         "parsed_pred": parsed_pred,
#         "pred_full": pred,
#         "is_correct": parsed_pred == correct_answer,
#         # Frame selection metadata
#         "frame_idx": doc.get("frame_idx", []),
#         "num_frames_selected": len(doc.get("frame_idx", [])),
#         "duration": doc.get("duration", "unknown"),
#         "video_path": doc.get("video_path", "unknown"),
#     }
    
#     return {
#         "lvb_custom_acc": lvb_custom_acc,
#     }


# def longvideobench_custom_aggregate_results(results):
#     """
#     Aggregate results for custom LongVideoBench.
#     """
#     if not results:
#         eval_logger.warning("No results to aggregate")
#         return 0.0
    
#     # Overall statistics
#     total_correct = 0
#     total_answered = 0
    
#     # Track by categories
#     by_duration_group = defaultdict(lambda: {"correct": 0, "total": 0})
#     by_question_category = defaultdict(lambda: {"correct": 0, "total": 0})
#     by_num_frames = defaultdict(lambda: {"correct": 0, "total": 0})
    
#     # Process each result
#     for result in results:
#         total_answered += 1
#         is_correct = result["is_correct"]
        
#         if is_correct:
#             total_correct += 1
        
#         # By duration group
#         duration_group = result.get("duration_group", "unknown")
#         by_duration_group[duration_group]["total"] += 1
#         if is_correct:
#             by_duration_group[duration_group]["correct"] += 1
        
#         # By question category
#         question_category = result.get("question_category", "unknown")
#         by_question_category[question_category]["total"] += 1
#         if is_correct:
#             by_question_category[question_category]["correct"] += 1
        
#         # By number of frames
#         num_frames = result.get("num_frames_selected", 0)
#         if num_frames < 32:
#             frame_bin = "< 32 frames"
#         elif num_frames < 64:
#             frame_bin = "32-64 frames"
#         else:
#             frame_bin = "> 64 frames"
        
#         by_num_frames[frame_bin]["total"] += 1
#         if is_correct:
#             by_num_frames[frame_bin]["correct"] += 1
    
#     # Calculate overall accuracy
#     overall_acc = 100 * total_correct / total_answered if total_answered > 0 else 0.0
    
#     # Logging
#     eval_logger.info("=" * 80)
#     eval_logger.info("LONGVIDEOBENCH CUSTOM RESULTS (Frame Selection)")
#     eval_logger.info("=" * 80)
#     eval_logger.info(f"Overall Accuracy: {overall_acc:.2f}% ({total_correct}/{total_answered})")
#     eval_logger.info("=" * 80)
    
#     # Log by duration group
#         # Log by duration group
#     if by_duration_group:
#         eval_logger.info("\nAccuracy by Duration Group:")
#         eval_logger.info("-" * 80)
#         for duration_group in sorted(by_duration_group.keys()):
#             stats = by_duration_group[duration_group]
#             acc = 100 * stats["correct"] / stats["total"] if stats["total"] > 0 else 0
#             eval_logger.info(
#                 f"  {str(duration_group):30s}: {acc:5.2f}% ({stats['correct']}/{stats['total']})"
#             )
    
#     # Log by question category
#     if by_question_category:
#         eval_logger.info("\nAccuracy by Question Category:")
#         eval_logger.info("-" * 80)
#         for category in sorted(by_question_category.keys()):
#             stats = by_question_category[category]
#             acc = 100 * stats["correct"] / stats["total"] if stats["total"] > 0 else 0
#             eval_logger.info(
#                 f"  {category:30s}: {acc:5.2f}% ({stats['correct']}/{stats['total']})"
#             )
    
#     # Log by number of frames
#     if by_num_frames:
#         eval_logger.info("\nAccuracy by Number of Frames:")
#         eval_logger.info("-" * 80)
#         for frame_bin in ["< 32 frames", "32-64 frames", "> 64 frames"]:
#             if frame_bin in by_num_frames:
#                 stats = by_num_frames[frame_bin]
#                 acc = 100 * stats["correct"] / stats["total"] if stats["total"] > 0 else 0
#                 eval_logger.info(
#                     f"  {frame_bin:30s}: {acc:5.2f}% ({stats['correct']}/{stats['total']})"
#                 )
    
#     # Frame selection statistics
#     eval_logger.info("\nFrame Selection Statistics:")
#     eval_logger.info("-" * 80)
#     total_frames = sum([r.get("num_frames_selected", 0) for r in results])
#     avg_frames = total_frames / len(results) if results else 0
#     frame_counts = [r.get("num_frames_selected", 0) for r in results if r.get("num_frames_selected", 0) > 0]
    
#     eval_logger.info(f"  Average frames selected: {avg_frames:.2f}")
#     if frame_counts:
#         eval_logger.info(f"  Min frames: {min(frame_counts)}")
#         eval_logger.info(f"  Max frames: {max(frame_counts)}")
#         eval_logger.info(f"  Median frames: {np.median(frame_counts):.2f}")
    
#     eval_logger.info("=" * 80)
    
#     return overall_acc