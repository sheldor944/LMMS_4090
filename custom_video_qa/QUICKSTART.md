# Quick Start Guide - Using Local JSON Files

This guide shows you how to use your JSON data directly without uploading to HuggingFace.

## Step 1: Prepare Your JSON File

Create your JSON file with an array of test examples:

**`./datasets/custom_video_qa/test.json`**:
```json
[
  {
    "video_id": "001",
    "duration": "short",
    "domain": "Knowledge",
    "sub_category": "Humanity & History",
    "url": "https://www.youtube.com/watch?v=fFjv93ACGo8",
    "videoID": "fFjv93ACGo8",
    "question_id": "001-1",
    "task_type": "Counting Problem",
    "question": "When demonstrating the Germany modern Christmas tree is initially decorated with apples, candles and berries, which kind of the decoration has the largest number?",
    "options": ["A. Apples.", "B. Candles.", "C. Berries.", "D. The three kinds are of the same number."],
    "answer": "C",
    "frame_idx": [0.0, 29.0, 58.0, ...],
    "frame_num": 64,
    "use_topk": true
  },
  {
    "video_id": "002",
    ...
  }
]
```

An example file is already created at: `./datasets/custom_video_qa/test.json`

## Step 2: Configure the Path (Optional)

The YAML is already configured to use `./datasets/custom_video_qa/test.json`.

**If you want to use a different path**, edit `custom_video_qa.yaml`:

```yaml
dataset_path: json
dataset_kwargs:
  data_files:
    test: /your/custom/path/data.json  # Change this
```

## Step 3: Prepare Video Files

Place your video files in the cache directory, named by `videoID`:

```
~/.cache/huggingface/custom_video_qa_cache/data/
├── fFjv93ACGo8.mp4  # Matches videoID from your JSON
├── 001.mp4          # Or use video_id if videoID not present
└── ...
```

**Alternative**: Enable YouTube auto-download in YAML:
```yaml
dataset_kwargs:
  From_YouTube: True
  video: True
```

## Step 4: Run the Evaluation

### On Local Machine:
```bash
lmms_eval \
  --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
  --tasks custom_video_qa \
  --batch_size 128 \
  --device cuda:0
```

### On HPC with SLURM:

Create `run_eval.sh`:
```bash
#!/bin/bash
#SBATCH --job-name=video_qa
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# Copy your JSON file to HPC first!
# scp ./datasets/custom_video_qa/test.json user@hpc:/path/to/lmms_eval/datasets/custom_video_qa/

lmms_eval \
  --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
  --tasks custom_video_qa \
  --batch_size 1 \
  --device cuda:0 \
  --output_path ./results/
```

Submit:
```bash
sbatch run_eval.sh
```

uv run lmms_eval \
  --model llava_vid \
  --model_args pretrained=/home/hpc4090/miraj/AKS/AKS/llava_eval/LLaVA-NeXT-Video-7B-Qwen2,conv_template=chatml_direct,video_decode_backend=decord,max_frames_num=16,overwrite=False \
  --tasks custom_video_qa \
  --batch_size 1 \
  --device cuda:0 \
  --output_path ./results/


#
## Key Points

1. ✅ **No HuggingFace upload needed** - Just use local JSON files
2. ✅ **Your exact JSON format works** - All fields are supported
3. ✅ **Flexible paths** - Use relative or absolute paths
4. ✅ **Easy to update** - Just edit the JSON file and re-run

## Troubleshooting

### "Video not found" error
- Check that videos are in: `~/.cache/huggingface/custom_video_qa_cache/data/`
- Filenames must match `videoID` field (e.g., `fFjv93ACGo8.mp4`)
- Supported formats: `.mp4`, `.mkv`, `.webm`, `.avi`

### "Dataset not found" error
- Check the path in `data_files` is correct
- Use absolute path if relative path doesn't work
- Ensure JSON file is valid (use `jq` to validate: `jq . test.json`)

### "No module named 'lmms_eval'" error
- Make sure you're in the lmms_eval directory
- Run: `uv sync` to setup environment
- Use: `uv run lmms_eval` instead of `lmms_eval`

## Example Output

```
================================================================================
Overall Accuracy: 67.50% (54/80)
================================================================================

Accuracy by Task Type:
--------------------------------------------------------------------------------
  Counting Problem              : 70.00% (14/20)
  Action Recognition            : 65.00% (13/20)

Accuracy by Domain:
--------------------------------------------------------------------------------
  Knowledge                     : 72.00% (18/25)
  Sports                        : 65.00% (13/20)
```

For more details, see [README.md](README.md).
