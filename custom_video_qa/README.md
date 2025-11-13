# Custom Video QA Task

This task evaluates video understanding capabilities using multiple-choice questions with metadata-rich annotations.

## Overview

The custom video QA task processes video-based multiple-choice questions with detailed metadata including:
- Task categorization (task_type, domain, sub_category)
- Video properties (duration, video_id, videoID)
- Frame sampling information (frame_idx, frame_num, use_topk)
- Question and answer options

## Dataset Format

Your local JSON file should be an array of objects, each containing the following fields:

```json
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
  "frame_idx": [0.0, 29.0, 58.0, 87.0, ...],
  "frame_num": 64,
  "use_topk": true
}
```

### Required Fields
- `question_id`: Unique identifier for each question
- `question`: The question text
- `options`: List of answer choices (can include letter prefixes or not)
- `answer`: Correct answer (letter A-D)
- `videoID` or `video_id`: Identifier for the video file

### Optional Fields
- `url`: YouTube URL (used if `From_YouTube: True` in config)
- `duration`: Video duration category (short/medium/long)
- `domain`: High-level category
- `sub_category`: More specific category
- `task_type`: Type of reasoning required
- `frame_idx`: Array of frame indices to sample
- `frame_num`: Number of frames to use
- `use_topk`: Whether to use top-k frame sampling

## Setup

### 1. Prepare Your JSON Data File

Create a JSON file containing your test data as an array:

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
    "frame_idx": [0.0, 29.0, 58.0, 87.0, 116.0, 145.0, 174.0, 203.0, 261.0, 290.0, 319.0, 377.0, 406.0, 435.0, 493.0, 522.0, 551.0, 580.0, 609.0, 638.0, 667.0, 696.0, 725.0, 783.0, 812.0, 841.0, 870.0, 899.0, 957.0, 986.0, 1044.0, 1073.0, 1102.0, 1131.0, 1160.0, 1189.0, 1218.0, 1247.0, 1276.0, 1334.0, 1363.0, 1392.0, 1421.0, 1450.0, 1508.0, 1537.0, 1566.0, 1595.0, 1653.0, 1682.0, 1711.0, 1740.0, 1769.0, 1798.0, 1856.0, 1885.0, 1914.0, 1943.0, 2001.0, 2030.0, 2059.0, 2088.0, 2117.0, 2146.0],
    "frame_num": 64,
    "use_topk": true
  },
  {
    "video_id": "002",
    ...
  }
]
```

The YAML is already configured to load from `./datasets/custom_video_qa/test.json`:

```yaml
dataset_path: json
dataset_kwargs:
  data_files:
    test: ./datasets/custom_video_qa/test.json
```

**To use a different path**, edit `custom_video_qa.yaml`:
```yaml
dataset_kwargs:
  data_files:
    test: /path/to/your/data.json  # Absolute or relative path
```

### 2. Prepare Video Files

#### Option A: Local Videos
Place video files in the cache directory:
```
~/.cache/huggingface/custom_video_qa_cache/data/
├── 001.mp4
├── 002.mp4
└── ...
```

Video files should be named using the `videoID` field (e.g., `001.mp4`).

#### Option B: YouTube Download
Enable automatic YouTube download in `custom_video_qa.yaml`:

```yaml
dataset_kwargs:
  From_YouTube: True
  cache_dir: custom_video_qa_cache
  video: True
```

The framework will download videos from the `url` field automatically.

### 3. Configure Frame Sampling (Optional)

Adjust frame sampling in `custom_video_qa.yaml`:

```yaml
lmms_eval_specific_kwargs:
  default:
    frame_num: 64  # Number of frames to sample
    use_topk: True  # Use top-k sampling strategy
```

## Running the Task

### Basic Usage

```bash
lmms_eval \
  --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
  --tasks custom_video_qa \
  --batch_size 128 \
  --device cuda:0
```

### With Custom Settings

```bash
lmms_eval \
  --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_pixels=12845056,attn_implementation=sdpa \
  --tasks custom_video_qa \
  --batch_size 64 \
  --limit 100 \
  --device cuda:0 \
  --output_path ./results/
```

### On HPC with SLURM

Create a SLURM script (`run_eval.sh`):

```bash
#!/bin/bash
#SBATCH --job-name=video_qa_eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G

# Load required modules
module load cuda/12.1
module load python/3.10

# Activate environment
source /path/to/your/venv/bin/activate

# Run evaluation
lmms_eval \
  --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
  --tasks custom_video_qa \
  --batch_size 128 \
  --device cuda:0 \
  --output_path $SCRATCH/results/
```

Submit with:
```bash
sbatch run_eval.sh
```

## Output and Metrics

### Metrics Computed

The task computes accuracy broken down by:
1. **Overall Accuracy**: Total correct answers / total questions
2. **By Task Type**: Accuracy for each task_type category
3. **By Domain**: Accuracy for each domain
4. **By Sub-Category**: Accuracy for each sub_category
5. **By Duration**: Accuracy for each duration category

### Sample Output

```
================================================================================
Overall Accuracy: 67.50% (54/80)
================================================================================

Accuracy by Task Type:
--------------------------------------------------------------------------------
  Counting Problem              : 70.00% (14/20)
  Action Recognition            : 65.00% (13/20)
  Spatial Reasoning             : 68.00% (17/25)
  Temporal Reasoning            : 66.67% (10/15)

Accuracy by Domain:
--------------------------------------------------------------------------------
  Knowledge                     : 72.00% (18/25)
  Sports                        : 65.00% (13/20)
  Daily Activities              : 64.00% (16/25)
  Science                       : 70.00% (7/10)

Accuracy by Sub-Category:
--------------------------------------------------------------------------------
  Humanity & History            : 75.00% (9/12)
  Basketball                    : 66.67% (8/12)
  Cooking                       : 62.50% (10/16)
  Physics                       : 70.00% (7/10)
```

## Customization

### Modify Prompt Format

Edit `custom_video_qa.yaml`:

```yaml
lmms_eval_specific_kwargs:
  default:
    pre_prompt: "Watch the video carefully and answer the following question.\n\n"
    post_prompt: "\n\nProvide only the letter of your answer (A, B, C, or D)."
```

### Change Answer Extraction Logic

Edit the `extract_answer()` function in `utils.py` to customize how answers are parsed from model outputs.

### Add New Metrics

Modify `custom_video_qa_aggregate_results()` in `utils.py` to compute additional metrics or breakdowns.

## Troubleshooting

### Video Not Found Error

```
Video path: /path/to/video.mp4 does not exist
```

**Solutions:**
1. Ensure videos are placed in the correct cache directory
2. Check that filenames match the `videoID` field
3. Enable YouTube download if using URLs
4. Verify supported video formats (.mp4, .mkv, .webm, .avi)

### Answer Extraction Issues

If answers are not being extracted correctly, check:
1. Model's output format
2. The `extract_answer()` function in `utils.py`
3. Whether options include letter prefixes

### HuggingFace Authentication

If your dataset is private:

```bash
# Login to HuggingFace
huggingface-cli login

# Or set token
export HF_TOKEN=your_token_here
```

## Files

- `custom_video_qa.yaml`: Task configuration
- `utils.py`: Processing functions
- `README.md`: This documentation

## References

This task implementation follows patterns from existing video tasks:
- VideoMME: Complex video understanding
- MVBench: Multi-choice video QA
- LongVideoBench: Long-form video analysis

For more information about the lmms_eval framework, see the [main documentation](https://github.com/EvolvingLMMs-Lab/lmms-eval).
