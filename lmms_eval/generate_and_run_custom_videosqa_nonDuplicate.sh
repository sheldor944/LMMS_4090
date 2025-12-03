#!/bin/bash

###############################################
# AUTO LMMS-EVAL MULTI-DATASET RUNNER
# Sequential with nohup background run
# WITH SKIP LOGIC FOR ALREADY PROCESSED FILES
###############################################

# 0) Resolve base paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

JSON_DIR="$SCRIPT_DIR/datasets/custom_video_qa"
YAML_FILE="$SCRIPT_DIR/lmms_eval/tasks/custom_video_qa/custom_video_qa.yaml"
RESULT_DIR="$SCRIPT_DIR/results/full_logs/vmme_param_tune"

MODEL_PATH="../LLaVA-NeXT-Video-7B-Qwen2"
MODEL_NAME="llava_vid"

# Master log file for the entire run
MASTER_LOG="$RESULT_DIR/master_run_$(date +"%Y%m%d_%H%M%S").log"

# Ensure dirs exist
mkdir -p "$RESULT_DIR"
mkdir -p "$(dirname "$YAML_FILE")"

# Function to log to both console and master log
log_message() {
    echo "$1" | tee -a "$MASTER_LOG"
}

log_message "======================================================"
log_message "AUTO LMMS-EVAL MULTI-DATASET RUNNER (NOHUP MODE)"
log_message "WITH SKIP LOGIC FOR ALREADY PROCESSED FILES"
log_message "======================================================"
log_message "SCRIPT_DIR = $SCRIPT_DIR"
log_message "JSON_DIR = $JSON_DIR"
log_message "YAML_FILE = $YAML_FILE"
log_message "RESULT_DIR = $RESULT_DIR"
log_message "MASTER_LOG = $MASTER_LOG"
log_message ""

###############################################
# 1) Collect JSON files
###############################################
shopt -s nullglob
json_files=("$JSON_DIR"/*.json)

if (( ${#json_files[@]} == 0 )); then
    log_message "ERROR: No JSON files found in $JSON_DIR"
    exit 1
fi

log_message "Found ${#json_files[@]} JSON files to process"
log_message ""

###############################################
# 2) Main loop over JSON configs
###############################################
dataset_count=0
skipped_count=0
processed_count=0
total_datasets=${#json_files[@]}

for json_path in "${json_files[@]}"; do
    ((dataset_count++))
    filename=$(basename "$json_path")
    dataset_name="${filename%.json}"

    log_message "--------------------------------------------------"
    log_message "[$dataset_count/$total_datasets] Checking dataset: $dataset_name"
    log_message "JSON (absolute): $json_path"
    
    ###############################################
    ### CHECK IF ALREADY PROCESSED ###
    ###############################################
    # Look for any files in RESULT_DIR that start with dataset_name
    existing_files=($(find "$RESULT_DIR" -type f -name "${dataset_name}_*" 2>/dev/null))
    
    if (( ${#existing_files[@]} > 0 )); then
        log_message "⊘ SKIPPING: Already processed (found ${#existing_files[@]} existing files)"
        log_message "   Example: $(basename "${existing_files[0]}")"
        log_message ""
        ((skipped_count++))
        continue
    fi
    
    log_message "✓ Not yet processed, will run"
    log_message "--------------------------------------------------"

    # Build the relative path from SCRIPT_DIR to the JSON file
    rel_json_path="datasets/custom_video_qa/$filename"

    # Extract frame number from kXX in the filename
    frame_num=$(echo "$filename" | grep -oP 'k\K[0-9]+')
    if [[ -z "$frame_num" ]]; then
        log_message "ERROR: Could not extract frame_num from $filename"
        log_message "Skipping..."
        log_message ""
        continue
    fi

    log_message "Detected frame_num = $frame_num"
    log_message "YAML will use test: $rel_json_path"

    # Verify the JSON file exists
    if [[ ! -f "$json_path" ]]; then
        log_message "ERROR: JSON file does not exist: $json_path"
        log_message "Skipping..."
        log_message ""
        continue
    fi

    ###############################################
    ###  3) Overwrite REAL YAML used by lmms_eval ###
    ###############################################
    log_message "Overwriting YAML: $YAML_FILE"

    cat > "$YAML_FILE" <<EOF
dataset_path: json
dataset_kwargs:
  data_files:
    test: $rel_json_path
  cache_dir: custom_video_qa_cache

task: "custom_video_qa"
test_split: test
output_type: generate_until
doc_to_visual: !function utils.custom_video_qa_doc_to_visual
doc_to_text: !function utils.custom_video_qa_doc_to_text
doc_to_target: "answer"

generation_kwargs:
  max_new_tokens: 16
  temperature: 0
  top_p: 1.0
  num_beams: 1
  do_sample: false

process_results: !function utils.custom_video_qa_process_results
metric_list:
  - metric: custom_video_qa_score
    aggregation: !function utils.custom_video_qa_aggregate_results
    higher_is_better: true
lmms_eval_specific_kwargs:
  default:
    frame_num: ${frame_num}
    use_topk: True
    pre_prompt: ""
    post_prompt: "\nAnswer with the option's letter from the given choices directly."

metadata:
  version: 1.0
  description: "Auto-generated Custom video QA task for ${dataset_name}"
EOF

    log_message "YAML overwritten successfully"
    log_message ""

    ###############################################
    ###  4) Run lmms_eval SEQUENTIALLY with nohup ###
    ###############################################
    PREFIX="${dataset_name}_$(date +"%Y%m%d_%H%M%S")"
    LOG_PATH="${RESULT_DIR}/${PREFIX}.log"
    OUT_PATH="${RESULT_DIR}/${PREFIX}_results"

    log_message "Starting lmms_eval for: $dataset_name"
    log_message "Using frame_num: $frame_num"
    log_message "Log file: $LOG_PATH"
    log_message "Output path: $OUT_PATH"
    log_message "Start time: $(date)"
    log_message ""

    # Change to SCRIPT_DIR before running lmms_eval
    cd "$SCRIPT_DIR"

    # Run with accelerate
    CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 \
    -m lmms_eval \
        --model "$MODEL_NAME" \
        --model_args "pretrained=$MODEL_PATH,conv_template=chatml_direct,video_decode_backend=decord,max_frames_num=${frame_num},overwrite=False" \
        --tasks custom_video_qa \
        --batch_size 1 \
        --device cuda \
        --output_path "$OUT_PATH" \
        --log_samples \
        --log_samples_suffix "$PREFIX" \
        --verbosity DEBUG \
        >> "$LOG_PATH" 2>&1
    
    # Check exit status
    exit_status=$?

    log_message ""
    log_message "Finished dataset: $dataset_name"
    log_message "End time: $(date)"
    log_message "Exit status: $exit_status"

    if [ $exit_status -eq 0 ]; then
        log_message "✓ SUCCESS"
        ((processed_count++))
    else
        log_message "✗ FAILED with exit code $exit_status"
    fi

    log_message "-------------------------------------------"
    log_message ""
done

log_message "======================================================"
log_message "ALL DATASETS PROCESSED SEQUENTIALLY."
log_message "======================================================"
log_message "Summary:"
log_message "  Total datasets found: $total_datasets"
log_message "  Skipped (already done): $skipped_count"
log_message "  Newly processed: $processed_count"
log_message "  Completed: $(date)"
log_message "======================================================"