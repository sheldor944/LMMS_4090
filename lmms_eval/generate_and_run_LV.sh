#!/bin/bash

###############################################
###  AUTO LMMS-EVAL LONGVIDEOBENCH RUNNER  ###
###  Sequential with nohup background run   ###
###############################################

# 0) Resolve base paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

JSON_DIR="$SCRIPT_DIR/datasets/longvideobench"
YAML_FILE="$SCRIPT_DIR/lmms_eval/tasks/longvideobench_custom/longvideobench_custom.yaml"
RESULT_DIR="$SCRIPT_DIR/results/full_logs/Fixed_radius"

MODEL_PATH="../LLaVA-NeXT-Video-7B-Qwen2"
MODEL_NAME="llava_vid"

# Master log file for the entire run
MASTER_LOG="$RESULT_DIR/master_longvideobench_with_adaptive_tuning_radius_run_$(date +"%Y%m%d_%H%M%S").log"

# Ensure dirs exist
mkdir -p "$RESULT_DIR"
mkdir -p "$(dirname "$YAML_FILE")"

# Function to log to both console and master log
log_message() {
    echo "$1" | tee -a "$MASTER_LOG"
}

log_message "======================================================"
log_message "AUTO LMMS-EVAL LONGVIDEOBENCH RUNNER (NOHUP MODE)"
log_message "======================================================"
log_message "SCRIPT_DIR = $SCRIPT_DIR"
log_message "JSON_DIR   = $JSON_DIR"
log_message "YAML_FILE  = $YAML_FILE"
log_message "RESULT_DIR = $RESULT_DIR"
log_message "MASTER_LOG = $MASTER_LOG"
log_message ""

###############################################
###  1) Collect JSON files                  ###
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
###  2) Main loop over JSON configs         ###
###############################################
dataset_count=0
total_datasets=${#json_files[@]}

for json_path in "${json_files[@]}"; do
    ((dataset_count++))
    filename=$(basename "$json_path")
    dataset_name="${filename%.json}"

    log_message "--------------------------------------------------"
    log_message "[$dataset_count/$total_datasets] Processing dataset: $dataset_name"
    log_message "JSON (absolute): $json_path"
    log_message "--------------------------------------------------"

    # Build the relative path from SCRIPT_DIR to the JSON file
    rel_json_path="datasets/longvideobench/$filename"

    # Extract frame number from kXX in the filename
    frame_num=$(echo "$filename" | grep -oP 'k\K[0-9]+')
    if [[ -z "$frame_num" ]]; then
        log_message "WARNING: Could not extract frame_num from $filename, using default=16"
        frame_num=64
    fi

    log_message "Detected frame_num = $frame_num"f
    log_message "YAML will use test: $rel_json_path"

    # Verify the JSON file exists
    if [[ ! -f "$json_path" ]]; then
        log_message "ERROR: JSON file does not exist: $json_path"
        log_message "Skipping..."
        log_message ""
        continue
    fi

    ###############################################
    ###  3) Overwrite YAML used by lmms_eval    ###
    ###############################################
    log_message "Overwriting YAML: $YAML_FILE"

    cat > "$YAML_FILE" << 'EOFYAML'
dataset_path: json
dataset_kwargs:
  data_files:
    test: REL_JSON_PATH
  cache_dir: longvideobench_custom_cache

task: "longvideobench_custom"
test_split: test
output_type: generate_until

doc_to_visual: !function utils.longvideobench_custom_doc_to_visual
doc_to_text: !function utils.longvideobench_custom_doc_to_text
doc_to_target: "correct_choice"

generation_kwargs:
  max_new_tokens: 128
  temperature: 0
  do_sample: false

process_results: !function utils.longvideobench_custom_process_results

metric_list:
  - metric: lvb_custom_acc
    aggregation: !function utils.longvideobench_custom_aggregate_results
    higher_is_better: true

lmms_eval_specific_kwargs:
  default:
    pre_prompt: ""
    post_prompt: "Answer with the option's letter from the given choices directly."

metadata:
  version: 1.0
  description: "DATASET_DESCRIPTION"
EOFYAML

    # Replace placeholders
    sed -i "s|REL_JSON_PATH|$rel_json_path|g" "$YAML_FILE"
    sed -i "s|DATASET_DESCRIPTION|Auto-generated LongVideoBench task for ${dataset_name}|g" "$YAML_FILE"

    log_message "YAML overwritten successfully"
    log_message ""

    ###############################################
    ###  4) Run lmms_eval SEQUENTIALLY          ###
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
    
    # Run with accelerate (sequential execution)
    # CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS="ignore" stdbuf -oL -eL \
    # accelerate launch --num_processes 1 --main_process_port 29500 \
    # -m lmms_eval \
    #     --model "$MODEL_NAME" \
    #     --model_args "pretrained=$MODEL_PATH,conv_template=chatml_direct,video_decode_backend=decord,max_frames_num=${frame_num},overwrite=False,device_map=auto" \
    #     --tasks longvideobench_custom \
    #     --batch_size 1 \
    #     --output_path "$OUT_PATH" \
    #     --log_samples \
    #     --log_samples_suffix "$PREFIX" \
    #     --verbosity DEBUG \
    #     --limit 3\
    #     >> "$LOG_PATH" 2>&1
    # CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS="ignore" stdbuf -oL -eL \
    # accelerate launch --num_processes 1 --main_process_port 29500 \
    # -m lmms_eval \
    #     --model "$MODEL_NAME" \
    #     --model_args "pretrained=$MODEL_PATH,conv_template=chatml_direct,video_decode_backend=decord,max_frames_num=${frame_num},overwrite=False,device_map=auto" \
    #     --tasks longvideobench_custom \
    #     --batch_size 1 \
    #     --output_path "$OUT_PATH" \
    #     --log_samples \
    #     --log_samples_suffix "$PREFIX" \
    #     --verbosity DEBUG \
    #     >> "$LOG_PATH" 2>&1
    CUDA_VISIBLE_DEVICES=0,1 PYTHONWARNINGS="ignore" stdbuf -oL -eL \
    accelerate launch --num_processes 2 --main_process_port 29500 \
    -m lmms_eval \
        --model "$MODEL_NAME" \
        --model_args "pretrained=$MODEL_PATH,conv_template=chatml_direct,video_decode_backend=decord,max_frames_num=${frame_num},overwrite=False" \
        --tasks longvideobench_custom \
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
    else
        log_message "✗ FAILED with exit code $exit_status"
    fi
    
    log_message "-------------------------------------------"
    log_message ""
    
done

log_message "======================================================"
log_message "ALL LONGVIDEOBENCH DATASETS PROCESSED SEQUENTIALLY."
log_message "Completed: $(date)"
log_message "======================================================"