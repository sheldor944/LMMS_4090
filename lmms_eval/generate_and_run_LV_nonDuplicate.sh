# #!/bin/bash

# ###############################################
# ###  AUTO LMMS-EVAL LONGVIDEOBENCH RUNNER  ###
# ###  Sequential with nohup background run   ###
# ###  WITH SKIP LOGIC AND CONDITIONAL GPU    ###
# ###############################################

# # 0) Resolve base paths
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# JSON_DIR="$SCRIPT_DIR/datasets/longvideobench"
# YAML_FILE="$SCRIPT_DIR/lmms_eval/tasks/longvideobench_custom/longvideobench_custom.yaml"
# RESULT_DIR="$SCRIPT_DIR/results/full_logs/FINAL_LV"

# MODEL_PATH="../LLaVA-NeXT-Video-7B-Qwen2"
# MODEL_NAME="llava_vid"

# # Master log file for the entire run
# MASTER_LOG="$RESULT_DIR/master_longvideobench_with_adaptive_tuning_radius_run_$(date +"%Y%m%d_%H%M%S").log"

# # Ensure dirs exist
# mkdir -p "$RESULT_DIR"
# mkdir -p "$(dirname "$YAML_FILE")"

# # Function to log to both console and master log
# log_message() {
#     echo "$1" | tee -a "$MASTER_LOG"
# }

# log_message "======================================================"
# log_message "AUTO LMMS-EVAL LONGVIDEOBENCH RUNNER (NOHUP MODE)"
# log_message "WITH SKIP LOGIC AND CONDITIONAL GPU SETUP"
# log_message "======================================================"
# log_message "SCRIPT_DIR = $SCRIPT_DIR"
# log_message "JSON_DIR   = $JSON_DIR"
# log_message "YAML_FILE  = $YAML_FILE"
# log_message "RESULT_DIR = $RESULT_DIR"
# log_message "MASTER_LOG = $MASTER_LOG"
# log_message ""

# ###############################################
# ###  1) Collect JSON files                  ###
# ###############################################
# shopt -s nullglob
# json_files=("$JSON_DIR"/*.json)

# if (( ${#json_files[@]} == 0 )); then
#     log_message "ERROR: No JSON files found in $JSON_DIR"
#     exit 1
# fi

# log_message "Found ${#json_files[@]} JSON files to process"
# log_message ""

# ###############################################
# ###  2) Main loop over JSON configs         ###
# ###############################################
# dataset_count=0
# skipped_count=0
# processed_count=0
# total_datasets=${#json_files[@]}

# for json_path in "${json_files[@]}"; do
#     ((dataset_count++))
#     filename=$(basename "$json_path")
#     dataset_name="${filename%.json}"

#     log_message "--------------------------------------------------"
#     log_message "[$dataset_count/$total_datasets] Checking dataset: $dataset_name"
#     log_message "JSON (absolute): $json_path"
    
#     ###############################################
#     ### CHECK IF ALREADY PROCESSED ###
#     ###############################################
#     # Look for any files in RESULT_DIR that start with dataset_name
#     existing_files=($(find "$RESULT_DIR" -type f -name "${dataset_name}_*" 2>/dev/null))
    
#     if (( ${#existing_files[@]} > 0 )); then
#         log_message "⊘ SKIPPING: Already processed (found ${#existing_files[@]} existing files)"
#         log_message "   Example: $(basename "${existing_files[0]}")"
#         log_message ""
#         ((skipped_count++))
#         continue
#     fi
    
#     log_message "✓ Not yet processed, will run"
#     log_message "--------------------------------------------------"

#     # Build the relative path from SCRIPT_DIR to the JSON file
#     rel_json_path="datasets/longvideobench/$filename"

#     # Extract frame number from kXX in the filename
#     frame_num=$(echo "$filename" | grep -oP 'k\K[0-9]+')
#     if [[ -z "$frame_num" ]]; then
#         log_message "WARNING: Could not extract frame_num from $filename, using default=64"
#         frame_num=64
#     fi

#     log_message "Detected frame_num = $frame_num"
#     log_message "YAML will use test: $rel_json_path"

#     # Verify the JSON file exists
#     if [[ ! -f "$json_path" ]]; then
#         log_message "ERROR: JSON file does not exist: $json_path"
#         log_message "Skipping..."
#         log_message ""
#         continue
#     fi

#     ###############################################
#     ###  3) Overwrite YAML used by lmms_eval    ###
#     ###############################################
#     log_message "Overwriting YAML: $YAML_FILE"

#     cat > "$YAML_FILE" << 'EOFYAML'
# dataset_path: json
# dataset_kwargs:
#   data_files:
#     test: REL_JSON_PATH
#   cache_dir: longvideobench_custom_cache

# task: "longvideobench_custom"
# test_split: test
# output_type: generate_until

# doc_to_visual: !function utils.longvideobench_custom_doc_to_visual
# doc_to_text: !function utils.longvideobench_custom_doc_to_text
# doc_to_target: "correct_choice"

# generation_kwargs:
#   max_new_tokens: 16
#   temperature: 0
#   do_sample: false

# process_results: !function utils.longvideobench_custom_process_results

# metric_list:
#   - metric: lvb_custom_acc
#     aggregation: !function utils.longvideobench_custom_aggregate_results
#     higher_is_better: true

# lmms_eval_specific_kwargs:
#   default:
#     pre_prompt: ""
#     post_prompt: "Answer with the option's letter from the given choices directly."

# metadata:
#   version: 1.0
#   description: "DATASET_DESCRIPTION"
# EOFYAML

#     # Replace placeholders
#     sed -i "s|REL_JSON_PATH|$rel_json_path|g" "$YAML_FILE"
#     sed -i "s|DATASET_DESCRIPTION|Auto-generated LongVideoBench task for ${dataset_name}|g" "$YAML_FILE"

#     log_message "YAML overwritten successfully"
#     log_message ""

#     ###############################################
#     ###  4) Run lmms_eval with conditional GPU  ###
#     ###############################################
#     PREFIX="${dataset_name}_$(date +"%Y%m%d_%H%M%S")"
#     LOG_PATH="${RESULT_DIR}/${PREFIX}.log"
#     OUT_PATH="${RESULT_DIR}/${PREFIX}_results"

#     log_message "Starting lmms_eval for: $dataset_name"
#     log_message "Using frame_num: $frame_num"
#     log_message "Log file: $LOG_PATH"
#     log_message "Output path: $OUT_PATH"
#     log_message "Start time: $(date)"

#     # Change to SCRIPT_DIR before running lmms_eval
#     cd "$SCRIPT_DIR"
    

    
#     # Conditional execution based on frame_num
#     CUDA_VISIBLE_DEVICES=0 PYTHONWARNINGS="ignore" stdbuf -oL -eL \
#       accelerate launch --num_processes 1 --main_process_port 29500 \
#       -m lmms_eval \
#           --model "$MODEL_NAME" \
#           --model_args "pretrained=$MODEL_PATH,conv_template=chatml_direct,video_decode_backend=decord,max_frames_num=${frame_num},overwrite=False" \
#           --tasks longvideobench_custom \
#           --batch_size 1 \
#           --device cuda \
#           --output_path "$OUT_PATH" \
#           --log_samples \
#           --log_samples_suffix "$PREFIX" \
#           --verbosity DEBUG \
#           --limit 5 \
#           >> "$LOG_PATH" 2>&1
    
#     exit_status=$?

#     log_message ""
#     log_message "Finished dataset: $dataset_name"
#     log_message "End time: $(date)"
#     log_message "Exit status: $exit_status"
    
#     if [ $exit_status -eq 0 ]; then
#         log_message "✓ SUCCESS"
#         ((processed_count++))
#     else
#         log_message "✗ FAILED with exit code $exit_status"
#     fi
    
#     log_message "-------------------------------------------"
#     log_message ""
    
# done

# log_message "======================================================"
# log_message "ALL LONGVIDEOBENCH DATASETS PROCESSED SEQUENTIALLY."
# log_message "======================================================"
# log_message "Summary:"
# log_message "  Total datasets found: $total_datasets"
# log_message "  Skipped (already done): $skipped_count"
# log_message "  Newly processed: $processed_count"
# log_message "  Completed: $(date)"
# log_message "======================================================"


















#!/bin/bash

###############################################
###  AUTO LMMS-EVAL LONGVIDEOBENCH RUNNER  ###
###  Sequential with minimal process mgmt   ###
###  WITH SKIP LOGIC AND TIMEOUT PROTECTION ###
###############################################

# 0) Resolve base paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

JSON_DIR="$SCRIPT_DIR/datasets/longvideobench"
YAML_FILE="$SCRIPT_DIR/lmms_eval/tasks/longvideobench/longvideobench_custom.yaml"
RESULT_DIR="$SCRIPT_DIR/results/full_logs/FINAL_LV"

MODEL_PATH="../LLaVA-NeXT-Video-7B-Qwen2"
MODEL_NAME="llava_vid"

# Master log file for the entire run
MASTER_LOG="$RESULT_DIR/master_longvideobench_with_adaptive_tuning_radius_run_$(date +"%Y%m%d_%H%M%S").log"

# Ensure dirs exist
mkdir -p "$RESULT_DIR"
mkdir -p "$(dirname "$YAML_FILE")"

# Function to log to both console and master log
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MASTER_LOG"
}

log_message "======================================================"
log_message "AUTO LMMS-EVAL LONGVIDEOBENCH RUNNER"
log_message "======================================================"
log_message "SCRIPT_DIR = $SCRIPT_DIR"
log_message "JSON_DIR   = $JSON_DIR"
log_message "YAML_FILE  = $YAML_FILE"
log_message "RESULT_DIR = $RESULT_DIR"
log_message "MASTER_LOG = $MASTER_LOG"
log_message ""

# Check GPU availability
log_message "Checking GPU availability..."
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv 2>&1 | tee -a "$MASTER_LOG"
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
skipped_count=0
processed_count=0
failed_count=0
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
    rel_json_path="datasets/longvideobench/$filename"

    # Extract frame number from kXX in the filename
    frame_num=$(echo "$filename" | grep -oP 'k\K[0-9]+')
    if [[ -z "$frame_num" ]]; then
        log_message "WARNING: Could not extract frame_num from $filename, using default=16"
        frame_num=16
    fi

    log_message "Detected frame_num = $frame_num"
    log_message "YAML will use test: $rel_json_path"

    # Verify the JSON file exists
    if [[ ! -f "$json_path" ]]; then
        log_message "ERROR: JSON file does not exist: $json_path"
        log_message "Skipping..."
        log_message ""
        ((failed_count++))
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
  max_new_tokens: 16
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
    ###  4) Run lmms_eval with conditional GPU  ###
    ###############################################
    PREFIX="${dataset_name}_$(date +"%Y%m%d_%H%M%S")"
    LOG_PATH="${RESULT_DIR}/${PREFIX}.log"
    OUT_PATH="${RESULT_DIR}/${PREFIX}_results"

    log_message "Starting lmms_eval for: $dataset_name"
    log_message "Using frame_num: $frame_num"
    log_message "Log file: $LOG_PATH"
    log_message "Output path: $OUT_PATH"
    log_message "Start time: $(date)"

    # Change to SCRIPT_DIR before running lmms_eval
    cd "$SCRIPT_DIR"
    
    # Conditional execution based on frame_num
    if [ "$frame_num" -le 16 ]; then
        log_message "GPU Strategy: Using 2 GPUs with 2 processes (frame_num <= 16)"
        
        CUDA_VISIBLE_DEVICES=0,1 PYTHONWARNINGS="ignore" stdbuf -oL -eL \
          accelerate launch --num_processes 2 --main_process_port 29500 \
          -m lmms_eval \
              --model "$MODEL_NAME" \
              --model_args "pretrained=$MODEL_PATH,conv_template=chatml_direct,video_decode_backend=decord,max_frames_num=${frame_num},overwrite=False,device_map=auto" \
              --tasks longvideobench_custom \
              --batch_size 1 \
              --output_path "$OUT_PATH" \
              --log_samples \
              --log_samples_suffix "$PREFIX" \
              --verbosity DEBUG \
              >> "$LOG_PATH" 2>&1 &
        
        eval_pid=$!
    else
        log_message "GPU Strategy: Using 2 GPUs with 1 process (frame_num > 16)"
        
        CUDA_VISIBLE_DEVICES=0,1 PYTHONWARNINGS="ignore" stdbuf -oL -eL \
          accelerate launch --num_processes 1 --main_process_port 29500 \
          -m lmms_eval \
              --model "$MODEL_NAME" \
              --model_args "pretrained=$MODEL_PATH,conv_template=chatml_direct,video_decode_backend=decord,max_frames_num=${frame_num},overwrite=False,device_map=auto" \
              --tasks longvideobench_custom \
              --batch_size 1 \
              --output_path "$OUT_PATH" \
              --log_samples \
              --log_samples_suffix "$PREFIX" \
              --verbosity DEBUG \
              >> "$LOG_PATH" 2>&1 &
        
        eval_pid=$!
    fi
    
    log_message "Process started with PID: $eval_pid"
    
    # Wait for process with 5-hour timeout
    timeout_seconds=18000  # 5 hours
    elapsed=0
    check_interval=60  # Check every minute
    
    while kill -0 $eval_pid 2>/dev/null; do
        sleep $check_interval
        elapsed=$((elapsed + check_interval))
        
        # Log progress every 30 minutes
        if (( elapsed % 1800 == 0 )); then
            log_message "Still running... Elapsed: $((elapsed / 3600))h $((elapsed % 3600 / 60))m"
        fi
        
        # Check timeout
        if (( elapsed >= timeout_seconds )); then
            log_message "⚠ TIMEOUT after 5 hours - Force killing PID $eval_pid"
            pkill -KILL -P $eval_pid 2>/dev/null
            kill -KILL $eval_pid 2>/dev/null
            log_message "✗ KILLED due to timeout"
            ((failed_count++))
            break
        fi
    done
    
    # Get exit status if process finished normally
    if kill -0 $eval_pid 2>/dev/null; then
        # Process was killed by timeout
        exit_status=124
    else
        wait $eval_pid
        exit_status=$?
    fi

    log_message ""
    log_message "Finished dataset: $dataset_name"
    log_message "End time: $(date)"
    log_message "Duration: $((elapsed / 3600))h $((elapsed % 3600 / 60))m"
    log_message "Exit status: $exit_status"
    
    # Verify output was actually created
    if [ $exit_status -eq 0 ]; then
        if [ -d "$OUT_PATH" ] || [ -f "$LOG_PATH" ]; then
            log_message "✓ SUCCESS - Output files verified"
            ((processed_count++))
        else
            log_message "✗ FAILED - Exit status 0 but no output files found"
            ((failed_count++))
        fi
    elif [ $exit_status -eq 124 ]; then
        log_message "✗ TIMEOUT - Killed after 5 hours"
    else
        log_message "✗ FAILED with exit code $exit_status"
        ((failed_count++))
    fi
    
    log_message "Waiting 10 seconds before next dataset..."
    sleep 10
    
    log_message "-------------------------------------------"
    log_message ""
    
done

log_message "======================================================"
log_message "ALL LONGVIDEOBENCH DATASETS PROCESSED SEQUENTIALLY."
log_message "======================================================"
log_message "Summary:"
log_message "  Total datasets found: $total_datasets"
log_message "  Skipped (already done): $skipped_count"
log_message "  Successfully processed: $processed_count"
log_message "  Failed: $failed_count"
log_message "  Completion time: $(date)"
log_message "======================================================"

if [ $failed_count -gt 0 ]; then
    log_message "⚠ WARNING: $failed_count datasets failed"
    exit 1
else
    log_message "✓ All datasets processed successfully"
    exit 0
fi