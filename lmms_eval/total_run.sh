#!/bin/bash

###############################################
###  MASTER RUNNER FOR MULTIPLE EXPERIMENTS ###
###############################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define the scripts to run
SCRIPT1="$SCRIPT_DIR/generate_and_run_custom_videoqa.sh"
SCRIPT2="$SCRIPT_DIR/generate_and_run_LV.sh"

# Master log file
MASTER_LOG="$SCRIPT_DIR/results/full_logs/master_all_experiments_$(date +"%Y%m%d_%H%M%S").log"
mkdir -p "$SCRIPT_DIR/results/full_logs"

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MASTER_LOG"
}

log_message "======================================================"
log_message "MASTER EXPERIMENT RUNNER STARTED"
log_message "======================================================"
log_message "Script 1: $SCRIPT1"
log_message "Script 2: $SCRIPT2"
log_message "Master Log: $MASTER_LOG"
log_message ""

###############################################
###  Run Script 1: Custom Video QA         ###
###############################################
log_message "------------------------------------------------------"
log_message "STARTING: Custom Video QA Experiments"
log_message "------------------------------------------------------"
log_message "Start time: $(date)"

if [[ ! -f "$SCRIPT1" ]]; then
    log_message "ERROR: Script not found: $SCRIPT1"
    exit 1
fi

chmod +x "$SCRIPT1"
bash "$SCRIPT1"
exit_code_1=$?

log_message ""
log_message "FINISHED: Custom Video QA Experiments"
log_message "End time: $(date)"
log_message "Exit code: $exit_code_1"

if [ $exit_code_1 -eq 0 ]; then
    log_message "✓ Custom Video QA completed successfully"
else
    log_message "✗ Custom Video QA FAILED with exit code $exit_code_1"
    log_message "Stopping execution. Fix the error before continuing."
    exit $exit_code_1
fi

log_message ""
log_message "======================================================"

###############################################
###  Run Script 2: LongVideoBench          ###
###############################################
log_message "------------------------------------------------------"
log_message "STARTING: LongVideoBench Experiments"
log_message "------------------------------------------------------"
log_message "Start time: $(date)"

if [[ ! -f "$SCRIPT2" ]]; then
    log_message "ERROR: Script not found: $SCRIPT2"
    exit 1
fi

chmod +x "$SCRIPT2"
bash "$SCRIPT2"
exit_code_2=$?

log_message ""
log_message "FINISHED: LongVideoBench Experiments"
log_message "End time: $(date)"
log_message "Exit code: $exit_code_2"

if [ $exit_code_2 -eq 0 ]; then
    log_message "✓ LongVideoBench completed successfully"
else
    log_message "✗ LongVideoBench FAILED with exit code $exit_code_2"
    exit $exit_code_2
fi

log_message ""
log_message "======================================================"
log_message "ALL EXPERIMENTS COMPLETED SUCCESSFULLY"
log_message "======================================================"
log_message "Total completion time: $(date)"
log_message ""
log_message "Summary:"
log_message "  Custom Video QA: ✓ SUCCESS (exit code: $exit_code_1)"
log_message "  LongVideoBench:  ✓ SUCCESS (exit code: $exit_code_2)"
log_message "======================================================"

exit 0