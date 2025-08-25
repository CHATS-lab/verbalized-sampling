#!/bin/bash

# Batch experiment runner for rlpref project
# This script runs experiments across multiple models and dataset variants

# Configuration
SAMPLES=2500
OUTPUT_FORMAT="pdf"
SEED=42

# Create timestamped results directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BATCH_DIR="batch_results_${TIMESTAMP}"
mkdir -p "$BATCH_DIR"

echo "Starting batch experiments in directory: $BATCH_DIR"
echo "========================================"

# Function to run a single experiment
run_experiment() {
    local model="$1"
    local use_4bit="$2"
    local samples="$3"
    local dataset="$4"
    
    # Create safe model name and dataset name for directory
    local model_safe=$(echo "$model" | sed 's/\//_/g')
    local dataset_safe=$(echo "$dataset" | sed 's/\//_/g')
    local experiment_dir="${BATCH_DIR}/${model_safe}_${dataset_safe}"
    mkdir -p "$experiment_dir"
    
    echo "Running: $model - $dataset ($samples samples)"
    echo "Output: $experiment_dir"
    
    # Build command
    local cmd="python main.py --model \"$model\" --dataset \"$dataset\" --samples $samples --seed $SEED --output \"$experiment_dir\" --output-format $OUTPUT_FORMAT --unload-model"
    
    if [ "$use_4bit" = "true" ]; then
        cmd="$cmd --use_4bit"
    fi
    
    echo "Command: $cmd"
    echo "----------------------------------------"
    
    # Run experiment and capture output
    if eval "$cmd" > "${experiment_dir}/stdout.log" 2> "${experiment_dir}/stderr.log"; then
        echo "✓ Experiment completed successfully"
        echo "SUCCESS: $model - $dataset" >> "${BATCH_DIR}/results_summary.txt"
    else
        echo "✗ Experiment failed (see ${experiment_dir}/stderr.log)"
        echo "FAILED: $model - $dataset" >> "${BATCH_DIR}/results_summary.txt"
    fi
    echo ""
}

# Model configurations: model_name:use_4bit
declare -A MODELS=(
    ["Qwen/Qwen3-4B-Base"]="false"
    ["google/gemma-3-4b-pt"]="false"
    ["meta-llama/Llama-3.1-8B"]="false"
    ["google/gemma-3-27b-pt"]="false"
)
DATASETS=(
    "HuggingFaceH4/summarize-from-feedback"
    "HuggingFaceH4/ultrafeedback_binarized"
    "nvidia/HelpSteer3"
    "Skywork/Skywork-Reward-Preference-80K-v0.2"
)

# Initialize summary file
echo "Batch experiment results - $(date)" > "${BATCH_DIR}/results_summary.txt"
echo "=====================================" >> "${BATCH_DIR}/results_summary.txt"

# Run experiments for each model and dataset combination
for model in "${!MODELS[@]}"; do
    use_4bit="${MODELS[$model]}"
    echo "=== Running experiments for $model ==="
    
    for dataset in "${DATASETS[@]}"; do
        run_experiment "$model" "$use_4bit" "$SAMPLES" "$dataset"
    done
done

# Print final summary
echo "========================================"
echo "BATCH EXPERIMENTS COMPLETED"
echo "Results directory: $BATCH_DIR"
echo ""
echo "Summary:"
cat "${BATCH_DIR}/results_summary.txt"

# Count successes and failures
SUCCESS_COUNT=$(grep -c "^SUCCESS:" "${BATCH_DIR}/results_summary.txt" || echo "0")
FAILED_COUNT=$(grep -c "^FAILED:" "${BATCH_DIR}/results_summary.txt" || echo "0")
TOTAL_COUNT=$((SUCCESS_COUNT + FAILED_COUNT))

echo ""
echo "Total experiments: $TOTAL_COUNT"
echo "Successful: $SUCCESS_COUNT"
echo "Failed: $FAILED_COUNT"

if [ "$FAILED_COUNT" -gt 0 ]; then
    exit 1
else
    exit 0
fi