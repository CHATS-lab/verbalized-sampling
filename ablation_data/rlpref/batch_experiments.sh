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
    # "nvidia/HelpSteer3"
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

# Generate markdown results table
generate_markdown_table() {
    local markdown_file="${BATCH_DIR}/experiment_results.md"
    
    echo "# Batch Experiment Results" > "$markdown_file"
    echo "" >> "$markdown_file"
    echo "**Run Date:** $(date)" >> "$markdown_file"
    echo "**Samples per experiment:** $SAMPLES" >> "$markdown_file"
    echo "**Random seed:** $SEED" >> "$markdown_file"
    echo "" >> "$markdown_file"
    
    # Create table header
    echo "| Model | Dataset | Status | Samples | Agreement Rate | Confidence Interval | Output Directory |" >> "$markdown_file"
    echo "|-------|---------|--------|---------|----------------|---------------------|------------------|" >> "$markdown_file"
    
    # Parse results and add to table
    for model in "${!MODELS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            local model_safe=$(echo "$model" | sed 's/\//_/g')
            local dataset_safe=$(echo "$dataset" | sed 's/\//_/g')
            local experiment_dir="${BATCH_DIR}/${model_safe}_${dataset_safe}"
            local results_file="${experiment_dir}/comparisons_results_${model_safe}_${dataset_safe}.json"
            
            # Check if experiment was successful
            if grep -q "SUCCESS: $model - $dataset" "${BATCH_DIR}/results_summary.txt"; then
                local status="✅"
                
                # Extract results from JSON file if it exists
                if [ -f "$results_file" ]; then
                    # Use python to parse JSON and extract key metrics
                    local metrics=$(python3 -c "
import json
import sys
try:
    with open('$results_file', 'r') as f:
        data = json.load(f)
    
    # Extract key metrics
    samples = data.get('num_samples', 'N/A')
    agreement = data.get('model_human_agreement', 'N/A')
    
    # Format agreement rate as percentage
    if agreement != 'N/A' and agreement is not None:
        agreement_pct = f'{float(agreement)*100:.1f}%'
    else:
        agreement_pct = 'N/A'
    
    # Extract confidence interval if available
    stats = data.get('agreement_stats', {})
    ci_lower = stats.get('confidence_interval_lower', None)
    ci_upper = stats.get('confidence_interval_upper', None)
    
    if ci_lower is not None and ci_upper is not None:
        ci = f'[{float(ci_lower)*100:.1f}%, {float(ci_upper)*100:.1f}%]'
    else:
        ci = 'N/A'
    
    print(f'{samples}|{agreement_pct}|{ci}')
    
except Exception as e:
    print('N/A|N/A|N/A')
" 2>/dev/null)
                    
                    # Split the metrics
                    local samples=$(echo "$metrics" | cut -d'|' -f1)
                    local agreement=$(echo "$metrics" | cut -d'|' -f2)
                    local ci=$(echo "$metrics" | cut -d'|' -f3)
                else
                    local samples="N/A"
                    local agreement="N/A"
                    local ci="N/A"
                fi
            else
                local status="❌"
                local samples="N/A"
                local agreement="N/A"
                local ci="N/A"
            fi
            
            # Add row to table
            echo "| $model | $dataset | $status | $samples | $agreement | $ci | \`${experiment_dir}\` |" >> "$markdown_file"
        done
    done
    
    # Add summary section
    echo "" >> "$markdown_file"
    echo "## Summary" >> "$markdown_file"
    echo "" >> "$markdown_file"
    echo "- **Total experiments:** $TOTAL_COUNT" >> "$markdown_file"
    echo "- **Successful:** $SUCCESS_COUNT" >> "$markdown_file"
    echo "- **Failed:** $FAILED_COUNT" >> "$markdown_file"
    echo "- **Success rate:** $(( SUCCESS_COUNT * 100 / TOTAL_COUNT ))%" >> "$markdown_file"
    
    # Find best performing model-dataset combination
    if [ "$SUCCESS_COUNT" -gt 0 ]; then
        echo "" >> "$markdown_file"
        echo "## Top Performing Combinations" >> "$markdown_file"
        echo "" >> "$markdown_file"
        
        # Use python to find top performers
        python3 -c "
import json
import os
import glob

results = []
for model in '${!MODELS[@]}'.split():
    for dataset in '${DATASETS[@]}'.split():
        model_safe = model.replace('/', '_')
        dataset_safe = dataset.replace('/', '_')
        results_file = '${BATCH_DIR}/' + model_safe + '_' + dataset_safe + '/comparisons_results_' + model_safe + '_' + dataset_safe + '.json'
        
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)
                
                agreement = data.get('model_human_agreement', 0)
                if agreement and agreement != 'N/A':
                    results.append({
                        'model': model,
                        'dataset': dataset,
                        'agreement': float(agreement),
                        'samples': data.get('num_samples', 0)
                    })
            except:
                continue

# Sort by agreement rate
results.sort(key=lambda x: x['agreement'], reverse=True)

# Show top 3
for i, result in enumerate(results[:3]):
    print(f'{i+1}. **{result[\"model\"]}** on **{result[\"dataset\"]}**: {result[\"agreement\"]*100:.1f}% agreement ({result[\"samples\"]} samples)')
" >> "$markdown_file" 2>/dev/null
    fi
    
    # Add details about failed experiments if any
    if [ "$FAILED_COUNT" -gt 0 ]; then
        echo "" >> "$markdown_file"
        echo "## Failed Experiments" >> "$markdown_file"
        echo "" >> "$markdown_file"
        grep "^FAILED:" "${BATCH_DIR}/results_summary.txt" | while read -r line; do
            local failed_exp=$(echo "$line" | sed 's/FAILED: //')
            local model_part=$(echo "$failed_exp" | cut -d' - ' -f1)
            local dataset_part=$(echo "$failed_exp" | cut -d' - ' -f2)
            local model_safe=$(echo "$model_part" | sed 's/\//_/g')
            local dataset_safe=$(echo "$dataset_part" | sed 's/\//_/g')
            local experiment_dir="${BATCH_DIR}/${model_safe}_${dataset_safe}"
            echo "- **$failed_exp**: Check \`${experiment_dir}/stderr.log\` for error details" >> "$markdown_file"
        done
    fi
    
    echo "" >> "$markdown_file"
    echo "## File Structure" >> "$markdown_file"
    echo "" >> "$markdown_file"
    echo "Each experiment directory contains:" >> "$markdown_file"
    echo "- \`stdout.log\` - Standard output from the experiment" >> "$markdown_file"
    echo "- \`stderr.log\` - Error output (if any)" >> "$markdown_file"
    echo "- \`comparisons_results_{model}_{dataset}.json\` - Experiment results (if successful)" >> "$markdown_file"
    echo "- \`*.pdf\` - Generated plots (if analysis was not skipped)" >> "$markdown_file"
    
    echo "Markdown summary created: $markdown_file"
}

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

# Generate markdown table
generate_markdown_table

if [ "$FAILED_COUNT" -gt 0 ]; then
    exit 1
else
    exit 0
fi