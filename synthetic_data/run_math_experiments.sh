#!/bin/bash

# Math Model Testing Experiments
# Run comprehensive math tests on Qwen models
# Assumes vLLM server is already running at localhost:8000

set -e  # Exit on any error

# Configuration
BASE_URL="http://localhost:8000"
NUM_SAMPLES=50  # Adjust based on how comprehensive you want the test
SEED=42
OUTPUT_DIR="math_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging
LOG_FILE="math_experiments_${TIMESTAMP}.log"

log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}


run_experiment() {
    local model_name="$1"
    local datasets="$2"
    local samples="$3"
    local experiment_name="$4"
    local experiment_type="${5:-test}"  # Default to test, can be "pipeline"

    log "\n${BLUE}🚀 Starting experiment: ${experiment_name}${NC}"
    log "${BLUE}Model: ${model_name}${NC}"
    log "${BLUE}Type: ${experiment_type}${NC}"
    log "${BLUE}Datasets: ${datasets}${NC}"
    log "${BLUE}Samples per dataset: ${samples}${NC}"

    if [[ "$experiment_type" == "pipeline" ]]; then
        # Use the pipeline-based approach
        log "${BLUE}Using pipeline-based generation${NC}"
        cd "$(dirname "$0")/../scripts/tasks"
        python run_math_simple.py 2>&1 | tee -a "$LOG_FILE"
        cd - > /dev/null
    else
        # Use the direct test approach
        # Create safe filename
        safe_model_name=$(echo "$model_name" | sed 's/\//_/g')
        output_file="${OUTPUT_DIR}/${experiment_name}_${safe_model_name}_${TIMESTAMP}"

        # Run the test
        python test_math.py \
            --models "$model_name" \
            --datasets $datasets \
            --num_samples "$samples" \
            --seed "$SEED" \
            --base_url "$BASE_URL" \
            --output "$output_file" \
            2>&1 | tee -a "$LOG_FILE"
    fi

    if [ $? -eq 0 ]; then
        log "${GREEN}✅ Experiment completed: ${experiment_name}${NC}"
    else
        log "${RED}❌ Experiment failed: ${experiment_name}${NC}"
        return 1
    fi
}

main() {
    # Parse command line arguments first (before any output)
    SELECTED_EXP="standard_test"

    while [[ $# -gt 0 ]]; do
        case $1 in
            --experiment)
                SELECTED_EXP="$2"
                shift 2
                ;;
            --samples)
                NUM_SAMPLES="$2"
                shift 2
                ;;
            --quick)
                SELECTED_EXP="quick_test"
                shift
                ;;
            --comprehensive)
                SELECTED_EXP="comprehensive_test"
                shift
                ;;
            --full)
                SELECTED_EXP="full_evaluation"
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --experiment NAME    Run specific experiment"
                echo "  --samples N          Override number of samples per dataset"
                echo "  --quick              Run quick test (10 samples, math+amc)"
                echo "  --comprehensive      Run comprehensive test (all datasets except olympiad)"
                echo "  --full               Run full evaluation (all datasets)"
                echo "  --help, -h           Show this help"
                echo ""
                echo "Available experiments:"
                for exp in "${EXPERIMENTS[@]}"; do
                    IFS='|' read -r name datasets samples <<< "$exp"
                    echo "  - ${name}: ${datasets} (${samples} samples)"
                done
                exit 0
                ;;
            *)
                log "${RED}Unknown option: $1${NC}"
                log "Use --help for usage information"
                exit 1
                ;;
        esac
    done

    # Now initialize logging and output
    log "${BLUE}🧮 Math Model Testing Experiments${NC}"
    log "${BLUE}Started at: $(date)${NC}"
    log "${BLUE}Configuration:${NC}"
    log "  Base URL: ${BASE_URL}"
    log "  Samples per dataset: ${NUM_SAMPLES}"
    log "  Random seed: ${SEED}"
    log "  Output directory: ${OUTPUT_DIR}"
    log "  Log file: ${LOG_FILE}"

    # Create output directory
    mkdir -p "$OUTPUT_DIR"

    # Check if vLLM server is running and get current model
    CURRENT_MODEL="Qwen/Qwen3-4B"

    # Experiments configuration
    # Format: "experiment_name|datasets|samples"
    EXPERIMENTS=(
        "quick_test|math|10"
        "standard_test|math aime amc|${NUM_SAMPLES}"
        "comprehensive_test|math aime amc minerva|${NUM_SAMPLES}"
        "full_evaluation|math aime amc minerva olympiad_bench|${NUM_SAMPLES}"
        "pipeline_simple|math|20|pipeline"
    )

    log "\n${YELLOW}Available experiments:${NC}"
    for i in "${!EXPERIMENTS[@]}"; do
        IFS='|' read -r name datasets samples <<< "${EXPERIMENTS[$i]}"
        log "  $((i+1)). ${name} - Datasets: ${datasets} (${samples} samples each)"
    done

    # Find and run selected experiment
    FOUND_EXP=false
    for exp in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r name datasets samples exp_type <<< "$exp"
        if [[ "$name" == "$SELECTED_EXP" ]]; then
            # Override samples if specified
            if [[ "$samples" == "${NUM_SAMPLES}" ]] && [[ "$NUM_SAMPLES" != "50" ]]; then
                samples="$NUM_SAMPLES"
            fi

            # Set default experiment type if not specified
            if [[ -z "$exp_type" ]]; then
                exp_type="test"
            fi

            log "\n${YELLOW}🎯 Running experiment: ${SELECTED_EXP}${NC}"
            run_experiment "$CURRENT_MODEL" "$datasets" "$samples" "$SELECTED_EXP" "$exp_type"
            FOUND_EXP=true
            break
        fi
    done

    if [[ "$FOUND_EXP" == "false" ]]; then
        log "${RED}❌ Unknown experiment: ${SELECTED_EXP}${NC}"
        log "${YELLOW}Available experiments: ${EXPERIMENTS[*]}${NC}"
        exit 1
    fi

    # Final summary
    log "\n${GREEN}🎉 All experiments completed!${NC}"
    log "${BLUE}Results saved in: ${OUTPUT_DIR}/${NC}"
    log "${BLUE}Log file: ${LOG_FILE}${NC}"
    log "${BLUE}Finished at: $(date)${NC}"

    # Show result files
    log "\n${YELLOW}Generated files:${NC}"
    find "$OUTPUT_DIR" -name "*${TIMESTAMP}*" -type f | while read -r file; do
        log "  📄 $(basename "$file")"
    done
}

# Script usage examples as comments
cat << 'EOF' > /dev/null
# Usage Examples:

# Quick test (10 samples on math and amc datasets)
./run_math_experiments.sh --quick

# Standard test with custom sample count
./run_math_experiments.sh --samples 30

# Comprehensive test (all datasets except olympiad_bench)
./run_math_experiments.sh --comprehensive

# Full evaluation (all datasets)
./run_math_experiments.sh --full

# Run specific experiment
./run_math_experiments.sh --experiment quick_test

# Before running, make sure vLLM server is started:
# vllm serve Qwen/Qwen3-4B-Base --port 8000
# OR
# vllm serve Qwen/Qwen3-4B-Thinking-2507 --port 8000

EOF

# Run main function
main "$@"