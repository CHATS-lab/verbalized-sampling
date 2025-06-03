# #!/bin/bash

MODELS=(
    openai/gpt-4.1
    openai/gpt-4.1-mini
    openai/o4-mini
    google/gemini-2.0-flash-001
    google/gemini-2.5-flash-preview-05-20
    # anthropic/claude-opus-4
    anthropic/claude-sonnet-4
    meta-llama/Llama-3.1-70B-Instruct
    meta-llama/Llama-3.3-70B-Instruct
)

# Create output directory
OUTPUT_DIR="plots/story_comparisons"
mkdir -p $OUTPUT_DIR

# First pass: analyze each model and save results
# for MODEL in ${MODELS[@]}; do
#     MODEL_NAME=$(basename $MODEL)
#     NUM_RESPONSES=10
#     PROMPT_TEMPLATE="story_all"
#     OUTPUT_FILE=outputs/${MODEL_NAME}/story_all_${NUM_RESPONSES}.jsonl
    
#     echo "Analyzing $MODEL_NAME..."
#     python tasks/story/analyze.py --output_file $OUTPUT_FILE --output_dir $OUTPUT_DIR
# done

# Second pass: create comparison plots using Python to handle JSON
# echo "Creating comparison plots..."
# python -c "
# import json
# import os
# import sys

# # Get all result files
# output_dir = '$OUTPUT_DIR'
# model_results = {}
# for filename in os.listdir(output_dir):
#     if filename.endswith('_results.json'):
#         model_name = filename.replace('_results.json', '')
#         model_results[model_name] = os.path.join(output_dir, filename)

# # Convert to JSON string
# print(json.dumps(model_results))
# " | python tasks/story/analyze.py --plot_comparisons --model_results "$(cat)" --output_dir $OUTPUT_DIR


# NUMS=(1 2 4 6 8 10)
TOTAL_RESPONSES=200
NUMS=(1 2 5 10 20)
MODEL=anthropic/claude-sonnet-4
MODEL_NAME=claude-sonnet-4
for NUM in ${NUMS[@]}; do
    python tasks/story/diversity_measure.py \
        --output_file outputs/${MODEL_NAME}/story_num_${NUM}.jsonl \
        --output_dir plots/story_num_${NUM}
done

python tasks/story/diversity_measure.py \
    --plot_comparisons \
    --model_results plots/story_num_1 plots/story_num_2 plots/story_num_5 plots/story_num_10 \
    --output_dir plots/story_num_main \
    --model_name ${MODEL_NAME}


python tasks/story/diversity_measure.py \
    --plot_comparisons \
    --model_results plots/story_num_5 plots/story_num_10 plots/story_num_20 \
    --output_dir plots/story_num_ablation \
    --model_name ${MODEL_NAME}

