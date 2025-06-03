export PYTHONPATH="$PYTHONPATH:/scratch/simon/verbalized_sampling"

MODEL=openai/gpt-4.1
MODEL=meta-llama/Llama-3.1-70B-Instruct
MODEL=anthropic/claude-sonnet-4
MODELS=(
    # openai/gpt-4.1
    # openai/gpt-4.1-mini
    # openai/o4-mini
    # google/gemini-2.0-flash-001
    # google/gemini-2.5-flash-preview-05-20
    # anthropic/claude-opus-4
    anthropic/claude-sonnet-4
    # meta-llama/Llama-3.1-70B-Instruct
    # meta-llama/Llama-3.3-70B-Instruct
)

for MODEL in ${MODELS[@]}; do
#     MODEL_NAME=$(basename $MODEL)
#     NUM_RESPONSES=10
#     PROMPT_TEMPLATE="story_all"
#     OUTPUT_FILE=outputs/${MODEL_NAME}/story_all_${NUM_RESPONSES}.jsonl

#     mkdir -p outputs/${MODEL_NAME}
#     python tasks/story/generate.py \
#         --model_name $MODEL \
#         --num_responses $NUM_RESPONSES \
#         --prompt_template $PROMPT_TEMPLATE \
#         --output_file $OUTPUT_FILE

    MODEL_NAME=$(basename $MODEL)
    TOTAL_RESPONSES=100
    PROMPT_TEMPLATE="story_num"
    # NUMS=(4 6 8 10)
    NUMS=(1 2 5 10 20)
    # NUMS=(1)
    for NUM in ${NUMS[@]}; do
        NUM_SAMPLES=$((TOTAL_RESPONSES / NUM))
        
        echo "Generating $NUM_SAMPLES samples for $NUM numbers"
        OUTPUT_FILE=outputs/${MODEL_NAME}/story_num_${NUM}.jsonl

        mkdir -p outputs/${MODEL_NAME}
        python tasks/story/generate.py \
            --model_name $MODEL \
            --num_responses $NUM \
            --num_samples $NUM_SAMPLES \
            --prompt_template $PROMPT_TEMPLATE \
            --output_file $OUTPUT_FILE \
            --prompt "Generate a creative story about potatoes with at least 5 sentences."
    done
done