# MODEL=meta-llama/Llama-3.3-70B-Instruct
MODEL=google/gemini-2.0-flash-001
MODELS=(
    openai/gpt-4.1
    openai/gpt-4.1-mini
    openai/o4-mini
)
MODEL=openai/o4-mini
MODEL=google/gemini-2.0-flash-001
MODEL=openai/gpt-4.1
MODEL=google/gemini-2.5-flash-preview-05-20

# MODEL=meta-llama/llama-3.3-70b-instruct
TOTAL=200
NUM_RESPONSES=10
NUM_SAMPLES=(1 3 5 10)
# NUM_SAMPLES=(50)
SIM_TYPE=sampling
TEMPERATURE=1.0
TOP_P=1.0
MODEL_NAME=$(basename $MODEL)
TASK=story
FORMATS=("direct" "seq" "structure" "structure_with_prob")
# FORMATS=("structure")
FORMATS=("seq" "structure_with_prob")
FORMATS=("direct" "seq" "structure_with_prob")
mkdir -p outputs/${MODEL_NAME}/${TASK}
for FORMAT in ${FORMATS[@]}; do
    if [ $FORMAT == "direct" ]; then
        NUM_SAMPLES=(1)
    else
        NUM_SAMPLES=(3 5 10)
    fi
    for NUM_SAMPLES in ${NUM_SAMPLES[@]}; do
        OUTPUT_FILE=outputs/${MODEL_NAME}/${TASK}/${FORMAT}_${TEMPERATURE}_${TOP_P}_${NUM_RESPONSES}_${NUM_SAMPLES}.jsonl
        NUM_RESPONSES=$(($TOTAL / $NUM_SAMPLES))
        python main.py \
        --model_name $MODEL \
        --format $FORMAT \
        --temperature $TEMPERATURE \
        --top_p $TOP_P \
        --num_responses $NUM_RESPONSES \
        --num_samples $NUM_SAMPLES \
        --output_file $OUTPUT_FILE \
        --task $TASK
    done
done

# stop_server
