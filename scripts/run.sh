#!/bin/bash


MODEL="openai/gpt-4.1"
MODEL_BASENAME=$(basename $MODEL)
# Run the CLI with the given parameters
mkdir -p outputs/$MODEL_BASENAME
TASK="poem"
METHOD="direct"
NUM_RESPONSES=20
NUM_SAMPLES=5
NUM_WORKERS=16
OUTPUT_FILE="outputs/$MODEL_BASENAME/$TASK/$METHOD.json"
SAMPLE_SIZE=5
RANDOM_SEED=42

python -m verbalized_sampling.cli run-experiment \
    --model-name $MODEL \
    --task $TASK \
    --method $METHOD \
    --num-responses $NUM_RESPONSES \
    --num-samples $NUM_SAMPLES \
    --num-workers $NUM_WORKERS \
    --output-file $OUTPUT_FILE \
    --sample-size $SAMPLE_SIZE \
    --random-seed $RANDOM_SEED

# python -m verbalized_sampling.cli evaluate \
#     --metric diversity \
#     --task creative_story \
#     --input-file outputs/$MODEL_BASENAME/creative_story_direct.json \
#     --output-file outputs/$MODEL_BASENAME/creative_story_direct_diversity.json

# python -m verbalized_sampling.cli evaluate \
#     --metric diversity \
#     --task creative_story \
#     --input-file outputs/$MODEL_BASENAME/creative_story_direct.json \
#     --output-file outputs/$MODEL_BASENAME/creative_story_direct_diversity.json