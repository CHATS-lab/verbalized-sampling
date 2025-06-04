#!/bin/bash


MODEL="openai/gpt-4.1"
MODEL_BASENAME=$(basename $MODEL)
# Run the CLI with the given parameters
mkdir -p outputs/$MODEL_BASENAME

python -m verbalized_sampling.cli run-experiment \
    --model-name $MODEL \
    --task creative_story \
    --method direct \
    --num-responses 5 \
    --num-samples 1 \
    --num-workers 128 \
    --output-file outputs/$MODEL_BASENAME/creative_story_direct.json