MODELS=gemini-2.5-flash
MODELS=gpt-4.1
POSITIVE_DATASET=simonycl/gsm8k_training_positive_1k_transformed
# POSITIVE_DATASET=simonycl/gsm8k_training_positive_direct_1k_transformed

# python processing/process_synthetic_negative.py \
#     "method_synthetic_negative_test/${MODELS}_synthetic_negative/generation/direct (samples=1)/responses.jsonl" \
#     --positive_dataset $POSITIVE_DATASET \
#     --output_dataset gsm8k_training_negative_direct_1k_${MODELS}_transformed \
#     --verbose

# python processing/process_synthetic_negative.py \
#     "method_synthetic_negative_test/${MODELS}_synthetic_negative/generation/sequence [strict] (samples=5)/responses.jsonl" \
#     --positive_dataset $POSITIVE_DATASET \
#     --output_dataset gsm8k_training_negative_sequence_1k_${MODELS}_transformed \
#     --verbose

python processing/process_synthetic_negative.py \
    "method_synthetic_negative_test/gpt-4.1_synthetic_negative/generation/multi_turn (samples=5)/responses.jsonl" \
    --positive_dataset $POSITIVE_DATASET \
    --output_dataset gsm8k_training_negative_multi_turn_1k_${MODELS}_transformed \
    --verbose

# python processing/process_synthetic_negative.py \
#     "method_synthetic_negative_test/${MODELS}_synthetic_negative/generation/structure_with_prob [strict] (samples=5)/responses.jsonl" \
#     --positive_dataset $POSITIVE_DATASET \
#     --output_dataset gsm8k_training_negative_vs_standard_1k_${MODELS}_transformed \
#     --verbose
python processing/process_synthetic_negative.py \
    "method_synthetic_negative_test/gpt-4.1_synthetic_negative/generation/chain_of_thought [strict] (samples=5)/responses.jsonl" \
    --positive_dataset $POSITIVE_DATASET \
    --output_dataset gsm8k_training_negative_chain_of_thought_1k_${MODELS}_transformed \
    --verbose


# python processing/process_synthetic_negative.py \
#     "method_synthetic_negative_test/${MODELS}_synthetic_negative/generation/combined [strict] (samples=5)/responses.jsonl" \
#     --positive_dataset $POSITIVE_DATASET \
#     --output_dataset gsm8k_training_negative_combined_1k_${MODELS}_transformed \
#     --verbose