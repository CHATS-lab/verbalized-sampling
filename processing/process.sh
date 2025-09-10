python processing/process_synthetic_negative.py \
    "method_synthetic_negative_test/gpt-4.1_synthetic_negative/generation/direct (samples=1)/responses.jsonl" \
    --positive_dataset simonycl/gsm8k_training_positive_direct_1k_transformed \
    --output_dataset gsm8k_training_negative_direct_1k_gpt-4.1_transformed \
    --verbose

python processing/process_synthetic_negative.py \
    "method_synthetic_negative_test/gpt-4.1_synthetic_negative/generation/sequence [strict] (samples=5)/responses.jsonl" \
    --positive_dataset simonycl/gsm8k_training_positive_direct_1k_transformed \
    --output_dataset gsm8k_training_negative_sequence_1k_gpt-4.1_transformed \
    --verbose

python processing/process_synthetic_negative.py \
    "method_synthetic_negative_test/gpt-4.1_synthetic_negative/generation/structure_with_prob [strict] (samples=5)/responses.jsonl" \
    --positive_dataset simonycl/gsm8k_training_positive_direct_1k_transformed \
    --output_dataset gsm8k_training_negative_vs_standard_1k_gpt-4.1_transformed \
    --verbose

python processing/process_synthetic_negative.py \
    "method_synthetic_negative_test/gpt-4.1_synthetic_negative/generation/combined [strict] (samples=5)/responses.jsonl" \
    --positive_dataset simonycl/gsm8k_training_positive_direct_1k_transformed \
    --output_dataset gsm8k_training_negative_combined_1k_gpt-4.1_transformed \
    --verbose