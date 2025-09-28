python process_aime_dataset.py \
    --local-dir synthetic_amc_aime/amc_aime_training_synthetic_positive_direct.json \
    --target simonycl/amc_aime_training_positive_direct_qwen3-32b

python process_aime_dataset.py \
    --local-dir synthetic_amc_aime/amc_aime_training_synthetic_positive_sequence.json \
    --target simonycl/amc_aime_training_positive_sequence_qwen3-32b

python process_aime_dataset.py \
    --local-dir synthetic_amc_aime/amc_aime_training_synthetic_positive_vs_standard.json \
    --target simonycl/amc_aime_training_positive_vs_standard_qwen3-32b

python process_aime_dataset.py \
    --local-dir synthetic_amc_aime/amc_aime_training_synthetic_positive_vs_cot.json \
    --target simonycl/amc_aime_training_positive_vs_cot_qwen3-32b