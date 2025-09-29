from datasets import load_dataset
from datasets import Dataset

dataset = load_dataset("EleanorZzz/gsm8k_training_synthetic_positive_direct_cot", split="train")

transformed_data = []
for example in dataset:
    messages = [
        {"role": "system", "content": example["system"]},
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]}
    ]
    transformed_data.append({"messages": messages})

new_dataset = Dataset.from_list(transformed_data)
new_dataset.push_to_hub("simonycl/gsm8k_training_positive_direct_cot_1k_transformed", split="train")