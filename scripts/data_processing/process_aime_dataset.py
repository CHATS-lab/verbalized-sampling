#!/usr/bin/env python3

import os
import json
from datasets import load_dataset, Dataset
from verbalized_sampling.llms.openai import OpenAILLM
from typing import List, Dict, Any
import argparse

def format_messages(system_content: str, instruction: str) -> List[Dict[str, str]]:
    """Format system and instruction into message format."""
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": instruction}
    ]

def process_local_dataset(json_file: str, target_dataset_name: str):
    """Process local JSON files and push to HuggingFace."""


    print(f"Loading {json_file}")

    # Load all data from JSON files
    all_data = []
    with open(json_file, 'r') as f:
        data = json.load(f)
        all_data.extend(data)

    print(f"Loaded {len(all_data)} examples total")

    # Examine first example
    if all_data:
        print("First example:")
        print(f"Instruction: {all_data[0]['instruction'][:100]}...")
        print(f"Output: {all_data[0]['output'][:100]}...")

    # Format data into HuggingFace dataset structure
    formatted_data = []
    for i, example in enumerate(all_data):
        if not all(key in example for key in ['instruction', 'output']):
            print(f"Warning: Missing required fields in example {i}")
            continue

        # Create the complete message chain
        messages = [
            # {"role": "system", "content": example['system']},
            {"role": "user", "content": example['instruction']},
            {"role": "assistant", "content": example['output']}
        ]

        formatted_data.append({"messages": messages})

    print(f"Formatted {len(formatted_data)} examples")

    # Create new dataset
    new_dataset = Dataset.from_list(formatted_data)

    print(f"Pushing to HuggingFace as: {target_dataset_name}")
    new_dataset.push_to_hub(target_dataset_name)

    print("Dataset processing complete!")
    return new_dataset

def process_dataset(dataset_name: str, target_dataset_name: str, num_workers: int = 10):
    """Process the dataset by generating GPT-4.1 responses."""

    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)

    # Get the train split
    train_data = dataset['train']
    print(f"Dataset loaded with {len(train_data)} examples")

    # Examine first few examples
    print("First example:")
    print(f"Instruction: {train_data[0]['instruction']}")

    # Initialize OpenAI LLM with GPT-4.1
    config = {
        'temperature': 0.7,
        'max_tokens': 4096,
    }

    llm = OpenAILLM(
        model_name="gpt-4.1",
        config=config,
        num_workers=num_workers
    )

    print(f"Using model: {llm.model_name} with {num_workers} workers")

    # Format all messages for parallel processing
    all_messages = []
    for example in train_data:
        messages = format_messages(example['system'], example['instruction'])
        all_messages.append(messages)

    print(f"Formatted {len(all_messages)} message sets")
    print("Generating responses with GPT-4.1...")

    # Generate responses in parallel
    responses = llm.chat(all_messages)

    print(f"Generated {len(responses)} responses")

    # Format data into final structure
    formatted_data = []
    for i, (example, response) in enumerate(zip(train_data, responses)):
        if response is None:
            print(f"Warning: No response for example {i}")
            continue

        # Create the complete message chain
        messages = [
            # {"role": "system", "content": example['system']},
            {"role": "user", "content": example['instruction']},
            {"role": "assistant", "content": response}
        ]

        formatted_data.append({"messages": messages})

    print(f"Formatted {len(formatted_data)} examples")

    # Create new dataset
    new_dataset = Dataset.from_list(formatted_data)

    print(f"Pushing to HuggingFace as: {target_dataset_name}")
    new_dataset.push_to_hub(target_dataset_name)

    print("Dataset processing complete!")
    return new_dataset

def main():
    parser = argparse.ArgumentParser(description="Process dataset and push to HuggingFace")
    parser.add_argument(
        "--local-dir",
        help="Path to local directory containing JSON files (e.g., synthetic_lcb/)"
    )
    parser.add_argument(
        "--source",
        default="EleanorZzz/gsm8k_training_positive_1k",
        help="Source dataset name (for HuggingFace datasets)"
    )
    parser.add_argument(
        "--target",
        default="simonycl/gsm8k_training_positive_1k_regenerated",
        help="Target dataset name"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of parallel workers (for GPT-4.1 generation)"
    )

    args = parser.parse_args()

    if args.local_dir:
        # Process local JSON files
        process_local_dataset(args.local_dir, args.target)
    else:
        # Check for OpenAI API key
        if not os.environ.get("OPENAI_API_KEY"):
            print("Error: OPENAI_API_KEY environment variable not set")
            return

        # Process HuggingFace dataset with GPT-4.1 generation
        process_dataset(args.source, args.target, args.workers)

if __name__ == "__main__":
    main()

# python process_gsm8k_dataset.py --local-dir synthetic_lcb/lcb_training_synthetic_positive_sequence.json --target simonycl/lcb_training_positive_sequence
# python process_gsm8k_dataset.py --local-dir synthetic_lcb/lcb_training_synthetic_positive_vs_standard.json --target simonycl/lcb_training_positive_vs_standard
# python process_gsm8k_dataset.py --local-dir synthetic_lcb/lcb_training_synthetic_positive_vs_cot.json --target simonycl/lcb_training_positive_vs_cot
# python process_gsm8k_dataset.py --local-dir synthetic_lcb/lcb_training_synthetic_positive_direct.json --target simonycl/lcb_training_synthetic_positive_direct
# python process_gsm8k_dataset.py --local-dir synthetic_lcb/lcb_training_synthetic_positive_direct_1k.json --target simonycl/lcb_training_positive_direct_1k
# python process_gsm8k_dataset.py --local-dir synthetic_lcb/lcb_training_synthetic_positive_direct_1k.json --target simonycl/lcb_training_positive_direct_1k