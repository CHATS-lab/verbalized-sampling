#!/usr/bin/env python3
"""
Script to load and process HarmBench dataset from walledai/HarmBench.

This script downloads the HarmBench safety dataset and processes it into
a format compatible with the verbalized sampling framework.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from datasets import load_dataset
import random


def load_harmbench_dataset(split: str = "train", config: str = "standard") -> List[Dict[str, Any]]:
    """
    Load the HarmBench dataset from Hugging Face.

    Args:
        split: Dataset split to load (default: "train")
        config: Dataset config to load (default: "standard")

    Returns:
        List of examples from the dataset
    """
    try:
        # Load the dataset from walledai/HarmBench
        dataset = load_dataset("walledai/HarmBench", config, split=split)
        examples = list(dataset)

        # Debug: print first example structure
        if examples:
            print(f"Dataset loaded with {len(examples)} examples")
            print(f"First example type: {type(examples[0])}")
            if isinstance(examples[0], dict):
                print(f"First example keys: {list(examples[0].keys())}")
            print(f"First example: {str(examples[0])[:200]}...")

        return examples
    except Exception as e:
        print(f"Error loading HarmBench dataset: {e}")
        print("Make sure you have internet connection and access to Hugging Face.")
        return []


def process_harmbench_data(examples: List[Dict[str, Any]],
                          max_samples: int = None,
                          random_seed: int = 42) -> List[str]:
    """
    Process HarmBench examples into safety prompts.

    Args:
        examples: Raw examples from HarmBench dataset
        max_samples: Maximum number of samples to process (None for all)
        random_seed: Random seed for sampling

    Returns:
        List of processed safety prompts
    """
    if not examples:
        return []

    # Set random seed for reproducible sampling
    if random_seed is not None:
        random.seed(random_seed)

    # Extract prompts/behaviors from the dataset
    prompts = []
    for example in examples:
        # Handle case where example is already a string
        if isinstance(example, str):
            prompts.append(example)
        # Handle case where example is a dict
        elif isinstance(example, dict):
            # The exact structure may vary, but typically HarmBench contains 'behavior' or 'prompt' fields
            if 'behavior' in example:
                prompts.append(example['behavior'])
            elif 'prompt' in example:
                prompts.append(example['prompt'])
            elif 'text' in example:
                prompts.append(example['text'])
            else:
                # Fallback: use the first string field found
                for value in example.values():
                    if isinstance(value, str) and len(value) > 10:  # Reasonable length check
                        prompts.append(value)
                        break
        else:
            # Debug: print the actual type and structure
            print(f"Unexpected example type: {type(example)}")
            print(f"Example content: {str(example)[:100]}...")
            continue

    # Remove duplicates while preserving order
    seen = set()
    unique_prompts = []
    for prompt in prompts:
        if prompt not in seen:
            seen.add(prompt)
            unique_prompts.append(prompt)

    # Sample if max_samples is specified
    if max_samples and len(unique_prompts) > max_samples:
        unique_prompts = random.sample(unique_prompts, max_samples)

    return unique_prompts


def save_safety_data(prompts: List[str], output_dir: str = "data"):
    """
    Save processed safety prompts to files.

    Args:
        prompts: List of safety prompts
        output_dir: Directory to save the data files
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Save as simple text file (one prompt per line)
    txt_file = output_path / "safety.txt"
    with open(txt_file, 'w', encoding='utf-8') as f:
        for prompt in prompts:
            # Clean the prompt (remove newlines within prompts for txt format)
            clean_prompt = prompt.replace('\n', ' ').replace('\r', ' ').strip()
            f.write(clean_prompt + '\n')

    # Also save as JSONL for more structured storage
    jsonl_file = output_path / "safety.jsonl"
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for i, prompt in enumerate(prompts):
            data = {
                "id": i,
                "prompt": prompt,
                "source": "harmbench"
            }
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"Saved {len(prompts)} safety prompts to:")
    print(f"  - {txt_file}")
    print(f"  - {jsonl_file}")


def main():
    """Main function to load and process HarmBench data."""
    parser = argparse.ArgumentParser(description="Load and process HarmBench safety dataset")
    parser.add_argument("--split", default="train", help="Dataset split to load")
    parser.add_argument("--config", default="standard", help="Dataset config to load (standard, contextual, copyright)")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples to process")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--output-dir", default="data", help="Output directory for processed data")

    args = parser.parse_args()

    print("Loading HarmBench dataset...")
    examples = load_harmbench_dataset(args.split, args.config)

    if not examples:
        print("Failed to load dataset. Exiting.")
        return

    print(f"Loaded {len(examples)} examples from HarmBench")

    print("Processing data...")
    prompts = process_harmbench_data(
        examples,
        max_samples=args.max_samples,
        random_seed=args.random_seed
    )

    if not prompts:
        print("No prompts extracted from dataset. Check dataset structure.")
        return

    print(f"Processed {len(prompts)} unique safety prompts")

    # Show a few examples
    print("\nExample prompts:")
    for i, prompt in enumerate(prompts[:3]):
        print(f"{i+1}. {prompt[:100]}{'...' if len(prompt) > 100 else ''}")

    save_safety_data(prompts, args.output_dir)
    print("\nData processing complete!")


if __name__ == "__main__":
    main()