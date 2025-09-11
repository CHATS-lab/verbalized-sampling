#!/usr/bin/env python3
"""
Post-hoc processing script for synthetic negative test results.
Processes negative data, combines with positive HF dataset, and pushes to Hub.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
from datasets import Dataset, load_dataset, concatenate_datasets


def extract_answer_from_text(text: str) -> Optional[str]:
    """Extract answer from text using the #### [answer] format."""
    pattern = r'####\s*(-?\d+(?:\.\d+)?)'
    match = re.search(pattern, text.strip())
    if match:
        return match.group(1).strip()
    return None


def extract_instruction_and_golden_answer(prompt: str) -> tuple[str, str]:
    """
    Extract instruction and golden answer from the prompt.
    Expected format: "Here's the math problem: [instruction]\nGolden answer: [golden_answer]"
    """
    # Split by "Golden answer:" to separate instruction from golden answer
    parts = prompt.split("Golden answer:")
    if len(parts) != 2:
        # Fallback: use entire prompt as instruction, empty golden answer
        return prompt.strip(), ""
    
    instruction_part = parts[0].strip()
    golden_answer_part = parts[1].strip()
    
    # Clean up instruction (remove "Here's the math problem:" prefix if present)
    instruction = re.sub(r'^Here\'s the math problem:\s*', '', instruction_part)
    
    return instruction.strip(), golden_answer_part.strip()


def process_jsonl_file(file_path: Path) -> List[Dict[str, Any]]:
    """Process a single JSONL file and return flattened records."""
    processed_records = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                # Extract instruction and golden answer from prompt
                instruction, golden_answer = extract_instruction_and_golden_answer(data['prompt'])
                
                # Extract golden answer (final answer after ####)
                golden_output = extract_answer_from_text(golden_answer)
                
                # Process each response
                for resp_idx, response in enumerate(data.get('responses', [])):
                    output_text = response.get('text', '')
                    
                    # Extract answer from output
                    output_answer = extract_answer_from_text(output_text)
                    
                    # Check if answer exists in required format
                    has_valid_format = output_answer is not None
                    
                    # Check if output answer differs from golden answer
                    answers_differ = (output_answer != golden_output) if (output_answer is not None and golden_output is not None) else None
                    
                    record = {
                        'instruction': instruction,
                        'golden_output': golden_answer,
                        'output': output_text,
                        'label': -1,
                        'golden_answer': golden_output,
                        'output_answer': output_answer,
                        'has_valid_format': has_valid_format,
                        'answers_differ': answers_differ,
                        'source_file': str(file_path),
                        'line_number': line_num,
                        'response_index': resp_idx
                    }
                    
                    processed_records.append(record)
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num} in {file_path}: {e}")
                continue
    
    return processed_records


def main():
    parser = argparse.ArgumentParser(description='Process synthetic negative test results')
    parser.add_argument('input_file', type=str, 
                       help='Path to the responses.jsonl file to process')
    parser.add_argument('--positive_dataset', type=str, 
                       default='simonycl/gsm8k_training_positive_1k_transformed',
                       help='HuggingFace positive dataset name')
    parser.add_argument('--output_dataset', type=str, required=True,
                       help='Output HuggingFace dataset name (under simonycl/)')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    
    args = parser.parse_args()
    
    input_file = Path(args.input_file)
    
    if not input_file.exists():
        print(f"Error: Input file {input_file} does not exist")
        return
    
    if args.verbose:
        print(f"Processing {input_file}")
    
    # Process the file
    all_records = process_jsonl_file(input_file)
    
    # Filter records: keep only those with valid format AND different answers
    filtered_records = []
    for record in all_records:
        if record['has_valid_format'] and record['answers_differ'] is True:
            output_text = record['output'].strip('\n')
            output_text = output_text.split("####")[0].strip()
            output_answer = record['output_answer']
            
            # Create truncated output format
            truncated_output = f"{output_text}\n#### {output_answer}"
            
            # Create messages format
            system_message = "You are an AI assistant who is an expert at solving math word problems. Think step by step then provide the numerical answer at the end after the delimiter '####', like '#### 24'."
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": record['instruction']},
                {"role": "assistant", "content": truncated_output}
            ]
            
            # Remove the extra fields used for filtering
            filtered_record = {
                'messages': messages,
                'label': -1
            }
            filtered_records.append(filtered_record)
    
    if not filtered_records:
        print("No records match the criteria (valid format + different answers)")
        return
    
    # Create negative dataset
    negative_dataset = Dataset.from_list(filtered_records)
    print(f"Created negative dataset with {len(negative_dataset)} examples")
    
    # Load positive dataset
    if args.verbose:
        print(f"Loading positive dataset: {args.positive_dataset}")
    
    positive_dataset = load_dataset(args.positive_dataset, split='train')
    print(f"Loaded positive dataset with {len(positive_dataset)} examples")
    
    # Add label=1 to positive dataset
    def add_positive_label(example):
        example['label'] = 1
        return example
    
    positive_dataset = positive_dataset.map(add_positive_label)
    
    # Combine datasets
    combined_dataset = concatenate_datasets([negative_dataset, positive_dataset])
    print(f"Combined dataset size: {len(combined_dataset)}")
    
    # Label distribution
    labels = combined_dataset['label']
    negative_count = sum(1 for label in labels if label == -1)
    positive_count = sum(1 for label in labels if label == 1)
    print(f"Label distribution: {negative_count} negative, {positive_count} positive")
    
    # Push to HuggingFace Hub
    output_name = f"simonycl/{args.output_dataset}"
    print(f"Pushing to HuggingFace Hub: {output_name}")
    combined_dataset.push_to_hub(output_name)
    print(f"Successfully pushed to {output_name}")
    
    if args.verbose:
        print(f"\nProcessing Summary:")
        print(f"  Total records processed: {len(all_records)}")
        print(f"  Negative examples (filtered): {len(filtered_records)}")
        print(f"  Positive examples: {len(positive_dataset)}")
        print(f"  Final dataset size: {len(combined_dataset)}")


if __name__ == '__main__':
    main()