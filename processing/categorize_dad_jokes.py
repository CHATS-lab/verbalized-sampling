#!/usr/bin/env python3
"""
Script to categorize dad jokes from the shuttie/reddit-dadjokes dataset using GPT-5.
Identifies joke categories/themes to collect 200 unique categories.
"""

import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from datasets import load_dataset
from openai import OpenAI
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading
from pydantic import BaseModel
from pydantic import Field


class JokeCategory(BaseModel):
    category: str = Field(description="The joke category, no explanation or reference to the joke in the category")
    explanation: str


def load_dadjokes_dataset(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load the dad jokes dataset."""
    dataset = load_dataset("shuttie/reddit-dadjokes", split="train")
    
    jokes = []
    for i, example in enumerate(dataset):
        if limit and i >= limit:
            break
            
        # Extract joke text - check different possible field names
        joke_text = ""
        if 'text' in example:
            joke_text = example['text']
        elif 'joke' in example:
            joke_text = example['joke']
        elif 'content' in example:
            joke_text = example['content']
        else:
            # Take the first string field we find
            for value in example.values():
                if isinstance(value, str) and len(value) > 10:
                    joke_text = value
                    break
        
        if joke_text.strip():
            jokes.append({
                'id': i,
                'joke': joke_text.strip(),
                'original_example': example
            })
    
    return jokes


def create_categorization_prompt(joke: str) -> str:
    """Create the prompt for GPT to categorize the joke."""
    prompt = f'''You are an expert in humor theory and comedy analysis. Analyze the following dad joke and identify its primary comedic category or theme that defines its humor mechanism.

Joke: "{joke}"

Think about what subject matter or domain does it involve? (family, animals, food, professions, etc.)

Provide a concise, specific category that captures the essence of this joke's humor. Be precise - avoid overly broad categories like "general humor" or "funny wordplay". Instead, use specific descriptors like "animal", "dad profession", or "food".'''
    return prompt


def call_gpt_with_structured_output(prompt: str, max_retries: int = 3) -> Optional[JokeCategory]:
    """Call GPT API with structured output using Pydantic."""
    client = OpenAI()
    
    for attempt in range(max_retries):
        try:
            response = client.responses.parse(
                model="gpt-4.1",
                input=[
                    {"role": "user", "content": prompt}
                ],
                text_format=JokeCategory
            )
            
            return response.output_parsed
            
        except Exception as e:
            if "rate_limit" in str(e).lower():
                time.sleep(1)  # Short delay for rate limits
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
    
    return None


def process_single_joke(joke_data: Dict[str, Any], pbar: tqdm, categories_seen: set, 
                       category_counts: Dict[str, int], lock: threading.Lock) -> Optional[Dict[str, Any]]:
    """Process a single joke and return the result."""
    joke = joke_data['joke']
    
    # Create prompt
    prompt = create_categorization_prompt(joke)
    
    # Call GPT with structured output
    result = call_gpt_with_structured_output(prompt)
    
    if result:
        category = result.category.strip()
        explanation = result.explanation.strip()
        
        # Normalize category (lowercase, remove extra spaces)
        normalized_category = ' '.join(category.lower().split())
        
        # Thread-safe update of shared data
        with lock:
            categories_seen.add(normalized_category)
            category_counts[normalized_category] += 1
            pbar.set_postfix({
                'categories': len(categories_seen),
                'current': category[:30] + '...' if len(category) > 30 else category
            })
        
        # Create output record
        output_record = {
            'id': joke_data['id'],
            'joke': joke,
            'category': category,
            'explanation': explanation,
            'normalized_category': normalized_category,
            'prompt': prompt,
            'response': result.model_dump()
        }
        
        pbar.update(1)
        return output_record
    else:
        pbar.update(1)
        return None


def main():
    parser = argparse.ArgumentParser(description='Categorize dad jokes using GPT-5')
    parser.add_argument('--limit', type=int, default=500,
                       help='Limit number of jokes to process (default: 500)')
    parser.add_argument('--output_file', type=str, 
                       default='processing/dad_jokes_categories.jsonl',
                       help='Output file for categorized jokes')
    parser.add_argument('--target_categories', type=int, default=200,
                       help='Target number of unique categories (default: 200)')
    parser.add_argument('--max_workers', type=int, default=64,
                       help='Maximum number of parallel workers (default: 10)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print verbose output')
    
    args = parser.parse_args()
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    
    # Load dataset
    print(f"Loading dad jokes dataset (limit: {args.limit})")
    jokes = load_dadjokes_dataset(args.limit)
    print(f"Loaded {len(jokes)} jokes")
    
    if not jokes:
        print("No jokes found in dataset")
        return
    
    # Create output directory
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Track categories
    categories_seen = set()
    category_counts = defaultdict(int)
    processed_jokes = []
    lock = threading.Lock()
    
    print(f"Processing jokes to find {args.target_categories} unique categories...")
    print(f"Using {args.max_workers} parallel workers")
    
    # Create progress bar
    pbar = tqdm(total=len(jokes), desc="Processing jokes", 
                unit="joke", dynamic_ncols=True)
    
    # Process jokes in parallel
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all jokes for processing
        future_to_joke = {
            executor.submit(process_single_joke, joke_data, pbar, categories_seen, category_counts, lock): joke_data
            for joke_data in jokes
        }
        
        # Process completed futures
        for future in as_completed(future_to_joke):
            # Check if we've reached our target
            if len(categories_seen) >= args.target_categories:
                # Cancel remaining futures
                for f in future_to_joke:
                    if not f.done():
                        f.cancel()
                break
                
            result = future.result()
            if result:
                processed_jokes.append(result)
                
                # Write incrementally (thread-safe)
                with lock:
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(result) + '\n')
    
    pbar.close()
    
    print(f"\nProcessing complete!")
    print(f"Total jokes processed: {len(processed_jokes)}")
    print(f"Unique categories found: {len(categories_seen)}")
    print(f"Results saved to: {output_file}")
    
    # Print top categories
    print(f"\nTop 20 most common categories:")
    sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    for category, count in sorted_categories[:20]:
        print(f"  {category}: {count} jokes")
    
    # Save summary
    summary_file = output_file.with_suffix('.summary.json')
    summary = {
        'total_jokes_processed': len(processed_jokes),
        'unique_categories': len(categories_seen),
        'target_categories': args.target_categories,
        'category_counts': dict(category_counts),
        'all_categories': sorted(list(categories_seen))
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_file}")


if __name__ == '__main__':
    main()