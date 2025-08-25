import random
from typing import List, Dict, Optional, Callable, Literal
from datasets import load_dataset, Dataset

def process_summarize_feedback_item(item: Dict) -> Dict:
    """Process a single item from HuggingFaceH4/summarize-from-feedback dataset."""
    return {
        "id": item["meta"]["id"],
        "post": item["meta"]["post"],
        "title": item["meta"]["title"],
        "subreddit": item["meta"]["subreddit"],
        "summaries": [
            {"text": s["text"].strip(), "policy": s["policy"]} 
            for s in item["responses"]
        ],
        "chosen_idx": item["label"],
        "chosen_summary": item["responses"][item["label"]]["text"],
        "rejected_summary": item["responses"][1 - item["label"]]["text"],
        "raw_item": item
    }

def process_ultrafeedback_item(item: Dict) -> Dict:
    """Process a single item from UltraFeedback dataset."""
    return {
        "id": item["id"],
        "post": item["post"],
        "title": item["title"],
        "raw_item": item
    }

def load_experiment_dataset(
    dataset_name: str = "HuggingFaceH4/summarize-from-feedback", # summarize-from-feedback, UltraFeedback, Helpsteer etc.
    split: str = "validation",
    count: int = 100,
    random_seed: int = 42,
    filter_fn: Optional[callable] = None
) -> List[Dict]:
    """
    Load and prepare the experiment dataset from OpenAI's Summarize From Feedback.
    
    Args:
        dataset_name: Which dataset variant to use - "HuggingFaceH4/summarize-from-feedback"
        split: Which split to use - "validation" or "test"
        count: Number of examples to sample from the dataset
        random_seed: Random seed for reproducibility
        filter_fn: Optional function to filter dataset entries
                  Should take a dataset item and return True to keep or False to discard
    
    Returns:
        A list of dictionaries containing the sampled data points

    Example:
        >>> # Load 50 examples from the axis variant
        >>> axis_data = load_experiment_dataset("axis", count=50)
        >>> 
        >>> # Load comparison data with a filter for specific subreddits
        >>> filter_fn = lambda x: x["info"]["subreddit"] == "relationship_advice"
        >>> rel_data = load_experiment_dataset("comparisons", filter_fn=filter_fn)
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Load the dataset
    dataset = load_dataset(dataset_name, split=split)
    
    # Apply filter if provided
    if filter_fn:
        dataset = dataset.filter(filter_fn)
    
    # Sample efficiently using shuffle + select
    if count >= len(dataset):
        sampled_dataset = dataset
    else:
        sampled_dataset = dataset.shuffle(seed=random_seed).select(range(count))
    
    # Choose processing function based on dataset
    if dataset_name == "HuggingFaceH4/summarize-from-feedback":
        process_fn = process_summarize_feedback_item
    elif dataset_name == "UltraFeedback":
        process_fn = process_ultrafeedback_item
    else:
        # Generic fallback - just return raw items
        process_fn = lambda x: {"raw_item": x, **x}
    
    # Apply processing function using map
    processed_dataset = sampled_dataset.map(process_fn)
    
    # Convert to list for return
    return list(processed_dataset)
