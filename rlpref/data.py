import random
from typing import List, Dict, Optional, Callable, Literal
from datasets import load_dataset, Dataset

def process_summarize_feedback_item(item: Dict) -> Dict:
    """Process a single item from HuggingFaceH4/summarize-from-feedback dataset."""
    # Extract prompt from post and title
    prompt = item["meta"]["post"]
    if item["meta"]["title"]:
        prompt = f"Title: {item['meta']['title']}\n\n{prompt}"
    
    return {
        "id": item["meta"]["id"],
        "prompt": prompt,
        "chosen": item["responses"][item["label"]]["text"].strip(),
        "rejected": item["responses"][1 - item["label"]]["text"].strip(),
        "raw_item": item
    }

def process_generic_preference_item(item: Dict) -> Dict:
    """Process a generic preference dataset item with prompt, chosen, rejected fields."""
    prompt = item["chosen"][0]["content"]
    chosen = item["chosen"][1]["content"]
    rejected = item["rejected"][1]["content"]
    processed = {
        "id": item.get("id", item.get("prompt_id", "unknown")),
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "raw_item": item
    }
    
    return processed

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
    
    assert dataset_name in [
        "HuggingFaceH4/summarize-from-feedback",
        "HuggingFaceH4/ultrafeedback_binarized",
        # "nvidia/HelpSteer3",
        "Skywork/Skywork-Reward-Preference-80K-v0.2"
    ]
    # Load the dataset
    if dataset_name == "HuggingFaceH4/summarize-from-feedback":
        dataset = load_dataset(dataset_name, split="validation")
    elif dataset_name == "HuggingFaceH4/ultrafeedback_binarized":
        dataset = load_dataset(dataset_name, split="test_prefs")
    # elif dataset_name == "nvidia/HelpSteer3":
    #     dataset = load_dataset(dataset_name, split="validation")
    elif dataset_name == "Skywork/Skywork-Reward-Preference-80K-v0.2":
        dataset = load_dataset(dataset_name, split="train")
    else:
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
    else:
        # Try to detect if it's a generic preference dataset
        # Sample one item to check structure
        sample_item = dataset[0]
        if all(key in sample_item for key in ["chosen", "rejected"]):
            process_fn = process_generic_preference_item
        else:
            # Fallback - assume it needs conversion to standard format
            def fallback_process(item):
                # Try common field mappings
                prompt = item.get("prompt", item.get("question", item.get("input", "")))
                chosen = item.get("chosen", item.get("response_1", item.get("answer_a", "")))
                rejected = item.get("rejected", item.get("response_2", item.get("answer_b", "")))
                
                return {
                    "id": item.get("id", item.get("prompt_id", "unknown")),
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "raw_item": item
                }
            process_fn = fallback_process
    
    # Apply processing function using map
    processed_dataset = sampled_dataset.map(process_fn)
    
    # Convert to list for return
    return list(processed_dataset)
