import random
from typing import List, Dict, Optional, Callable, Literal
from datasets import load_dataset

def load_experiment_dataset(
    variant: Literal["axis", "comparisons"] = "axis",
    count: int = 100,
    random_seed: int = 42,
    filter_fn: Optional[callable] = None
) -> List[Dict]:
    """
    Load and prepare the experiment dataset from OpenAI's Summarize From Feedback.
    
    Args:
        variant: Which dataset variant to use - "axis" (individual summary ratings)
                or "comparisons" (pairwise summary ratings)
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
    dataset = load_dataset("openai/summarize_from_feedback", variant)
    
    # Convert to list for easier processing
    data_list = list(dataset["validation"])
    
    # Apply filter if provided
    if filter_fn:
        data_list = [item for item in data_list if filter_fn(item)]
    
    # Sample the requested number of examples (or all if count > len)
    if count >= len(data_list):
        sampled_data = data_list
    else:
        sampled_data = random.sample(data_list, count)
    
    # Process the data based on variant
    processed_data = []
    
    if variant == "axis":
        for item in sampled_data:
            processed_item = {
                "id": item["info"]["id"],
                "post": item["info"]["post"],
                "title": item["info"]["title"],
                "subreddit": item["info"]["subreddit"],
                "summary": item["summary"]["text"].strip(),
                "policy": item["summary"]["policy"],
                "rating": item["summary"]["axes"]["overall"],
                "accuracy": item["summary"]["axes"]["accuracy"],
                "coverage": item["summary"]["axes"]["coverage"],
                "coherence": item["summary"]["axes"]["coherence"],
                "raw_item": item  # Keep the original item for reference
            }
            processed_data.append(processed_item)
    
    elif variant == "comparisons":
        for item in sampled_data:
            processed_item = {
                "id": item["info"]["id"],
                "post": item["info"]["post"],
                "title": item["info"]["title"],
                "subreddit": item["info"]["subreddit"],
                "summaries": [
                    {"text": s["text"].strip(), "policy": s["policy"]} 
                    for s in item["summaries"]
                ],
                "chosen_idx": item["choice"],
                "chosen_summary": item["summaries"][item["choice"]]["text"],
                "rejected_summary": item["summaries"][1 - item["choice"]]["text"],
                "raw_item": item  # Keep the original item for reference
            }
            processed_data.append(processed_item)
    
    return processed_data
