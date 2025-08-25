"""
Token log probability calculation and analysis for language models.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict
import copy

import numpy as np
from utilities import format_prompt, prepare_for_serialization, save_json_results


def get_token_logprobs(
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
    prompt: str,
    response: str
) -> List[Tuple[str, float]]:
    """
    Get the log probabilities for each token in the response given a prompt.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: The prompt text
        response: The response text to compute logprobs for
            
    Returns:
        A list of tuples, each containing (token_text, log_probability)
        
    Example:
        >>> model, tokenizer = load_model("distilgpt2", use_4bit=False)
        >>> token_logprobs = get_token_logprobs(model, tokenizer, prompt="Question", response="Answer")
    """
    # Create prompts using the generic format
    prompt_only = format_prompt(prompt=prompt)
    full_prompt_with_response = format_prompt(prompt=prompt, response=response)
    
    # Tokenize both prompts
    prompt_tokens = tokenizer(prompt_only, return_tensors="pt").to(model.device)
    full_tokens = tokenizer(full_prompt_with_response, return_tensors="pt").to(model.device)
    
    # Get the length of the prompt without response
    prompt_length = len(prompt_tokens.input_ids[0])
    
    # Get model outputs for the full sequence
    with torch.no_grad():
        outputs = model(full_tokens.input_ids)
        
    # Get the logits (model's raw predictions)
    logits = outputs.logits[0]
    
    # Calculate log probabilities using log_softmax
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    
    # Extract logprobs for the response tokens
    token_logprobs = []
    input_ids = full_tokens.input_ids[0]
    
    # Start from where the response begins
    # The prompt ends at prompt_length - 1, so start at prompt_length
    for i in range(prompt_length, len(input_ids)):
        token_id = input_ids[i].item()
        token_text = tokenizer.decode([token_id])
        # The logprob for position i comes from position i-1
        logprob = log_probs[i-1, token_id].item()
        token_logprobs.append((token_text, logprob))
    
    return token_logprobs


def extract_token_logprobs(analysis_results: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract token-level average log probabilities from experiment results.
    
    Args:
        analysis_results: Dictionary containing experiment results, 
                          either from run_axis_experiment or run_comparisons_experiment
                          
    Returns:
        Dictionary mapping tokens to their average log probability across all samples
        
    Example:
        >>> results = run_axis_experiment(...)
        >>> token_logprobs = extract_token_logprobs(results)
        >>> print(token_logprobs["the"])  # Average logprob for token "the"
    """
    # Dictionary to store token counts and sum of logprobs
    token_sums = defaultdict(float)  # Sum of log probabilities for each token
    token_counts = defaultdict(int)  # Number of occurrences of each token
    
    # Check if we have sample_results field (present in both experiment types)
    if "sample_results" not in analysis_results:
        print("Error: No sample_results found in the analysis_results")
        return {}
    
    # Function to process a list of token logprobs
    def process_token_logprobs(token_logprobs_list):
        for token, logprob in token_logprobs_list:
            # Skip empty tokens
            if not token or token.isspace():
                continue
                
            # Add to running sums and counts
            token_sums[token] += logprob
            token_counts[token] += 1
    
    # Determine the experiment type
    is_axis = False
    is_comparisons = False
    
    # Check first item to determine experiment type
    if analysis_results["sample_results"]:
        first_item = analysis_results["sample_results"][0]
        is_axis = "token_logprobs" in first_item
        is_comparisons = "chosen_token_logprobs" in first_item
    
    # Process tokens based on experiment type
    if is_axis:
        print("Processing axis experiment results...")
        for item in analysis_results["sample_results"]:
            if "token_logprobs" in item:
                process_token_logprobs(item["token_logprobs"])
    
    elif is_comparisons:
        print("Processing comparisons experiment results...")
        for item in analysis_results["sample_results"]:
            # Process both chosen and rejected summaries
            if "chosen_token_logprobs" in item:
                process_token_logprobs(item["chosen_token_logprobs"])
            if "rejected_token_logprobs" in item:
                process_token_logprobs(item["rejected_token_logprobs"])
    
    else:
        print("Unknown experiment type in the provided results")
        return {}
    
    # Calculate average log probability for each token
    avg_logprobs = {}
    for token, total in token_sums.items():
        count = token_counts[token]
        if count > 0:
            avg_logprobs[token] = total / count
    
    # Print some statistics
    print(f"Extracted average log probabilities for {len(avg_logprobs)} unique tokens")
    
    # Show a few examples of token probabilities (sorted by frequency)
    sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
    if sorted_tokens:
        print("\nTop 10 most frequent tokens and their average log probabilities:")
        for token, count in sorted_tokens[:10]:
            print(f"  '{token}' (count: {count}): {avg_logprobs[token]:.4f}")
    
    return avg_logprobs


def filter_tokens_by_logprob(analysis_results: Dict[str, Any], threshold: float) -> Dict[str, Any]:
    """
    Filter out tokens with average log probabilities above a threshold.
    
    Args:
        analysis_results: Dictionary containing experiment results
        threshold: Minimum log probability threshold for tokens to keep
        
    Returns:
        A new analysis_results dictionary with filtered token_logprobs lists
        
    Example:
        >>> results = run_axis_experiment(...)
        >>> filtered_results = filter_tokens_by_logprob(results, -3.0)
    """
    # Make a deep copy of the analysis_results to avoid modifying the original
    filtered_results = copy.deepcopy(analysis_results)
    
    # No sample_results means nothing to filter
    if "sample_results" not in filtered_results:
        print("Error: No sample_results found in the analysis_results")
        return filtered_results
    
    # Get token-level average logprobs to determine which tokens to keep
    token_logprobs = extract_token_logprobs(analysis_results)
    
    # Create a set of tokens to keep (those below threshold)
    tokens_to_keep = {token for token, logprob in token_logprobs.items() if logprob <= threshold}
    
    # Track statistics
    total_tokens = 0
    kept_tokens = 0
    
    # Function to filter token_logprobs list
    def filter_token_list(token_logprobs_list):
        nonlocal total_tokens, kept_tokens
        filtered_list = []
        for token, logprob in token_logprobs_list:
            total_tokens += 1
            if token in tokens_to_keep:
                filtered_list.append((token, logprob))
                kept_tokens += 1
        return filtered_list
    
    # Determine experiment type
    is_axis = False
    is_comparisons = False
    
    if filtered_results["sample_results"]:
        first_item = filtered_results["sample_results"][0]
        is_axis = "token_logprobs" in first_item
        is_comparisons = "chosen_token_logprobs" in first_item
    
    # Filter based on experiment type
    if is_axis:
        for item in filtered_results["sample_results"]:
            if "token_logprobs" in item:
                item["token_logprobs"] = filter_token_list(item["token_logprobs"])
                # Update token count and average/sum logprobs
                if item["token_logprobs"]:
                    item["token_count"] = len(item["token_logprobs"])
                    item["sum_logprob"] = sum(lp for _, lp in item["token_logprobs"])
                    item["avg_logprob"] = item["sum_logprob"] / item["token_count"] if item["token_count"] > 0 else 0
                else:
                    # Handle empty case
                    item["token_count"] = 0
                    item["sum_logprob"] = 0
                    item["avg_logprob"] = 0
    
    elif is_comparisons:
        for item in filtered_results["sample_results"]:
            if "chosen_token_logprobs" in item:
                item["chosen_token_logprobs"] = filter_token_list(item["chosen_token_logprobs"])
                # Update chosen token count and logprobs
                if item["chosen_token_logprobs"]:
                    item["chosen_token_count"] = len(item["chosen_token_logprobs"])
                    item["chosen_sum_logprob"] = sum(lp for _, lp in item["chosen_token_logprobs"])
                    item["chosen_avg_logprob"] = item["chosen_sum_logprob"] / item["chosen_token_count"] if item["chosen_token_count"] > 0 else 0
                else:
                    item["chosen_token_count"] = 0
                    item["chosen_sum_logprob"] = 0
                    item["chosen_avg_logprob"] = 0
                
            if "rejected_token_logprobs" in item:
                item["rejected_token_logprobs"] = filter_token_list(item["rejected_token_logprobs"])
                # Update rejected token count and logprobs
                if item["rejected_token_logprobs"]:
                    item["rejected_token_count"] = len(item["rejected_token_logprobs"])
                    item["rejected_sum_logprob"] = sum(lp for _, lp in item["rejected_token_logprobs"]) 
                    item["rejected_avg_logprob"] = item["rejected_sum_logprob"] / item["rejected_token_count"] if item["rejected_token_count"] > 0 else 0
                else:
                    item["rejected_token_count"] = 0
                    item["rejected_sum_logprob"] = 0 
                    item["rejected_avg_logprob"] = 0
                
                # Update model_preferred_chosen based on new average logprobs
                item["model_preferred_chosen"] = item["chosen_avg_logprob"] > item["rejected_avg_logprob"]
    
    # Print statistics about the filtering
    if total_tokens > 0:
        keep_percentage = (kept_tokens / total_tokens) * 100
        print(f"Token filtering: keeping {kept_tokens} out of {total_tokens} tokens ({keep_percentage:.1f}%)")
        print(f"Unique tokens kept: {len(tokens_to_keep)} with average logprob >= {threshold}")
    
    # Add metadata about the filtering
    filtered_results["filtering"] = {
        "threshold": threshold,
        "total_tokens_before": total_tokens,
        "tokens_kept": kept_tokens,
        "unique_tokens_kept": len(tokens_to_keep),
        "keep_percentage": (kept_tokens / total_tokens) * 100 if total_tokens > 0 else 0
    }
    
    return filtered_results


def analyze_logprobs(token_logprobs: List[Tuple[str, float]]) -> Dict[str, Any]:
    """
    Analyze log probabilities to calculate summary statistics.
    
    Args:
        token_logprobs: List of (token, logprob) tuples
        
    Returns:
        Dictionary with summary statistics
    """
    if not token_logprobs:
        return {
            "sum_logprob": 0.0,
            "avg_logprob": 0.0,
            "min_logprob": 0.0,
            "max_logprob": 0.0,
            "token_count": 0
        }
    
    logprobs = [lp for _, lp in token_logprobs]
    
    return {
        "sum_logprob": sum(logprobs),
        "avg_logprob": sum(logprobs) / len(logprobs),
        "min_logprob": min(logprobs),
        "max_logprob": max(logprobs),
        "token_count": len(logprobs)
    }


def save_token_logprobs(token_logprobs: Dict[str, float], output_path: str, sort_by: str = "frequency") -> None:
    """
    Save token logprobs to a JSON file with sorting options.
    
    Args:
        token_logprobs: Dictionary mapping tokens to their average log probabilities
        output_path: Path to save the JSON file
        sort_by: Sorting method: "frequency" (default), "token", "logprob", or "none"
    """
    # Prepare the data with probability information
    token_data = {}
    for token, logprob in token_logprobs.items():
        token_data[token] = {
            "logprob": logprob,
            "prob": np.exp(logprob)
        }
    
    # Sort the data according to the specified method
    if sort_by == "token":
        # Sort alphabetically by token
        sorted_items = sorted(token_data.items())
    elif sort_by == "logprob":
        # Sort by descending log probability 
        sorted_items = sorted(token_data.items(), key=lambda x: x[1]["logprob"], reverse=True)
    elif sort_by == "frequency":
        # We don't have frequency information here, so we'll just sort by token
        sorted_items = sorted(token_data.items())
    else:
        # No sorting, just use the dictionary as is
        sorted_items = token_data.items()
    
    # Convert to a dictionary to maintain sort order
    sorted_dict = {k: v for k, v in sorted_items}
    
    # Save to file using our utility function
    save_json_results(sorted_dict, output_path)

