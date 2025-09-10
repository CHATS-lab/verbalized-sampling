"""
Implementation of the experiments specified in README.md.

This module contains functions for running various experiment variants:
- Axis experiment: correlation between individual summary ratings and model probabilities
- Comparisons experiment: agreement between model probabilities and human preferences
"""

import random
import torch
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
from typing import List, Dict, Union, Literal, Optional, Tuple, Any
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import pearsonr, spearmanr, binomtest
import statsmodels.stats.proportion as smp
from tqdm import tqdm

from data import load_experiment_dataset
from model import load_model, unload_model
from logprobs import get_token_logprobs, analyze_logprobs
from utilities import log_execution_time, save_json_results
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from results import analyze_axis_results, plot_axis_results, analyze_comparisons_results, plot_comparisons_results



def run_axis_experiment(
    model_name: str,
    model=None,
    tokenizer=None,
    dataset_name: str = "HuggingFaceH4/summarize-from-feedback",
    num_samples: int = 100,
    random_seed: int = 42,
    use_4bit: bool = False,
    results_dir: str = ".",
    skip_analysis: bool = False,
    output_format: str = "pdf",
    jitter_seed: int = 42
) -> Dict[str, Any]:
    """
    Run the axis variant of the experiment.
    
    Args:
        model_name: The name of the model (for reporting)
        model: Pre-loaded model (if None, will load the model)
        tokenizer: Pre-loaded tokenizer (if None, will load the tokenizer)
        num_samples: Number of samples to use from the dataset
        random_seed: Random seed for reproducibility
        use_4bit: Whether to use 4-bit quantization for the model (only used if model is None)
        results_dir: Directory to save results and plots
        skip_analysis: If True, skips analysis and plotting (just collects data)
        output_format: Format for saving plots ('pdf' or 'png')
        jitter_seed: Seed for jittering points in scatter plots
    
    Returns:
        Dictionary with experiment results
    """
    start_time = time.time()
    print(f"Starting axis experiment with model: {model_name}")
    print(f"Samples: {num_samples}, Random seed: {random_seed}")
    
    # Load the dataset  
    print("Loading dataset...")
    data = load_experiment_dataset(dataset_name, count=num_samples, random_seed=random_seed)
    print(f"Loaded {len(data)} examples")
    
    # Load the model if not provided
    if model is None or tokenizer is None:
        print(f"Model not provided, loading model: {model_name}")
        model, tokenizer = load_model(model_name, use_4bit=use_4bit)
    
    # Process each example
    results = []
    
    print("Processing examples...")
    try:
        # Use tqdm for a progress bar
        for i, example in enumerate(tqdm(data, desc="Processing", unit="example")):
            try:
                # Get logprobs for the chosen response
                token_logprobs = get_token_logprobs(
                    model, 
                    tokenizer, 
                    prompt=example["prompt"],
                    response=example["chosen"]
                )
                
                # Analyze the logprobs
                logprob_stats = analyze_logprobs(token_logprobs)
                
                # Add results to the example
                result = {
                    "id": example["id"],
                    "rating": example.get("rating", example.get("score", 0)),
                    "token_count": logprob_stats["token_count"],
                    "sum_logprob": logprob_stats["sum_logprob"],
                    "avg_logprob": logprob_stats["avg_logprob"],
                    "min_logprob": logprob_stats["min_logprob"],
                    "max_logprob": logprob_stats["max_logprob"],
                    "token_logprobs": token_logprobs,  # Store individual token logprobs
                    "response": example["chosen"],
                    "prompt": example["prompt"]
                }
                
                results.append(result)
            except Exception as e:
                print(f"Error processing example {i+1}: {e}")
                # Continue with the next example
    except KeyboardInterrupt:
        print("\nExperiment interrupted. Saving partial results...")
    
    if not results:
        print("No results were obtained. Exiting.")
        return None
    
    print(f"Processed {len(results)}/{len(data)} examples successfully")
    
    # Create results summary
    experiment_results = {
        "model": model_name,
        "num_samples": len(results),
        "execution_time": time.time() - start_time,
        "sample_results": results
    }
    
    # Run analysis if requested
    if not skip_analysis:
        analysis_results = analyze_axis_results(results, model_name)
        if analysis_results:
            # Add analysis results to the main results dictionary
            for key, value in analysis_results.items():
                if key != "data":  # Don't duplicate the data
                    experiment_results[key] = value
            
            # Generate plots
            plots = plot_axis_results(
                analysis_results, 
                model_name, 
                results_dir, 
                jitter_seed=jitter_seed,
                output_format=output_format
            )
            experiment_results["plots"] = plots
    
    log_execution_time(start_time, "Axis experiment")
    
    return experiment_results


def run_comparisons_experiment(
    model_name: str,
    model=None,
    tokenizer=None,
    dataset_name: str = "HuggingFaceH4/summarize-from-feedback",
    num_samples: int = 100,
    random_seed: int = 42,
    use_4bit: bool = False,
    results_dir: str = ".",
    skip_analysis: bool = False,
    output_format: str = "pdf",
    jitter_seed: int = 42
) -> Dict[str, Any]:
    """
    Run the comparisons variant of the experiment.
    
    Args:
        model_name: The name of the model (for reporting)
        model: Pre-loaded model (if None, will load the model)
        tokenizer: Pre-loaded tokenizer (if None, will load the tokenizer)
        num_samples: Number of samples to use from the dataset
        random_seed: Random seed for reproducibility
        use_4bit: Whether to use 4-bit quantization for the model (only used if model is None)
        results_dir: Directory to save results and plots
        skip_analysis: If True, skips analysis and plotting (just collects data)
        output_format: Format for saving plots ('pdf' or 'png')
        jitter_seed: Seed for jittering points in scatter plots
    
    Returns:
        Dictionary with experiment results
    """
    start_time = time.time()
    print(f"Starting comparisons experiment with model: {model_name}")
    print(f"Samples: {num_samples}, Random seed: {random_seed}")
    
    # Load the dataset
    print("Loading dataset...")
    data = load_experiment_dataset(dataset_name, count=num_samples, random_seed=random_seed)
    print(f"Loaded {len(data)} examples")
    
    # Load the model if not provided
    if model is None or tokenizer is None:
        print(f"Model not provided, loading model: {model_name}")
        model, tokenizer = load_model(model_name, use_4bit=use_4bit)
    
    # Process each example
    results = []
    
    print("Processing examples...")
    try:
        for i, example in enumerate(tqdm(data, desc="Processing", unit="example")):
            try:
                # Get logprobs for both responses
                chosen_logprobs = get_token_logprobs(
                    model, 
                    tokenizer, 
                    prompt=example["prompt"],
                    response=example["chosen"]
                )
                rejected_logprobs = get_token_logprobs(
                    model, 
                    tokenizer, 
                    prompt=example["prompt"],
                    response=example["rejected"]
                )
                
                # Analyze the logprobs
                chosen_stats = analyze_logprobs(chosen_logprobs)
                rejected_stats = analyze_logprobs(rejected_logprobs)
                
                # Determine if the model preferred the human-chosen summary
                model_preferred_chosen = chosen_stats["avg_logprob"] > rejected_stats["avg_logprob"]
                
                # Add results to the example
                result = {
                    "id": example["id"],
                    "prompt": example["prompt"],
                    "chosen": example["chosen"],
                    "rejected": example["rejected"],
                    "chosen_token_count": chosen_stats["token_count"],
                    "chosen_sum_logprob": chosen_stats["sum_logprob"],
                    "chosen_avg_logprob": chosen_stats["avg_logprob"],
                    "rejected_token_count": rejected_stats["token_count"],
                    "rejected_sum_logprob": rejected_stats["sum_logprob"],
                    "rejected_avg_logprob": rejected_stats["avg_logprob"],
                    "model_preferred_chosen": model_preferred_chosen,
                    "chosen_token_logprobs": chosen_logprobs,
                    "rejected_token_logprobs": rejected_logprobs
                }
                
                results.append(result)
            except Exception as e:
                print(f"Error processing example {i+1}: {e}")
                # Continue with the next example
    except KeyboardInterrupt:
        print("\nExperiment interrupted. Saving partial results...")
    
    if not results:
        print("No results were obtained. Exiting.")
        return None
    
    print(f"Processed {len(results)}/{len(data)} examples successfully")
    
    # Create results summary
    experiment_results = {
        "model": model_name,
        "num_samples": len(results),
        "execution_time": time.time() - start_time,
        "sample_results": results
    }
    
    # Run analysis if requested
    if not skip_analysis:
        analysis_results = analyze_comparisons_results(results, model_name)
        if analysis_results:
            # Add analysis results to the main results dictionary
            for key, value in analysis_results.items():
                if key != "data":  # Don't duplicate the data
                    experiment_results[key] = value
            
            # Generate plots
            plots = plot_comparisons_results(
                analysis_results, 
                model_name, 
                results_dir, 
                jitter_seed=jitter_seed,
                output_format=output_format
            )
            experiment_results["plots"] = plots
    
    log_execution_time(start_time, "Comparisons experiment")
    
    return experiment_results

