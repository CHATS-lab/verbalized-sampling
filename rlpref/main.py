"""
Main module for the rlpref experiment.

This module contains the entry point for running experiments and reanalyzing results.
"""
import sys
from typing import Optional, Dict, Any, Tuple
import os
import re
import time
from datetime import datetime

from model import load_model, unload_model
from logprobs import filter_tokens_by_logprob, extract_token_logprobs, save_token_logprobs
from results import analyze_axis_results, plot_axis_results, analyze_comparisons_results, plot_comparisons_results
from experiments import run_axis_experiment, run_comparisons_experiment
from test import run_tests
from utilities import format_prompt, create_timestamp_dir, log_execution_time, save_json_results, safe_file_operation
from config import ExperimentConfig, config_from_args, setup_matplotlib


def reanalyze_results(
    results_file: str,
    output_dir: str = None,
    token_filter_threshold: Optional[float] = None,
    jitter_seed: int = 42,
    output_format: str = "pdf"
) -> Tuple[bool, str]:
    """
    Reanalyze existing experiment results from a JSON file.
    
    Args:
        results_file: Path to the JSON results file
        output_dir: Directory to save new plots and analysis
        token_filter_threshold: Optional threshold for filtering tokens by log probability
                               (e.g., -2.0 means only keep tokens with avg logprob >= -2.0)
        jitter_seed: Seed for jittering points in scatter plots
        output_format: Format for saving plots ('pdf' or 'png')
        
    Returns:
        Tuple of (success, results_dir) indicating whether the reanalysis was successful
    """
    start_time = time.time()
    print(f"Reanalyzing results from: {results_file}")
    
    # Create output directory if not provided
    if output_dir is None:
        results_dir = create_timestamp_dir(prefix="reanalysis")
    else:
        results_dir = output_dir
        os.makedirs(results_dir, exist_ok=True)
    
    # Load the JSON file
    try:
        with open(results_file, 'r') as f:
            results_data = f.read()
            import json
            results_data = json.loads(results_data)
    except Exception as e:
        print(f"Error loading results file: {e}")
        return False, results_dir
    
    # Extract model name from the filename
    filename = os.path.basename(results_file)
    # Try to extract model name from a filename pattern like "axis_results_model_name.json"
    # or "comparisons_results_model_name.json"
    match = re.search(r'(?:axis|comparisons)_results_(.+)\.json$', filename)
    if match:
        extracted_model = match.group(1)
        model_name = extracted_model.replace('_', '/')  # Convert back any underscores used for slashes
        print(f"Extracted model name from filename: {model_name}")
    else:
        # Fallback to the model field in the JSON data
        model_name = results_data.get('model', 'unknown_model')
        print(f"Using model name from data: {model_name}")
    
    # Determine which type of experiment (axis or comparisons)
    if 'sample_results' not in results_data:
        print(f"Error: Invalid results file format - missing 'sample_results' field")
        return False, results_dir
    
    # Check if it's an axis experiment
    is_axis = False
    is_comparisons = False
    
    if len(results_data['sample_results']) > 0:
        sample = results_data['sample_results'][0]
        is_axis = all(key in sample for key in ['rating', 'avg_logprob'])
        is_comparisons = all(key in sample for key in ['chosen_avg_logprob', 'rejected_avg_logprob'])
    
    success = False
    
    # Apply token filtering if a threshold was provided
    filtered_data = None
    if token_filter_threshold is not None:
        print(f"Applying token filtering with threshold: {token_filter_threshold}")
        filtered_data = filter_tokens_by_logprob(results_data, token_filter_threshold)
        
        # Save the filtered results to a JSON file
        filtered_filename = os.path.join(results_dir, f"filtered_results_threshold_{token_filter_threshold}.json")
        save_json_results(filtered_data, filtered_filename)
    
    # Use filtered data if available, otherwise use original
    working_data = filtered_data if filtered_data is not None else results_data
    
    # Create appropriate suffix based on whether filtering was applied
    suffix_base = f"reanalysis_{model_name.replace('/', '_')}"
    file_suffix = f"{suffix_base}_filtered_{token_filter_threshold}" if token_filter_threshold is not None else suffix_base
    
    # Rerun the appropriate analysis and plotting functions
    if is_axis:
        print(f"Detected axis experiment results for model: {model_name}")
        try:
            # Extract the results in the format expected by analyze_axis_results
            analysis_results = analyze_axis_results(working_data['sample_results'], model_name)
            if analysis_results:
                plots = plot_axis_results(
                    analysis_results, 
                    file_suffix, 
                    results_dir, 
                    jitter_seed=jitter_seed,
                    output_format=output_format
                )
                print(f"Generated {len(plots)} new plots in {results_dir}/")
                success = True
                
                # Save token-level log probabilities if requested
                if token_filter_threshold is not None:
                    token_logprobs = extract_token_logprobs(working_data)
                    token_logprobs_file = os.path.join(results_dir, f"token_logprobs_{file_suffix}.json")
                    save_token_logprobs(token_logprobs, token_logprobs_file, sort_by="logprob")
        except Exception as e:
            print(f"Error reanalyzing axis results: {e}")
    
    elif is_comparisons:
        print(f"Detected comparisons experiment results for model: {model_name}")
        try:
            # Extract the results in the format expected by analyze_comparisons_results
            analysis_results = analyze_comparisons_results(working_data['sample_results'], model_name)
            if analysis_results:
                plots = plot_comparisons_results(
                    analysis_results, 
                    file_suffix, 
                    results_dir, 
                    jitter_seed=jitter_seed,
                    output_format=output_format
                )
                print(f"Generated {len(plots)} new plots in {results_dir}/")
                success = True
                
                # Save token-level log probabilities if requested
                if token_filter_threshold is not None:
                    token_logprobs = extract_token_logprobs(working_data)
                    token_logprobs_file = os.path.join(results_dir, f"token_logprobs_{file_suffix}.json")
                    save_token_logprobs(token_logprobs, token_logprobs_file, sort_by="logprob")
        except Exception as e:
            print(f"Error reanalyzing comparisons results: {e}")
    
    else:
        print("Could not determine experiment type from the results file")
    
    if success:
        print(f"\nReanalysis completed successfully. New plots saved to {results_dir}/")
    else:
        print("\nReanalysis was not successful.")
    
    log_execution_time(start_time, "Reanalysis")
    return success, results_dir


def main(config: ExperimentConfig = None) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Run the experiment with the specified settings.
    
    Args:
        config: Configuration object with experiment settings
        
    Returns:
        Tuple of (experiment_success, results_dir, results_dict) where:
            - experiment_success: Boolean indicating if any experiment variant succeeded
            - results_dir: Path to the directory where results are saved
            - results_dict: Dictionary containing experiment results by variant
    """
    start_time = time.time()
    
    # Use default config if none provided
    if config is None:
        from config import DEFAULT_CONFIG
        config = DEFAULT_CONFIG
    
    # Set up matplotlib with project settings
    setup_matplotlib()
    
    # Check if we're just reanalyzing existing results
    if hasattr(config, 'reanalyze_file') and config.reanalyze_file:
        success, results_dir = reanalyze_results(
            config.reanalyze_file, 
            output_dir=config.results_dir,
            token_filter_threshold=config.token_filter_threshold,
            jitter_seed=config.jitter_seed,
            output_format=config.output_format
        )
        return success, results_dir, {}
    
    # Create timestamped results directory if not provided
    if config.results_dir is None:
        results_dir = create_timestamp_dir(prefix="results")
    else:
        results_dir = config.results_dir
        os.makedirs(results_dir, exist_ok=True)
    
    # Run tests if requested
    if hasattr(config, 'run_tests_only') and config.run_tests_only:
        run_tests()
        return True, results_dir, {}
    
    # Print experiment info
    print(f"=== Running experiment with settings ===")
    print(f"Model: {config.model_name}")
    print(f"Variant: {config.dataset_variant}")
    print(f"Samples: {config.num_samples}")
    print(f"Seed: {config.random_seed}")
    print(f"Results directory: {results_dir}")
    print("======================================")
    
    # Check for warnings
    if config.num_samples < 30:
        print("WARNING: Sample size is less than 30. Statistical results may not be reliable.")
    
    # Run the appropriate experiment variant
    experiment_success = False
    axis_results = None
    comparisons_results = None
    
    # Load model once if needed for both experiments
    model = None
    tokenizer = None
    if config.dataset_variant in ["axis", "comparisons", "both"]:
        try:
            print(f"4-bit quantization: {'enabled' if config.use_4bit else 'disabled'}")
            model, tokenizer = load_model(
                config.model_name, 
                use_4bit=config.use_4bit, 
                force_reload=getattr(config, 'force_reload', False)
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            return False, results_dir, {}
    
    if config.dataset_variant in ["axis", "both"]:
        try:
            axis_results = run_axis_experiment(
                config.model_name, 
                model=model,
                tokenizer=tokenizer,
                num_samples=config.num_samples, 
                random_seed=config.random_seed,
                use_4bit=config.use_4bit,
                results_dir=results_dir,
                skip_analysis=config.skip_analysis,
                output_format=config.output_format,
                jitter_seed=config.jitter_seed
            )
            
            if axis_results:
                experiment_success = True
                # Save results
                results_file = os.path.join(results_dir, f"axis_results_{config.model_name.replace('/', '_')}.json")
                save_json_results(axis_results, results_file)
                print(f"Axis results saved to {results_file}")
        except Exception as e:
            print(f"Error in axis experiment: {e}")
            print("Continuing with other experiment variants...")
    
    if config.dataset_variant in ["comparisons", "both"]:
        try:
            comparisons_results = run_comparisons_experiment(
                config.model_name, 
                model=model,
                tokenizer=tokenizer,
                num_samples=config.num_samples, 
                random_seed=config.random_seed,
                use_4bit=config.use_4bit,
                results_dir=results_dir,
                skip_analysis=config.skip_analysis,
                output_format=config.output_format,
                jitter_seed=config.jitter_seed
            )
            
            if comparisons_results:
                experiment_success = True
                # Save results
                results_file = os.path.join(results_dir, f"comparisons_results_{config.model_name.replace('/', '_')}.json")
                save_json_results(comparisons_results, results_file)
                print(f"Comparisons results saved to {results_file}")
        except Exception as e:
            print(f"Error in comparisons experiment: {e}")
            
    # Free up memory if explicitly requested
    # Note: We keep the model cached by default for future calls
    if model is not None and not config.model_cache:
        try:
            unload_model(config.model_name, config.use_4bit)
        except:
            pass  # If it fails, just continue
    
    if experiment_success:
        print(f"\nExperiment completed successfully. Results saved to {results_dir}/")
        print(f"To view results, check the JSON files and {config.output_format} plots in the results directory.")
    else:
        print("\nNo experiment variants completed successfully.")
    
    log_execution_time(start_time, "Experiment")
    return experiment_success, results_dir, {"axis": axis_results, "comparisons": comparisons_results}


def run_with_params(
    model_name="distilgpt2",
    variant="both", 
    num_samples=100, 
    random_seed=42, 
    use_4bit=False, 
    output_dir=None,
    run_tests_only=False,
    keep_model_loaded=True,
    force_reload=False,
    skip_analysis=False,
    reanalyze_file=None,
    token_filter_threshold=None,
    output_format="pdf",
    jitter_seed=42
):
    """
    Compatibility wrapper for running with individual parameters.
    
    Args:
        Individual parameters instead of a config object
        
    Returns:
        Tuple of (experiment_success, results_dir, results_dict)
    """
    config = ExperimentConfig(
        model_name=model_name,
        dataset_variant=variant,
        num_samples=num_samples,
        random_seed=random_seed,
        use_4bit=use_4bit,
        results_dir=output_dir,
        skip_analysis=skip_analysis,
        token_filter_threshold=token_filter_threshold,
        output_format=output_format,
        jitter_seed=jitter_seed,
        model_cache=keep_model_loaded
    )
    
    # Add additional attributes for backward compatibility
    setattr(config, 'reanalyze_file', reanalyze_file)
    setattr(config, 'run_tests_only', run_tests_only)
    setattr(config, 'force_reload', force_reload)
    
    return main(config)


if __name__ == "__main__":
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run LLM probability vs. human preference experiment")
    parser.add_argument("--test", action="store_true", help="Run tests instead of the experiment")
    parser.add_argument("--model", default="distilgpt2", help="Model name to use")
    parser.add_argument("--variant", choices=["axis", "comparisons", "both"], default="both", 
                        help="Which experiment variant to run")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_4bit", action="store_true", help="Enable 4-bit quantization (requires bitsandbytes)")
    parser.add_argument("--output", type=str, default=None, 
                       help="Output directory for results (default: timestamped directory)")
    parser.add_argument("--force-reload", action="store_true", 
                       help="Force reload the model even if it's already in cache")
    parser.add_argument("--unload-model", action="store_true", 
                       help="Unload the model after the experiment (default: keep it in cache)")
    parser.add_argument("--skip-analysis", action="store_true",
                       help="Skip analysis and plotting (just collect data)")
    parser.add_argument("--reanalyze", type=str, default=None,
                       help="Reanalyze an existing results JSON file, skipping data collection")
    parser.add_argument("--token-filter", type=float, default=None,
                       help="Filter out tokens with average log probability above this threshold")
    parser.add_argument("--output-format", choices=["pdf", "png"], default="pdf",
                       help="Format for saving plots")
    
    args = parser.parse_args()
    
    # Create configuration from command line arguments
    config = config_from_args(args)
    
    # Add additional attributes needed for backward compatibility
    setattr(config, 'reanalyze_file', args.reanalyze)
    setattr(config, 'run_tests_only', args.test)
    setattr(config, 'force_reload', args.force_reload)
    
    # Set output format from args or use default
    if hasattr(args, 'output_format'):
        config.output_format = args.output_format
    
    # Run the main function with configuration
    success, _, _ = main(config)
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)