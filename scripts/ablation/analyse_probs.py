#!/usr/bin/env python3
"""
Probability Distribution Analysis Script for Poem Generation Experiments

This script analyzes the probability distribution of responses in the 
structure_with_prob experiment results across all models.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import pandas as pd
from typing import List, Dict, Any
import argparse
import glob

def load_jsonl_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of JSON objects from the file
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def extract_probabilities(data: List[Dict[str, Any]]) -> List[float]:
    """
    Extract all probability values from the data.
    
    Args:
        data: List of JSON objects from the JSONL file
        
    Returns:
        List of all probability values
    """
    probabilities = []
    for item in data:
        if 'responses' in item:
            for response in item['responses']:
                if 'probability' in response:
                    probabilities.append(response['probability'])
    return probabilities

def analyze_probability_distribution(probabilities: List[float]) -> Dict[str, Any]:
    """
    Perform comprehensive analysis of probability distribution.
    
    Args:
        probabilities: List of probability values
        
    Returns:
        Dictionary containing various statistical measures
    """
    if not probabilities:
        return {}
    
    probs_array = np.array(probabilities)
    
    analysis = {
        'count': len(probabilities),
        'mean': float(np.mean(probs_array)),
        'median': float(np.median(probs_array)),
        'std': float(np.std(probs_array)),
        'min': float(np.min(probs_array)),
        'max': float(np.max(probs_array)),
        'range': float(np.max(probs_array) - np.min(probs_array)),
        'q25': float(np.percentile(probs_array, 25)),
        'q75': float(np.percentile(probs_array, 75)),
        'iqr': float(np.percentile(probs_array, 75) - np.percentile(probs_array, 25)),
        'skewness': float(pd.Series(probs_array).skew()),
        'kurtosis': float(pd.Series(probs_array).kurtosis()),
    }
    
    return analysis

def analyze_per_prompt_distributions(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze probability distributions for each prompt separately.
    
    Args:
        data: List of JSON objects from the JSONL file
        
    Returns:
        Dictionary containing per-prompt analysis
    """
    prompt_analyses = {}
    
    for i, item in enumerate(data):
        if 'responses' in item:
            prompt_probs = []
            for response in item['responses']:
                if 'probability' in response:
                    prompt_probs.append(response['probability'])
            
            if prompt_probs:
                prompt_analyses[f'prompt_{i+1}'] = {
                    'probabilities': prompt_probs,
                    'mean': float(np.mean(prompt_probs)),
                    'std': float(np.std(prompt_probs)),
                    'min': float(np.min(prompt_probs)),
                    'max': float(np.max(prompt_probs)),
                    'range': float(np.max(prompt_probs) - np.min(prompt_probs)),
                    'entropy': float(-sum(p * np.log2(p) for p in prompt_probs if p > 0))
                }
    
    return prompt_analyses

def find_all_experiments(base_dir: str = "poem_experiments_final") -> Dict[str, str]:
    """
    Find all structure_with_prob experiments across all models.
    
    Args:
        base_dir: Base directory containing experiment results
        
    Returns:
        Dictionary mapping model names to their responses.jsonl file paths
    """
    experiments = {}
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Warning: Base directory {base_dir} not found.")
        return experiments
    
    # Look for VS-Standard (vs_standard) directories in each model folder
    for model_dir in base_path.iterdir():
        if model_dir.is_dir():
            model_name = model_dir.name
            
            # Look for the specific experiment directory - handle the space in directory name
            experiment_dir_name = "structure_with_prob [strict] (samples=5)"
            experiment_path = model_dir / f"{model_name}_poem" / "generation" / experiment_dir_name / "responses.jsonl"
            
            if experiment_path.exists():
                experiments[model_name] = str(experiment_path)
                print(f"Found experiment for {model_name}: {experiment_path}")
            else:
                print(f"No structure_with_prob experiment found for {model_name}")
                print(f"Path: {experiment_path}")
    
    return experiments

def create_visualizations(probabilities: List[float], output_dir: str = "analysis_output", 
                         model_name: str = "unknown", xlim: tuple = (0, 1)):
    """
    Create various visualizations of the probability distribution.
    
    Args:
        probabilities: List of probability values
        output_dir: Directory to save plots
        model_name: Name of the model for plot titles
        xlim: X-axis limits for probability plots
    """
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Probability Distribution Analysis - {model_name}', fontsize=16, fontweight='bold')
    
    # 1. Histogram
    axes[0, 0].hist(probabilities, bins=20, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Histogram of Probabilities')
    axes[0, 0].set_xlabel('Probability')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_xlim(xlim)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Box plot
    axes[0, 1].boxplot(probabilities, patch_artist=True)
    axes[0, 1].set_title('Box Plot of Probabilities')
    axes[0, 1].set_ylabel('Probability')
    axes[0, 1].set_ylim(xlim)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Violin plot
    axes[0, 2].violinplot(probabilities)
    axes[0, 2].set_title('Violin Plot of Probabilities')
    axes[0, 2].set_ylabel('Probability')
    axes[0, 2].set_ylim(xlim)
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Cumulative distribution
    sorted_probs = np.sort(probabilities)
    cumulative = np.arange(1, len(sorted_probs) + 1) / len(sorted_probs)
    axes[1, 0].plot(sorted_probs, cumulative, linewidth=2)
    axes[1, 0].set_title('Cumulative Distribution')
    axes[1, 0].set_xlabel('Probability')
    axes[1, 0].set_ylabel('Cumulative Probability')
    axes[1, 0].set_xlim(xlim)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Q-Q plot (against uniform distribution)
    try:
        from scipy import stats
        stats.probplot(probabilities, dist="uniform", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot vs Uniform Distribution')
        axes[1, 1].set_xlim(xlim)
        axes[1, 1].grid(True, alpha=0.3)
    except ImportError:
        axes[1, 1].text(0.5, 0.5, 'scipy not available', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Q-Q Plot vs Uniform Distribution')
    
    # 6. Probability density estimation
    try:
        sns.kdeplot(data=probabilities, ax=axes[1, 2], fill=True)
        axes[1, 2].set_title('Kernel Density Estimation')
        axes[1, 2].set_xlabel('Probability')
        axes[1, 2].set_ylabel('Density')
        axes[1, 2].set_xlim(xlim)
        axes[1, 2].grid(True, alpha=0.3)
    except:
        axes[1, 2].hist(probabilities, bins=20, alpha=0.7, density=True, edgecolor='black')
        axes[1, 2].set_title('Probability Density (Histogram)')
        axes[1, 2].set_xlabel('Probability')
        axes[1, 2].set_ylabel('Density')
        axes[1, 2].set_xlim(xlim)
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_name}_probability_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create additional detailed plots
    create_detailed_plots(probabilities, output_dir, model_name, xlim)

def create_detailed_plots(probabilities: List[float], output_dir: str, model_name: str, xlim: tuple = (0, 1)):
    """
    Create additional detailed visualizations.
    
    Args:
        probabilities: List of probability values
        output_dir: Directory to save plots
        model_name: Name of the model for plot titles
        xlim: X-axis limits for probability plots
    """
    # 1. Distribution comparison with theoretical distributions
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram with theoretical uniform distribution
    axes[0].hist(probabilities, bins=20, alpha=0.7, density=True, label='Observed', edgecolor='black')
    axes[0].axhline(y=1, color='red', linestyle='--', label='Uniform (0,1)')
    axes[0].set_title(f'Distribution vs Uniform - {model_name}')
    axes[0].set_xlabel('Probability')
    axes[0].set_ylabel('Density')
    axes[0].set_xlim(xlim)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Log scale histogram
    axes[1].hist(probabilities, bins=20, alpha=0.7, edgecolor='black')
    axes[1].set_yscale('log')
    axes[1].set_title(f'Log-Scale Histogram - {model_name}')
    axes[1].set_xlabel('Probability')
    axes[1].set_ylabel('Frequency (log scale)')
    axes[1].set_xlim(xlim)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_name}_detailed_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_comparison_plot(all_results: Dict[str, Dict], output_dir: str = "analysis_output"):
    """
    Create comparison plots across all models.
    
    Args:
        all_results: Dictionary containing results for all models
        output_dir: Directory to save plots
    """
    if len(all_results) < 2:
        print("Need at least 2 models for comparison plots.")
        return
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Comparison - Probability Distributions', fontsize=16, fontweight='bold')
    
    # Prepare data for plotting
    model_names = list(all_results.keys())
    all_probabilities = [all_results[model]['probabilities'] for model in model_names]
    
    # 1. Box plot comparison
    axes[0, 0].boxplot(all_probabilities, labels=model_names, patch_artist=True)
    axes[0, 0].set_title('Box Plot Comparison')
    axes[0, 0].set_ylabel('Probability')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Violin plot comparison
    axes[0, 1].violinplot(all_probabilities, labels=model_names)
    axes[0, 1].set_title('Violin Plot Comparison')
    axes[0, 1].set_ylabel('Probability')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Mean comparison
    means = [all_results[model]['analysis']['mean'] for model in model_names]
    axes[1, 0].bar(model_names, means, alpha=0.7)
    axes[1, 0].set_title('Mean Probability Comparison')
    axes[1, 0].set_ylabel('Mean Probability')
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Standard deviation comparison
    stds = [all_results[model]['analysis']['std'] for model in model_names]
    axes[1, 1].bar(model_names, stds, alpha=0.7, color='orange')
    axes[1, 1].set_title('Standard Deviation Comparison')
    axes[1, 1].set_ylabel('Standard Deviation')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_comparison_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def print_analysis_results(analysis: Dict[str, Any], prompt_analyses: Dict[str, Any], model_name: str = "Unknown"):
    """
    Print formatted analysis results.
    
    Args:
        analysis: Overall probability distribution analysis
        prompt_analyses: Per-prompt analysis results
        model_name: Name of the model being analyzed
    """
    print("=" * 60)
    print(f"PROBABILITY DISTRIBUTION ANALYSIS - {model_name.upper()}")
    print("=" * 60)
    
    if not analysis:
        print("No data found for analysis.")
        return
    
    print(f"\nOVERALL STATISTICS:")
    print(f"  Total samples: {analysis['count']}")
    print(f"  Mean: {analysis['mean']:.4f}")
    print(f"  Median: {analysis['median']:.4f}")
    print(f"  Standard deviation: {analysis['std']:.4f}")
    print(f"  Range: {analysis['min']:.4f} - {analysis['max']:.4f}")
    print(f"  IQR: {analysis['iqr']:.4f}")
    print(f"  Skewness: {analysis['skewness']:.4f}")
    print(f"  Kurtosis: {analysis['kurtosis']:.4f}")
    
    print(f"\nQUARTILES:")
    print(f"  Q1 (25th percentile): {analysis['q25']:.4f}")
    print(f"  Q2 (50th percentile): {analysis['median']:.4f}")
    print(f"  Q3 (75th percentile): {analysis['q75']:.4f}")
    
    print(f"\nPER-PROMPT ANALYSIS:")
    print(f"  Number of prompts: {len(prompt_analyses)}")
    
    if prompt_analyses:
        prompt_means = [stats['mean'] for stats in prompt_analyses.values()]
        prompt_stds = [stats['std'] for stats in prompt_analyses.values()]
        
        print(f"  Average mean across prompts: {np.mean(prompt_means):.4f}")
        print(f"  Average std across prompts: {np.mean(prompt_stds):.4f}")
        print(f"  Mean range across prompts: {np.max(prompt_means) - np.min(prompt_means):.4f}")

def save_analysis_to_file(all_results: Dict[str, Dict], output_dir: str = "analysis_output"):
    """
    Save analysis results to JSON file.
    
    Args:
        all_results: Dictionary containing results for all models
        output_dir: Directory to save results
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    results = {
        'all_models_analysis': all_results,
        'metadata': {
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'total_models': len(all_results),
            'models_analyzed': list(all_results.keys())
        }
    }
    
    with open(f'{output_dir}/all_models_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAnalysis results saved to: {output_dir}/all_models_analysis_results.json")

def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(description='Analyze probability distributions from poem generation experiments')
    parser.add_argument('--input-dir', '-i', 
                       default='poem_experiments_final',
                       help='Base directory containing experiment results')
    parser.add_argument('--output', '-o', 
                       default='analysis_output',
                       help='Output directory for analysis results')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots')
    parser.add_argument('--single-model', '-s',
                       help='Analyze only a specific model (e.g., anthropic_claude-4-sonnet)')
    
    args = parser.parse_args()
    
    # Find all experiments
    experiments = find_all_experiments(args.input_dir)
    
    if not experiments:
        print(f"No experiments found in {args.input_dir}")
        return
    
    # Filter for single model if specified
    if args.single_model:
        if args.single_model in experiments:
            experiments = {args.single_model: experiments[args.single_model]}
        else:
            print(f"Model {args.single_model} not found. Available models: {list(experiments.keys())}")
            return
    
    print(f"Found {len(experiments)} experiments to analyze:")
    for model, path in experiments.items():
        print(f"  - {model}: {path}")
    
    all_results = {}
    
    try:
        for model_name, file_path in experiments.items():
            print(f"\n{'='*60}")
            print(f"Analyzing {model_name}...")
            print(f"{'='*60}")
            
            # Load data
            data = load_jsonl_data(file_path)
            print(f"Loaded {len(data)} prompts from JSONL file")
            
            # Extract probabilities
            probabilities = extract_probabilities(data)
            print(f"Extracted {len(probabilities)} probability values")
            
            if not probabilities:
                print(f"No probability values found for {model_name}. Skipping...")
                continue
            
            # Perform analysis
            analysis = analyze_probability_distribution(probabilities)
            prompt_analyses = analyze_per_prompt_distributions(data)
            
            # Store results
            all_results[model_name] = {
                'analysis': analysis,
                'prompt_analyses': prompt_analyses,
                'probabilities': probabilities,
                'file_path': file_path
            }
            
            # Print results
            print_analysis_results(analysis, prompt_analyses, model_name)
            
            # Create visualizations
            if not args.no_plots:
                print(f"\nGenerating visualizations for {model_name}...")
                create_visualizations(probabilities, args.output, model_name, xlim=(0, 1))
                print(f"Plots saved to: {args.output}/")
        
        # Save all results
        save_analysis_to_file(all_results, args.output)
        
        # Create comparison plots if multiple models
        if len(all_results) > 1 and not args.no_plots:
            print(f"\nGenerating comparison plots...")
            create_comparison_plot(all_results, args.output)
            print(f"Comparison plots saved to: {args.output}/")
        
        print(f"\nAnalysis complete! Results saved to: {args.output}/")
        print(f"Analyzed {len(all_results)} models: {list(all_results.keys())}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
