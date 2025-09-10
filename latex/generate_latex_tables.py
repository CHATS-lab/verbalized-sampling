#!/usr/bin/env python3
"""
Unified LaTeX table generator for both poem and story experiments.
Parses results and generates LaTeX-formatted tables for publications.
"""

import json
import os
import numpy as np
import argparse
from pathlib import Path

METHODS = {
    "Baseline": "direct (samples=1)",
    "Baseline CoT": "direct_cot [strict] (samples=1)",
    "Sequence": "sequence [strict] (samples=5)", 
    "Multi-turn": "multi_turn [strict] (samples=5)",
    "Standard": "structure_with_prob [strict] (samples=5)",
    "CoT": "chain_of_thought [strict] (samples=5)",
    "Combined": "combined [strict] (samples=5)"
}

def load_metric_with_std(model_dir, method, metric_file, avg_key, std_key):
    """Load a specific metric with standard deviation from a results file"""
    file_path = os.path.join(model_dir, "evaluation", method, metric_file)
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        overall_metrics = data.get('overall_metrics', {})
        avg_result = overall_metrics.get(avg_key, None)
        std_result = overall_metrics.get(std_key, None)
        return avg_result, std_result
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        return None, None

def get_model_results(model_dir, model_name):
    """Extract all metrics for a model with standard deviations"""
    methods = METHODS
    
    results = {"model": model_name}
    
    for method_name, method_dir in methods.items():
        # Get diversity (higher is better)
        diversity_avg, diversity_std = load_metric_with_std(model_dir, method_dir, "diversity_results.json", "avg_diversity", "std_diversity")
        
        # Get Rouge-L (lower is better - convert to percentage and multiply by 100)
        rouge_l_avg, rouge_l_std = load_metric_with_std(model_dir, method_dir, "ngram_results.json", "avg_rouge_l", "std_rouge_l")
        
        # Get quality score (convert from 0-1 scale to 0-100 scale)
        quality_avg, quality_std = load_metric_with_std(model_dir, method_dir, "creative_writing_v3_results.json", "avg_score", "std_score")
        
        results[method_name] = {
            "diversity": diversity_avg * 100 if diversity_avg is not None else None,
            "diversity_std": diversity_std * 100 if diversity_std is not None else None,
            "rouge_l": rouge_l_avg * 100 if rouge_l_avg is not None else None,
            "rouge_l_std": rouge_l_std * 100 if rouge_l_std is not None else None,
            "quality": quality_avg * 100 if quality_avg is not None else None,
            "quality_std": quality_std * 100 if quality_std is not None else None
        }
    
    return results

def calculate_improvements(baseline, method):
    """Calculate percentage improvement over baseline"""
    improvements = {}
    
    if baseline["diversity"] is not None and method["diversity"] is not None:
        improvements["diversity"] = ((method["diversity"] - baseline["diversity"]) / baseline["diversity"]) * 100
    
    if baseline["rouge_l"] is not None and method["rouge_l"] is not None:
        # For Rouge-L, lower is better, so improvement is negative change
        improvements["rouge_l"] = ((baseline["rouge_l"] - method["rouge_l"]) / baseline["rouge_l"]) * 100
    
    if baseline["quality"] is not None and method["quality"] is not None:
        improvements["quality"] = ((method["quality"] - baseline["quality"]) / baseline["quality"]) * 100
    
    return improvements

def format_metric_with_std(value, std_value, is_best=False):
    """Format metric value with standard deviation for LaTeX table"""
    if value is None or std_value is None:
        return "N/A"
    
    formatted = f"{value:.1f}$_{{\\pm{{{std_value:.1f}}}}}$"
    if is_best:
        formatted = f"\\textbf{{{formatted}}}"
    
    return formatted

def generate_latex_table(task_type):
    """Generate LaTeX table from all model results"""
    
    # Model directory mapping
    models = {
        "Claude-4-Sonnet": "anthropic_claude-4-sonnet",
        "Claude-3.7-Sonnet": "anthropic_claude-3.7-sonnet", 
        "Gemini-2.5-Pro": "google_gemini-2.5-pro",
        "Gemini-2.5-Flash": "google_gemini-2.5-flash",
        "GPT-4.1": "openai_gpt-4.1",
        "GPT-4.1-Mini": "openai_gpt-4.1-mini",
        "Llama-3.1-70B": "meta-llama_Llama-3.1-70B-Instruct",
        # "Llama-3.1-8B": "meta-llama_Llama-3.1-8B-Instruct",
        "DeepSeek-R1": "deepseek_deepseek-r1-0528",
        "GPT-o3": "openai_o3",
    }
    
    # Task-specific configuration
    if task_type == "poem":
        base_dir = "poem_experiments_final"
        task_suffix = "poem"
    elif task_type == "story":
        base_dir = "story_experiments_final"
        task_suffix = "book"
    else:
        raise ValueError("task_type must be 'poem' or 'story'")
    
    all_results = {}
    
    # Collect results for all models
    for model_name, model_dir_name in models.items():
        model_path = os.path.join(base_dir, model_dir_name, f"{model_dir_name}_{task_suffix}")
        if os.path.exists(model_path):
            results = get_model_results(model_path, model_name)
            all_results[model_name] = results
            print(f"Processed {model_name}")
        else:
            print(f"Warning: Directory not found for {model_name}: {model_path}")
    
    # Generate LaTeX table
    # print("\n" + "="*80)
    # print(f"LATEX TABLE DATA - {task_type.upper()}")
    # print("="*80)
    
    for model_name, results in all_results.items():
        # print(f"\n{model_name}:")
        # print("-" * 40)
        
        baseline = results.get("Baseline")
        if baseline is None:
            print("No baseline data available")
            continue
            
        # Count available methods to determine multirow span
        available_methods = []
        method_order = ["Baseline", "Baseline CoT", "Sequence", "Multi-turn", "Standard", "CoT", "Combined"]
        
        for method in method_order:
            if method in results and results[method] and any(v is not None for v in [results[method].get('diversity'), results[method].get('rouge_l'), results[method].get('quality')]):
                available_methods.append(method)
        
        if not available_methods:
            print("No valid methods found")
            continue
            
        # Find best values across all methods for highlighting
        all_diversity = [results[method]["diversity"] for method in available_methods 
                        if results[method]["diversity"] is not None]
        all_rouge_l = [results[method]["rouge_l"] for method in available_methods 
                      if results[method]["rouge_l"] is not None]
        all_quality = [results[method]["quality"] for method in available_methods 
                      if results[method]["quality"] is not None]
        
        best_diversity = max(all_diversity) if all_diversity else None
        best_rouge_l = min(all_rouge_l) if all_rouge_l else None  # Lower is better for Rouge-L
        best_quality = max(all_quality) if all_quality else None
        
        # Print with multirow
        multirow_span = len(available_methods)
        
        for i, method in enumerate(available_methods):
            data = results[method]
            
            # Format method name
            if method == "Baseline":
                method_display = "Direct"
            elif method == "Baseline CoT":
                method_display = "CoT"
            elif method == "Standard":
                method_display = "$\\hookrightarrow$ Standard"
            elif method == "CoT":
                method_display = "$\\hookrightarrow$ CoT"
            elif method == "Combined":
                method_display = "$\\hookrightarrow$ Combined"
            else:
                method_display = method
            
            # Add Verbalized Sampling header before first VS method
            if method == "Standard" and i > 0:
                print("& \\textbf{Verbalized Sampling} \\\\")
            
            # Format metrics with standard deviations
            diversity_formatted = format_metric_with_std(
                data["diversity"], data["diversity_std"], 
                data["diversity"] == best_diversity if data["diversity"] is not None else False
            )
            rouge_l_formatted = format_metric_with_std(
                data["rouge_l"], data["rouge_l_std"], 
                data["rouge_l"] == best_rouge_l if data["rouge_l"] is not None else False
            )
            quality_formatted = format_metric_with_std(
                data["quality"], data["quality_std"], 
                data["quality"] == best_quality if data["quality"] is not None else False
            )
            
            # Print row with multirow on first iteration
            if i == 0:
                print(f"\\multirow{{{multirow_span + 1}}}{{*}}{{{model_name}}}")
                print(f"& {method_display} & {diversity_formatted} & {rouge_l_formatted} & {quality_formatted} \\\\")
            else:
                print(f"& {method_display} & {diversity_formatted} & {rouge_l_formatted} & {quality_formatted} \\\\")
        
        # Calculate improvements for the best VS method (find the one with best overall performance)
        vs_methods = ["Standard", "CoT", "Combined"]
        best_vs_method = None
        best_vs_score = -float('inf')
        
        for method in vs_methods:
            data = results.get(method)
            if data and all(v is not None for v in [data.get('diversity'), data.get('rouge_l'), data.get('quality')]):
                # Simple scoring: normalize each metric and sum (diversity + quality - rouge_l)
                score = (data["diversity"] / 100) + (data["quality"] / 100) - (data["rouge_l"] / 100)
                if score > best_vs_score:
                    best_vs_score = score
                    best_vs_method = method
        
        if best_vs_method and baseline:
            improvements = calculate_improvements(baseline, results[best_vs_method])
            improvement_str = []
            if "diversity" in improvements:
                improvement_str.append(f"+{improvements['diversity']:.1f}%")
            if "rouge_l" in improvements: 
                improvement_str.append(f"-{improvements['rouge_l']:.1f}%")
            if "quality" in improvements:
                improvement_str.append(f"+{improvements['quality']:.1f}%")
            
            print(f"% Best VS method ({best_vs_method}) improvements: {', '.join(improvement_str)}")
        
        print("\\midrule")

    # Also generate a summary table with key statistics
    print(f"\n\nSUMMARY STATISTICS:")
    print("-" * 40)
    
    # Calculate average improvements across all models for best VS method
    total_diversity_imp = []
    total_rouge_l_imp = []
    total_quality_imp = []
    
    for model_name, results in all_results.items():
        baseline = results.get("Baseline")
        if not baseline:
            continue
            
        # Find best VS method for this model
        vs_methods = ["Standard", "CoT", "Combined"]
        best_vs_method = None
        best_vs_score = -float('inf')
        
        for method in vs_methods:
            data = results.get(method)
            if data and all(v is not None for v in [data.get('diversity'), data.get('rouge_l'), data.get('quality')]):
                score = (data["diversity"] / 100) + (data["quality"] / 100) - (data["rouge_l"] / 100)
                if score > best_vs_score:
                    best_vs_score = score
                    best_vs_method = method
        
        if best_vs_method:
            improvements = calculate_improvements(baseline, results[best_vs_method])
            if "diversity" in improvements:
                total_diversity_imp.append(improvements["diversity"])
            if "rouge_l" in improvements:
                total_rouge_l_imp.append(improvements["rouge_l"]) 
            if "quality" in improvements:
                total_quality_imp.append(improvements["quality"])
    
    if total_diversity_imp:
        print(f"Average Diversity Improvement: +{np.mean(total_diversity_imp):.1f}% (±{np.std(total_diversity_imp):.1f}%)")
    if total_rouge_l_imp:
        print(f"Average Rouge-L Improvement: -{np.mean(total_rouge_l_imp):.1f}% (±{np.std(total_rouge_l_imp):.1f}%)")
    if total_quality_imp:
        print(f"Average Quality Improvement: +{np.mean(total_quality_imp):.1f}% (±{np.std(total_quality_imp):.1f}%)")

def main():
    parser = argparse.ArgumentParser(description='Generate LaTeX tables for poem or story experiments')
    parser.add_argument('--task', choices=['poem', 'story', 'both'], default='both',
                        help='Which task to generate tables for (default: both)')
    
    args = parser.parse_args()
    
    if args.task == 'both':
        print("Generating LaTeX tables for both poem and story experiments...")
        generate_latex_table('poem')
        print("\n" + "="*80 + "\n")
        generate_latex_table('story')
    else:
        generate_latex_table(args.task)

if __name__ == "__main__":
    main()