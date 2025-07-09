#!/usr/bin/env python3

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import argparse
from scipy import stats

def load_metric(model_dir, method, metric_file, metric_key):
    """Load a specific metric from a results file"""
    file_path = os.path.join(model_dir, "evaluation", method, metric_file)
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data.get('overall_metrics', {}).get(metric_key, None)
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None

def get_model_results(model_dir, model_name, experiment_type="poem"):
    """Extract all metrics for a model"""
    methods = {
        "Baseline": "direct (samples=1)",
        "Sequence": "sequence [strict] (samples=5)", 
        "Multi-turn": "multi_turn [strict] (samples=5)",
        "VS-Standard": "structure_with_prob [strict] (samples=5)",
        "VS-CoT": "chain_of_thought [strict] (samples=5)",
        "VS-Combined": "combined [strict] (samples=5)"
    }
    
    # Choose quality file and metric keys based on experiment type
    if experiment_type == "poem":
        quality_file = "creative_writing_v3_results.json"
        quality_avg_key = "avg_score"
        quality_std_key = "std_score"
    else:  # joke
        quality_file = "joke_quality_results.json"
        quality_avg_key = "avg_normalized_overall"
        quality_std_key = "std_normalized_overall"
    
    results = {"model": model_name}
    
    for method_name, method_dir in methods.items():
        # Get diversity (higher is better)
        diversity_avg = load_metric(model_dir, method_dir, "diversity_results.json", "avg_diversity")
        diversity_std = load_metric(model_dir, method_dir, "diversity_results.json", "std_diversity")
        
        # Get Rouge-L (lower is better)
        rouge_l_avg = load_metric(model_dir, method_dir, "ngram_results.json", "avg_rouge_l")
        rouge_l_std = load_metric(model_dir, method_dir, "ngram_results.json", "std_rouge_l")
        
        # Get quality score (0-1 scale)
        quality_avg = load_metric(model_dir, method_dir, quality_file, quality_avg_key)
        quality_std = load_metric(model_dir, method_dir, quality_file, quality_std_key)
        
        results[method_name] = {
            "diversity": diversity_avg * 100 if diversity_avg is not None else None,
            "diversity_std": diversity_std * 100 if diversity_std is not None else None,
            "rouge_l": rouge_l_avg * 100 if rouge_l_avg is not None else None,
            "rouge_l_std": rouge_l_std * 100 if rouge_l_std is not None else None,
            "quality": quality_avg * 100 if quality_avg is not None else None,
            "quality_std": quality_std * 100 if quality_std is not None else None
        }
    
    return results

def format_metric(value, std_value=None, is_best=False):
    """Format metric value for LaTeX table with std as subscript"""
    if value is None:
        return "N/A"
    
    if std_value is not None:
        formatted = f"{value:.1f}$_{{\\pm{{{std_value:.1f}}}}}$"
    else:
        formatted = f"{value:.1f}"
    
    if is_best:
        formatted = f"\\textbf{{{formatted}}}"
    
    return formatted

def find_best_values_per_model(results):
    """Find the best value for each metric within a single model"""
    method_names = ["Baseline", "Sequence", "Multi-turn", "VS-Standard", "VS-CoT", "VS-Combined"]
    
    # Get all valid values for each metric
    diversity_values = []
    rouge_l_values = []
    quality_values = []
    
    for method in method_names:
        if results.get(method):
            data = results[method]
            if data["diversity"] is not None:
                diversity_values.append(data["diversity"])
            if data["rouge_l"] is not None:
                rouge_l_values.append(data["rouge_l"])
            if data["quality"] is not None:
                quality_values.append(data["quality"])
    
    # Find best values (max for diversity/quality, min for rouge_l)
    best_diversity = max(diversity_values) if diversity_values else None
    best_rouge_l = min(rouge_l_values) if rouge_l_values else None  # Lower is better
    best_quality = max(quality_values) if quality_values else None
    
    return best_diversity, best_rouge_l, best_quality

def plot_model_comparison(all_results, output_dir="plots", experiment_type="poem"):
    """Create bar charts comparing models across different metrics"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    method_names = ["Baseline", "Sequence", "Multi-turn", "VS-Standard", "VS-CoT", "VS-Combined"]
    
    # Filter out models with insufficient data
    valid_models = {}
    for model_name, results in all_results.items():
        if all(results.get(method) and 
               all(results[method][metric] is not None for metric in ['diversity', 'rouge_l', 'quality'])
               for method in method_names):
            valid_models[model_name] = results
    
    quality_label = "Joke Quality Score (%)" if experiment_type == "joke" else "Quality Score (%)"
    
    metrics = [
        ('diversity', 'Diversity (%)', 'Higher is Better'),
        ('rouge_l', 'Rouge-L (%)', 'Lower is Better'),
        ('quality', quality_label, 'Higher is Better')
    ]
    
    for metric_key, metric_title, direction in metrics:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        model_names = list(valid_models.keys())
        x = np.arange(len(model_names))
        width = 0.13  # Width of bars
        
        # Plot bars for each method
        for i, method in enumerate(method_names):
            values = []
            errors = []
            
            for model_name in model_names:
                value = valid_models[model_name][method][metric_key]
                error = valid_models[model_name][method][f'{metric_key}_std']
                values.append(value if value is not None else 0)
                errors.append(error if error is not None else 0)
            
            bars = ax.bar(x + i * width, values, width, 
                         label=method, color=colors[i % len(colors)],
                         yerr=errors, capsize=3, alpha=0.8)
            
            # Add value labels on bars
            for j, (bar, value, error) in enumerate(zip(bars, values, errors)):
                if value > 0:  # Only label non-zero values
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + error + 0.5,
                           f'{value:.1f}', ha='center', va='bottom', fontsize=8, rotation=0)
        
        ax.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_title, fontsize=12, fontweight='bold')
        title_suffix = f" ({experiment_type.capitalize()} Experiments)"
        ax.set_title(f'{metric_title} Comparison Across Models{title_suffix}\n({direction})', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x + width * 2.5)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/model_comparison_{metric_key}_{experiment_type}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_dir}/model_comparison_{metric_key}_{experiment_type}.pdf', bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved {metric_title} model comparison plot for {experiment_type}")

def plot_method_averages(all_results, output_dir="plots", experiment_type="poem"):
    """Create bar charts showing average performance across all models for each method"""
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    method_names = ["Baseline", "Sequence", "Multi-turn", "VS-Standard", "VS-CoT", "VS-Combined"]
    
    # Calculate averages and std across all models for each method
    method_stats = {}
    
    for method in method_names:
        method_stats[method] = {
            'diversity': [], 'rouge_l': [], 'quality': []
        }
    
    # Collect data from all models
    for model_name, results in all_results.items():
        for method in method_names:
            if results.get(method):
                data = results[method]
                for metric in ['diversity', 'rouge_l', 'quality']:
                    if data[metric] is not None:
                        method_stats[method][metric].append(data[metric])
    
    # Calculate means and stds
    method_means = {}
    method_stds = {}
    
    for method in method_names:
        method_means[method] = {}
        method_stds[method] = {}
        for metric in ['diversity', 'rouge_l', 'quality']:
            values = method_stats[method][metric]
            if values:
                method_means[method][metric] = np.mean(values)
                method_stds[method][metric] = np.std(values)
            else:
                method_means[method][metric] = 0
                method_stds[method][metric] = 0
    
    # Find best VS method for each metric
    vs_methods = ["VS-Standard", "VS-CoT", "VS-Combined"]
    baseline_methods = ["Baseline", "Sequence", "Multi-turn"]
    
    quality_label = "Average Joke Quality Score (%)" if experiment_type == "joke" else "Average Quality Score (%)"
    
    metrics = [
        ('diversity', 'Average Diversity (%)', 'Higher is Better'),
        ('rouge_l', 'Average Rouge-L (%)', 'Lower is Better'),
        ('quality', quality_label, 'Higher is Better')
    ]
    
    for metric_key, metric_title, direction in metrics:
        fig, ax = plt.subplots(figsize=(12, 8))  # Increased height for p-value text
        
        means = [method_means[method][metric_key] for method in method_names]
        stds = [method_stds[method][metric_key] for method in method_names]
        
        bars = ax.bar(method_names, means, yerr=stds, capsize=5, 
                     color=colors[:len(method_names)], alpha=0.8, 
                     edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.5,
                   f'{mean:.1f}±{std:.1f}', ha='center', va='bottom', 
                   fontsize=10, fontweight='bold')
        
        # Find best VS method for this metric
        vs_means = [method_means[method][metric_key] for method in vs_methods]
        if metric_key == 'rouge_l':  # Lower is better
            best_vs_idx = np.argmin(vs_means)
        else:  # Higher is better
            best_vs_idx = np.argmax(vs_means)
        
        best_vs_method = vs_methods[best_vs_idx]
        best_vs_data = method_stats[best_vs_method][metric_key]
        
        # Perform t-tests against baseline methods
        p_values = {}
        for baseline_method in baseline_methods:
            baseline_data = method_stats[baseline_method][metric_key]
            if len(baseline_data) > 1 and len(best_vs_data) > 1:
                # Perform two-sample t-test
                t_stat, p_val = stats.ttest_ind(best_vs_data, baseline_data)
                p_values[baseline_method] = p_val
            else:
                p_values[baseline_method] = None
        
        # Add significance annotations
        y_max = max([m + s for m, s in zip(means, stds)]) * 1.1
        significance_y = y_max + (y_max - min(means)) * 0.05
        
        # Add significance text
        sig_text = f"Statistical Tests (Best VS: {best_vs_method}):\n"
        for baseline_method in baseline_methods:
            p_val = p_values[baseline_method]
            if p_val is not None:
                if p_val < 0.001:
                    sig_level = "***"
                elif p_val < 0.01:
                    sig_level = "**"
                elif p_val < 0.05:
                    sig_level = "*"
                else:
                    sig_level = "ns"
                sig_text += f"vs {baseline_method}: p={p_val:.4f} {sig_level}\n"
            else:
                sig_text += f"vs {baseline_method}: insufficient data\n"
        
        # Add significance text to plot
        ax.text(0.02, 0.98, sig_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Highlight best performing method
        if metric_key == 'rouge_l':  # Lower is better
            best_idx = np.argmin(means)
        else:  # Higher is better
            best_idx = np.argmax(means)
        
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(3)
        
        # Highlight best VS method
        best_vs_idx_in_all = method_names.index(best_vs_method)
        bars[best_vs_idx_in_all].set_facecolor('lightgreen')
        bars[best_vs_idx_in_all].set_alpha(0.9)
        
        ax.set_xlabel('Methods', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_title, fontsize=12, fontweight='bold')
        title_suffix = f" ({experiment_type.capitalize()} Experiments)"
        ax.set_title(f'{metric_title} - Average Across All Models{title_suffix}\n({direction})', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Adjust layout to accommodate significance text
        plt.subplots_adjust(left=0.15, bottom=0.15, top=0.85)
        plt.tight_layout()
        
        plt.savefig(f'{output_dir}/method_average_{metric_key}_{experiment_type}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_dir}/method_average_{metric_key}_{experiment_type}.pdf', bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved {metric_title} method average plot for {experiment_type}")
        print(f"  Best VS method: {best_vs_method}")
        for baseline_method in baseline_methods:
            p_val = p_values[baseline_method]
            if p_val is not None:
                print(f"  vs {baseline_method}: p-value = {p_val:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Generate plots and LaTeX tables for poem or joke experiments')
    parser.add_argument('--experiment', choices=['poem', 'joke'], default='poem',
                        help='Type of experiment to process (default: poem)')
    parser.add_argument('--output-dir', default='plots',
                        help='Output directory for plots (default: plots)')
    
    args = parser.parse_args()
    
    # Model directory mapping - same for both experiments
    models = {
        "Claude-4-Sonnet": "claude-4-sonnet",
        "Gemini-2.5-Pro": "gemini-2.5-pro",
        "Gemini-2.5-Flash": "gemini-2.5-flash",
        "GPT-4.1": "gpt-4.1",
        "GPT-4.1-Mini": "gpt-4.1-mini",
        "Llama-3.1-70B": "llama-3.1-70b-instruct",
        "DeepSeek-R1": "deepseek-r1",
        "GPT-o3": "o3",
    }
    
    base_dir = f"method_results_{args.experiment}"
    suffix = f"_{args.experiment}"
    all_results = {}
    
    # Output file
    output_file = f"latex_table_results_{args.experiment}.txt"
    
    print(f"Processing {args.experiment} experiments...")
    print(f"Base directory: {base_dir}")
    
    with open(output_file, 'w') as f:
        # Collect results for all models
        for model_name, model_dir_name in models.items():
            model_path = os.path.join(base_dir, model_dir_name, f"{model_dir_name}{suffix}")
            print(f"Checking path: {model_path}")
            if os.path.exists(model_path):
                results = get_model_results(model_path, model_name, args.experiment)
                all_results[model_name] = results
                print(f"✓ Processed {model_name}")
                f.write(f"% Processed {model_name}\n")
            else:
                print(f"✗ Directory not found for {model_name}: {model_path}")
                f.write(f"% Directory not found for {model_name}: {model_path}\n")
        
        if not all_results:
            print("❌ No valid model results found!")
            return
        
        # Generate LaTeX table
        f.write("\n" + "="*80 + "\n")
        f.write(f"LATEX TABLE DATA ({args.experiment.upper()} EXPERIMENTS)\n")
        f.write("="*80 + "\n")
        
        for model_name, results in all_results.items():
            f.write(f"\n\\multirow{{7}}{{*}}{{{model_name}}}\n")
            
            baseline = results.get("Baseline")
            if baseline is None:
                f.write("% No baseline data available\n")
                continue
            
            # Find best values for this specific model
            best_diversity, best_rouge_l, best_quality = find_best_values_per_model(results)
            
            # Print baseline
            if all(v is not None for v in [baseline["diversity"], baseline["rouge_l"], baseline["quality"]]):
                f.write(f"& Baseline & {format_metric(baseline['diversity'], baseline['diversity_std'], baseline['diversity'] == best_diversity)} & {format_metric(baseline['rouge_l'], baseline['rouge_l_std'], baseline['rouge_l'] == best_rouge_l)} & {format_metric(baseline['quality'], baseline['quality_std'], baseline['quality'] == best_quality)} \\\\\n")
            
            # Print other methods
            for method in ["Sequence", "Multi-turn"]:
                data = results.get(method)
                if data and all(v is not None for v in [data["diversity"], data["rouge_l"], data["quality"]]):
                    f.write(f"& {method} & {format_metric(data['diversity'], data['diversity_std'], data['diversity'] == best_diversity)} & {format_metric(data['rouge_l'], data['rouge_l_std'], data['rouge_l'] == best_rouge_l)} & {format_metric(data['quality'], data['quality_std'], data['quality'] == best_quality)} \\\\\n")
            
            # Print Verbalized Sampling methods
            f.write("& \\textbf{Verbalized Sampling} \\\\\n")
            for method, display_name in [("VS-Standard", "$\\hookrightarrow$ Standard"), ("VS-CoT", "$\\hookrightarrow$ CoT"), ("VS-Combined", "$\\hookrightarrow$ Combined")]:
                data = results.get(method)
                if data and all(v is not None for v in [data["diversity"], data["rouge_l"], data["quality"]]):
                    f.write(f"& {display_name} & {format_metric(data['diversity'], data['diversity_std'], data['diversity'] == best_diversity)} & {format_metric(data['rouge_l'], data['rouge_l_std'], data['rouge_l'] == best_rouge_l)} & {format_metric(data['quality'], data['quality_std'], data['quality'] == best_quality)} \\\\\n")
            
            f.write("\\midrule\n")
        
        # Generate summary statistics
        f.write("\n% SUMMARY STATISTICS\n")
        f.write("% " + "="*60 + "\n")
        
        # Calculate average improvements across all models
        diversity_improvements = []
        rouge_l_improvements = []
        quality_improvements = []
        
        for model_name, results in all_results.items():
            baseline = results.get("Baseline")
            if not baseline or any(v is None for v in [baseline["diversity"], baseline["rouge_l"], baseline["quality"]]):
                continue
            
            # Find best performing verbalized sampling method for this model
            best_vs_score = -float('inf')
            best_vs_method = None
            
            for method in ["VS-Standard", "VS-CoT", "VS-Combined"]:
                data = results.get(method)
                if data and all(v is not None for v in [data["diversity"], data["rouge_l"], data["quality"]]):
                    # Composite score: higher diversity + higher quality + lower rouge_l
                    score = data["diversity"] + data["quality"] - data["rouge_l"]
                    if score > best_vs_score:
                        best_vs_score = score
                        best_vs_method = method
            
            if best_vs_method:
                best_data = results[best_vs_method]
                diversity_imp = ((best_data["diversity"] - baseline["diversity"]) / baseline["diversity"]) * 100
                rouge_l_imp = ((baseline["rouge_l"] - best_data["rouge_l"]) / baseline["rouge_l"]) * 100  # Improvement is reduction
                quality_imp = ((best_data["quality"] - baseline["quality"]) / baseline["quality"]) * 100
                
                diversity_improvements.append(diversity_imp)
                rouge_l_improvements.append(rouge_l_imp)
                quality_improvements.append(quality_imp)
        
        if diversity_improvements:
            f.write(f"% Average Diversity Improvement: +{np.mean(diversity_improvements):.1f}% ± {np.std(diversity_improvements):.1f}%\n")
        if rouge_l_improvements:
            f.write(f"% Average Rouge-L Improvement: -{np.mean(rouge_l_improvements):.1f}% ± {np.std(rouge_l_improvements):.1f}%\n")
        if quality_improvements:
            f.write(f"% Average Quality Improvement: +{np.mean(quality_improvements):.1f}% ± {np.std(quality_improvements):.1f}%\n")
    
    print(f"\n✓ LaTeX table data written to: {output_file}")
    print(f"You can now copy the content from {output_file} into your LaTeX document.")
    
    # Generate plots
    print("\n" + "="*50)
    print(f"GENERATING PLOTS FOR {args.experiment.upper()} EXPERIMENTS")
    print("="*50)
    
    plot_model_comparison(all_results, args.output_dir, args.experiment)
    plot_method_averages(all_results, args.output_dir, args.experiment)
    
    print(f"\n✓ All plots saved to '{args.output_dir}/' directory")
    print("✓ Generated files:")
    print(f"  - model_comparison_diversity_{args.experiment}.png/pdf")
    print(f"  - model_comparison_rouge_l_{args.experiment}.png/pdf")  
    print(f"  - model_comparison_quality_{args.experiment}.png/pdf")
    print(f"  - method_average_diversity_{args.experiment}.png/pdf")
    print(f"  - method_average_rouge_l_{args.experiment}.png/pdf")
    print(f"  - method_average_quality_{args.experiment}.png/pdf")

if __name__ == "__main__":
    main()