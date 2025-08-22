#!/usr/bin/env python3
"""
Unified plotting script for both poem and story experiments.
Generates publication-quality figures for analysis and visualization.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import argparse
from pathlib import Path

def load_metric(model_dir, method, metric_file, metric_key):
    """Load a specific metric from a results file"""
    file_path = os.path.join(model_dir, "evaluation", method, metric_file)
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        overall_metrics = data.get('overall_metrics', {})
        return overall_metrics.get(metric_key, None)
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None

def get_model_results(model_dir, model_name):
    """Extract all metrics for a model"""
    methods = {
        "Direct": "direct (samples=1)",
        "CoT": "direct_cot [strict] (samples=1)",
        "Sequence": "sequence [strict] (samples=5)", 
        "Multi-turn": "multi_turn [strict] (samples=5)",
        "VS-Standard": "structure_with_prob [strict] (samples=5)",
        "VS-CoT": "chain_of_thought [strict] (samples=5)",
        "VS-Multi": "combined [strict] (samples=5)"
    }
    
    results = {"model": model_name}
    
    for method_name, method_dir in methods.items():
        # Get diversity (higher is better)
        diversity_avg = load_metric(model_dir, method_dir, "diversity_results.json", "avg_diversity")
        diversity_std = load_metric(model_dir, method_dir, "diversity_results.json", "std_diversity")
        
        # Get Rouge-L (lower is better)
        rouge_l_avg = load_metric(model_dir, method_dir, "ngram_results.json", "avg_rouge_l")
        rouge_l_std = load_metric(model_dir, method_dir, "ngram_results.json", "std_rouge_l")
        
        # Get quality score (0-1 scale)
        quality_avg = load_metric(model_dir, method_dir, "creative_writing_v3_results.json", "avg_score")
        quality_std = load_metric(model_dir, method_dir, "creative_writing_v3_results.json", "std_score")
        
        results[method_name] = {
            "diversity": diversity_avg * 100 if diversity_avg is not None else None,
            "diversity_std": diversity_std * 100 if diversity_std is not None else None,
            "rouge_l": rouge_l_avg * 100 if rouge_l_avg is not None else None,
            "rouge_l_std": rouge_l_std * 100 if rouge_l_std is not None else None,
            "quality": quality_avg * 100 if quality_avg is not None else None,
            "quality_std": quality_std * 100 if quality_std is not None else None
        }
    
    return results

def plot_diversity_vs_quality_individual(results, model_name, task_type, output_dir):
    """Create individual diversity vs quality scatter plot for a model"""
    
    # Create task-specific subdirectory
    task_output_dir = os.path.join(output_dir, task_type, "individual_models")
    os.makedirs(task_output_dir, exist_ok=True)
    
    # Academic style settings with News Gothic MT font
    plt.rcParams.update({
        'font.family': 'News Gothic MT',
        'font.size': 11,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'lines.linewidth': 2.0,
        'lines.markersize': 8
    })
    
    method_names = ["Direct", "CoT", "Sequence", "Multi-turn", 
                   "VS-Standard", "VS-CoT", "VS-Multi"]
    
    # Academic color palette with high contrast
    colors = {
        'Direct': '#1f77b4',       # Blue
        'CoT': '#ff7f0e',          # Orange  
        'Sequence': '#2ca02c',     # Green
        'Multi-turn': '#d62728',   # Red
        'VS-Standard': '#9467bd',  # Purple
        'VS-CoT': '#8c564b',       # Brown
        'VS-Multi': '#e377c2'   # Pink
    }
    
    # Marker styles for better distinction
    markers = {
        'Direct': 'o',
        'CoT': 's', 
        'Sequence': '^',
        'Multi-turn': 'D',
        'VS-Standard': 'v',
        'VS-CoT': 'p',
        'VS-Multi': '*'
    }
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot each method
    for method in method_names:
        data = results.get(method)
        if data and data["diversity"] is not None and data["quality"] is not None:
            # Main point
            ax.scatter(data["diversity"], data["quality"], 
                      color=colors[method], marker=markers[method], 
                      s=100, alpha=0.8, 
                      edgecolors='white', linewidth=1.5,
                      label=method, zorder=5)
            
            # Error bars if std is available
            if data["diversity_std"] is not None and data["quality_std"] is not None:
                ax.errorbar(data["diversity"], data["quality"],
                           xerr=data["diversity_std"], yerr=data["quality_std"],
                           color=colors[method], alpha=0.4, capsize=3, capthick=1.5,
                           linestyle='none', zorder=3, elinewidth=3, markeredgewidth=10)
    
    # Add directional arrows to indicate "better"
    ax.text(0.98, 0.02, '→ Diversity', 
            ha='right', va='bottom',
            transform=ax.transAxes, 
            fontsize=10, 
            color='gray', 
            alpha=0.7,
            style='italic')
    
    ax.text(0.02, 0.98, '↑ Quality', 
            ha='left', va='top',
            transform=ax.transAxes, 
            fontsize=10, 
            color='gray', 
            alpha=0.7,
            style='italic')
    
    # Add a subtle annotation for the optimal region
    ax.text(0.98, 0.98, 'Optimal', 
            ha='right', va='top',
            transform=ax.transAxes, 
            fontsize=9, 
            color='gray', 
            alpha=0.5,
            style='italic')
    
    # Set labels with units
    ax.set_xlabel('Diversity (%)', fontsize=13, fontweight='normal')
    ax.set_ylabel('Quality (%)', fontsize=13, fontweight='normal')
    
    task_display = task_type.capitalize() + " Generation"
    ax.set_title(f'{model_name} - {task_display}', fontsize=14, fontweight='bold', pad=15)
    
    # Set axis limits with some padding
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_padding = (x_max - x_min) * 0.05
    y_padding = (y_max - y_min) * 0.05
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    
    # Configure grid
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Configure spines
    for spine in ax.spines.values():
        spine.set_color('#666666')
        spine.set_linewidth(1.0)
    
    # Legend configuration
    legend = ax.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fancybox=False,
        shadow=False,
        ncol=1,
        borderpad=1,
        columnspacing=1,
        handletextpad=0.8,
        framealpha=0.95,
        edgecolor='#666666'
    )
    legend.get_frame().set_linewidth(0.8)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    safe_model_name = model_name.replace(' ', '_').replace('.', '_')
    plt.savefig(f'{task_output_dir}/diversity_vs_quality_{safe_model_name}.png', 
               dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(f'{task_output_dir}/diversity_vs_quality_{safe_model_name}.pdf', 
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✓ Saved diversity vs quality plot for {model_name} ({task_type})")

def plot_method_averages(all_results, task_type, output_dir):
    """Create bar charts showing average performance across all models for each method"""
    
    # Create task-specific subdirectory
    task_output_dir = os.path.join(output_dir, task_type, "method_averages")
    os.makedirs(task_output_dir, exist_ok=True)
    
    plt.style.use('seaborn-v0_8')
    plt.rcParams.update({
        'font.family': 'News Gothic MT',
        'font.size': 12,
        'font.weight': 'heavy'
    })
    # Set up the plotting style
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    method_names = ["Direct", "CoT", "Sequence", "Multi-turn", "VS-Standard", "VS-CoT", "VS-Multi"]
    
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
    vs_methods = ["VS-Standard", "VS-CoT", "VS-Multi"]
    baseline_methods = ["Direct", "CoT", "Sequence", "Multi-turn"]
    
    metrics = [
        ('diversity', 'Average Diversity (%)', 'Higher is Better'),
        ('rouge_l', 'Average Rouge-L (%)', 'Lower is Better'),
        ('quality', 'Average Quality Score (%)', 'Higher is Better')
    ]
    
    for metric_key, metric_title, direction in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        means = [method_means[method][metric_key] for method in method_names]
        stds = [method_stds[method][metric_key] for method in method_names]
        
        # Create bars with hatches for VS methods
        bars = ax.bar(method_names, means, yerr=stds, capsize=5, 
                     color=colors[:len(method_names)], alpha=0.8,
                     ecolor='black', error_kw={'markeredgewidth':1})
        
        # Add hatches to VS methods (last 3 bars)
        for i, bar in enumerate(bars[-3:], start=len(bars)-3):
            bar.set_hatch('///')
            bar._hatch_color = (0.0, 0.0, 0.0)

        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.5,
                   f'{mean:.1f}±{std:.1f}', ha='center', va='bottom', 
                   fontsize=16, fontweight='bold')
        
        # Find best VS method for this metric
        vs_means = [method_means[method][metric_key] for method in vs_methods]
        if metric_key == 'rouge_l':  # Lower is better
            best_vs_idx = np.argmin(vs_means)
        else:  # Higher is better
            best_vs_idx = np.argmax(vs_means)
        
        best_vs_method = vs_methods[best_vs_idx]
        best_vs_method = "VS-Standard"
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
        
        # Add p-test results annotation in top left (for diversity only to avoid clutter)
        if metric_key in ['diversity']:
            p_text_lines = ["VS-Standard $p$-values:", 
            # f"Best VS: {best_vs_method}"
            ]
            for baseline_method in baseline_methods:
                p_val = p_values[baseline_method]
                if p_val is not None:
                    if p_val < 0.05:
                        sig_marker = f'{p_val:.2f} (p < 0.05)'
                    else:
                        sig_marker = f'{p_val:.2f} (p ≥ 0.05)'
                    p_text_lines.append(f"{baseline_method}: {sig_marker}")
                else:
                    p_text_lines.append(f"{baseline_method}: insufficient data")
            
            p_text = '\n'.join(p_text_lines)
            ax.text(0.02, 0.98, p_text, transform=ax.transAxes, fontsize=15, 
                   verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
                   fontweight='bold')
        
        # Highlight best performing method
        if metric_key == 'rouge_l':  # Lower is better
            best_idx = np.argmin(means)
        else:  # Higher is better
            best_idx = np.argmax(means)
        
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(3)
        
        # ax.set_xlabel('Methods', fontsize=16, fontweight='bold')
        ax.set_ylabel(metric_title, fontsize=20, fontweight='bold')
        
        task_display = task_type.capitalize()
        # ax.set_title(f'{metric_title} -- Average Across All Models ({task_display})', 
                    # fontsize=24, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        
        max_y = max(means) * 1.45
        # NO rotation for x-axis labels
        plt.xticks(rotation=0)
        plt.ylim(0, max_y)
        plt.tight_layout()
        
        plt.savefig(f'{task_output_dir}/method_average_{metric_key}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{task_output_dir}/method_average_{metric_key}.pdf', bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved {metric_title} method average plot for {task_type} experiments")
        print(f"  Best VS method: {best_vs_method}")

def plot_all_models_comparison(all_results, task_type, output_dir):
    """Create comparison plot across all models"""
    
    # Create task-specific subdirectory
    task_output_dir = os.path.join(output_dir, task_type, "model_comparisons")
    os.makedirs(task_output_dir, exist_ok=True)
    
    # Set up seaborn style
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.titlesize': 18
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Metrics to plot
    metrics = [
        ('diversity', 'Diversity (%)', 'higher'),
        ('rouge_l', 'Rouge-L (%)', 'lower'), 
        ('quality', 'Quality (%)', 'higher')
    ]
    
    methods_order = ["Direct", "CoT", "Sequence", "Multi-turn", 
                    "VS-Standard", "VS-CoT", "VS-Multi"]
    
    # Colors for methods
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
              '#9467bd', '#8c564b', '#e377c2']
    
    for idx, (metric, ylabel, better_direction) in enumerate(metrics):
        ax = axes[idx]
        
        # Prepare data for each method
        method_data = {method: [] for method in methods_order}
        model_names = []
        
        for model_name, results in all_results.items():
            model_names.append(model_name)
            for method in methods_order:
                if method in results and results[method][metric] is not None:
                    method_data[method].append(results[method][metric])
                else:
                    method_data[method].append(None)
        
        # Create bar plot
        x_pos = np.arange(len(model_names))
        bar_width = 0.11
        
        for i, method in enumerate(methods_order):
            values = method_data[method]
            # Replace None with 0 for plotting
            plot_values = [v if v is not None else 0 for v in values]
            
            ax.bar(x_pos + i * bar_width, plot_values, bar_width, 
                  label=method, color=colors[i], alpha=0.8, edgecolor='none')
        
        ax.set_xlabel('Models', fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(f'{ylabel} Comparison', fontweight='bold')
        ax.set_xticks(x_pos + bar_width * 3)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        
        if idx == 0:  # Only show legend for first subplot
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Remove the fourth subplot (we only have 3 metrics)
    fig.delaxes(axes[3])
    
    task_display = task_type.capitalize()
    plt.suptitle(f'{task_display} Generation Results - All Models Comparison', 
                fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    plt.savefig(f'{task_output_dir}/all_models_comparison.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f'{task_output_dir}/all_models_comparison.pdf', 
               bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✓ Saved all models comparison plot")

def generate_all_plots(task_type):
    """Generate all plots for specified task"""
    
    # Model directory mapping
    models = {
        "Claude-4-Sonnet": "anthropic_claude-4-sonnet",
        "Claude-3.7-Sonnet": "anthropic_claude-3.7-sonnet", 
        "Gemini-2.5-Pro": "google_gemini-2.5-pro",
        "Gemini-2.5-Flash": "google_gemini-2.5-flash",
        "GPT-4.1": "openai_gpt-4.1",
        "GPT-4.1-Mini": "openai_gpt-4.1-mini",
        "GPT-o3": "openai_o3",
        "Llama-3.1-70B": "meta-llama_Llama-3.1-70B-Instruct",
        "DeepSeek-R1": "deepseek_deepseek-r1-0528"
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
    
    # Standardized output directory structure
    base_output_dir = "latex_figures"
    all_results = {}
    
    print(f"Loading {task_type} experiment results...")
    
    # Collect results for all models
    for model_name, model_dir_name in models.items():
        model_path = os.path.join(base_dir, model_dir_name, f"{model_dir_name}_{task_suffix}")
        if os.path.exists(model_path):
            results = get_model_results(model_path, model_name)
            all_results[model_name] = results
            print(f"✓ Processed {model_name}")
        else:
            print(f"⚠ Directory not found for {model_name}: {model_path}")
    
    if not all_results:
        print("❌ No results found. Check directory structure.")
        return
    
    print(f"\n🎯 Generating plots for {len(all_results)} models...")
    
    # Generate individual plots for each model
    for model_name, results in all_results.items():
        plot_diversity_vs_quality_individual(results, model_name, task_type, base_output_dir)
    
    # Generate comparison plots
    plot_all_models_comparison(all_results, task_type, base_output_dir)
    plot_method_averages(all_results, task_type, base_output_dir)
    
    print(f"\n🎉 All plots generated successfully!")
    print(f"📁 Check the 'latex_figures/{task_type}/' directory for results")
    print(f"📊 Directory structure:")
    print(f"  - latex_figures/{task_type}/individual_models/")
    print(f"  - latex_figures/{task_type}/method_averages/")
    print(f"  - latex_figures/{task_type}/model_comparisons/")

def main():
    parser = argparse.ArgumentParser(description='Generate plots for poem or story experiments')
    parser.add_argument('--task', choices=['poem', 'story', 'both'], default='both',
                        help='Which task to generate plots for (default: both)')
    
    args = parser.parse_args()
    
    if args.task == 'both':
        print("Generating plots for both poem and story experiments...")
        generate_all_plots('poem')
        print("\n" + "="*80 + "\n")
        generate_all_plots('story')
    else:
        generate_all_plots(args.task)

if __name__ == "__main__":
    main()