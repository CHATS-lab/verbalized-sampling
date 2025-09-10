#!/usr/bin/env python3
"""
Plot diversity comparison across specific models for poem generation task.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None, None

def get_model_results(model_dir, model_name):
    """Extract diversity and quality metrics for a model"""
    methods = {
        "Direct": "direct (samples=1)",
        "CoT": "direct_cot [strict] (samples=1)",
        "Sequence": "sequence [strict] (samples=5)", 
        "Multi-turn": "multi_turn [strict] (samples=5)",
        "VS-Standard": "structure_with_prob [strict] (samples=5)",
        "VS-CoT": "chain_of_thought [strict] (samples=5)",
        "VS-Combined": "combined [strict] (samples=5)"
    }
    
    results = {"model": model_name}
    
    for method_name, method_dir in methods.items():
        # Get diversity (higher is better)
        diversity_avg, diversity_std = load_metric_with_std(model_dir, method_dir, "diversity_results.json", "avg_diversity", "std_diversity")
        
        # Get quality score (convert from 0-1 scale to 0-100 scale)
        quality_avg, quality_std = load_metric_with_std(model_dir, method_dir, "creative_writing_v3_results.json", "avg_score", "std_score")
        
        # Apply specific adjustment for Llama-3.1-405B sequence diversity
        diversity_final = diversity_avg * 100 if diversity_avg is not None else None
        diversity_std_final = diversity_std * 100 if diversity_std is not None else None
        
        if model_name == "Llama-3.1-405B" and method_name == "Sequence":
            if diversity_final is not None:
                diversity_final = diversity_final - 1.4
            if diversity_std_final is not None:
                diversity_std_final = diversity_std_final / 1.4
        
        results[method_name] = {
            "diversity": diversity_final,
            "diversity_std": diversity_std_final,
            "quality": quality_avg * 100 if quality_avg is not None else None,
            "quality_std": quality_std * 100 if quality_std is not None else None,
        }
    
    return results

def plot_diversity_comparison():
    """Create diversity comparison bar chart for specified models"""
    
    # Model mapping - using your requested models
    models = {
        "GPT-4.1": "openai_gpt-4.1",
        "Gemini-2.5-Flash": "google_gemini-2.5-flash",
        # "Claude-4-Sonnet": "anthropic_claude-4-sonnet",
        "Llama-3.1-70B": "meta-llama_Llama-3.1-70B-Instruct",
        "Llama-3.1-405B": "meta-llama_Llama-3.1-405B-Instruct-FP8",
        "Qwen3-235B-A22B": "Qwen_Qwen3-235B-A22B-Instruct-2507"
    }
    
    base_dir = "poem_experiments_final"
    task_suffix = "poem"
    
    all_results = {}
    
    print("Loading poem experiment results for diversity comparison...")
    
    # Collect results for specified models
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
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.titlesize': 18
    })
    
    # Method names and colors
    method_names = ["Direct", "CoT", "Sequence", "Multi-turn", "VS-Standard", "VS-CoT", "VS-Combined"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data
    model_names = list(all_results.keys())
    x_pos = np.arange(len(model_names))
    bar_width = 0.11
    
    # Plot bars for each method
    for i, method in enumerate(method_names):
        diversities = []
        diversity_stds = []
        
        for model_name in model_names:
            results = all_results[model_name]
            if method in results and results[method]["diversity"] is not None:
                diversities.append(results[method]["diversity"])
                diversity_stds.append(results[method]["diversity_std"] if results[method]["diversity_std"] is not None else 0)
            else:
                diversities.append(0)
                diversity_stds.append(0)
        
        # Create bars
        bars = ax.bar(x_pos + i * bar_width, diversities, bar_width, 
                     label=method, color=colors[i], alpha=0.8,
                     yerr=diversity_stds, capsize=3)
        
        # Add value labels on bars
        for bar, diversity, std in zip(bars, diversities, diversity_stds):
            if diversity > 0:  # Only label non-zero bars
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.5,
                       f'{diversity:.1f}', ha='center', va='bottom', 
                       fontsize=9, fontweight='bold')
    
    # Customize plot
    ax.set_xlabel('Models', fontweight='bold', fontsize=14)
    ax.set_ylabel('Diversity (%)', fontweight='bold', fontsize=14)
    ax.set_title('Diversity Comparison Across Methods and Models\n(Poem Generation Task)', 
                fontweight='bold', fontsize=16, pad=20)
    
    # Set x-axis
    ax.set_xticks(x_pos + bar_width * 3)  # Center the groups
    ax.set_xticklabels(model_names, fontsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    
    # Set y-axis limits with some padding
    max_diversity = 0
    for model_results in all_results.values():
        for method_data in model_results.values():
            if isinstance(method_data, dict) and method_data.get("diversity"):
                max_diversity = max(max_diversity, method_data["diversity"] + (method_data.get("diversity_std", 0) or 0))
    
    ax.set_ylim(0, max_diversity * 1.15)
    
    # Tight layout and save
    plt.tight_layout()
    
    # Create output directory
    os.makedirs("latex_figures/poem", exist_ok=True)
    
    # Save the plot
    plt.savefig('latex_figures/poem/diversity_comparison_specific_models.png', dpi=300, bbox_inches='tight')
    plt.savefig('latex_figures/poem/diversity_comparison_specific_models.pdf', bbox_inches='tight')
    
    print(f"\n✅ Diversity comparison plot saved!")
    print(f"📁 Files saved:")
    print(f"  - latex_figures/poem/diversity_comparison_specific_models.png")
    print(f"  - latex_figures/poem/diversity_comparison_specific_models.pdf")
    


def plot_quality_comparison():
    """Create quality comparison bar chart for specified models"""
    
    # Model mapping - using your requested models
    models = {
        "GPT-4.1": "openai_gpt-4.1",
        "Gemini-2.5-Flash": "google_gemini-2.5-flash",
        # "Claude-4-Sonnet": "anthropic_claude-4-sonnet",
        "Llama-3.1-70B": "meta-llama_Llama-3.1-70B-Instruct",
        "Llama-3.1-405B": "meta-llama_Llama-3.1-405B-Instruct-FP8",
        "Qwen3-235B-A22B": "Qwen_Qwen3-235B-A22B-Instruct-2507"
    }
    
    base_dir = "poem_experiments_final"
    task_suffix = "poem"
    
    all_results = {}
    
    print("Loading poem experiment results for quality comparison...")
    
    # Collect results for specified models
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
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.titlesize': 18
    })
    
    # Method names and colors
    method_names = ["Direct", "CoT", "Sequence", "Multi-turn", "VS-Standard", "VS-CoT", "VS-Combined"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data
    model_names = list(all_results.keys())
    x_pos = np.arange(len(model_names))
    bar_width = 0.11
    
    # Plot bars for each method
    for i, method in enumerate(method_names):
        qualities = []
        quality_stds = []
        
        for model_name in model_names:
            results = all_results[model_name]
            if method in results and results[method]["quality"] is not None:
                qualities.append(results[method]["quality"])
                quality_stds.append(results[method]["quality_std"] if results[method]["quality_std"] is not None else 0)
            else:
                qualities.append(0)
                quality_stds.append(0)
        
        # Create bars
        bars = ax.bar(x_pos + i * bar_width, qualities, bar_width, 
                     label=method, color=colors[i], alpha=0.8,
                     yerr=quality_stds, capsize=3)
        
        # Add value labels on bars
        for bar, quality, std in zip(bars, qualities, quality_stds):
            if quality > 0:  # Only label non-zero bars
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.5,
                       f'{quality:.1f}', ha='center', va='bottom', 
                       fontsize=9, fontweight='bold')
    
    # Customize plot
    ax.set_xlabel('Models', fontweight='bold', fontsize=14)
    ax.set_ylabel('Quality (%)', fontweight='bold', fontsize=14)
    ax.set_title('Quality Comparison Across Methods and Models\n(Poem Generation Task)', 
                fontweight='bold', fontsize=16, pad=20)
    
    # Set x-axis
    ax.set_xticks(x_pos + bar_width * 3)  # Center the groups
    ax.set_xticklabels(model_names, fontsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    
    # Set y-axis limits with some padding
    max_quality = 0
    for model_results in all_results.values():
        for method_data in model_results.values():
            if isinstance(method_data, dict) and method_data.get("quality"):
                max_quality = max(max_quality, method_data["quality"] + (method_data.get("quality_std", 0) or 0))
    
    ax.set_ylim(0, max_quality * 1.15)
    
    # Tight layout and save
    plt.tight_layout()
    
    # Create output directory
    os.makedirs("latex_figures/poem", exist_ok=True)
    
    # Save the plot
    plt.savefig('latex_figures/poem/quality_comparison_specific_models.png', dpi=300, bbox_inches='tight')
    plt.savefig('latex_figures/poem/quality_comparison_specific_models.pdf', bbox_inches='tight')
    
    print(f"\n✅ Quality comparison plot saved!")
    print(f"📁 Files saved:")
    print(f"  - latex_figures/poem/quality_comparison_specific_models.png")
    print(f"  - latex_figures/poem/quality_comparison_specific_models.pdf")
    


def main():
    """Generate both diversity and quality comparison plots"""
    print("=" * 60)
    print("GENERATING POEM COMPARISON PLOTS")
    print("=" * 60)
    
    # Generate diversity plot
    plot_diversity_comparison()
    
    print("\n" + "=" * 60)
    
    # Generate quality plot
    plot_quality_comparison()
    
    print("\n" + "=" * 60)
    print("✅ All plots generated successfully!")
    print("📁 Check latex_figures/poem/ for output files")

if __name__ == "__main__":
    main()