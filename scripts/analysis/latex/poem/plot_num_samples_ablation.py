#!/usr/bin/env python3
"""
Plot num samples ablation study results showing how diversity and quality 
change with the number of samples across different methods.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_metric_with_std(base_dir, method, num_samples, metric_file, avg_key, std_key):
    """Load a specific metric with standard deviation from a results file"""
    method_dir = f"{method}_num_samples_{num_samples}"
    file_path = os.path.join(base_dir, "evaluation", method_dir, metric_file)
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        overall_metrics = data.get('overall_metrics', {})
        avg_result = overall_metrics.get(avg_key, None)
        std_result = overall_metrics.get(std_key, None)
        return avg_result, std_result
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None, None

def get_ablation_results(base_dir, model_name):
    """Extract ablation results for diversity and quality across different methods and sample sizes"""
    
    # Available methods in the ablation study
    methods = ["direct", "sequence", "multi_turn", "vs_standard"]
    
    # Sample sizes to test (note: direct only has 1 sample)
    sample_sizes = [1, 3, 5, 10, 20]
    
    results = {}
    
    for method in methods:
        results[method] = {
            "sample_sizes": [],
            "diversity": [],
            "diversity_std": [],
            "quality": [],
            "quality_std": []
        }
        
        for num_samples in sample_sizes:
            # Direct method only has 1 sample
            if method == "direct" and num_samples != 1:
                continue
                
            # Get diversity
            diversity_avg, diversity_std = load_metric_with_std(
                base_dir, method, num_samples, "diversity_results.json", 
                "avg_diversity", "std_diversity"
            )
            
            # Get quality
            quality_avg, quality_std = load_metric_with_std(
                base_dir, method, num_samples, "creative_writing_v3_results.json", 
                "avg_score", "std_score"
            )
            
            # Only add if we have both metrics
            if diversity_avg is not None and quality_avg is not None:
                results[method]["sample_sizes"].append(num_samples)
                results[method]["diversity"].append(diversity_avg * 100)  # Convert to percentage
                results[method]["diversity_std"].append(diversity_std * 100 if diversity_std else 0)
                results[method]["quality"].append(quality_avg * 100)  # Convert to percentage
                results[method]["quality_std"].append(quality_std * 100 if quality_std else 0)
    
    return results

def plot_ablation_diversity(results, model_name):
    """Plot diversity vs number of samples"""
    
    plt.style.use('default')
    sns.set_palette("husl")
    
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Colors and markers for each method
    method_styles = {
        "direct": {"color": "#1f77b4", "marker": "o", "label": "Direct"},
        "sequence": {"color": "#ff7f0e", "marker": "s", "label": "Sequence"},
        "multi_turn": {"color": "#2ca02c", "marker": "^", "label": "Multi-turn"},
        "vs_standard": {"color": "#d62728", "marker": "D", "label": "VS-Standard"}
    }
    
    for method, data in results.items():
        if not data["sample_sizes"]:  # Skip if no data
            continue
            
        style = method_styles[method]
        
        # Plot line with error bars
        ax.errorbar(data["sample_sizes"], data["diversity"], 
                   yerr=data["diversity_std"],
                   color=style["color"], marker=style["marker"], 
                   label=style["label"], linewidth=2, markersize=8,
                   capsize=5, capthick=2)
    
    ax.set_xlabel('Number of Samples', fontweight='bold', fontsize=14)
    ax.set_ylabel('Diversity (%)', fontweight='bold', fontsize=14)
    ax.set_title(f'Diversity vs Number of Samples\n{model_name} - Poem Generation', 
                fontweight='bold', fontsize=16, pad=20)
    
    # Set x-axis to show all sample sizes
    ax.set_xlim(0, 21)
    ax.set_xticks([1, 3, 5, 10, 20])
    
    # Add grid
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    # Legend
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs("latex_figures/poem/ablation", exist_ok=True)
    plt.savefig(f'latex_figures/poem/ablation/{model_name.lower().replace("-", "_")}_diversity_ablation.png', 
               dpi=300, bbox_inches='tight')
    plt.savefig(f'latex_figures/poem/ablation/{model_name.lower().replace("-", "_")}_diversity_ablation.pdf', 
               bbox_inches='tight')
    
    print(f"✅ Diversity ablation plot saved for {model_name}")
    
    # plt.show()

def plot_ablation_quality(results, model_name):
    """Plot quality vs number of samples"""
    
    plt.style.use('default')
    sns.set_palette("husl")
    
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Colors and markers for each method
    method_styles = {
        "direct": {"color": "#1f77b4", "marker": "o", "label": "Direct"},
        "sequence": {"color": "#ff7f0e", "marker": "s", "label": "Sequence"},
        "multi_turn": {"color": "#2ca02c", "marker": "^", "label": "Multi-turn"},
        "vs_standard": {"color": "#d62728", "marker": "D", "label": "VS-Standard"}
    }
    
    for method, data in results.items():
        if not data["sample_sizes"]:  # Skip if no data
            continue
            
        style = method_styles[method]
        
        # Plot line with error bars
        ax.errorbar(data["sample_sizes"], data["quality"], 
                   yerr=data["quality_std"],
                   color=style["color"], marker=style["marker"], 
                   label=style["label"], linewidth=2, markersize=8,
                   capsize=5, capthick=2)
    
    ax.set_xlabel('Number of Samples', fontweight='bold', fontsize=14)
    ax.set_ylabel('Quality (%)', fontweight='bold', fontsize=14)
    ax.set_title(f'Quality vs Number of Samples\n{model_name} - Poem Generation', 
                fontweight='bold', fontsize=16, pad=20)
    
    # Set x-axis to show all sample sizes
    ax.set_xlim(0, 21)
    ax.set_xticks([1, 3, 5, 10, 20])
    
    # Add grid
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    # Legend
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs("latex_figures/poem/ablation", exist_ok=True)
    plt.savefig(f'latex_figures/poem/ablation/{model_name.lower().replace("-", "_")}_quality_ablation.png', 
               dpi=300, bbox_inches='tight')
    plt.savefig(f'latex_figures/poem/ablation/{model_name.lower().replace("-", "_")}_quality_ablation.pdf', 
               bbox_inches='tight')
    
    print(f"✅ Quality ablation plot saved for {model_name}")
    
    # plt.show()

def plot_combined_ablation(results, model_name):
    """Plot both diversity and quality in a single figure with subplots"""
    
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
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Colors and markers for each method
    method_styles = {
        "direct": {"color": "#1f77b4", "marker": "o", "label": "Direct"},
        "sequence": {"color": "#ff7f0e", "marker": "s", "label": "Sequence"},
        "multi_turn": {"color": "#2ca02c", "marker": "^", "label": "Multi-turn"},
        "vs_standard": {"color": "#d62728", "marker": "D", "label": "VS-Standard"}
    }
    
    # Plot diversity
    for method, data in results.items():
        if not data["sample_sizes"]:  # Skip if no data
            continue
            
        style = method_styles[method]
        
        ax1.errorbar(data["sample_sizes"], data["diversity"], 
                    yerr=data["diversity_std"],
                    color=style["color"], marker=style["marker"], 
                    label=style["label"], linewidth=2, markersize=8,
                    capsize=5, capthick=2)
    
    ax1.set_xlabel('Number of Samples', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Diversity (%)', fontweight='bold', fontsize=14)
    ax1.set_title('Diversity vs Number of Samples', fontweight='bold', fontsize=14)
    ax1.set_xlim(0, 21)
    ax1.set_xticks([1, 3, 5, 10, 20])
    ax1.grid(True, alpha=0.3)
    ax1.set_axisbelow(True)
    
    # Plot quality
    for method, data in results.items():
        if not data["sample_sizes"]:  # Skip if no data
            continue
            
        style = method_styles[method]
        
        ax2.errorbar(data["sample_sizes"], data["quality"], 
                    yerr=data["quality_std"],
                    color=style["color"], marker=style["marker"], 
                    label=style["label"], linewidth=2, markersize=8,
                    capsize=5, capthick=2)
    
    ax2.set_xlabel('Number of Samples', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Quality (%)', fontweight='bold', fontsize=14)
    ax2.set_title('Quality vs Number of Samples', fontweight='bold', fontsize=14)
    ax2.set_xlim(0, 21)
    ax2.set_xticks([1, 3, 5, 10, 20])
    ax2.grid(True, alpha=0.3)
    ax2.set_axisbelow(True)
    
    # Add legend to the right side
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.98, 0.5), fontsize=12)
    
    # Overall title
    fig.suptitle(f'Ablation Study: {model_name} - Poem Generation', 
                fontweight='bold', fontsize=18, y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    
    # Save plot
    os.makedirs("latex_figures/poem/ablation", exist_ok=True)
    plt.savefig(f'latex_figures/poem/ablation/{model_name.lower().replace("-", "_")}_combined_ablation.png', 
               dpi=300, bbox_inches='tight')
    plt.savefig(f'latex_figures/poem/ablation/{model_name.lower().replace("-", "_")}_combined_ablation.pdf', 
               bbox_inches='tight')
    
    print(f"✅ Combined ablation plot saved for {model_name}")
    
    # plt.show()

def main():
    """Generate ablation plots for all available models"""
    
    # Available models in ablation study
    models = {
        "GPT-4.1": "openai_gpt-4.1_poem_num_samples_ablation",
        "Gemini-2.5-Flash": "google_gemini-2.5-flash_poem_num_samples_ablation"
    }
    
    base_dir = "num_samples_ablation_results"
    
    print("=" * 60)
    print("GENERATING NUM SAMPLES ABLATION PLOTS")
    print("=" * 60)
    
    for model_name, model_dir in models.items():
        model_path = os.path.join(base_dir, model_dir)
        
        if not os.path.exists(model_path):
            print(f"⚠ Directory not found for {model_name}: {model_path}")
            continue
            
        print(f"\n📊 Processing {model_name}...")
        
        # Load results
        results = get_ablation_results(model_path, model_name)
        
        # Generate plots
        plot_ablation_diversity(results, model_name)
        plot_ablation_quality(results, model_name)
        plot_combined_ablation(results, model_name)
        
        print(f"✅ All plots generated for {model_name}")
    
    print("\n" + "=" * 60)
    print("✅ All ablation plots generated successfully!")
    print("📁 Check latex_figures/poem/ablation/ for output files")

if __name__ == "__main__":
    main()