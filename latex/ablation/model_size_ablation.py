#!/usr/bin/env python3
"""
Model Size Ablation Study for Poem Generation Tasks
Analyzes how model size affects diversity-quality trade-offs and method effectiveness.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from scipy import stats
from scipy.spatial import ConvexHull
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

def load_all_results():
    """Load results for all models and categorize by size"""
    
    # Model categorization by size
    model_groups = {
        "large": {
            "GPT-4.1": "openai_gpt-4.1",
            "Gemini-2.5-Pro": "google_gemini-2.5-pro", 
            # "GPT-o3": "openai_o3",
            # "Claude-4-Sonnet": "anthropic_claude-4-sonnet",
            # "Llama-3.1-70B": "meta-llama_Llama-3.1-70B-Instruct",
            # "DeepSeek-R1": "deepseek_deepseek-r1-0528"
        },
        "small": {
            "GPT-4.1-Mini": "openai_gpt-4.1-mini",
            "Gemini-2.5-Flash": "google_gemini-2.5-flash",
            # "Llama-3.1-8B": "meta-llama_Llama-3.1-8B-Instruct"
        }
    }
    
    base_dir, task = "poem_experiments_final", "poem"
    # base_dir, task = "story_experiments_final", "book"
    results_by_size = {"large": {}, "small": {}}
    
    print("Loading model results by size...")
    
    for size_category, models in model_groups.items():
        print(f"\n{size_category.upper()} MODELS:")
        for model_name, model_dir_name in models.items():
            model_path = os.path.join(base_dir, model_dir_name, f"{model_dir_name}_{task}")
            if os.path.exists(model_path):
                results = get_model_results(model_path, model_name)
                results_by_size[size_category][model_name] = results
                print(f"  ✓ {model_name}")
            else:
                print(f"  ⚠ {model_name}: Directory not found")
    
    return results_by_size

def calculate_pareto_efficiency(diversity_values, quality_values):
    """Calculate Pareto efficiency metrics"""
    if len(diversity_values) < 3:
        return None
    
    points = np.column_stack((diversity_values, quality_values))
    
    # Find Pareto optimal points
    pareto_mask = np.zeros(len(points), dtype=bool)
    for i, point in enumerate(points):
        dominated = False
        for j, other_point in enumerate(points):
            if i != j:
                if (other_point[0] >= point[0] and other_point[1] >= point[1] and
                    (other_point[0] > point[0] or other_point[1] > point[1])):
                    dominated = True
                    break
        pareto_mask[i] = not dominated
    
    pareto_points = points[pareto_mask]
    
    if len(pareto_points) < 3:
        return None
    
    # Sort by diversity for area calculation
    sorted_indices = np.argsort(pareto_points[:, 0])
    sorted_pareto = pareto_points[sorted_indices]
    
    # Calculate area under Pareto curve using trapezoidal rule
    area = np.trapz(sorted_pareto[:, 1], sorted_pareto[:, 0])
    
    return {
        'area': area,
        'pareto_points': pareto_points,
        'n_pareto_points': len(pareto_points)
    }

def plot_size_comparison_scatter(results_by_size, output_dir="latex_figures"):
    """Create side-by-side scatter plots comparing model sizes"""
    
    # Create ablation-specific subdirectory
    ablation_output_dir = os.path.join(output_dir, "ablation", "model_size")
    os.makedirs(ablation_output_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'axes.linewidth': 1.2
    })
    
    method_names = ["Direct", "CoT", "Sequence", "Multi-turn", 
                   "VS-Standard", "VS-CoT", "VS-Multi"]
    
    # Colors and markers
    colors = {
        'Direct': '#1f77b4', 'CoT': '#ff7f0e', 'Sequence': '#2ca02c', 
        'Multi-turn': '#d62728', 'VS-Standard': '#9467bd', 
        'VS-CoT': '#8c564b', 'VS-Multi': '#e377c2'
    }
    
    markers = {
        'Direct': 'o', 'CoT': 's', 'Sequence': '^', 'Multi-turn': 'D',
        'VS-Standard': 'v', 'VS-CoT': 'p', 'VS-Multi': '*'
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    size_titles = {"small": "Small Models", "large": "Large Models"}
    axes = {"small": ax1, "large": ax2}
    
    pareto_stats = {}
    
    for size_category, ax in axes.items():
        results = results_by_size[size_category]
        
        # Collect all points for Pareto analysis
        all_diversity = []
        all_quality = []
        
        # Plot each method across all models in this size category
        for method in method_names:
            diversity_values = []
            quality_values = []
            
            for model_name, model_results in results.items():
                data = model_results.get(method)
                if data and data["diversity"] is not None and data["quality"] is not None:
                    diversity_values.append(data["diversity"])
                    quality_values.append(data["quality"])
                    all_diversity.append(data["diversity"])
                    all_quality.append(data["quality"])
            
            if diversity_values:
                ax.scatter(diversity_values, quality_values, 
                          color=colors[method], marker=markers[method], 
                          s=80, alpha=0.7, label=method, zorder=5,
                          edgecolors='white', linewidth=1)
        
        # Calculate and plot Pareto frontier
        pareto_result = calculate_pareto_efficiency(all_diversity, all_quality)
        if pareto_result:
            pareto_stats[size_category] = pareto_result
            pareto_points = pareto_result['pareto_points']
            
            # Sort for line plotting
            sorted_indices = np.argsort(pareto_points[:, 0])
            sorted_pareto = pareto_points[sorted_indices]
            
            ax.plot(sorted_pareto[:, 0], sorted_pareto[:, 1], 'r--', 
                   linewidth=2, alpha=0.8, label='Pareto Frontier', zorder=6)
            
            # Highlight Pareto points
            ax.scatter(pareto_points[:, 0], pareto_points[:, 1], 
                      facecolors='none', edgecolors='red', s=100, 
                      linewidth=2, alpha=0.8, zorder=7)
        
        # Formatting
        ax.set_xlabel('Diversity (%)', fontweight='bold')
        ax.set_ylabel('Quality (%)', fontweight='bold')
        ax.set_title(f'{size_titles[size_category]}', fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Add directional indicators
        ax.text(0.98, 0.02, '→ Better', ha='right', va='bottom',
                transform=ax.transAxes, fontsize=10, color='gray', alpha=0.7)
        ax.text(0.02, 0.98, '↑ Better', ha='left', va='top',
                transform=ax.transAxes, fontsize=10, color='gray', alpha=0.7)
    
    # Add legend to the right
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5),
               frameon=True, fancybox=False, shadow=False)
    
    # Add model classifications and Pareto efficiency comparison text
    if pareto_stats:
        # Get model lists for classification
        large_models_list = list(results_by_size['large'].keys())
        small_models_list = list(results_by_size['small'].keys())
        
        efficiency_text = "MODEL CLASSIFICATIONS:\n\n"
        efficiency_text += f"LARGE MODELS ({len(large_models_list)}):\n"
        efficiency_text += ", ".join(large_models_list[:3])  # Show first 3
        if len(large_models_list) > 3:
            efficiency_text += f", +{len(large_models_list)-3} more"
        efficiency_text += "\n\n"
        
        efficiency_text += f"SMALL MODELS ({len(small_models_list)}):\n"
        efficiency_text += ", ".join(small_models_list)
        efficiency_text += "\n\n"
        
        efficiency_text += "PARETO EFFICIENCY:\n"
        for size, stats in pareto_stats.items():
            area = stats['area']
            n_points = stats['n_pareto_points']
            efficiency_text += f"{size.title()}: Area={area:.1f}, Points={n_points}\n"
        
        fig.text(0.02, 0.02, efficiency_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8),
                verticalalignment='bottom')
    
    plt.suptitle('Model Size Ablation: Diversity vs Quality Trade-offs', 
                fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.15, right=0.85)
    
    plt.savefig(f'{ablation_output_dir}/model_size_diversity_quality_comparison.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f'{ablation_output_dir}/model_size_diversity_quality_comparison.pdf', 
               bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✓ Saved model size comparison scatter plots")
    return pareto_stats

def plot_cognitive_burden_analysis_subfigures(results_by_size, output_dir="latex_figures"):
    """Generate subfigures with separate legend for LaTeX integration"""
    
    # Create ablation-specific subdirectory
    ablation_output_dir = os.path.join(output_dir, "ablation", "model_size")
    os.makedirs(ablation_output_dir, exist_ok=True)
    
    method_names = ["Direct", "CoT", "Sequence", "Multi-turn", 
                   "VS-Standard", "VS-CoT", "VS-Multi"]
    
    # Define cognitive complexity levels for each method
    cognitive_complexity = {
        "Direct": 1,           # Baseline - no extra burden
        "CoT": 2,             # Think step-by-step - moderate burden
        "Sequence": 3,        # Generate multiple responses - higher burden
        "Multi-turn": 4,      # Conversational format - complex burden
        "VS-Standard": 5,     # JSON + confidence - high burden
        "VS-CoT": 6,         # JSON + reasoning + confidence - very high burden
        "VS-Multi": 7      # Most complex format - maximum burden
    }
    
    # Calculate performance changes relative to Direct baseline
    size_method_deltas = {}
    
    for size_category, results in results_by_size.items():
        size_method_deltas[size_category] = {}
        
        # Calculate deltas for each method
        for method_name in method_names[1:]:  # Skip Direct
            diversity_deltas = []
            quality_deltas = []
            
            for model_name, model_results in results.items():
                method_data = model_results.get(method_name)
                direct_data = model_results.get("Direct")
                
                if (method_data and method_data["quality"] is not None and 
                    direct_data and direct_data["quality"] is not None):
                    
                    div_delta = method_data["diversity"] - direct_data["diversity"]
                    qual_delta = method_data["quality"] - direct_data["quality"]
                    
                    diversity_deltas.append(div_delta)
                    quality_deltas.append(qual_delta)
            
            if diversity_deltas and quality_deltas:
                size_method_deltas[size_category][method_name] = {
                    'diversity_delta_mean': np.mean(diversity_deltas),
                    'diversity_delta_std': np.std(diversity_deltas),
                    'quality_delta_mean': np.mean(quality_deltas),
                    'quality_delta_std': np.std(quality_deltas),
                    'n_models': len(diversity_deltas),
                    'complexity': cognitive_complexity[method_name]
                }
    
    # Set up plotting style
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 16,
        'axes.linewidth': 1.2
    })
    
    # Methods to show (excluding Direct since we're showing deltas)
    methods_subset = ["Sequence", "Multi-turn", "VS-Standard", "VS-CoT", "VS-Multi"]
    x_methods = np.arange(len(methods_subset))
    width = 0.35
    
    # Prepare data
    large_diversity_changes = []
    small_diversity_changes = []
    large_div_errors = []
    small_div_errors = []
    large_quality_changes = []
    small_quality_changes = []
    large_qual_errors = []
    small_qual_errors = []
    
    for method in methods_subset:
        # Diversity data
        large_div_val = size_method_deltas.get('large', {}).get(method, {}).get('diversity_delta_mean', 0)
        small_div_val = size_method_deltas.get('small', {}).get(method, {}).get('diversity_delta_mean', 0)
        large_div_err = size_method_deltas.get('large', {}).get(method, {}).get('diversity_delta_std', 0)
        small_div_err = size_method_deltas.get('small', {}).get(method, {}).get('diversity_delta_std', 0)
        
        large_diversity_changes.append(large_div_val)
        small_diversity_changes.append(small_div_val)
        large_div_errors.append(large_div_err)
        small_div_errors.append(small_div_err)
        
        # Quality data
        large_qual_val = size_method_deltas.get('large', {}).get(method, {}).get('quality_delta_mean', 0)
        small_qual_val = size_method_deltas.get('small', {}).get(method, {}).get('quality_delta_mean', 0)
        large_qual_err = size_method_deltas.get('large', {}).get(method, {}).get('quality_delta_std', 0)
        small_qual_err = size_method_deltas.get('small', {}).get(method, {}).get('quality_delta_std', 0)
        
        large_quality_changes.append(large_qual_val)
        small_quality_changes.append(small_qual_val)
        large_qual_errors.append(large_qual_err)
        small_qual_errors.append(small_qual_err)
    
    # ============================================
    # Option 1: Create separate legend-only figure
    # ============================================
    
    # Create diversity plot WITHOUT legend
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    
    bars1 = ax1.bar(x_methods - width/2, small_diversity_changes, width, 
                   yerr=small_div_errors, capsize=3, 
                   color='#B565A7', alpha=0.8, label='Small Models (GPT-4.1-Mini, Gemini-2.5-Flash)')
    bars2 = ax1.bar(x_methods + width/2, large_diversity_changes, width, 
                   yerr=large_div_errors, capsize=3, 
                   color='#5B9BD5', alpha=0.8, label='Large Models (GPT-4.1, Gemini-2.5-Pro)')
    
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax1.set_ylabel('Diversity Change vs Direct (Δ)', fontweight='bold', fontsize=16)
    ax1.set_title('Diversity Impact by Model Size', fontweight='bold', fontsize=18)
    ax1.set_xticks(x_methods)
    ax1.set_xticklabels(methods_subset, fontsize=14)
    ax1.tick_params(axis='y', labelsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add subfigure label
    # ax1.text(-0.12, 1.02, '(a)', transform=ax1.transAxes, fontsize=16, fontweight='bold')
    
    fig1.savefig(f'{ablation_output_dir}/diversity_no_legend.pdf', bbox_inches='tight', facecolor='white')
    fig1.savefig(f'{ablation_output_dir}/diversity_no_legend.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig1)
    
    # Create quality plot WITHOUT legend
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    
    bars3 = ax2.bar(x_methods - width/2, small_quality_changes, width, 
                   yerr=small_qual_errors, capsize=3, 
                   color='#B565A7', alpha=0.8, label='Small Models')
    bars4 = ax2.bar(x_methods + width/2, large_quality_changes, width, 
                   yerr=large_qual_errors, capsize=3, 
                   color='#5B9BD5', alpha=0.8, label='Large Models')
    
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.set_ylabel('Quality Change vs Direct (Δ)', fontweight='bold', fontsize=16)
    ax2.set_title('Quality Impact by Model Size', fontweight='bold', fontsize=18)
    ax2.set_xticks(x_methods)
    ax2.set_xticklabels(methods_subset, fontsize=14)
    ax2.tick_params(axis='y', labelsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add subfigure label
    # ax2.text(-0.12, 1.02, '(b)', transform=ax2.transAxes, fontsize=16, fontweight='bold')
    
    fig2.savefig(f'{ablation_output_dir}/quality_no_legend.pdf', bbox_inches='tight', facecolor='white')
    fig2.savefig(f'{ablation_output_dir}/quality_no_legend.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    
    # Create a separate figure with ONLY the legend (same width as plot figures)
    fig_legend = plt.figure(figsize=(8, 0.8))
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.axis('off')
    
    # Create simple patches for cleaner legend
    legend_elements = [
        Patch(facecolor='#B565A7', label='Small Models (GPT-4.1-Mini, Gemini-2.5-Flash)'),
        Patch(facecolor='#5B9BD5', label='Large Models (GPT-4.1, Gemini-2.5-Pro)')
    ]
    
    # Add legend with clean formatting
    ax_legend.legend(handles=legend_elements, loc='center', ncol=2, frameon=False, fontsize=16,
                    handlelength=2, handletextpad=0.5, columnspacing=2)
    
    fig_legend.savefig(f'{ablation_output_dir}/legend_only.pdf', bbox_inches='tight', pad_inches=0, facecolor='white')
    fig_legend.savefig(f'{ablation_output_dir}/legend_only.png', dpi=300, bbox_inches='tight', pad_inches=0, facecolor='white')
    plt.close(fig_legend)
    
    # ============================================
    # Option 2: Combined figure with custom layout
    # ============================================
    
    # Create figure with GridSpec for custom layout
    fig_combined = plt.figure(figsize=(16, 8))
    gs = fig_combined.add_gridspec(3, 2, height_ratios=[1, 20, 1], width_ratios=[1, 1], hspace=0.4, wspace=0.3)
    
    # Legend axis (spans both columns at top)
    ax_legend_combined = fig_combined.add_subplot(gs[0, :])
    ax_legend_combined.axis('off')
    
    # Plot axes
    ax1_combined = fig_combined.add_subplot(gs[1, 0])
    ax2_combined = fig_combined.add_subplot(gs[1, 1])
    
    # Plot diversity data
    bars1_combined = ax1_combined.bar(x_methods - width/2, small_diversity_changes, width, 
                                     yerr=small_div_errors, capsize=3, 
                                     color='#B565A7', alpha=0.8, label='Small Models (GPT-4.1-Mini, Gemini-2.5-Flash)')
    bars2_combined = ax1_combined.bar(x_methods + width/2, large_diversity_changes, width, 
                                     yerr=large_div_errors, capsize=3, 
                                     color='#5B9BD5', alpha=0.8, label='Large Models (GPT-4.1, Gemini-2.5-Pro)')
    
    ax1_combined.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax1_combined.set_ylabel('Diversity Change vs Direct (Δ)', fontweight='bold', fontsize=16)
    ax1_combined.set_title('Diversity Impact by Model Size', fontweight='bold', fontsize=18)
    ax1_combined.set_xticks(x_methods)
    ax1_combined.set_xticklabels(methods_subset, fontsize=14)
    ax1_combined.tick_params(axis='y', labelsize=14)
    ax1_combined.grid(True, alpha=0.3, axis='y')
    
    # Plot quality data
    bars3_combined = ax2_combined.bar(x_methods - width/2, small_quality_changes, width, 
                                     yerr=small_qual_errors, capsize=3, 
                                     color='#B565A7', alpha=0.8)
    bars4_combined = ax2_combined.bar(x_methods + width/2, large_quality_changes, width, 
                                     yerr=large_qual_errors, capsize=3, 
                                     color='#5B9BD5', alpha=0.8)
    
    ax2_combined.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2_combined.set_ylabel('Quality Change vs Direct (Δ)', fontweight='bold', fontsize=16)
    ax2_combined.set_title('Quality Impact by Model Size', fontweight='bold', fontsize=18)
    ax2_combined.set_xticks(x_methods)
    ax2_combined.set_xticklabels(methods_subset, fontsize=14)
    ax2_combined.tick_params(axis='y', labelsize=14)
    ax2_combined.grid(True, alpha=0.3, axis='y')
    
    # Create legend in the top center
    handles = [bars1_combined, bars2_combined]
    labels = ['Small Models (GPT-4.1-Mini, Gemini-2.5-Flash)', 
              'Large Models (GPT-4.1, Gemini-2.5-Pro)']
    ax_legend_combined.legend(handles, labels, loc='center', ncol=2, frameon=False, 
                             bbox_to_anchor=(0.5, 0.5), fontsize=16)
    
    # Add subfigure labels
    ax1_combined.text(-0.12, 1.02, '(a)', transform=ax1_combined.transAxes, fontsize=16, fontweight='bold')
    ax2_combined.text(-0.12, 1.02, '(b)', transform=ax2_combined.transAxes, fontsize=16, fontweight='bold')
    
    fig_combined.savefig(f'{ablation_output_dir}/combined_figure_with_legend.pdf', bbox_inches='tight', facecolor='white')
    fig_combined.savefig(f'{ablation_output_dir}/combined_figure_with_legend.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_combined)
    
    print("✓ Saved cognitive burden analysis subfigures")
    print("📁 Generated files:")
    print("  - diversity_no_legend.pdf/png (subfigure a)")
    print("  - quality_no_legend.pdf/png (subfigure b)")
    print("  - legend_only.pdf/png (separate legend)")
    print("  - combined_figure_with_legend.pdf/png (all-in-one version)")
    print("📊 Methods analyzed: Sequence, Multi-turn, VS-Standard, VS-CoT, VS-Multi")
    
    return size_method_deltas

def plot_cognitive_burden_analysis(results_by_size, output_dir="latex_figures"):
    """Analyze cognitive burden effects focusing on Sequence, Multi-turn, and VS-Standard methods"""
    
    # Create ablation-specific subdirectory
    ablation_output_dir = os.path.join(output_dir, "ablation", "model_size")
    os.makedirs(ablation_output_dir, exist_ok=True)
    
    # Focus on VS methods plus key comparison methods for cognitive burden analysis
    focus_methods = ["Direct", "Sequence", "Multi-turn", "VS-Standard", "VS-CoT", "VS-Multi"]
    method_names = ["Direct", "CoT", "Sequence", "Multi-turn", 
                   "VS-Standard", "VS-CoT", "VS-Multi"]
    
    # Define cognitive complexity levels for each method
    cognitive_complexity = {
        "Direct": 1,           # Baseline - no extra burden
        "CoT": 2,             # Think step-by-step - moderate burden
        "Sequence": 3,        # Generate multiple responses - higher burden
        "Multi-turn": 4,      # Conversational format - complex burden
        "VS-Standard": 5,     # JSON + confidence - high burden
        "VS-CoT": 6,         # JSON + reasoning + confidence - very high burden
        "VS-Multi": 7      # Most complex format - maximum burden
    }
    
    # Calculate performance changes relative to Direct baseline
    size_method_deltas = {}
    
    for size_category, results in results_by_size.items():
        size_method_deltas[size_category] = {}
        
        # Get Direct baseline for this model size
        direct_metrics = []
        for model_name, model_results in results.items():
            direct_data = model_results.get("Direct")
            if direct_data and direct_data["quality"] is not None:
                direct_metrics.append({
                    'diversity': direct_data["diversity"],
                    'quality': direct_data["quality"]
                })
        
        if not direct_metrics:
            continue
            
        # Calculate average baseline performance
        avg_direct_diversity = np.mean([m['diversity'] for m in direct_metrics])
        avg_direct_quality = np.mean([m['quality'] for m in direct_metrics])
        
        # Calculate deltas for each method
        for method_name in method_names[1:]:  # Skip Direct
            diversity_deltas = []
            quality_deltas = []
            
            for model_name, model_results in results.items():
                method_data = model_results.get(method_name)
                direct_data = model_results.get("Direct")
                
                if (method_data and method_data["quality"] is not None and 
                    direct_data and direct_data["quality"] is not None):
                    
                    div_delta = method_data["diversity"] - direct_data["diversity"]
                    qual_delta = method_data["quality"] - direct_data["quality"]
                    
                    diversity_deltas.append(div_delta)
                    quality_deltas.append(qual_delta)
            
            if diversity_deltas and quality_deltas:
                size_method_deltas[size_category][method_name] = {
                    'diversity_delta_mean': np.mean(diversity_deltas),
                    'diversity_delta_std': np.std(diversity_deltas),
                    'quality_delta_mean': np.mean(quality_deltas),
                    'quality_delta_std': np.std(quality_deltas),
                    'n_models': len(diversity_deltas),
                    'complexity': cognitive_complexity[method_name]
                }
    
    # Create focused cognitive burden analysis plots - simplified to 2 key plots
    # Larger figure size for better LaTeX rendering
    fig = plt.figure(figsize=(18, 6))
    
    # Create a 1x2 grid for focused analysis
    gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Diversity Impact by Model Size - Left plot 
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Methods to show (excluding Direct since we're showing deltas)
    methods_subset = ["Sequence", "Multi-turn", "VS-Standard", "VS-CoT", "VS-Multi"]
    x_methods = np.arange(len(methods_subset))
    width = 0.35
    
    large_diversity_changes = []
    small_diversity_changes = []
    large_div_errors = []
    small_div_errors = []
    
    for method in methods_subset:
        large_val = size_method_deltas.get('large', {}).get(method, {}).get('diversity_delta_mean', 0)
        small_val = size_method_deltas.get('small', {}).get(method, {}).get('diversity_delta_mean', 0)
        large_err = size_method_deltas.get('large', {}).get(method, {}).get('diversity_delta_std', 0)
        small_err = size_method_deltas.get('small', {}).get(method, {}).get('diversity_delta_std', 0)
        
        large_diversity_changes.append(large_val)
        small_diversity_changes.append(small_val)
        large_div_errors.append(large_err)
        small_div_errors.append(small_err)
    
    bars1 = ax1.bar(x_methods - width/2, small_diversity_changes, width, 
                   yerr=small_div_errors, capsize=3, label='Small Models (GPT-4.1-Mini, Gemini-2.5-Flash)', 
                   color='#A23B72', alpha=0.8)
    bars2 = ax1.bar(x_methods + width/2, large_diversity_changes, width, 
                   yerr=large_div_errors, capsize=3, label='Large Models (GPT-4.1, Gemini-2.5-Pro)', 
                   color='#2E86AB', alpha=0.8)
    
    # Keep consistent colors from legend regardless of positive/negative values
    
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    # ax1.set_xlabel('Methods', fontweight='bold', fontsize=18)
    ax1.set_ylabel('Diversity Change vs Direct (Δ)', fontweight='bold', fontsize=18)
    ax1.set_title('Diversity Impact by Model Size', fontweight='bold', fontsize=20)
    ax1.set_xticks(x_methods)
    ax1.set_xticklabels(methods_subset, fontsize=14)  # Larger for LaTeX readability
    ax1.tick_params(axis='y', labelsize=14)  # Make y-axis labels larger
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Quality Impact by Model Size - Right plot
    ax2 = fig.add_subplot(gs[0, 1])
    
    large_quality_changes = []
    small_quality_changes = []
    large_errors = []
    small_errors = []
    
    for method in methods_subset:
        large_val = size_method_deltas.get('large', {}).get(method, {}).get('quality_delta_mean', 0)
        small_val = size_method_deltas.get('small', {}).get(method, {}).get('quality_delta_mean', 0)
        large_err = size_method_deltas.get('large', {}).get(method, {}).get('quality_delta_std', 0)
        small_err = size_method_deltas.get('small', {}).get(method, {}).get('quality_delta_std', 0)
        
        large_quality_changes.append(large_val)
        small_quality_changes.append(small_val)
        large_errors.append(large_err)
        small_errors.append(small_err)
    
    bars3 = ax2.bar(x_methods - width/2, small_quality_changes, width, 
                   yerr=small_errors, capsize=3, label='Small Models', 
                   color='#A23B72', alpha=0.8)
    bars4 = ax2.bar(x_methods + width/2, large_quality_changes, width, 
                   yerr=large_errors, capsize=3, label='Large Models', 
                   color='#2E86AB', alpha=0.8)
    
    # Keep consistent colors from legend regardless of positive/negative values
    
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    # ax2.set_xlabel('Methods', fontweight='bold', fontsize=18)
    ax2.set_ylabel('Quality Change vs Direct (Δ)', fontweight='bold', fontsize=18)
    ax2.set_title('Quality Impact by Model Size', fontweight='bold', fontsize=20)
    ax2.set_xticks(x_methods)
    ax2.set_xticklabels(methods_subset, fontsize=14)  # Larger for LaTeX readability
    ax2.tick_params(axis='y', labelsize=14)  # Make y-axis labels larger
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add a single unified legend at the top middle
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.93), 
               ncol=2, fontsize=16, frameon=True, fancybox=False, shadow=False)  # Larger legend font
    
    # Set overall title and save the plots  
    # plt.suptitle('Cognitive Burden Analysis: Quality & Diversity Impact by Model Size', 
    #             fontsize=18, fontweight='bold', y=0.88)  # Lower title to make room for legend
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.78)  # Make room for title and legend
    
    plt.savefig(f'{ablation_output_dir}/cognitive_burden_analysis.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f'{ablation_output_dir}/cognitive_burden_analysis.pdf', 
               bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✓ Saved cognitive burden analysis plots")
    print("📊 Focus methods analyzed: Sequence, Multi-turn, VS-Standard, VS-CoT, VS-Multi")
    return size_method_deltas

def plot_method_effectiveness_by_size(results_by_size, output_dir="latex_figures"):
    """Compare method effectiveness between model sizes"""
    
    # Create ablation-specific subdirectory
    ablation_output_dir = os.path.join(output_dir, "ablation", "model_size")
    os.makedirs(ablation_output_dir, exist_ok=True)
    
    method_names = ["Direct", "CoT", "Sequence", "Multi-turn", 
                   "VS-Standard", "VS-CoT", "VS-Multi"]
    
    # Calculate average improvements for each method and size
    size_method_stats = {}
    
    for size_category, results in results_by_size.items():
        size_method_stats[size_category] = {}
        
        for method in method_names:
            diversity_values = []
            quality_values = []
            
            for model_name, model_results in results.items():
                data = model_results.get(method)
                if data and data["diversity"] is not None and data["quality"] is not None:
                    diversity_values.append(data["diversity"])
                    quality_values.append(data["quality"])
            
            if diversity_values:
                size_method_stats[size_category][method] = {
                    'diversity_mean': np.mean(diversity_values),
                    'diversity_std': np.std(diversity_values),
                    'quality_mean': np.mean(quality_values),
                    'quality_std': np.std(quality_values),
                    'n_models': len(diversity_values)
                }
    
    # Create comparison plots - now 2x3 layout to accommodate quality improvements
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 12))
    
    # Plot 1: Diversity comparison
    methods = list(size_method_stats['large'].keys())
    large_div = [size_method_stats['large'][m]['diversity_mean'] for m in methods]
    small_div = [size_method_stats['small'][m]['diversity_mean'] if m in size_method_stats['small'] else 0 for m in methods]
    large_div_std = [size_method_stats['large'][m]['diversity_std'] for m in methods]
    small_div_std = [size_method_stats['small'][m]['diversity_std'] if m in size_method_stats['small'] else 0 for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, large_div, width, yerr=large_div_std, 
                   label='Large Models', color='#2E86AB', alpha=0.8, capsize=3)
    bars2 = ax1.bar(x + width/2, small_div, width, yerr=small_div_std,
                   label='Small Models', color='#A23B72', alpha=0.8, capsize=3)
    
    ax1.set_xlabel('Methods', fontweight='bold')
    ax1.set_ylabel('Average Diversity (%)', fontweight='bold')
    ax1.set_title('Diversity by Model Size', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Quality comparison
    large_qual = [size_method_stats['large'][m]['quality_mean'] for m in methods]
    small_qual = [size_method_stats['small'][m]['quality_mean'] if m in size_method_stats['small'] else 0 for m in methods]
    large_qual_std = [size_method_stats['large'][m]['quality_std'] for m in methods]
    small_qual_std = [size_method_stats['small'][m]['quality_std'] if m in size_method_stats['small'] else 0 for m in methods]
    
    bars3 = ax2.bar(x - width/2, large_qual, width, yerr=large_qual_std,
                   label='Large Models', color='#2E86AB', alpha=0.8, capsize=3)
    bars4 = ax2.bar(x + width/2, small_qual, width, yerr=small_qual_std,
                   label='Small Models', color='#A23B72', alpha=0.8, capsize=3)
    
    ax2.set_xlabel('Methods', fontweight='bold')
    ax2.set_ylabel('Average Quality (%)', fontweight='bold')
    ax2.set_title('Quality by Model Size', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Quality improvement over Direct method
    direct_large_qual = size_method_stats['large']['Direct']['quality_mean']
    direct_small_qual = size_method_stats['small']['Direct']['quality_mean'] if 'Direct' in size_method_stats['small'] else 0
    
    large_qual_improvements = [(size_method_stats['large'][m]['quality_mean'] - direct_large_qual) / direct_large_qual * 100 
                              for m in methods[1:]]  # Skip Direct
    small_qual_improvements = [(size_method_stats['small'][m]['quality_mean'] - direct_small_qual) / direct_small_qual * 100 
                              if m in size_method_stats['small'] and direct_small_qual > 0 else 0 
                              for m in methods[1:]]
    
    x_imp = np.arange(len(methods[1:]))
    
    bars7 = ax3.bar(x_imp - width/2, large_qual_improvements, width,
                   label='Large Models', color='#2E86AB', alpha=0.8)
    bars8 = ax3.bar(x_imp + width/2, small_qual_improvements, width,
                   label='Small Models', color='#A23B72', alpha=0.8)
    
    ax3.set_xlabel('Methods', fontweight='bold')
    ax3.set_ylabel('Quality Improvement over Direct (%)', fontweight='bold')
    ax3.set_title('Method Effectiveness: Quality Gains', fontweight='bold')
    ax3.set_xticks(x_imp)
    ax3.set_xticklabels(methods[1:], rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Plot 4: Diversity improvement over Direct method
    direct_large_div = size_method_stats['large']['Direct']['diversity_mean']
    direct_small_div = size_method_stats['small']['Direct']['diversity_mean'] if 'Direct' in size_method_stats['small'] else 0
    
    large_div_improvements = [(size_method_stats['large'][m]['diversity_mean'] - direct_large_div) / direct_large_div * 100 
                              for m in methods[1:]]  # Skip Direct
    small_div_improvements = [(size_method_stats['small'][m]['diversity_mean'] - direct_small_div) / direct_small_div * 100 
                             if m in size_method_stats['small'] and direct_small_div > 0 else 0 
                             for m in methods[1:]]
    
    x_imp = np.arange(len(methods[1:]))
    
    bars5 = ax4.bar(x_imp - width/2, large_div_improvements, width,
                   label='Large Models', color='#2E86AB', alpha=0.8)
    bars6 = ax4.bar(x_imp + width/2, small_div_improvements, width,
                   label='Small Models', color='#A23B72', alpha=0.8)
    
    ax4.set_xlabel('Methods', fontweight='bold')
    ax4.set_ylabel('Diversity Improvement over Direct (%)', fontweight='bold')
    ax4.set_title('Method Effectiveness: Diversity Gains', fontweight='bold')
    ax4.set_xticks(x_imp)
    ax4.set_xticklabels(methods[1:], rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Plot 5: Model size classifications and statistical significance
    # Add model classifications and statistical tests
    large_models = list(results_by_size['large'].keys())
    small_models = list(results_by_size['small'].keys())
    
    # Create model classification text
    classification_text = "MODEL SIZE CLASSIFICATIONS:\n\n"
    classification_text += "LARGE MODELS:\n"
    for model in large_models:
        classification_text += f"• {model}\n"
    classification_text += f"\nTotal: {len(large_models)} models\n\n"
    
    classification_text += "SMALL MODELS:\n"
    for model in small_models:
        classification_text += f"• {model}\n"
    classification_text += f"\nTotal: {len(small_models)} models"
    
    ax5.text(0.05, 0.95, classification_text, transform=ax5.transAxes, fontsize=11,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
             verticalalignment='top', fontfamily='monospace')
    ax5.set_title('Model Size Classifications', fontweight='bold')
    ax5.axis('off')
    
    # Plot 6: Statistical significance tests for VS methods
    vs_methods = ["VS-Standard", "VS-CoT", "VS-Multi"]
    p_values_div = []
    p_values_qual = []
    
    for vs_method in vs_methods:
        if vs_method in size_method_stats['large'] and vs_method in size_method_stats['small']:
            # Get individual data points for statistical test
            large_div_data = []
            small_div_data = []
            large_qual_data = []
            small_qual_data = []
            
            for model_name, model_results in results_by_size['large'].items():
                data = model_results.get(vs_method)
                if data and data["diversity"] is not None:
                    large_div_data.append(data["diversity"])
                    large_qual_data.append(data["quality"])
            
            for model_name, model_results in results_by_size['small'].items():
                data = model_results.get(vs_method)
                if data and data["diversity"] is not None:
                    small_div_data.append(data["diversity"])
                    small_qual_data.append(data["quality"])
            
            if len(large_div_data) > 1 and len(small_div_data) > 1:
                _, p_div = stats.ttest_ind(large_div_data, small_div_data)
                _, p_qual = stats.ttest_ind(large_qual_data, small_qual_data)
                p_values_div.append(p_div)
                p_values_qual.append(p_qual)
            else:
                p_values_div.append(1.0)
                p_values_qual.append(1.0)
    
    # Display significance results
    sig_text = "STATISTICAL SIGNIFICANCE\n(Large vs Small Models):\n\n"
    sig_text += "Method          | Diversity | Quality\n"
    sig_text += "----------------|-----------|--------\n"
    for i, method in enumerate(vs_methods):
        if i < len(p_values_div):
            div_sig = "***" if p_values_div[i] < 0.001 else "**" if p_values_div[i] < 0.01 else "*" if p_values_div[i] < 0.05 else "ns"
            qual_sig = "***" if p_values_qual[i] < 0.001 else "**" if p_values_qual[i] < 0.01 else "*" if p_values_qual[i] < 0.05 else "ns"
            method_short = method.replace("VS-", "")
            sig_text += f"{method_short:<15} | {div_sig:>9} | {qual_sig:>7}\n"
    
    sig_text += "\nLegend: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant"
    
    ax6.text(0.05, 0.95, sig_text, transform=ax6.transAxes, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
             verticalalignment='top', fontfamily='monospace')
    ax6.set_title('Statistical Significance Tests', fontweight='bold')
    ax6.axis('off')
    
    plt.suptitle('Method Effectiveness Analysis by Model Size', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    plt.savefig(f'{ablation_output_dir}/method_effectiveness_by_size.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f'{ablation_output_dir}/method_effectiveness_by_size.pdf', 
               bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✓ Saved method effectiveness comparison plots")
    return size_method_stats

def generate_summary_statistics(results_by_size, pareto_stats, method_stats):
    """Generate comprehensive summary statistics"""
    
    print("\n" + "="*80)
    print("MODEL SIZE ABLATION SUMMARY")
    print("="*80)
    
    # Overall model count
    print(f"\nMODEL COUNTS:")
    print(f"Large models: {len(results_by_size['large'])}")
    print(f"Small models: {len(results_by_size['small'])}")
    
    # Pareto efficiency comparison
    if pareto_stats:
        print(f"\nPARETO EFFICIENCY:")
        for size, stats in pareto_stats.items():
            area = stats['area']
            n_points = stats['n_pareto_points']
            print(f"{size.title()} models: Area under curve = {area:.1f}, Pareto points = {n_points}")
        
        if 'large' in pareto_stats and 'small' in pareto_stats:
            efficiency_ratio = pareto_stats['large']['area'] / pareto_stats['small']['area']
            print(f"Large models are {efficiency_ratio:.2f}x more Pareto efficient")
    
    # Method effectiveness comparison
    if method_stats:
        print(f"\nMETHOD EFFECTIVENESS COMPARISON:")
        vs_methods = ["VS-Standard", "VS-CoT", "VS-Multi"]
        
        for vs_method in vs_methods:
            if vs_method in method_stats['large'] and vs_method in method_stats['small']:
                large_div = method_stats['large'][vs_method]['diversity_mean']
                small_div = method_stats['small'][vs_method]['diversity_mean']
                large_qual = method_stats['large'][vs_method]['quality_mean']
                small_qual = method_stats['small'][vs_method]['quality_mean']
                
                div_improvement = (large_div - small_div) / small_div * 100 if small_div > 0 else 0
                qual_improvement = (large_qual - small_qual) / small_qual * 100 if small_qual > 0 else 0
                
                print(f"{vs_method}:")
                print(f"  Diversity: Large {large_div:.1f}% vs Small {small_div:.1f}% ({div_improvement:+.1f}%)")
                print(f"  Quality: Large {large_qual:.1f}% vs Small {small_qual:.1f}% ({qual_improvement:+.1f}%)")
    
    # Best methods by size
    print(f"\nBEST METHODS BY SIZE:")
    for size_category, results in results_by_size.items():
        method_scores = {}
        
        for method_name in ["Direct", "CoT", "Sequence", "Multi-turn", "VS-Standard", "VS-CoT", "VS-Multi"]:
            diversity_values = []
            quality_values = []
            
            for model_name, model_results in results.items():
                data = model_results.get(method_name)
                if data and data["diversity"] is not None and data["quality"] is not None:
                    diversity_values.append(data["diversity"])
                    quality_values.append(data["quality"])
            
            if diversity_values:
                # Combined score: diversity + quality (both higher is better)
                combined_score = np.mean(diversity_values) + np.mean(quality_values)
                method_scores[method_name] = combined_score
        
        if method_scores:
            best_method = max(method_scores.keys(), key=lambda k: method_scores[k])
            print(f"{size_category.title()} models: Best method is {best_method} (score: {method_scores[best_method]:.1f})")

def main():
    parser = argparse.ArgumentParser(description='Model size ablation study for poem generation')
    parser.add_argument('--output-dir', default='latex_figures', 
                        help='Output directory for plots (default: latex_figures)')
    
    args = parser.parse_args()
    
    print("🔬 Starting Model Size Ablation Study...")
    
    # Load all results
    results_by_size = load_all_results()
    
    if not results_by_size['large'] and not results_by_size['small']:
        print("❌ No model results found. Check data directory structure.")
        return
    
    print(f"\n📊 Generating visualizations...")
    
    # Generate plots and analyses
    pareto_stats = plot_size_comparison_scatter(results_by_size, args.output_dir)
    method_stats = plot_method_effectiveness_by_size(results_by_size, args.output_dir)
    
    # NEW: Generate cognitive burden analysis with subfigures
    cognitive_stats = plot_cognitive_burden_analysis_subfigures(results_by_size, args.output_dir)
    
    # Also generate original combined plot
    plot_cognitive_burden_analysis(results_by_size, args.output_dir)
    
    # Generate summary statistics
    generate_summary_statistics(results_by_size, pareto_stats, method_stats)
    
    print(f"\n🎉 Model size ablation study complete!")
    print(f"📁 Results saved to: {args.output_dir}/ablation/model_size/")
    print("📊 Directory structure:")
    print("  - latex_figures/ablation/model_size/")
    print("📋 Generated files:")
    print("  - model_size_diversity_quality_comparison.png/pdf")
    print("  - method_effectiveness_by_size.png/pdf")
    print("  - cognitive_burden_analysis.png/pdf (Combined figure)")
    print("  - diversity_no_legend.pdf/png (Subfigure a)")
    print("  - quality_no_legend.pdf/png (Subfigure b)")
    print("  - legend_only.pdf/png (Separate legend)")
    print("  - combined_figure_with_legend.pdf/png (GridSpec layout)")

if __name__ == "__main__":
    main()