#!/usr/bin/env python3

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from scipy import stats
from scipy.spatial import ConvexHull

def load_metric(model_dir, method, metric_file, metric_key):
    """Load a specific metric from a results file"""
    file_path = os.path.join(model_dir, "evaluation", method, metric_file)
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data.get('overall_metrics', {}).get(metric_key, None)
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
        print(f"Processing {method_name} with directory {method_dir}")
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
    method_names = ["Direct", "CoT", "Sequence", "Multi-turn", "VS-Standard", "VS-CoT", "VS-Multi"]
    
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

def plot_model_comparison(all_results, output_dir="plots"):
    """Create bar charts comparing models across different metrics"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    method_names = ["Direct", "CoT", "Sequence", "Multi-turn", "VS-Standard", "VS-CoT", "VS-Multi"]
    
    # Filter out models with insufficient data
    valid_models = {}
    for model_name, results in all_results.items():
        if all(results.get(method) and 
               all(results[method][metric] is not None for metric in ['diversity', 'rouge_l', 'quality'])
               for method in method_names):
            valid_models[model_name] = results
    
    metrics = [
        ('diversity', 'Diversity (%)', 'Higher is Better'),
        ('rouge_l', 'Rouge-L (%)', 'Lower is Better'),
        ('quality', 'Quality Score (%)', 'Higher is Better')
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
        ax.set_title(f'{metric_title} Comparison Across Models\n({direction})', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x + width * 2.5)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/model_comparison_{metric_key}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_dir}/model_comparison_{metric_key}.pdf', bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved {metric_title} model comparison plot")

def plot_individual_models(all_results, output_dir="plots"):
    """Create 2x1 subplot for each model showing diversity and quality bar charts"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plotting style with muted colors
    sns.set_style("whitegrid")
    # Use muted gray/blue palette
    base_colors = ['#4472C4', '#70AD47', '#FFC000', '#C5504B', '#264478', '#375623', '#7F6000']
    method_names = ["Direct", "CoT", "Sequence", "Multi-turn", "VS-Standard", "VS-CoT", "VS-Multi"]
    
    for model_name, results in all_results.items():
        # Filter methods with complete data
        valid_methods = []
        diversity_values = []
        diversity_errors = []
        quality_values = []
        quality_errors = []
        colors = []
        hatches = []
        
        for method in method_names:
            data = results.get(method)
            if data and data["diversity"] is not None and data["quality"] is not None:
                valid_methods.append(method)
                diversity_values.append(data["diversity"])
                diversity_errors.append(data["diversity_std"] if data["diversity_std"] is not None else 0)
                quality_values.append(data["quality"])
                quality_errors.append(data["quality_std"] if data["quality_std"] is not None else 0)
                
                # Set colors and hatches
                method_idx = method_names.index(method)
                colors.append(base_colors[method_idx])
                # Add hatch for VS methods
                if method.startswith("VS-"):
                    hatches.append('///')
                else:
                    hatches.append(None)
        
        if len(valid_methods) < 2:  # Skip if insufficient data
            continue
        
        # Create 2x1 subplot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Top subplot: Diversity
        bars1 = ax1.bar(valid_methods, diversity_values, color=colors, hatch=hatches, 
                        edgecolor='black', linewidth=0.8, alpha=0.8)
        ax1.errorbar(range(len(valid_methods)), diversity_values, yerr=diversity_errors, 
                     fmt='none', color='black', capsize=3, capthick=1)
        ax1.set_title(f'{model_name} - Diversity', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Diversity (%)', fontsize=12)
        ax1.tick_params(axis='x', rotation=0)
        
        # Bottom subplot: Quality
        bars2 = ax2.bar(valid_methods, quality_values, color=colors, hatch=hatches,
                        edgecolor='black', linewidth=0.8, alpha=0.8)
        ax2.errorbar(range(len(valid_methods)), quality_values, yerr=quality_errors, 
                     fmt='none', color='black', capsize=3, capthick=1)
        ax2.set_title(f'{model_name} - Quality', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Quality (%)', fontsize=12)
        ax2.set_xlabel('Methods', fontsize=12)
        ax2.tick_params(axis='x', rotation=0)
        
        plt.tight_layout()
        safe_model_name = model_name.replace(' ', '_').replace('.', '_')
        plt.savefig(f'{output_dir}/individual_{safe_model_name}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_dir}/individual_{safe_model_name}.pdf', bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved individual model plot for {model_name}")

# def plot_diversity_vs_quality_scatter(all_results, output_dir="plots"):
#     """Create scatter plot with diversity vs quality for each model separately"""
    
#     # Create output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Set up the plotting style
#     sns.set_style("whitegrid")
#     method_names = ["Direct", "CoT", "Sequence", "Multi-turn", "VS-Standard", "VS-CoT", "VS-Combined"]
#     # Use consistent markers with same colors
#     base_colors = ['#4472C4', '#70AD47', '#FFC000', '#C5504B', '#264478', '#375623', '#7F6000']
#     markers = ['o', 'o', 's', 's', '^', '^', 'X']  # Circle, circle, square, square, triangle, triangle, cross
    
#     # Create separate plot for each model
#     for model_name, results in all_results.items():
#         fig, ax = plt.subplots(figsize=(10, 8))
        
#         for i, method in enumerate(method_names):
#             data = results.get(method)
#             if data and data["diversity"] is not None and data["quality"] is not None:
#                 div = data["diversity"]
#                 qual = data["quality"]
#                 div_std = data["diversity_std"] if data["diversity_std"] is not None else 0
#                 qual_std = data["quality_std"] if data["quality_std"] is not None else 0
                
#                 # Plot the main point with consistent marker for each method
#                 marker_style = markers[i]
#                 marker_size = 300
#                 # Use filled markers for VS methods, unfilled for others
#                 if method.startswith("VS-"):
#                     ax.scatter(div, qual, color=base_colors[i], marker=marker_style, 
#                               s=marker_size, alpha=0.9, label=method, zorder=5, 
#                               edgecolors='black', linewidth=2)
#                 else:
#                     ax.scatter(div, qual, facecolors='none', edgecolors=base_colors[i], 
#                               marker=marker_style, s=marker_size, alpha=0.9, label=method, 
#                               zorder=5, linewidth=2)
        
#         # Add arrows to indicate direction of improvement
#         ax.annotate('', xy=(1, 0), xytext=(0, 0),
#                    xycoords='axes fraction', textcoords='axes fraction',
#                    arrowprops=dict(arrowstyle='->', lw=2, color='red'),
#                    annotation_clip=False)
#         ax.text(0.5, -0.08, 'Higher Diversity →', ha='center', va='top',
#                 transform=ax.transAxes, fontsize=10, color='red', fontweight='bold')
        
#         ax.annotate('', xy=(0, 1), xytext=(0, 0),
#                    xycoords='axes fraction', textcoords='axes fraction',
#                    arrowprops=dict(arrowstyle='->', lw=2, color='red'),
#                    annotation_clip=False)
#         ax.text(-0.12, 0.5, 'Higher\nQuality\n↑', ha='center', va='center',
#                 transform=ax.transAxes, fontsize=10, color='red', fontweight='bold',
#                 rotation=90)
        
#         ax.set_xlabel('Diversity (%)', fontsize=12, fontweight='bold')
#         ax.set_ylabel('Quality (%)', fontsize=12, fontweight='bold')
#         # ax.set_title(f'{model_name} - Diversity vs Quality', 
#         #             fontsize=14, fontweight='bold')
#         ax.legend(bbox_to_anchor=(0.5, 1.02), loc='lower center', ncol=len(method_names))
#         ax.grid(True, alpha=0.3)
        
#         plt.tight_layout()
#         safe_model_name = model_name.replace(' ', '_').replace('.', '_')
#         plt.savefig(f'{output_dir}/diversity_vs_quality_{safe_model_name}.png', dpi=300, bbox_inches='tight')
#         plt.savefig(f'{output_dir}/diversity_vs_quality_{safe_model_name}.pdf', bbox_inches='tight')
#         plt.close()
        
#         print(f"✓ Saved diversity vs quality scatter plot for {model_name}")

def plot_diversity_vs_quality_scatter_academic(all_results, output_dir="plots"):
    """Create publication-quality scatter plot with diversity vs quality"""
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plotting style for academic papers
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 11,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'lines.linewidth': 1.5,
        'lines.markersize': 8,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.8
    })
    
    # Method names and styling
    method_names = ["Direct", "CoT", "Sequence", "Multi-turn", 
                "VS-Standard", "VS-CoT", "VS-Multi"]
    
    # Academic color palette (colorblind-friendly)
    base_colors = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', 
                   '#56B4E9', '#F0E442', '#D55E00']
    
    # Distinct markers for each method
    markers = ['o', 's', 'D', '^', 'v', 'p', '*']
    marker_sizes = [120, 120, 100, 120, 120, 140, 180]
    
    # Create separate plot for each model
    for model_name, results in all_results.items():
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot each method
        for i, method in enumerate(method_names):
            data = results.get(method)
            if data and data["diversity"] is not None and data["quality"] is not None:
                div = data["diversity"]
                qual = data["quality"]
                
                # Plot the point
                if method.startswith("VS-"):
                    # Filled markers for VS methods
                    ax.scatter(div, qual, 
                             color=base_colors[i], 
                             marker=markers[i],
                             s=marker_sizes[i], 
                             alpha=0.85, 
                             label=method, 
                             zorder=5,
                             edgecolors='black', 
                             linewidth=1.0)
                else:
                    # Unfilled markers for baseline methods
                    ax.scatter(div, qual, 
                             facecolors='none', 
                             edgecolors=base_colors[i],
                             marker=markers[i], 
                             s=marker_sizes[i], 
                             alpha=0.9, 
                             label=method,
                             zorder=5, 
                             linewidth=2.0)
        
        # Add subtle directional indicators in the corners
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
        plt.savefig(f'{output_dir}/diversity_vs_quality_{safe_model_name}_academic.png', 
                   dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.savefig(f'{output_dir}/diversity_vs_quality_{safe_model_name}_academic.pdf', 
                   bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.savefig(f'{output_dir}/diversity_vs_quality_{safe_model_name}_academic.eps', 
                   format='eps', bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✓ Saved academic-quality diversity vs quality plot for {model_name}")


# Alternative minimalist version for space-constrained publications
def plot_diversity_vs_quality_scatter(all_results, output_dir="plots"):
    """Create minimalist scatter plot for space-constrained academic papers"""
    
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Minimal style settings
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 9,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
    })
    
    method_names = ["Direct", "CoT", "Sequence", "Multi-turn", 
                   "VS-Standard", "VS-CoT", "VS-Multi"]
    
    # Grayscale-friendly colors
    colors = ['#000000', '#333333', '#666666', '#999999', 
              '#0066CC', '#CC0000', '#00CC66']
    markers = ['o', 's', 'D', '^', 'o', 's', 'D']
    
    for model_name, results in all_results.items():
        fig, ax = plt.subplots(figsize=(3.5, 3))
        
        # Plot data points
        for i, method in enumerate(method_names):
            data = results.get(method)
            if data and data["diversity"] is not None and data["quality"] is not None:
                if method.startswith("VS-"):
                    ax.scatter(data["diversity"], data["quality"],
                             color=colors[i], marker=markers[i], s=60,
                             alpha=0.9, label=method, zorder=5)
                else:
                    ax.scatter(data["diversity"], data["quality"],
                             facecolors='none', edgecolors=colors[i],
                             marker=markers[i], s=60, alpha=0.9,
                             label=method, zorder=5, linewidth=1.5)
        
        ax.set_xlabel('Diversity (%)')
        ax.set_ylabel('Quality (%)')
        ax.grid(True, alpha=0.15, linestyle=':')
        
        # Compact legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                 frameon=False, borderaxespad=0)
        
        plt.tight_layout()
        safe_model_name = model_name.replace(' ', '_').replace('.', '_')
        plt.savefig(f'{output_dir}/diversity_quality_{safe_model_name}_minimal.pdf',
                   bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"✓ Saved minimal diversity vs quality plot for {model_name}")

def plot_diversity_vs_quality_average(all_results, output_dir="plots"):
    """Create scatter plot showing average diversity vs quality across all models for each method"""
    
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plotting style (exactly like 2x2)
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 12,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'axes.linewidth': 1.0,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.8
    })
    
    method_names = ["Direct", "CoT", "Sequence", "Multi-turn", "VS-Standard", "VS-CoT", "VS-Multi"]
    # Use colorful palette for all methods (exactly like 2x2)
    base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                   '#9467bd', '#8c564b', '#e377c2']
    markers = ['x', 'x', 'x', 'x', '*', '*', '*']
    
    # Calculate averages across all models for each method
    method_averages = {}
    
    for method in method_names:
        diversity_values = []
        quality_values = []
        
        # Collect values from all models
        for model_name, results in all_results.items():
            data = results.get(method)
            if data and data["diversity"] is not None and data["quality"] is not None:
                diversity_values.append(data["diversity"])
                quality_values.append(data["quality"])
        
        if diversity_values and quality_values:
            method_averages[method] = {
                'diversity': np.mean(diversity_values),
                'quality': np.mean(quality_values)
            }
    
    # Create single subplot (mimic 2x2 structure but single plot)
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # First, collect all points to find Pareto optimal ones (exactly like 2x2)
    all_points = []
    for i, method in enumerate(method_names):
        if method in method_averages:
            avg_data = method_averages[method]
            all_points.append((avg_data['diversity'], avg_data['quality'], method, i))
    
    # Find Pareto optimal points
    pareto_mask = find_pareto_frontier([p[0] for p in all_points], [p[1] for p in all_points])
    
    # Plot each method (exactly like 2x2 implementation)
    for i, method in enumerate(method_names):
        if method in method_averages:
            avg_data = method_averages[method]
            div = avg_data['diversity']
            qual = avg_data['quality']
            
            # Check if this point is Pareto optimal
            point_idx = next((j for j, p in enumerate(all_points) if p[2] == method), -1)
            is_pareto_optimal = point_idx >= 0 and pareto_mask[point_idx]
            
            # Plot the main point with consistent marker for each method
            marker_style = markers[i]
            base_size = 120
            base_linewidth = 1.5
            
            # Make Pareto optimal points larger and bolder
            if is_pareto_optimal:
                size = base_size * 1.5
                linewidth = base_linewidth
                edge_color = 'black'
                edge_width = 1.5
            else:
                size = base_size
                linewidth = base_linewidth
                edge_color = None
                edge_width = 0
            
            # Use different markers and hatches for VS methods vs others (exactly like 2x2)
            if method.startswith("VS-"):
                # VS methods: hatched star markers
                ax.scatter(div, qual, color=base_colors[i], marker=marker_style, 
                          s=size, alpha=0.9, label=method, zorder=5, 
                          linewidth=linewidth, hatch='///')
            else:
                # Baseline methods: solid x markers
                ax.scatter(div, qual, color=base_colors[i], marker=marker_style, 
                          s=size, alpha=0.7, label=method, zorder=5, 
                          linewidth=linewidth)
            
            # Add bold outline for Pareto optimal points
            if is_pareto_optimal:
                ax.scatter(div, qual, facecolors='none', edgecolors=edge_color, 
                          marker=marker_style, s=size + 50, linewidth=edge_width, 
                          alpha=1.0, zorder=6)
    
    # Configure subplot (exactly like 2x2)
    ax.set_xlabel('Diversity (%)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Quality (%)', fontsize=16, fontweight='bold')
    # ax.set_title('Average Diversity vs Quality Across All Models', fontsize=18, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add Pareto optimal arrow with independent text positioning (exactly like 2x2)
    # First add the arrow
    ax.annotate('', xy=(0.95, 0.95), xytext=(0.85, 0.85),
               xycoords='axes fraction', textcoords='axes fraction',
               arrowprops=dict(arrowstyle='->', lw=3, color='red', alpha=0.7))
    # Then add the text separately
    ax.text(0.85, 0.88, 'Pareto optimal', transform=ax.transAxes, 
           fontsize=12, color='red', fontweight='bold', ha='right')
    
    # Add more headroom to axes (exactly like 2x2)
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_range = x_max - x_min
    y_range = y_max - y_min
    ax.set_xlim(x_min - 0.05 * x_range, x_max + 0.20 * x_range)
    ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.20 * y_range)
    
    # Add legend above the plot (matching 2x2 style)
    handles, labels = ax.get_legend_handles_labels()
    if handles:  # Only add legend if we have data
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.00), 
                  ncol=4, fontsize=12, frameon=False, 
                  borderaxespad=0)
    
    # Adjust layout to make room for legend (exactly like 2x2)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    # Save the figure
    plt.savefig(f'{output_dir}/diversity_vs_quality_average.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/diversity_vs_quality_average.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved average diversity vs quality plot")
    
    return method_averages

def plot_diversity_vs_quality_2x2(all_results, output_dir="plots"):
    """Create 2x2 subplot layout with Claude-4, Gemini-2.5-Pro, GPT-4.1, and Deepseek-R1"""
    
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 12,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'axes.linewidth': 1.0,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.8
    })
    
    method_names = ["Direct", "CoT", "Sequence", "Multi-turn", "VS-Standard", "VS-CoT", "VS-Combined"]
    # Use colorful palette for all methods
    base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                   '#9467bd', '#8c564b', '#e377c2']
    markers = ['x', 'x', 'x', 'x', '*', '*', '*']
    
    # Define the 4 models we want to plot
    target_models = ["Claude-4-Sonnet", "Gemini-2.5-Pro", "GPT-4.1", "Deepseek-R1"]
    
    # Create 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    axes = axes.flatten()
    
    # Plot each model
    plotted_models = []
    for idx, model_name in enumerate(target_models):
        if idx >= 4:  # Safety check
            break
            
        ax = axes[idx]
        
        # Find the model in results (handle potential name variations)
        model_results = None
        for key in all_results.keys():
            if model_name.lower() in key.lower() or key.lower() in model_name.lower():
                model_results = all_results[key]
                plotted_models.append(key)
                break
        
        if model_results is None:
            # If exact match not found, try to find similar names
            for key in all_results.keys():
                if any(part in key.lower() for part in model_name.lower().split('-')):
                    model_results = all_results[key]
                    plotted_models.append(key)
                    break
        
        if model_results is not None:
            # First, collect all points to find Pareto optimal ones
            all_points = []
            for i, method in enumerate(method_names):
                data = model_results.get(method)
                if data and data["diversity"] is not None and data["quality"] is not None:
                    all_points.append((data["diversity"], data["quality"], method, i))
            
            # Find Pareto optimal points
            pareto_mask = find_pareto_frontier([p[0] for p in all_points], [p[1] for p in all_points])
            
            # Plot each method for this model
            for i, method in enumerate(method_names):
                data = model_results.get(method)
                if data and data["diversity"] is not None and data["quality"] is not None:
                    div = data["diversity"]
                    qual = data["quality"]
                    
                    # Check if this point is Pareto optimal
                    point_idx = next((j for j, p in enumerate(all_points) if p[2] == method), -1)
                    is_pareto_optimal = point_idx >= 0 and pareto_mask[point_idx]
                    
                    # Plot the main point with consistent marker for each method
                    marker_style = markers[i]
                    base_size = 120
                    base_linewidth = 1.5
                    
                    # Make Pareto optimal points larger and bolder
                    if is_pareto_optimal:
                        size = base_size * 1.5
                        linewidth = base_linewidth
                        edge_color = 'black'
                        edge_width = 1.5
                    else:
                        size = base_size
                        linewidth = base_linewidth
                        edge_color = None
                        edge_width = 0
                    
                    # Use different markers and hatches for VS methods vs others
                    if method.startswith("VS-"):
                        # VS methods: hatched star markers
                        scatter = ax.scatter(div, qual, color=base_colors[i], marker=marker_style, 
                                  s=size, alpha=0.9, label=method if idx == 0 else "", zorder=5, 
                                  linewidth=linewidth, hatch='///')
                    else:
                        # Baseline methods: solid x markers
                        scatter = ax.scatter(div, qual, color=base_colors[i], marker=marker_style, 
                                  s=size, alpha=0.7, label=method if idx == 0 else "", zorder=5, 
                                  linewidth=linewidth)
                    
                    # Add bold outline for Pareto optimal points
                    if is_pareto_optimal:
                        ax.scatter(div, qual, facecolors='none', edgecolors=edge_color, 
                                  marker=marker_style, s=size + 50, linewidth=edge_width, 
                                  alpha=1.0, zorder=6)
            
            # Configure subplot
            ax.set_xlabel('Diversity (%)', fontsize=16, fontweight='bold')
            ax.set_ylabel('Quality (%)', fontsize=16, fontweight='bold')
            ax.set_title(f'{model_name}', fontsize=18, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add Pareto optimal arrow with independent text positioning
            # First add the arrow
            ax.annotate('', xy=(0.95, 0.95), xytext=(0.85, 0.85),
                       xycoords='axes fraction', textcoords='axes fraction',
                       arrowprops=dict(arrowstyle='->', lw=3, color='red', alpha=0.7))
            # Then add the text separately
            ax.text(0.85, 0.88, 'Pareto optimal', transform=ax.transAxes, 
                   fontsize=12, color='red', fontweight='bold', ha='right')
            
            # Add more headroom to axes
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            x_range = x_max - x_min
            y_range = y_max - y_min
            ax.set_xlim(x_min - 0.05 * x_range, x_max + 0.20 * x_range)
            ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.20 * y_range)
            
        else:
            # If model not found, show empty plot with message
            ax.text(0.5, 0.5, f'{model_name}\n(No data available)', 
                   ha='center', va='center', transform=ax.transAxes, 
                   fontsize=12, color='gray')
            ax.set_xlabel('Diversity (%)', fontsize=16, fontweight='bold')
            ax.set_ylabel('Quality (%)', fontsize=16, fontweight='bold')
            ax.set_title(f'{model_name}', fontsize=18, fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    # Add legend above the entire plot (matching original style)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:  # Only add legend if we have data
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.00), 
                  ncol=4, fontsize=12, frameon=False, 
                  borderaxespad=0)
    
    # Adjust layout to make room for legend
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    # Save the figure
    plt.savefig(f'{output_dir}/diversity_vs_quality_2x2.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/diversity_vs_quality_2x2.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved 2x2 diversity vs quality plot")
    print(f"  Models plotted: {plotted_models}")

def find_pareto_frontier(diversity_values, quality_values):
    """Find points on the Pareto frontier (maximize both diversity and quality)"""
    points = np.column_stack((diversity_values, quality_values))
    
    # Find Pareto optimal points
    pareto_mask = np.zeros(len(points), dtype=bool)
    
    for i, point in enumerate(points):
        # A point is Pareto optimal if no other point dominates it
        # (has both higher diversity AND higher quality)
        dominated = False
        for j, other_point in enumerate(points):
            if i != j:
                if (other_point[0] >= point[0] and other_point[1] >= point[1] and
                    (other_point[0] > point[0] or other_point[1] > point[1])):
                    dominated = True
                    break
        pareto_mask[i] = not dominated
    
    return pareto_mask

def plot_combined_pareto_scatter(all_results, output_dir="plots"):
    """Create VS-Multi (vs_multi) scatter plot showing Pareto frontier across all models"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plotting style
    sns.set_style("whitegrid")
    method_names = ["Direct", "CoT", "Sequence", "Multi-turn", "VS-Standard", "VS-CoT", "VS-Combined"]
    base_colors = ['#4472C4', '#70AD47', '#FFC000', '#C5504B', '#264478', '#375623', '#7F6000']
    markers = ['o', 'o', 's', 's', '^', '^', 'X']  # Circle, circle, square, square, triangle, triangle, cross
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Collect all points for Pareto frontier calculation
    all_diversity = []
    all_quality = []
    all_methods = []
    all_models = []
    
    # Plot each method across all models
    for i, method in enumerate(method_names):
        diversity_values = []
        quality_values = []
        model_labels = []
        
        # Collect data for this method across all models
        for model_name, results in all_results.items():
            data = results.get(method)
            if data and data["diversity"] is not None and data["quality"] is not None:
                div = data["diversity"]
                qual = data["quality"]
                diversity_values.append(div)
                quality_values.append(qual)
                model_labels.append(model_name)
                
                # Add to global collections for Pareto analysis
                all_diversity.append(div)
                all_quality.append(qual)
                all_methods.append(method)
                all_models.append(model_name)
        
        if diversity_values:  # Only plot if we have data
            # Different styling for VS methods vs baselines
            if method.startswith("VS-"):
                # VS methods: filled markers with border
                ax.scatter(diversity_values, quality_values, color=base_colors[i], 
                          marker=markers[i], s=150, alpha=0.9, label=method, 
                          zorder=5, edgecolors='black', linewidth=2)
            else:
                # Baseline methods: unfilled markers
                ax.scatter(diversity_values, quality_values, facecolors='none', 
                          edgecolors=base_colors[i], marker=markers[i], s=150, 
                          alpha=0.9, label=method, zorder=5, linewidth=2)
    
    # Find and highlight Pareto frontier
    if all_diversity and all_quality:
        pareto_mask = find_pareto_frontier(all_diversity, all_quality)
        pareto_diversity = np.array(all_diversity)[pareto_mask]
        pareto_quality = np.array(all_quality)[pareto_mask]
        pareto_methods = np.array(all_methods)[pareto_mask]
        
        # Sort Pareto points by diversity for proper line drawing
        sort_idx = np.argsort(pareto_diversity)
        pareto_diversity_sorted = pareto_diversity[sort_idx]
        pareto_quality_sorted = pareto_quality[sort_idx]
        pareto_methods_sorted = pareto_methods[sort_idx]
        
        # Draw Pareto frontier
        ax.plot(pareto_diversity_sorted, pareto_quality_sorted, 'r--', 
                linewidth=2, alpha=0.7, label='Pareto Frontier', zorder=3)
        
        # Highlight Pareto optimal points
        ax.scatter(pareto_diversity, pareto_quality, facecolors='none', 
                  edgecolors='red', s=200, linewidth=3, alpha=0.8, zorder=6)
        
        # Count VS methods on Pareto frontier
        vs_on_frontier = sum(1 for method in pareto_methods if method.startswith("VS-"))
        total_on_frontier = len(pareto_methods)
        
        # Add text annotation about Pareto dominance
        ax.text(0.02, 0.98, f'Pareto Frontier: {vs_on_frontier}/{total_on_frontier} points are VS methods', 
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                verticalalignment='top')
    
    ax.set_xlabel('Diversity (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Quality (%)', fontsize=14, fontweight='bold')
    ax.set_title('Diversity vs Quality: Pareto Frontier Analysis\\n(All Models Combined)', 
                fontsize=16, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pareto_frontier_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/pareto_frontier_analysis.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved Pareto frontier analysis plot")
    return vs_on_frontier, total_on_frontier

def perform_statistical_tests(all_results):
    """Perform statistical significance tests between VS-CoT and all baseline methods"""
    method_names = ["Direct", "CoT", "Sequence", "Multi-turn", "VS-Standard", "VS-CoT", "VS-Combined"]
    
    # Collect data for each method across all models
    method_data = {}
    for method in method_names:
        method_data[method] = {
            'diversity': [],
            'quality': []
        }
    
    for model_name, results in all_results.items():
        for method in method_names:
            data = results.get(method)
            if data and data["diversity"] is not None and data["quality"] is not None:
                method_data[method]['diversity'].append(data["diversity"])
                method_data[method]['quality'].append(data["quality"])
    
    # Compare VS-CoT against all baseline methods
    best_vs_method = "VS-CoT"
    baseline_methods = ["Direct", "CoT", "Sequence", "Multi-turn"]
    statistical_results = {}
    
    if method_data[best_vs_method]['diversity'] and method_data[best_vs_method]['quality']:
        vs_div = method_data[best_vs_method]['diversity']
        vs_qual = method_data[best_vs_method]['quality']
        
        for baseline_method in baseline_methods:
            if method_data[baseline_method]['diversity']:
                baseline_div = method_data[baseline_method]['diversity']
                baseline_qual = method_data[baseline_method]['quality']
                
                # Perform t-tests (assuming normal distribution)
                div_stat, div_pvalue = stats.ttest_ind(vs_div, baseline_div, alternative='greater')
                qual_stat, qual_pvalue = stats.ttest_ind(vs_qual, baseline_qual, alternative='greater')
                
                statistical_results[baseline_method] = {
                    'diversity_pvalue': div_pvalue,
                    'quality_pvalue': qual_pvalue,
                    'diversity_improvement': np.mean(vs_div) - np.mean(baseline_div),
                    'quality_improvement': np.mean(vs_qual) - np.mean(baseline_qual)
                }
    
    return statistical_results

def plot_statistical_comparison(all_results, output_dir="plots"):
    """Create box plots showing statistical comparison"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plotting style
    sns.set_style("whitegrid")
    method_names = ["Direct", "CoT", "Sequence", "Multi-turn", "VS-Standard", "VS-CoT", "VS-Combined"]
    base_colors = ['#4472C4', '#70AD47', '#FFC000', '#C5504B', '#264478', '#375623', '#7F6000']
    
    # Collect data for plotting
    diversity_data = []
    quality_data = []
    method_labels = []
    
    for method in method_names:
        for model_name, results in all_results.items():
            data = results.get(method)
            if data and data["diversity"] is not None and data["quality"] is not None:
                diversity_data.append(data["diversity"])
                quality_data.append(data["quality"])
                method_labels.append(method)
    
    # Create DataFrame for easier plotting
    import pandas as pd
    df = pd.DataFrame({
        'Method': method_labels,
        'Diversity': diversity_data,
        'Quality': quality_data
    })
    
    # Perform statistical tests
    stat_results = perform_statistical_tests(all_results)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Diversity box plot
    sns.boxplot(data=df, x='Method', y='Diversity', ax=ax1, palette=base_colors)
    ax1.set_title('Diversity Distribution Across Models', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Diversity (%)', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    
    # Quality box plot
    sns.boxplot(data=df, x='Method', y='Quality', ax=ax2, palette=base_colors)
    ax2.set_title('Quality Distribution Across Models', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Quality (%)', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add significance annotations
    def get_significance_marker(p_value):
        if p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**'
        elif p_value < 0.05:
            return '*'
        else:
            return 'ns'
    
    # Add statistical significance text
    if stat_results:
        text_lines = ["Statistical Significance vs Direct:"]
        for method, results in stat_results.items():
            div_sig = get_significance_marker(results['diversity_pvalue'])
            qual_sig = get_significance_marker(results['quality_pvalue'])
            div_imp = results['diversity_improvement']
            qual_imp = results['quality_improvement']
            text_lines.append(f"{method}: Div {div_sig} (+{div_imp:.1f}%), Qual {qual_sig} (+{qual_imp:.1f}%)")
        
        fig.text(0.02, 0.02, '\\n'.join(text_lines), fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
                verticalalignment='bottom')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # Make room for statistical text
    plt.savefig(f'{output_dir}/statistical_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/statistical_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved statistical comparison box plots")
    return stat_results

def plot_method_averages(all_results, output_dir="plots"):
    """Create bar charts showing average performance across all models for each method"""
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    method_names = ["Direct", "CoT", "Sequence", "Multi-turn", "VS-Standard", "VS-CoT", "VS-Combined"]
    
    # Perform statistical tests
    stat_results = perform_statistical_tests(all_results)
    
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
    
    metrics = [
        ('diversity', 'Average Diversity (%)', ''),
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
                     edgecolor='black', linewidth=1)
        
        # Add hatches to VS methods (last 3 bars)
        for i, bar in enumerate(bars[-3:], start=len(bars)-3):
            bar.set_hatch('///')
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.5,
                   f'{mean:.1f}±{std:.1f}', ha='center', va='bottom', 
                   fontsize=12, fontweight='bold')
        
        ax.set_xlabel('Methods', fontsize=16, fontweight='bold')
        ax.set_ylabel(metric_title, fontsize=16, fontweight='bold')
        ax.set_title(f'{metric_title} - Average Across All Models', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=0)
        
        # Highlight best performing method
        if metric_key == 'rouge_l':  # Lower is better
            best_idx = np.argmin(means)
        else:  # Higher is better
            best_idx = np.argmax(means)
        
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(3)
        
        # Add p-test results annotation in top left
        if stat_results and metric_key in ['diversity', 'quality']:
            p_text_lines = ["Statistical test results:", "VS-CoT"]
            for baseline_method, results in stat_results.items():
                p_val = results[f'{metric_key}_pvalue']
                if p_val < 0.001:
                    sig_marker = '***'
                elif p_val < 0.01:
                    sig_marker = '**'
                elif p_val < 0.05:
                    sig_marker = '*'
                else:
                    sig_marker = 'ns'
                p_text_lines.append(f"vs {baseline_method}: {sig_marker} (p={p_val:.3f})")
            
            p_text = '\n'.join(p_text_lines)
            ax.text(0.02, 0.98, p_text, transform=ax.transAxes, fontsize=14, 
                   verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
                   fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/method_average_{metric_key}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_dir}/method_average_{metric_key}.pdf', bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved {metric_title} method average plot")


def plot_diversity_vs_quality_model(all_results, model_name, output_dir="plots"):
    """
    Plot the diversity vs quality scatter plot for a specific model.
    """
    plot_diversity_vs_quality_average({model_name: all_results[model_name]})
    

def main():
    # Model directory mapping
    models = {
        "Claude-4-Sonnet": "anthropic_claude-4-sonnet",
        "Claude-3.7-Sonnet": "anthropic_claude-3.7-sonnet", 
        "Gemini-2.5-Pro": "google_gemini-2.5-pro",
        "Gemini-2.5-Flash": "google_gemini-2.5-flash",
        "GPT-4.1": "openai_gpt-4.1",
        "GPT-4.1-Mini": "openai_gpt-4.1-mini",
        "Llama-3.1-70B": "meta-llama_Llama-3.1-70B-Instruct",
        "DeepSeek-R1": "deepseek_deepseek-r1-0528",
        "GPT-o3": "openai_o3",
    }
    
    base_dir = "poem_experiments_final"
    # base_dir = "method_results_bias"
    all_results = {}
    
    # Output file
    output_file = "latex_table_results.txt"
    
    with open(output_file, 'w') as f:
        # Collect results for all models
        for model_name, model_dir_name in models.items():
            model_path = os.path.join(base_dir, model_dir_name, f"{model_dir_name}_poem")
            if os.path.exists(model_path):
                results = get_model_results(model_path, model_name)
                all_results[model_name] = results
                print(f"✓ Processed {model_name}")
                f.write(f"% Processed {model_name}\n")
            else:
                print(f"✗ Directory not found for {model_name}: {model_path}")
                f.write(f"% Directory not found for {model_name}: {model_path}\n")
        
        # Generate LaTeX table
        f.write("\n" + "="*80 + "\n")
        f.write("LATEX TABLE DATA\n")
        f.write("="*80 + "\n")
        
        for model_name, results in all_results.items():
            f.write(f"\n\\multirow{{8}}{{*}}{{{model_name}}}\n")
            
            baseline = results.get("Direct")
            if baseline is None:
                f.write("% No baseline data available\n")
                continue
            
            # Find best values for this specific model
            best_diversity, best_rouge_l, best_quality = find_best_values_per_model(results)
            
            # Print baseline
            # if all(v is not None for v in [baseline["diversity"], baseline["rouge_l"], baseline["quality"]]):
                # f.write(f"& Baseline & {format_metric(baseline['diversity'], baseline['diversity_std'], baseline['diversity'] == best_diversity)} & {format_metric(baseline['rouge_l'], baseline['rouge_l_std'], baseline['rouge_l'] == best_rouge_l)} & {format_metric(baseline['quality'], baseline['quality_std'], baseline['quality'] == best_quality)} \\\\\n")
            
            # Print other methods
            for method in ["Direct", "CoT", "Sequence", "Multi-turn"]:
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
            baseline = results.get("Direct")
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
    print("GENERATING PLOTS")
    print("="*50)
    
    # Generate original plots
    # plot_model_comparison(all_results)
    # plot_method_averages(all_results)
    # plot_individual_models(all_results)
    # plot_diversity_vs_quality_scatter(all_results)
    # plot_diversity_vs_quality_average(all_results)
    # plot_diversity_vs_quality_2x2(all_results)
    plot_diversity_vs_quality_model(all_results, "GPT-o3") 

    # Generate advanced analysis plots
    print("\n" + "="*30)
    print("ADVANCED ANALYSIS")
    print("="*30)
    
    vs_frontier, total_frontier = plot_combined_pareto_scatter(all_results)
    stat_results = plot_statistical_comparison(all_results)
    
    print(f"\n📊 PARETO ANALYSIS RESULTS:")
    print(f"   VS methods dominate {vs_frontier}/{total_frontier} points on Pareto frontier")
    print(f"   That's {100*vs_frontier/total_frontier:.1f}% of optimal trade-offs!")
    
    if stat_results:
        print(f"\n📈 STATISTICAL SIGNIFICANCE RESULTS:")
        for method, results in stat_results.items():
            div_sig = "✓" if results['diversity_pvalue'] < 0.05 else "✗"
            qual_sig = "✓" if results['quality_pvalue'] < 0.05 else "✗"
            print(f"   {method}: Diversity {div_sig} (p={results['diversity_pvalue']:.3f}), Quality {qual_sig} (p={results['quality_pvalue']:.3f})")
    
    print(f"\n✓ All plots saved to 'plots/' directory")
    print("✓ Generated files:")
    print("  - model_comparison_diversity.png/pdf")
    print("  - model_comparison_rouge_l.png/pdf")  
    print("  - model_comparison_quality.png/pdf")
    print("  - method_average_diversity.png/pdf")
    print("  - method_average_rouge_l.png/pdf")
    print("  - method_average_quality.png/pdf")
    print("  - individual_[model_name].png/pdf (for each model)")
    print("  - diversity_vs_quality_[model_name].png/pdf (for each model)")
    print("  - diversity_vs_quality_2x2.png/pdf (2x2 subplot for 4 main models)")
    print("  - pareto_frontier_analysis.png/pdf (KEY FIGURE)")
    print("  - statistical_comparison.png/pdf (SIGNIFICANCE TESTS)")

    
if __name__ == "__main__":
    main()