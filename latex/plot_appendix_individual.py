import matplotlib.pyplot as plt
import json
import os
import numpy as np
import re
from pathlib import Path
from scipy import stats
from scipy.stats import ttest_ind
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from config import COLORS, EDGE_COLORS

def perform_statistical_tests(task_data, task_name, metric):
    """Perform t-tests comparing baselines against VS-Standard"""
    print(f"\n=== Statistical Tests for {task_name} ({metric}) ===")

    # Collect individual model data for VS-Standard
    vs_standard_values = []
    for model_name, model_results in task_data.items():
        if 'VS-Standard' in model_results and model_results['VS-Standard'][metric] is not None:
            vs_standard_values.append(model_results['VS-Standard'][metric])

    if not vs_standard_values:
        print(f"No VS-Standard data found for {task_name}")
        return {}

    baseline_methods = ['Direct', 'CoT', 'Sequence', 'Multi-turn']
    significant_results = {}

    for method in baseline_methods:
        # Collect individual model data for this baseline method
        baseline_values = []
        for model_name, model_results in task_data.items():
            if method in model_results and model_results[method][metric] is not None:
                baseline_values.append(model_results[method][metric])

        if len(baseline_values) < 2 or len(vs_standard_values) < 2:
            print(f"Insufficient data for {method} vs VS-Standard comparison")
            continue

        # Perform two-sample t-test (one-tailed: VS-Standard > baseline for diversity, < baseline for quality)
        if metric == 'diversity':
            t_stat, p_value = ttest_ind(vs_standard_values, baseline_values, alternative='greater')
        else:  # quality - we want higher quality, so VS-Standard should be greater
            t_stat, p_value = ttest_ind(vs_standard_values, baseline_values, alternative='greater')

        vs_mean = np.mean(vs_standard_values)
        baseline_mean = np.mean(baseline_values)

        # Determine significance level and marker
        if p_value < 0.001:
            significance_marker = "***"
            significant_results[method] = "***"
        elif p_value < 0.01:
            significance_marker = "**"
            significant_results[method] = "**"
        elif p_value < 0.05:
            significance_marker = "*"
            significant_results[method] = "*"
        else:
            significance_marker = ""
            significant_results[method] = ""

        print(f"{method}{significance_marker}: VS-Standard ({vs_mean:.2f}) vs {method} ({baseline_mean:.2f}), t={t_stat:.3f}, p={p_value:.4f}")

    return significant_results

def parse_latex_table_data(file_path):
    """Parse LaTeX table data from .tex files to extract metrics"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract model data using regex patterns
    model_results = {}
    
    # Pattern to match model sections
    model_pattern = r'\\multirow\{[0-9]+\}\{\*\}\{([^}]+)\}(.*?)(?=\\multirow|\\bottomrule)'
    model_matches = re.findall(model_pattern, content, re.DOTALL)
    
    for model_name, model_content in model_matches:
        model_results[model_name] = {}
        
        # Extract method data - handle both bold and non-bold formatting
        method_patterns = [
            (r'& Direct & (?:\\textbf\{)?([0-9.]+)\$_\{\\pm\{([0-9.]+)\}\}\$(?:\})? & (?:\\textbf\{)?([0-9.]+)\$_\{\\pm\{([0-9.]+)\}\}\$(?:\})? & (?:\\textbf\{)?([0-9.]+)\$_\{\\pm\{([0-9.]+)\}\}\$(?:\})?', 'Direct'),
            (r'& CoT & (?:\\textbf\{)?([0-9.]+)\$_\{\\pm\{([0-9.]+)\}\}\$(?:\})? & (?:\\textbf\{)?([0-9.]+)\$_\{\\pm\{([0-9.]+)\}\}\$(?:\})? & (?:\\textbf\{)?([0-9.]+)\$_\{\\pm\{([0-9.]+)\}\}\$(?:\})?', 'CoT'),
            (r'& Sequence & (?:\\textbf\{)?([0-9.]+)\$_\{\\pm\{([0-9.]+)\}\}\$(?:\})? & (?:\\textbf\{)?([0-9.]+)\$_\{\\pm\{([0-9.]+)\}\}\$(?:\})? & (?:\\textbf\{)?([0-9.]+)\$_\{\\pm\{([0-9.]+)\}\}\$(?:\})?', 'Sequence'),
            (r'& Multi-turn & (?:\\textbf\{)?([0-9.]+)\$_\{\\pm\{([0-9.]+)\}\}\$(?:\})? & (?:\\textbf\{)?([0-9.]+)\$_\{\\pm\{([0-9.]+)\}\}\$(?:\})? & (?:\\textbf\{)?([0-9.]+)\$_\{\\pm\{([0-9.]+)\}\}\$(?:\})?', 'Multi-turn'),
            (r'& \$\\hookrightarrow\$ Standard & (?:\\textbf\{)?([0-9.]+)\$_\{\\pm\{([0-9.]+)\}\}\$(?:\})? & (?:\\textbf\{)?([0-9.]+)\$_\{\\pm\{([0-9.]+)\}\}\$(?:\})? & (?:\\textbf\{)?([0-9.]+)\$_\{\\pm\{([0-9.]+)\}\}\$(?:\})?', 'VS-Standard'),
            (r'& \$\\hookrightarrow\$ CoT & (?:\\textbf\{)?([0-9.]+)\$_\{\\pm\{([0-9.]+)\}\}\$(?:\})? & (?:\\textbf\{)?([0-9.]+)\$_\{\\pm\{([0-9.]+)\}\}\$(?:\})? & (?:\\textbf\{)?([0-9.]+)\$_\{\\pm\{([0-9.]+)\}\}\$(?:\})?', 'VS-CoT'),
            (r'& \$\\hookrightarrow\$ Multi & (?:\\textbf\{)?([0-9.]+)\$_\{\\pm\{([0-9.]+)\}\}\$(?:\})? & (?:\\textbf\{)?([0-9.]+)\$_\{\\pm\{([0-9.]+)\}\}\$(?:\})? & (?:\\textbf\{)?([0-9.]+)\$_\{\\pm\{([0-9.]+)\}\}\$(?:\})?', 'VS-Multi')
        ]
        
        for pattern, method_name in method_patterns:
            match = re.search(pattern, model_content)
            if match:
                diversity_val, diversity_std, rouge_val, rouge_std, quality_val, quality_std = match.groups()
                model_results[model_name][method_name] = {
                    'diversity': float(diversity_val) * 2,
                    'diversity_std': float(diversity_std),
                    'rouge_l': float(rouge_val),
                    'rouge_l_std': float(rouge_std),
                    'quality': float(quality_val),
                    'quality_std': float(quality_std)
                }
    
    return model_results

def create_individual_plot(task_data, task_name, metric, output_dir="latex_figures"):
    """Create individual bar plot for a specific task and metric"""
    
    # Set up professional styling with larger text
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'font.size': 20,
        'axes.labelsize': 20,
        'axes.titlesize': 22,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'legend.fontsize': 20,
        'axes.linewidth': 1.2,
        'axes.edgecolor': '#333333',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'lines.linewidth': 2.5,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white'
    })
    
    # Same color scheme as unified figure: our methods (red), baselines (blue)
    method_names = ["Direct", "CoT", "Sequence", "Multi-turn", "VS-Standard", "VS-CoT", "VS-Multi"]
    
    # Edge colors for better distinction

    
    # Create figure with appropriate size
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Calculate average scores across all models for each method
    method_averages = []
    method_stds = []
    
    for method in method_names:
        values = []
        
        for model_name, model_results in task_data.items():
            if method in model_results and model_results[method][metric] is not None:
                values.append(model_results[method][metric])
        
        if values:
            method_averages.append(np.mean(values))
            method_stds.append(np.std(values))
        else:
            method_averages.append(0)
            method_stds.append(0)
    
    # Create bars with refined styling
    x_pos = np.arange(len(method_names))
    bars = []
    
    for i, (method, avg, std) in enumerate(zip(method_names, method_averages, method_stds)):
        bar = ax.bar(i, avg, yerr=std,
                    color=COLORS[method], edgecolor=EDGE_COLORS[method], 
                    linewidth=1.5, width=0.7, alpha=0.9,
                    error_kw={'elinewidth': 2.0, 'capsize': 4, 'capthick': 2.0, 'alpha': 0.8})
        bars.append(bar)
    
    # Perform statistical tests and add significance markers
    significance_results = perform_statistical_tests(task_data, task_name, metric)
    
    # Add *, **, *** markers above error bars for statistically significant differences
    method_labels = []
    for i, method in enumerate(method_names):
        method_labels.append(method)
        if method in significance_results and significance_results[method]:
            # Add significance marker above the error bar
            significance_marker = significance_results[method]
            y_pos = method_averages[i] + method_stds[i] + (max(method_averages) * 0.05)
            ax.text(i, y_pos, significance_marker, ha='center', va='bottom',
                   fontsize=20, fontweight='bold', color='red')
    
    # Clean formatting
    metric_label = 'Diversity Score' if metric == 'diversity' else 'Quality Score'
    ax.set_title(f'{task_name.title()} - {metric_label}', fontweight='bold', pad=50, fontsize=28)
    ax.set_ylabel(metric_label, fontweight='bold', fontsize=24)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(method_labels, rotation=45, ha='right', fontsize=24)
    # Format y-axis ticks as integers with max 6 labels
    # Set y-axis to have exactly 6 integer tick labels
    y_min, y_max = ax.get_ylim()

    # Make tick labels (numbers) bigger
    ax.tick_params(axis='both', which='major', labelsize=22)
    
    # Subtle grid
    ax.grid(True, alpha=0.2, axis='y', linestyle='-', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Clean spines
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    
    # Set y-axis limits with intelligent scaling (including error bars)
    if max(method_averages) > 0:
        # Calculate the actual min/max including error bars
        min_with_error = min([avg - std for avg, std in zip(method_averages, method_stds) if avg > 0])
        max_with_error = max([avg + std for avg, std in zip(method_averages, method_stds)])
        
        # Add extra space for significance markers and value labels
        range_val = max_with_error - min_with_error
        y_min = max(0, min_with_error - range_val * 0.1)
        y_max = max_with_error + range_val * 0.15  # Extra space for ** markers and value labels
        ax.set_ylim(y_min, y_max)
    
        tick_min = int(np.floor(y_min))
        tick_max = int(np.ceil(y_max))
        tick_values = np.linspace(tick_min, tick_max, 8)

        ax.yaxis.set_major_locator(plt.FixedLocator(tick_values))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))
    # Add value labels on bars
    for i, (avg, std) in enumerate(zip(method_averages, method_stds)):
        if avg > 0:
            y_pos = avg + std + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01
            ax.text(i, y_pos, f'{avg:.1f}', ha='center', va='bottom', 
                   fontsize=20, fontweight='600', alpha=0.9)
    
    # Add legend
    legend_patches = []
    baseline_methods = ['Direct', 'CoT', 'Sequence', 'Multi-turn']
    vs_methods = ['VS-Standard', 'VS-CoT', 'VS-Multi']
    
    legend_patches.append(Patch(color='#4A90E2', alpha=0.7, label='Baseline Methods'))
    legend_patches.append(Patch(color='#FF6B6B', alpha=0.7, label='Our Methods (Verbalized Sampling)'))
    
    ax.legend(handles=legend_patches, loc='upper center', fontsize=20, framealpha=0.9, ncol=2, bbox_to_anchor=(0.5, 1.12))
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    filename = f'{task_name.lower()}_{metric}_appendix'
    
    plt.tight_layout()
    fig.savefig(f'{output_dir}/{filename}.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(f'{output_dir}/{filename}.pdf', 
               bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Generated {task_name} {metric} plot: {filename}")

def create_all_appendix_plots(output_dir="latex_figures"):
    """Create all 6 individual plots for the appendix"""
    
    # Load data from LaTeX tables
    poem_data = parse_latex_table_data("latex/results/poem.tex")
    joke_data = parse_latex_table_data("latex/results/joke.tex")
    story_data = parse_latex_table_data("latex/results/story.tex")
    
    tasks = [
        ("Poem", poem_data),
        ("Story", story_data),
        ("Joke", joke_data)
    ]
    
    metrics = ['diversity', 'quality']
    
    print("Creating individual appendix plots...")
    print("=" * 50)
    
    # Create all 6 plots
    for task_name, task_data in tasks:
        for metric in metrics:
            create_individual_plot(task_data, task_name, metric, output_dir)
    
    print("\n" + "=" * 50)
    print("ALL APPENDIX PLOTS COMPLETE")
    print("=" * 50)
    print(f"📁 All plots saved to: {output_dir}/")

if __name__ == "__main__":
    create_all_appendix_plots()