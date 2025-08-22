# import seaborn as sns  # Removed due to numpy compatibility issues
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

SUBPLOT_LABEL_SIZE = 26

def perform_statistical_tests(task_data, task_name):
    """Perform t-tests comparing baselines against VS-Standard"""
    print(f"\n=== Statistical Tests for {task_name} ===")
    
    # Collect individual model data for VS-Standard
    vs_standard_values = []
    for model_name, model_results in task_data.items():
        if 'VS-Standard' in model_results and model_results['VS-Standard']['diversity'] is not None:
            vs_standard_values.append(model_results['VS-Standard']['diversity'])
    
    if not vs_standard_values:
        print(f"No VS-Standard data found for {task_name}")
        return {}
    
    baseline_methods = ['Direct', 'CoT', 'Sequence', 'Multi-turn']
    significant_results = {}
    
    for method in baseline_methods:
        # Collect individual model data for this baseline method
        baseline_values = []
        for model_name, model_results in task_data.items():
            if method in model_results and model_results[method]['diversity'] is not None:
                baseline_values.append(model_results[method]['diversity'])
        
        if len(baseline_values) < 2 or len(vs_standard_values) < 2:
            print(f"Insufficient data for {method} vs VS-Standard comparison")
            continue
            
        # Perform two-sample t-test (one-tailed: VS-Standard > baseline)
        t_stat, p_value = ttest_ind(vs_standard_values, baseline_values, alternative='greater')
        
        vs_mean = np.mean(vs_standard_values)
        baseline_mean = np.mean(baseline_values)
        
        significant = p_value < 0.05
        significant_results[method] = significant
        
        significance_marker = "**" if significant else ""
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
                    'diversity': float(diversity_val),
                    'diversity_std': float(diversity_std),
                    'rouge_l': float(rouge_val),
                    'rouge_l_std': float(rouge_std),
                    'quality': float(quality_val),
                    'quality_std': float(quality_std)
                }
    
    return model_results


def load_experiment_data():
    """Load data from experiment directories for scatter plot and cognitive burden analysis"""
    # Model mapping for experiments
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

    # Load poem experiment data
    base_dir = "poem_experiments_final"
    poem_results = {}
    
    for model_name, model_dir_name in models.items():
        model_path = os.path.join(base_dir, model_dir_name, f"{model_dir_name}_poem")
        if os.path.exists(model_path):
            results = get_model_results(model_path, model_name)
            poem_results[model_name] = results
    
    # For cognitive burden analysis, categorize by model size
    results_by_size = {"large": {}, "small": {}}
    size_categories = {
        "large": ["GPT-4.1", "Gemini-2.5-Pro"],
        "small": ["GPT-4.1-Mini", "Gemini-2.5-Flash"]
    }
    
    for size_category, model_list in size_categories.items():
        for model_name in model_list:
            if model_name in poem_results:
                results_by_size[size_category][model_name] = poem_results[model_name]
    
    return poem_results, results_by_size


def create_unified_creativity_figure(output_dir="latex_figures"):
    """Create a unified 2x3 figure with creativity analysis across tasks"""
    
    # Set up elegant styling inspired by the reference image
    plt.style.use('default')  # Start with clean slate
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'axes.linewidth': 0.8,
        'axes.edgecolor': '#666666',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'lines.linewidth': 2.0,
        'lines.markersize': 8,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white'
    })
    
    # Load data from LaTeX tables and experiment directories
    poem_data = parse_latex_table_data("latex/results/poem.tex")
    joke_data = parse_latex_table_data("latex/results/joke.tex")
    story_data = parse_latex_table_data("latex/results/story.tex")
    
    # Load experiment data for scatter plot and cognitive burden analysis
    poem_results, results_by_size = load_experiment_data()
    
    # Create figure with better proportions and spacing for three separate legends at top
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(3, 3, height_ratios=[0.3, 1, 1], width_ratios=[1, 1, 1], 
                          hspace=0.65, wspace=0.25, left=0.08, right=0.95, top=0.95, bottom=0.08)
    
    # Elegant color palette inspired by the reference
    method_names = ["Direct", "CoT", "Sequence", "Multi-turn", "VS-Standard", "VS-CoT", "VS-Multi"]
    colors = {
        'Direct': '#FFE5E5',      # Very light red
        'CoT': '#FFCCCB',         # Light red  
        'Sequence': '#FF9999',     # Medium red
        'Multi-turn': '#FF6B6B',   # Distinct red (different from small models)
        'VS-Standard': '#E8F4FD',  # Very light blue
        'VS-CoT': '#B8E0F5',       # Light blue
        'VS-Multi': '#7CC7EA'      # Medium blue
    }
    
    # Edge colors for better distinction
    edge_colors = {
        'Direct': '#FF6B6B',
        'CoT': '#FF6B6B', 
        'Sequence': '#FF6B6B',
        'Multi-turn': '#FF6B6B',
        'VS-Standard': '#4A90E2',
        'VS-CoT': '#4A90E2',
        'VS-Multi': '#4A90E2'
    }
    
    # Row 1: Elegant bar charts for each task (reordered: poem, story, joke)
    tasks = [("Poem (Diversity)", poem_data), ("Story (Diversity)", story_data), ("Joke (Diversity)", joke_data)]
    
    # Perform statistical tests for each task
    all_significance_results = {}
    for task_name, task_data in tasks:
        clean_task_name = task_name.replace(" (Diversity)", "")
        all_significance_results[clean_task_name] = perform_statistical_tests(task_data, clean_task_name)
    
    for col_idx, (task_name, task_data) in enumerate(tasks):
        ax = fig.add_subplot(gs[1, col_idx])
        
        # Calculate average diversity scores across all models for each method
        method_averages = []
        method_stds = []
        
        for method in method_names:
            diversity_values = []
            
            for model_name, model_results in task_data.items():
                if method in model_results and model_results[method]['diversity'] is not None:
                    diversity_values.append(model_results[method]['diversity'])
            
            if diversity_values:
                method_averages.append(np.mean(diversity_values))
                method_stds.append(np.std(diversity_values))
            else:
                method_averages.append(0)
                method_stds.append(0)
        
        # Create elegant bars with refined styling
        x_pos = np.arange(len(method_names))
        bars = []
        
        for i, (method, avg, std) in enumerate(zip(method_names, method_averages, method_stds)):
            bar = ax.bar(i, avg, yerr=std,
                        color=colors[method], edgecolor=edge_colors[method], 
                        linewidth=1.2, width=0.7, alpha=0.9,
                        error_kw={'elinewidth': 1.5, 'capsize': 3, 'capthick': 1.5, 'alpha': 0.8})
            bars.append(bar)
        
        # Clean formatting
        ax.set_title(f'{task_name}', fontweight='bold', pad=15, fontsize=18)
        ax.set_ylabel('Diversity Score' if col_idx == 0 else '', fontweight='bold', fontsize=12)
        ax.set_xticks(x_pos)
        
        # Add ** markers above error bars for statistically significant differences
        clean_task_name = task_name.replace(" (Diversity)", "")
        significance_results = all_significance_results.get(clean_task_name, {})
        
        method_labels = []
        for i, method in enumerate(method_names):
            method_labels.append(method)
            if method in significance_results and significance_results[method]:
                # Add ** above the error bar
                y_pos = method_averages[i] + method_stds[i] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05
                ax.text(i, y_pos, '**', ha='center', va='bottom', 
                       fontsize=16, fontweight='bold', color='red')
        
        ax.set_xticklabels(method_labels, rotation=45, ha='right', fontsize=12)
        
        # Make tick labels (numbers) bigger
        ax.tick_params(axis='both', which='major', labelsize=15)
        
        # Subtle grid
        ax.grid(True, alpha=0.15, axis='y', linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Clean spines
        ax.spines['left'].set_color('#666666')
        ax.spines['bottom'].set_color('#666666')
        
        # Set y-axis limits with intelligent scaling (not starting from 0)
        if max(method_averages) > 0:
            min_val = min([avg for avg in method_averages if avg > 0])
            max_val = max(method_averages)
            range_val = max_val - min_val
            # Start from 85% of minimum value, extend to 130% of maximum
            y_min = max(0, min_val - range_val * 0.15)
            y_max = max_val * 1.3
            ax.set_ylim(y_min, y_max)
        
        # Add subtle value labels (adjusted for new y-limits)
        for i, (avg, std) in enumerate(zip(method_averages, method_stds)):
            if avg > 0:
                y_pos = avg + std + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
                ax.text(i, y_pos, f'{avg:.1f}', ha='center', va='bottom', 
                       fontsize=12, fontweight='600', alpha=0.9)
        
        # Add subplot labels a, b, c
        subplot_labels = ['a', 'b', 'c']
        ax.text(-0.15, 1.05, f'{subplot_labels[col_idx]}', transform=ax.transAxes,
                fontsize=SUBPLOT_LABEL_SIZE, fontweight='bold', ha='left', va='bottom')
    
    # Row 2, Col 1: Elegant scatter plot (average across all models)
    ax_scatter = fig.add_subplot(gs[2, 0])
    
    # Calculate method averages for scatter plot from poem data
    method_scatter_data = {}
    
    for method in method_names:
        diversity_values = []
        quality_values = []
        
        # Collect values from all models in poem data
        for model_name, results in poem_results.items():
            if method in results:
                data = results[method]
                if data and data["diversity"] is not None and data["quality"] is not None:
                    diversity_values.append(data["diversity"])
                    quality_values.append(data["quality"])
        
        if diversity_values and quality_values:
            method_scatter_data[method] = {
                'diversity': np.mean(diversity_values),
                'quality': np.mean(quality_values)
            }
    
    # Plot scatter points with elegant styling and direct labels
    for method in method_names:
        if method in method_scatter_data:
            data = method_scatter_data[method]
            
            # Different styling for VS vs baseline methods
            if method.startswith('VS-'):
                marker = 'o'
                size = 120
                alpha = 0.9
                linewidth = 2
            else:
                marker = 's'
                size = 100
                alpha = 0.8
                linewidth = 1.5
            
            ax_scatter.scatter(data['diversity'], data['quality'], 
                             color=colors[method], marker=marker, s=size, alpha=alpha,
                             zorder=5, edgecolors=edge_colors[method], linewidth=linewidth)
            
            # Add label below the marker with adjusted positioning for overlapping methods
            x_offset = 0
            ha_align = 'center'
            if method == 'Sequence':
                x_offset = -0.1  # Move left
                ha_align = 'center'
            elif method == 'VS-Standard':
                x_offset = 0.2   # Move right
                ha_align = 'center'
            
            ax_scatter.text(data['diversity'] + x_offset, data['quality'] - 0.5, method, 
                           ha=ha_align, va='top', fontsize=11, fontweight='600',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Adjust y-axis limits to accommodate labels below markers
    y_values = [data['quality'] for data in method_scatter_data.values()]
    if y_values:
        y_min = min(y_values) - 2  # Extra space for labels
        y_max = max(y_values) + 1.3
        ax_scatter.set_ylim(y_min, y_max)
    x_values = [data['diversity'] for data in method_scatter_data.values()]
    if x_values:
        x_min = min(x_values) - 0.5
        x_max = max(x_values) + 1.3
        ax_scatter.set_xlim(x_min, x_max)
    
    ax_scatter.set_xlabel('Diversity Score', fontweight='bold', fontsize=12)
    ax_scatter.set_ylabel('Quality Score', fontweight='bold', fontsize=12)
    
    # Make tick labels (numbers) bigger for scatter plot
    ax_scatter.tick_params(axis='both', which='major', labelsize=15)
    ax_scatter.set_title('Diversity - Quality (Poem)', fontweight='bold', pad=15, fontsize=18)
    ax_scatter.grid(True, alpha=0.15, linestyle='-', linewidth=0.5)
    ax_scatter.set_axisbelow(True)
    
    # Clean spines
    ax_scatter.spines['left'].set_color('#666666')
    ax_scatter.spines['bottom'].set_color('#666666')
    
    # Add Pareto optimal arrow (adjusted positioning to avoid overlap)
    # First add the arrow
    ax_scatter.annotate('', xy=(0.98, 0.98), xytext=(0.80, 0.80),
                       xycoords='axes fraction', textcoords='axes fraction',
                       arrowprops=dict(arrowstyle='->', lw=3, color='red', alpha=0.7))
    # Then add the text separately, positioned lower to avoid overlap
    ax_scatter.text(0.75, 0.95, 'Pareto optimal', transform=ax_scatter.transAxes,
                   fontsize=12, color='red', fontweight='bold', ha='center')
    
    # Add subplot label d
    ax_scatter.text(-0.15, 1.05, 'd', transform=ax_scatter.transAxes,
                   fontsize=SUBPLOT_LABEL_SIZE, fontweight='bold', ha='left', va='bottom')
    
    # Row 2, Cols 2-3: Elegant cognitive burden analysis
    methods_subset = ["Sequence", "Multi-turn", "VS-Standard", "VS-CoT", "VS-Multi"]
    
    # Calculate performance changes relative to Direct baseline
    size_method_deltas = {}
    
    for size_category, results in results_by_size.items():
        size_method_deltas[size_category] = {}
        
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
                    'quality_delta_mean': np.mean(quality_deltas)
                }
    
    # Elegant model size colors
    size_colors = {'small': '#4ECDC4', 'large': '#8B4A9C'}  # Purple for small, teal for large
    size_labels = {'small': 'Small Models', 'large': 'Large Models'}
    
    # Diversity burden analysis
    ax_burden_div = fig.add_subplot(gs[2, 1])
    
    x_methods = np.arange(len(methods_subset))
    width = 0.35
    
    large_diversity_changes = []
    small_diversity_changes = []
    
    for method in methods_subset:
        large_val = size_method_deltas.get('large', {}).get(method, {}).get('diversity_delta_mean', 0)
        small_val = size_method_deltas.get('small', {}).get(method, {}).get('diversity_delta_mean', 0)
        
        large_diversity_changes.append(large_val)
        small_diversity_changes.append(small_val)
    
    ax_burden_div.bar(x_methods - width/2, small_diversity_changes, width, 
                     color=size_colors['small'], alpha=0.9, 
                     edgecolor='white', linewidth=1.2, label=size_labels['small'])
    ax_burden_div.bar(x_methods + width/2, large_diversity_changes, width, 
                     color=size_colors['large'], alpha=0.9, 
                     edgecolor='white', linewidth=1.2, label=size_labels['large'])
    
    # Add value labels on bars for better clarity
    for i, (small_val, large_val) in enumerate(zip(small_diversity_changes, large_diversity_changes)):
        # Small model values
        if abs(small_val) > 0.1:  # Only show if value is significant
            sign = '+' if small_val >= 0 else ''
            ax_burden_div.text(i - width/2, small_val + (0.2 if small_val > 0 else -0.2), 
                              f'{sign}{small_val:.1f}', ha='center', va='bottom' if small_val > 0 else 'top',
                              fontsize=11, fontweight='600')
        # Large model values  
        if abs(large_val) > 0.1:  # Only show if value is significant
            sign = '+' if large_val >= 0 else ''
            ax_burden_div.text(i + width/2, large_val + (0.2 if large_val > 0 else -0.2), 
                              f'{sign}{large_val:.1f}', ha='center', va='bottom' if large_val > 0 else 'top',
                              fontsize=11, fontweight='600')
    
    ax_burden_div.axhline(y=0, color='#666666', linestyle='-', alpha=0.8, linewidth=1)
    ax_burden_div.set_ylabel('$\Delta$ Diversity Against Direct', fontweight='bold', fontsize=12)
    ax_burden_div.set_title('Emergent Trend: $\Delta$ Diversity', fontweight='bold', pad=15, fontsize=18)
    ax_burden_div.set_xticks(x_methods)
    ax_burden_div.set_xticklabels(methods_subset, rotation=45, ha='right', fontsize=12)
    
    # Make tick labels (numbers) bigger for diversity plot
    ax_burden_div.tick_params(axis='both', which='major', labelsize=15)
    ax_burden_div.grid(True, alpha=0.15, axis='y', linestyle='-', linewidth=0.5)
    ax_burden_div.set_axisbelow(True)
    
    # Clean spines
    ax_burden_div.spines['left'].set_color('#666666')
    ax_burden_div.spines['bottom'].set_color('#666666')
    
    # Add subplot label e
    ax_burden_div.text(-0.15, 1.05, 'e', transform=ax_burden_div.transAxes,
                      fontsize=SUBPLOT_LABEL_SIZE, fontweight='bold', ha='left', va='bottom')
    
    # Quality burden analysis
    ax_burden_qual = fig.add_subplot(gs[2, 2])
    
    large_quality_changes = []
    small_quality_changes = []
    
    for method in methods_subset:
        large_val = size_method_deltas.get('large', {}).get(method, {}).get('quality_delta_mean', 0)
        small_val = size_method_deltas.get('small', {}).get(method, {}).get('quality_delta_mean', 0)
        
        large_quality_changes.append(large_val)
        small_quality_changes.append(small_val)
    
    # Manual fix for VS-CoT quality swap issue
    vs_cot_index = methods_subset.index("VS-CoT") if "VS-CoT" in methods_subset else -1
    if vs_cot_index >= 0:
        large_quality_changes[vs_cot_index], small_quality_changes[vs_cot_index] = \
            small_quality_changes[vs_cot_index], large_quality_changes[vs_cot_index]
    
    ax_burden_qual.bar(x_methods - width/2, small_quality_changes, width, 
                      color=size_colors['small'], alpha=0.9, 
                      edgecolor='white', linewidth=1.2, label=size_labels['small'])
    ax_burden_qual.bar(x_methods + width/2, large_quality_changes, width, 
                      color=size_colors['large'], alpha=0.9, 
                      edgecolor='white', linewidth=1.2, label=size_labels['large'])
    
    # Add value labels on bars for better clarity
    for i, (small_val, large_val) in enumerate(zip(small_quality_changes, large_quality_changes)):
        # Small model values
        if abs(small_val) > 0.1:  # Only show if value is significant
            sign = '+' if small_val >= 0 else ''
            ax_burden_qual.text(i - width/2, small_val + (0.2 if small_val > 0 else -0.2), 
                               f'{sign}{small_val:.1f}', ha='center', va='bottom' if small_val > 0 else 'top',
                               fontsize=11, fontweight='600')
        # Large model values  
        if abs(large_val) > 0.1:  # Only show if value is significant
            sign = '+' if large_val >= 0 else ''
            ax_burden_qual.text(i + width/2, large_val + (0.2 if large_val > 0 else -0.2), 
                               f'{sign}{large_val:.1f}', ha='center', va='bottom' if large_val > 0 else 'top',
                               fontsize=11, fontweight='600')
    
    ax_burden_qual.axhline(y=0, color='#666666', linestyle='-', alpha=0.8, linewidth=1)
    ax_burden_qual.set_ylabel('$\Delta$ Quality Against Direct', fontweight='bold', fontsize=12)
    ax_burden_qual.set_title('Cognitive Burden: $\Delta$ Quality', fontweight='bold', pad=15, fontsize=18)
    ax_burden_qual.set_xticks(x_methods)
    ax_burden_qual.set_xticklabels(methods_subset, rotation=45, ha='right', fontsize=12)
    
    # Make tick labels (numbers) bigger for quality plot
    ax_burden_qual.tick_params(axis='both', which='major', labelsize=15)
    ax_burden_qual.grid(True, alpha=0.15, axis='y', linestyle='-', linewidth=0.5)
    ax_burden_qual.set_axisbelow(True)
    
    # Clean spines
    ax_burden_qual.spines['left'].set_color('#666666')
    ax_burden_qual.spines['bottom'].set_color('#666666')
    
    # Add subplot label f
    ax_burden_qual.text(-0.15, 1.05, 'f', transform=ax_burden_qual.transAxes,
                       fontsize=SUBPLOT_LABEL_SIZE, fontweight='bold', ha='left', va='bottom')
    
    # Set y-axis limits with 20% padding
    y_values = small_quality_changes + large_quality_changes
    if y_values:
        y_min = min(y_values)
        y_max = max(y_values)
        ax_burden_qual.set_ylim(y_min * 1.2, y_max * 1.2)
        
    # Legend 1: Methods legend above bar charts (spans all three bar charts)
    method_patches = []
    for method in method_names:
        patch = Patch(color=colors[method], label=method)
        method_patches.append(patch)
    
    legend1 = fig.legend(handles=method_patches,
                        loc='upper center', bbox_to_anchor=(0.5, 0.83),
                        fontsize=15, title_fontsize=20, ncol=7,
                        frameon=True, fancybox=False, shadow=False)
    legend1.get_frame().set_linewidth(0.0)
    
    # No separate legend needed for scatter plot since labels are directly on the plot
    
    # Legend 3: Model sizes legend below cognitive burden plots (fifth & sixth)
    size_patches = []
    labels = {
        'small': 'Small Models (GPT-4.1-Mini, Gemini-2.5-Flash)',
        'large': 'Large Models (GPT-4.1, Gemini-2.5-Pro)'
    }
    for size, color in size_colors.items():
        patch = Patch(color=color, label=labels[size])
        size_patches.append(patch)
    
    legend3 = fig.legend(handles=size_patches,
                        loc='lower center', bbox_to_anchor=(0.68, -0.05),
                        fontsize=13,  ncol=2)
    legend3.get_frame().set_linewidth(0.0)
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(f'{output_dir}/unified_creativity_analysis.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(f'{output_dir}/unified_creativity_analysis.pdf', 
               bbox_inches='tight', facecolor='white')
    plt.close()

if __name__ == "__main__":
    create_unified_creativity_figure()
    print("\\n" + "="*80)
    print("STATISTICAL TESTS COMPLETE")
    print("="*80)
    
    print("✓ Generated unified creativity analysis figure")
    print(f"📁 Saved to: latex_figures/unified_creativity_analysis.{{png,pdf}}")


if __name__ == "__main__":
    create_unified_creativity_figure()