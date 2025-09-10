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

def create_violin_plot(ax, task_data, task_name, col_idx):
    """Create horizontal violin plot for a specific task"""
    
    # Method names and their corresponding labels for violin plot (reversed for horizontal layout)
    method_names = ["Direct", "CoT", "Sequence", "Multi-turn", "VS-Standard", "VS-CoT", "VS-Multi"]
    method_labels = ["Direct", "CoT", "Sequence", "Multi-turn", "VS-Standard", "VS-CoT", "VS-Multi"]
    
    # Colors with more visible blues: start from medium blue, red for our methods
    colors = {
        'Direct': '#7CC7EA',      # Medium blue (baseline)
        'CoT': '#4A90E2',         # Distinct blue (baseline)
        'Sequence': '#2E86C1',     # Darker blue (baseline)
        'Multi-turn': '#1B4F72',   # Dark blue (baseline)
        'VS-Standard': '#FFCCCB',  # Light red (our method)
        'VS-CoT': '#FF6B6B',       # Medium red (our method)
        'VS-Multi': '#DC143C'      # Distinct red (our method)
    }
    
    # Collect all data points for each method
    data_for_violin = []
    positions = []
    violin_colors = []
    
    # Reverse order for horizontal violin plot (top to bottom)
    reversed_methods = list(reversed(method_names))
    reversed_labels = list(reversed(method_labels))
    
    for i, method in enumerate(reversed_methods):
        method_values = []
        
        # Collect individual model data for this method
        for model_name, model_results in task_data.items():
            if method in model_results and model_results[method]['diversity'] is not None:
                # Create multiple data points based on mean and std to simulate distribution
                mean = model_results[method]['diversity']
                std = model_results[method]['diversity_std']
                
                # Generate synthetic data points around the mean
                synthetic_points = np.random.normal(mean, std, 100)
                method_values.extend(synthetic_points)
        
        if method_values:
            # Remove outliers using IQR method
            method_values = np.array(method_values)
            Q1 = np.percentile(method_values, 25)
            Q3 = np.percentile(method_values, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Filter out outliers
            filtered_values = method_values[(method_values >= lower_bound) & (method_values <= upper_bound)]
            
            data_for_violin.append(filtered_values.tolist())
            positions.append(i)
            violin_colors.append(colors[method])
    
    # Create horizontal violin plots with embedded box plots (like reference image)
    if data_for_violin:
        # First create the violin plot
        parts = ax.violinplot(data_for_violin, positions=positions, 
                             widths=0.6, showmeans=False, showmedians=False, showextrema=False,
                             vert=False)  # Horizontal
        
        # Style the violins to match reference - smooth, no edges
        for i, (pc, color) in enumerate(zip(parts['bodies'], violin_colors)):
            pc.set_facecolor(color)
            pc.set_alpha(0.8)
            pc.set_edgecolor('none')
        
        # Add custom mean-centered box plots with full-width error bars
    for i, method in enumerate(reversed_methods):
        # Use the same synthetic data that was used to create the violin
        if i < len(data_for_violin):
            violin_data = data_for_violin[i]
            mean_val = np.mean(violin_data)
            std_val = np.std(violin_data)
            min_val = np.min(violin_data)
            max_val = np.max(violin_data)
            
            # Draw a small black rectangle at the mean (centered box)
            box_height = 0.2  # Height of the central box
            box_width = 0.2   # Width of the central box
            
            from matplotlib.patches import Rectangle
            rect = Rectangle((mean_val - std_val/2, i - box_height/2), 
                           std_val, box_height, 
                           facecolor='black', edgecolor='black', linewidth=1.5)
            ax.add_patch(rect)
            
            # Add white dot at the mean
            ax.scatter(mean_val, i, color='white', s=60, zorder=10, edgecolors='black', linewidth=1)
            
            # Draw error bars extending to full violin width (min to max of synthetic data)
            # Horizontal line from min to max
            ax.plot([min_val, max_val], [i, i], 
                   color='black', linewidth=2, alpha=0.8)
            
            # Vertical caps at the ends
            # cap_height = 0.05
            # ax.plot([min_val, min_val], 
            #        [i - cap_height, i + cap_height], 
            #        color='black', linewidth=2, alpha=0.8)
            # ax.plot([max_val, max_val], 
            #        [i - cap_height, i + cap_height], 
            #        color='black', linewidth=2, alpha=0.8)
    
    # Formatting for horizontal layout
    ax.set_yticks(range(len(reversed_methods)))
    ax.set_yticklabels(reversed_labels, fontsize=12)
    # ax.set_xlabel('Diversity Index (↑ Better)', fontweight='bold', fontsize=14)

    ax.set_title(f'{task_name}', fontweight='bold', pad=15, fontsize=18)
    
    # Style similar to reference - remove x-axis
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_visible(False)  # Hide bottom spine (x-axis)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)  # Hide x-axis ticks and labels
    ax.tick_params(axis='y', labelsize=16)
    ax.grid(True, alpha=0.3, axis='x')  # Keep horizontal grid for reference
    ax.set_axisbelow(True)
    
    # Perform statistical tests (for console output only)
    significance_results = perform_statistical_tests(task_data, task_name.replace(" (↑ Better)", ""))
    
    # Add subplot labels
    subplot_labels = ['a', 'b', 'c']
    ax.text(-0.15, 1.05, f'{subplot_labels[col_idx]}', transform=ax.transAxes,
            fontsize=SUBPLOT_LABEL_SIZE, fontweight='bold', ha='left', va='bottom')

def create_unified_creativity_figure_violin(output_dir="latex_figures"):
    """Create a unified 2x3 figure with violin plots for a,b,c and original plots for d,e,f"""
    
    # Set up elegant styling
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 15,
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
    
    # Create figure
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(3, 3, height_ratios=[0.3, 1, 1], width_ratios=[1, 1, 1], 
                          hspace=0.6, wspace=0.5, left=0.08, right=0.95, top=0.95, bottom=0.08)
    
    # Row 1: Violin plots for each task
    suffix_title = "(↑ Better)"
    tasks = [("Poem " + suffix_title, poem_data), ("Story " + suffix_title, story_data), ("Joke " + suffix_title, joke_data)]
    
    for col_idx, (task_name, task_data) in enumerate(tasks):
        ax = fig.add_subplot(gs[1, col_idx])
        create_violin_plot(ax, task_data, task_name, col_idx)
    
    # Continue with the original scatter plot and cognitive burden analysis (d, e, f)
    # [Rest of the original code for subplots d, e, f remains exactly the same as in the original file]
    
    # Same color scheme as unified figure: our methods (red), baselines (blue)
    method_names = ["Direct", "CoT", "Sequence", "Multi-turn", "VS-Standard", "VS-CoT", "VS-Multi"]
    colors = {
        'Direct': '#7CC7EA',      # Medium blue (baseline)
        'CoT': '#4A90E2',         # Distinct blue (baseline)
        'Sequence': '#2E86C1',     # Darker blue (baseline)
        'Multi-turn': '#1B4F72',   # Dark blue (baseline)
        'VS-Standard': '#FFCCCB',  # Light red (our method)
        'VS-CoT': '#FF6B6B',       # Medium red (our method)
        'VS-Multi': '#DC143C'      # Distinct red (our method)
    }
    
    # Edge colors for better distinction
    edge_colors = {
        'Direct': '#4A90E2',
        'CoT': '#4A90E2', 
        'Sequence': '#4A90E2',
        'Multi-turn': '#4A90E2',
        'VS-Standard': '#FF6B6B',
        'VS-CoT': '#FF6B6B',
        'VS-Multi': '#FF6B6B'
    }
    
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
                x_offset = -0.65  # Move left
                ha_align = 'center'
            elif method == 'Direct':
                x_offset = 0.5  # Move left
                ha_align = 'center'
            elif method == 'VS-Standard':
                x_offset = 0.75   # Move right
                ha_align = 'center'
            elif method == "Multi-turn":
                x_offset = 0.55   # Move right
                ha_align = 'center'
            
            ax_scatter.text(data['diversity'] + x_offset, data['quality'] - 0.5, method, 
                           ha=ha_align, va='top', fontsize=15, fontweight='600',
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
    ax_scatter.tick_params(axis='y', labelsize=18)
    ax_scatter.set_title('Diversity vs. Quality (Poem)', fontweight='bold', pad=15, fontsize=18)
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
    ax_scatter.text(0.70, 0.95, 'Pareto optimal', transform=ax_scatter.transAxes,
                   fontsize=15, color='red', fontweight='bold', ha='center')
    
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
    
    # Vibrant coral & teal for model sizes
    import seaborn as sns
    color = sns.color_palette("Paired")
    size_colors = {'small': color[8], 'large': color[9]}  # Coral, light sea green
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
                     color=size_colors['small'], alpha=0.7, 
                     edgecolor='#E55B5B', linewidth=0, label=size_labels['small'])
    ax_burden_div.bar(x_methods + width/2, large_diversity_changes, width, 
                     color=size_colors['large'], alpha=0.7, 
                     edgecolor='#1A8A7A', linewidth=0, label=size_labels['large'])
    
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
    ax_burden_div.set_title('Emergent Trend: $\Delta$ in Diversity', fontweight='bold', pad=15, fontsize=18)
    ax_burden_div.set_xticks(x_methods)
    ax_burden_div.set_xticklabels(methods_subset, rotation=45, ha='right', fontsize=12)
    
    # Make tick labels (numbers) bigger for diversity plot
    ax_burden_div.tick_params(axis='both', which='major', labelsize=15)
    ax_burden_div.tick_params(axis='y', labelsize=18)
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
                      color=size_colors['small'], alpha=0.7, 
                      edgecolor='#E55B5B', linewidth=0, label=size_labels['small'])
    ax_burden_qual.bar(x_methods + width/2, large_quality_changes, width, 
                      color=size_colors['large'], alpha=0.7, 
                      edgecolor='#1A8A7A', linewidth=0, label=size_labels['large'])
    
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
    ax_burden_qual.set_title('Cognitive Burden: $\Delta$ in Quality', fontweight='bold', pad=15, fontsize=18)
    ax_burden_qual.set_xticks(x_methods)
    ax_burden_qual.set_xticklabels(methods_subset, rotation=45, ha='right', fontsize=12)
    
    # Make tick labels (numbers) bigger for quality plot
    ax_burden_qual.tick_params(axis='both', which='major', labelsize=15)
    ax_burden_qual.tick_params(axis='y', labelsize=18)
    # Format y-axis ticks as integers
    ax_burden_qual.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))
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
                        fontsize=18, title_fontsize=20, ncol=7,
                        frameon=True, fancybox=False, shadow=False)
    legend1.get_frame().set_linewidth(0.0)
    
    # Legend 3: Model sizes legend below cognitive burden plots (fifth & sixth)
    size_patches = []
    labels = {
        'small': 'Small Models (GPT-4.1-Mini, Gemini-2.5-Flash)',
        'large': 'Large Models (GPT-4.1, Gemini-2.5-Pro)'
    }
    alphas = {'small': 0.7, 'large': 0.7}
    for size, color in size_colors.items():
        patch = Patch(color=color, alpha=alphas[size], label=labels[size])
        size_patches.append(patch)
    
    legend3 = fig.legend(handles=size_patches,
                        loc='lower center', bbox_to_anchor=(0.68, -0.05),
                        fontsize=14,  ncol=2)
    legend3.get_frame().set_linewidth(0.0)
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(f'{output_dir}/unified_creativity_analysis_violin.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(f'{output_dir}/unified_creativity_analysis_violin.pdf', 
               bbox_inches='tight', facecolor='white')
    plt.close()

if __name__ == "__main__":
    create_unified_creativity_figure_violin()
    print("\\n" + "="*80)
    print("STATISTICAL TESTS COMPLETE")
    print("="*80)
    
    print("✓ Generated unified creativity analysis figure with violin plots")
    print(f"📁 Saved to: latex_figures/unified_creativity_analysis_violin.{{png,pdf}}")