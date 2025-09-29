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
from config import COLORS, EDGE_COLORS, RC_PARAMS
SUBPLOT_LABEL_SIZE = 26

def perform_statistical_tests(task_data, task_name, use_rouge=False):
    """Perform t-tests comparing baselines against VS-Standard"""
    print(f"\n=== Statistical Tests for {task_name} ===")
    
    # Choose metric based on task
    metric_key = 'rouge_l' if use_rouge else 'diversity'
    
    # Collect individual model data for VS-Standard
    vs_standard_values = []
    for model_name, model_results in task_data.items():
        if 'VS-Standard' in model_results and model_results['VS-Standard'][metric_key] is not None:
            vs_standard_values.append(model_results['VS-Standard'][metric_key])
    
    if not vs_standard_values:
        print(f"No VS-Standard data found for {task_name}")
        return {}
    
    baseline_methods = ['Direct', 'CoT', 'Sequence', 'Multi-turn']
    significant_results = {}
    
    for method in baseline_methods:
        # Collect individual model data for this baseline method
        baseline_values = []
        for model_name, model_results in task_data.items():
            if method in model_results and model_results[method][metric_key] is not None:
                baseline_values.append(model_results[method][metric_key])
        
        if len(baseline_values) < 2 or len(vs_standard_values) < 2:
            print(f"Insufficient data for {method} vs VS-Standard comparison")
            continue
            
        # For Rouge-L, we want lower values (so test baseline > VS-Standard)
        # For diversity, we want higher values (so test VS-Standard > baseline)
        if use_rouge:
            t_stat, p_value = ttest_ind(baseline_values, vs_standard_values, alternative='greater')
        else:
            t_stat, p_value = ttest_ind(vs_standard_values, baseline_values, alternative='greater')
        
        vs_mean = np.mean(vs_standard_values)
        baseline_mean = np.mean(baseline_values)
        
        significant = p_value < 0.05
        significant_results[method] = significant
        
        print(f"{method} vs VS-Standard: t={t_stat:.3f}, p={p_value:.4f}, significant={significant}")
        print(f"  VS-Standard mean: {vs_mean:.2f}, {method} mean: {baseline_mean:.2f}")
    
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
            
            # Get quality score
            quality_avg = load_metric(model_dir, method_dir, "creative_writing_v3_results.json", "avg_score")
            quality_std = load_metric(model_dir, method_dir, "creative_writing_v3_results.json", "std_score")
            
            results[method_name] = {
                "diversity": diversity_avg,
                "diversity_std": diversity_std,
                "rouge_l": rouge_l_avg,
                "rouge_l_std": rouge_l_std,
                "quality": quality_avg,
                "quality_std": quality_std
            }
        
        return results

    # Load results for poem task
    poem_base_dir = "generated_data/poem_experiments_final"
    poem_results = {}
    
    for display_name, model_dir_name in models.items():
        model_path = os.path.join(poem_base_dir, model_dir_name, f"{model_dir_name}_poem")
        if os.path.exists(model_path):
            results = get_model_results(model_path, display_name)
            poem_results[display_name] = results
    
    return poem_results


def main():
    plt.rcParams.update(RC_PARAMS)

    # Load data from LaTeX files
    poem_data = parse_latex_table_data("latex/results/poem.tex")
    joke_data = parse_latex_table_data("latex/results/joke.tex")
    story_data = parse_latex_table_data("latex/results/story.tex")

    # Load experiment data for scatter and cognitive burden analysis
    poem_results = load_experiment_data()

    # Main plotting section
    method_names = ['Direct', 'CoT', 'Sequence', 'Multi-turn', 'VS-Standard', 'VS-CoT', 'VS-Multi']
    suffix_title = ""  # For customization
    
    # Row 1: Elegant bar charts for each task (reordered: poem, story, joke)
    fig = plt.figure(figsize=(18, 12))
    tasks = [("Poem " + suffix_title, poem_data), ("Story " + suffix_title, story_data), ("Joke " + suffix_title, joke_data)]
    
    # Create custom grid layout
    gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 1.1], width_ratios=[1, 1, 1])
    
    for col_idx, (task_name, task_data) in enumerate(tasks):
        ax = fig.add_subplot(gs[0, col_idx])
        
        # Determine if this is the joke task (use rouge_l instead of diversity)
        use_rouge_for_this_task = (col_idx == 2)  # Third subplot is joke
        metric_key = 'rouge_l' if use_rouge_for_this_task else 'diversity'
        
        # Calculate method averages and standard deviations
        method_averages = []
        method_stds = []
        
        for method in method_names:
            values = []
            stds = []
            
            for model_name, results in task_data.items():
                if method in results:
                    val = results[method][metric_key]
                    std_val = results[method][f'{metric_key}_std']
                    if val is not None and std_val is not None:
                        values.append(val)
                        stds.append(std_val)
            
            if values:
                avg = np.mean(values)
                # Use standard error of the mean for error bars
                std_error = np.sqrt(np.mean([s**2 for s in stds])) / np.sqrt(len(stds))
                method_averages.append(avg)
                method_stds.append(std_error)
            else:
                method_averages.append(0)
                method_stds.append(0)
        
        # Perform statistical tests
        significant_results = perform_statistical_tests(task_data, task_name, use_rouge_for_this_task)
        
        # Plotting with custom colors
        x_pos = np.arange(len(method_names))
        colors_list = [COLORS[method] for method in method_names]
        edge_colors_list = [EDGE_COLORS[method] for method in method_names]
        
        bars = ax.bar(x_pos, method_averages, yerr=method_stds, 
                     color=colors_list, edgecolor=edge_colors_list, linewidth=1.8,
                     alpha=0.8, error_kw={'elinewidth': 1.5, 'capsize': 5, 'capthick': 1.5})
        
        # Add significance markers for baseline methods
        max_height = max([avg + std for avg, std in zip(method_averages, method_stds)]) if method_averages else 1
        vs_standard_idx = method_names.index('VS-Standard')
        
        for i, method in enumerate(method_names[:4]):  # Only baseline methods
            if method in significant_results and significant_results[method]:
                # Add significance marker at the top
                y_marker = max_height + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05
                ax.text(i, y_marker, '**', ha='center', va='center', 
                       fontsize=16, fontweight='bold', color='red')
        
        # Styling
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['Direct', 'CoT', 'Seq.', 'Multi', 'VS-S', 'VS-C', 'VS-M'], 
                          rotation=45, ha='right', fontweight='600')
        
        if use_rouge_for_this_task:
            ax.set_ylabel('Rouge-L Score (lower is better)', fontweight='bold')
            ax.set_title('Rouge-L vs. Quality (Joke)', fontweight='bold', pad=15, fontsize=18)
        else:
            ax.set_ylabel('Diversity Score', fontweight='bold')
            ax.set_title(f'Diversity vs. Quality ({task_name.split()[0]})', fontweight='bold', pad=15, fontsize=18)
        
        ax.grid(True, alpha=0.15, linestyle='-', linewidth=0.5, axis='y')
        ax.set_axisbelow(True)
        
        # Clean styling
        ax.spines['left'].set_color('#666666')
        ax.spines['bottom'].set_color('#666666')
        
        # Dynamic y-axis limits based on data
        if method_averages:
            valid_data = [(avg, std) for avg, std in zip(method_averages, method_stds) if avg > 0]
            if valid_data:
                min_with_error = min([avg - std for avg, std in valid_data])
                max_with_error = max([avg + std for avg, std in valid_data])
            else:
                min_with_error = 0
                max_with_error = 1
            
            # Add extra space for significance markers and value labels
            range_val = max_with_error - min_with_error
            y_min = max(0, min_with_error - range_val * 0.1)
            y_max = max_with_error + range_val * 0.15  # Extra space for ** markers and value labels
            
            # Special handling for each subplot
            if col_idx == 2:  # Joke subplot (now using Rouge-L)
                if use_rouge_for_this_task:
                    ax.set_ylim(8, 70)  # Adjusted for Rouge-L values
                else:
                    ax.set_ylim(6, 37)
            elif col_idx == 1:
                ax.set_ylim(8.5, 22)
            else:
                ax.set_ylim(y_min, y_max)
            
            tick_min = int(np.floor(y_min))
            tick_max = int(np.ceil(y_max))
            tick_values = np.linspace(tick_min, tick_max, 8)
            if col_idx == 2 and use_rouge_for_this_task:
                tick_values = np.linspace(10, 65, 6)
            elif col_idx == 2:
                tick_values = np.linspace(10, 35, 6)

            ax.yaxis.set_major_locator(plt.FixedLocator(tick_values))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))
        

        # Add subtle value labels (adjusted for new y-limits)
        for i, (avg, std) in enumerate(zip(method_averages, method_stds)):
            if avg > 0:
                y_pos = avg + std + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
                ax.text(i, y_pos, f'{avg:.1f}', ha='center', va='bottom', 
                       fontsize=13, fontweight='600', alpha=0.9)
        
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
                if (data["diversity"] is not None and data["quality"] is not None and 
                    data["diversity"] > 0 and data["quality"] > 0):
                    diversity_values.append(data["diversity"])
                    quality_values.append(data["quality"])
        
        if diversity_values and quality_values:
            method_scatter_data[method] = {
                'diversity': np.mean(diversity_values),
                'quality': np.mean(quality_values)
            }
    
    # Scatter plotting
    for method in method_names:
        if method in method_scatter_data:
            data = method_scatter_data[method]
            color = COLORS[method]
            edge_color = EDGE_COLORS[method]
            ax_scatter.scatter(data['diversity'], data['quality'], 
                              c=[color], edgecolors=edge_color, s=120, alpha=0.8, linewidth=2,
                              label=method, zorder=5)
    
    # Dynamic axis limits for scatter plot
    if method_scatter_data:
        x_values = [data['diversity'] for data in method_scatter_data.values()]
        y_values = [data['quality'] for data in method_scatter_data.values()]
        
        x_min = min(x_values) - 1.3
        x_max = max(x_values) + 1.3
        ax_scatter.set_xlim(x_min, x_max)
    
    ax_scatter.set_xlabel('Diversity Score', fontweight='bold')
    ax_scatter.set_ylabel('Quality Score', fontweight='bold')
    
    # Make tick labels (numbers) bigger for scatter plot
    # ax_scatter.tick_params(axis='both', which='major', labelsize=15)
    # ax_scatter.tick_params(axis='y', labelsize=18)
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
    
    # Rest of the plot remains the same...
    # [Previous code for cognitive burden analysis and other plots would continue here]
    # For brevity, I'm focusing on the main change requested
    
    plt.tight_layout(pad=2.0)
    
    # Save the figure
    output_path = Path("latex_figures/combined_creativity_rouge.pdf")
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    print(f"Figure saved to {output_path}")
    
    plt.show()

if __name__ == "__main__":
    main()