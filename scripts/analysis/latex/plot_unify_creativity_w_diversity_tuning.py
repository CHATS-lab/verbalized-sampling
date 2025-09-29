#!/usr/bin/env python3

import matplotlib.pyplot as plt
import json
import os
import numpy as np
import re
import argparse
from pathlib import Path
from scipy import stats
from scipy.stats import ttest_ind
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from config import COLORS, EDGE_COLORS, RC_PARAMS

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

        # Determine significance level and marker
        if p_value < 0.001:
            significance_marker = "***"
            significant = True
        elif p_value < 0.01:
            significance_marker = "**"
            significant = True
        elif p_value < 0.05:
            significance_marker = "*"
            significant = True
        else:
            significance_marker = ""
            significant = False

        significant_results[method] = significance_marker if significant else False
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
                "diversity": diversity_avg * 100 * 2 if diversity_avg is not None else None,
                "diversity_std": diversity_std * 100 if diversity_std is not None else None,
                "rouge_l": rouge_l_avg * 100 if rouge_l_avg is not None else None,
                "rouge_l_std": rouge_l_std * 100 if rouge_l_std is not None else None,
                "quality": quality_avg * 100 if quality_avg is not None else None,
                "quality_std": quality_std * 100 if quality_std is not None else None
            }

        return results

    # Load poem experiment data
    base_dir = "generated_data/poem_experiments_final"
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

# Diversity tuning functions from plot_diversity_tuning.py
def load_diversity_data(base_path, model, task, method, prob_tuning_values):
    """Load diversity data for a specific method and prob_tuning values"""
    diversities = []
    actual_prob_values = []

    for prob_val in prob_tuning_values:
        if prob_val == -1:
            # prob_tuning=-1 corresponds to no probability tuning
            prob_str = "prob_tuning=-1"
        else:
            prob_str = f"prob_tuning={prob_val}"

        file_path = os.path.join(
            base_path,
            model,
            f"{model}_{task}",
            "evaluation",
            f"{method} [strict] (samples=5) ({prob_str})",
            "diversity_results.json"
        )

        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    diversity = data['overall_metrics']['avg_diversity'] * 2
                    diversities.append(diversity)
                    actual_prob_values.append(prob_val)
            except (json.JSONDecodeError, KeyError) as e:
                pass

    return actual_prob_values, diversities

def load_baseline_data(base_path, model, task, baseline_type):
    """Load baseline diversity data"""
    if baseline_type == "direct":
        baseline_path = os.path.join(
            base_path,
            model,
            f"{model}_{task}",
            "evaluation",
            "direct (samples=1)",
            "diversity_results.json"
        )
    elif baseline_type == "sequence":
        baseline_path = os.path.join(
            base_path,
            model,
            f"{model}_{task}",
            "evaluation",
            "sequence [strict] (samples=5)",
            "diversity_results.json"
        )
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")

    if os.path.exists(baseline_path):
        try:
            with open(baseline_path, 'r') as f:
                data = json.load(f)
                return data['overall_metrics']['avg_diversity'] * 2
        except (json.JSONDecodeError, KeyError) as e:
            return None
    else:
        return None

def plot_single_model_diversity_tuning(ax, base_path, model, task, prob_values, title, colors, edge_colors):
    """Plot diversity tuning for a single model with styling aligned to plot_unify_creativity.py"""

    # Load data for VS-Standard (structure_with_prob) and VS-Multi (VS-Multi (vs_multi))
    vs_standard_probs, vs_standard_divs = load_diversity_data(
        base_path, model, task, "vs_standard", prob_values
    )
    vs_multi_probs, vs_multi_divs = load_diversity_data(
        base_path, model, task, "vs_multi", prob_values
    )

    # Load baseline data
    direct_diversity = load_baseline_data(base_path, model, task, "direct")
    sequence_diversity = load_baseline_data(base_path, model, task, "sequence")
    
    # Print experiment results
    print(f"\n=== Diversity Tuning Results for {task.upper()} ===")
    print(f"Model: {model}")
    print(f"Base path: {base_path}")
    
    # Print baseline results
    if direct_diversity is not None:
        print(f"Direct baseline: {direct_diversity * 100:.2f}")
    if sequence_diversity is not None:
        print(f"Sequence baseline: {sequence_diversity * 100:.2f}")
    
    # Print VS-Standard results
    if vs_standard_probs and vs_standard_divs:
        print(f"\nVS-Standard (structure_with_prob) results:")
        for prob, div in zip(vs_standard_probs, vs_standard_divs):
            prob_display = 1.0 if prob == -1 else abs(prob)
            print(f"  Prob {prob_display:.3f}: {div * 100:.2f}")
    
    # Print VS-Multi results
    if vs_multi_probs and vs_multi_divs:
        print(f"\nVS-Multi (combined) results:")
        for prob, div in zip(vs_multi_probs, vs_multi_divs):
            prob_display = 1.0 if prob == -1 else abs(prob)
            print(f"  Prob {prob_display:.3f}: {div * 100:.2f}")
    
    # Print best results
    all_divs = []
    if vs_standard_divs:
        all_divs.extend(vs_standard_divs)
    if vs_multi_divs:
        all_divs.extend(vs_multi_divs)
    if all_divs:
        best_div = max(all_divs) * 100
        print(f"\nBest diversity score: {best_div:.2f}")
    
    print("=" * 50)

    # Convert prob values to x-axis values (handle prob_tuning=-1 as 1.0)
    def prob_to_x(prob_val):
        if prob_val == -1:
            return 1.0  # No tuning maps to 10^0
        return abs(prob_val)  # Use absolute value

    # Create x values for plotting
    vs_standard_x = []
    vs_multi_x = []

    if vs_standard_probs:
        vs_standard_x = [prob_to_x(p) for p in vs_standard_probs]
    if vs_multi_probs:
        vs_multi_x = [prob_to_x(p) for p in vs_multi_probs]

    # Check if we need a broken axis - only for joke task with large gaps
    use_broken_axis = False
    if (task == "joke" and direct_diversity is not None and
        sequence_diversity is not None):
        gap_size = abs((sequence_diversity - direct_diversity) * 100)
        # Only use broken axis if gap is very large (>10 points)
        if gap_size > 10:
            use_broken_axis = True

    if use_broken_axis:
        # Create broken y-axis: show Direct at bottom, then break, then main data at top
        direct_val = direct_diversity * 100

        # Determine main data range
        all_main_values = []
        if vs_standard_divs:
            all_main_values.extend([d * 100 for d in vs_standard_divs])
        if vs_multi_divs:
            all_main_values.extend([d * 100 for d in vs_multi_divs])
        all_main_values.append(sequence_diversity * 100)

        main_min = min(all_main_values) - 1
        main_max = max(all_main_values) + 1

        # Set up broken axis: Direct section (minimal) + break + main section
        break_height = 2.0  # Height of the break section
        direct_section_height = 0.5  # Minimal height for Direct baseline section

        # Transform function for y values
        def transform_y(y_val):
            if y_val <= direct_val + 1:
                # Direct section: map to bottom of plot
                return y_val - direct_val + direct_section_height
            else:
                # Main section: map to top of plot (after break)
                return y_val - main_min + direct_section_height + break_height

        # Set overall y limits
        total_height = direct_section_height + break_height + (main_max - main_min)
        ax.set_ylim(0, total_height)

    # Plot lines (multiply by 100 for percentage)
    if vs_standard_probs and vs_standard_divs:
        y_vals = [d * 100 for d in vs_standard_divs]
        if use_broken_axis:
            y_vals = [transform_y(y) for y in y_vals]

        ax.plot(vs_standard_x, y_vals, 'o-',
                linewidth=2, markersize=6,
                color=colors['VS-Standard'], markeredgecolor=edge_colors['VS-Standard'],
                markeredgewidth=1.2, alpha=0.9)

    if vs_multi_probs and vs_multi_divs:
        y_vals = [d * 100 for d in vs_multi_divs]
        if use_broken_axis:
            y_vals = [transform_y(y) for y in y_vals]

        ax.plot(vs_multi_x, y_vals, 's-',
                linewidth=2, markersize=6,
                color=colors['VS-Multi'], markeredgecolor=edge_colors['VS-Multi'],
                markeredgewidth=1.2, alpha=0.9)

    # Plot baseline horizontal lines (multiply by 100 for percentage) with annotations
    if direct_diversity is not None:
        y_pos = direct_diversity * 100
        if use_broken_axis:
            y_pos = transform_y(y_pos)

        ax.axhline(y=y_pos, color=colors['Direct'], linestyle='--',
                  linewidth=2, alpha=0.8)
        # Add annotation on the right side
        ax.text(0.0009, y_pos, 'Direct',
                verticalalignment='bottom', horizontalalignment='left',
                fontsize=14, fontweight='bold', color=colors['Direct'],
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.95, edgecolor='none'))

    if sequence_diversity is not None:
        y_pos = sequence_diversity * 100
        if use_broken_axis:
            y_pos = transform_y(y_pos)

        ax.axhline(y=y_pos, color=colors['Sequence'], linestyle='--',
                  linewidth=2, alpha=0.8)
        # Add annotation on the right side
        ax.text(0.0009, y_pos, 'Sequence',
                verticalalignment='bottom', horizontalalignment='left',
                fontsize=14, fontweight='bold', color=colors['Sequence'],
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.95, edgecolor='none'))

    # Add break indicators for broken axis
    if use_broken_axis:
        # Add zigzag break lines on y-axis
        break_y = direct_section_height + break_height / 2

        # Draw break symbols (diagonal lines) on the y-axis
        break_width = 0.015
        break_line_height = 0.4

        # Two diagonal lines to indicate break
        line1_y = break_y - break_line_height/2
        line2_y = break_y + break_line_height/2

        # First diagonal line
        ax.plot([-break_width, break_width], [line1_y, line1_y + break_line_height],
                'k-', linewidth=3, transform=ax.get_yaxis_transform(), clip_on=False)
        # Second diagonal line
        ax.plot([-break_width, break_width], [line2_y - break_line_height, line2_y],
                'k-', linewidth=3, transform=ax.get_yaxis_transform(), clip_on=False)

        # Custom y-axis ticks for broken axis
        # Direct section - show the exact Direct value
        direct_tick_pos = transform_y(direct_val)

        # Main section - show clear ticks around the main data
        main_tick_positions = []
        main_tick_labels = []

        # Add ticks for main data range with good coverage
        tick_values = []

        # Start from a nice round number just below sequence
        seq_val = sequence_diversity * 100
        start_val = int(seq_val) - 1
        end_val = int(main_max) + 1

        # Add ticks every 1-2 units depending on range
        step = 1 if (end_val - start_val) <= 6 else 2
        for val in range(start_val, end_val + 1, step):
            if val >= main_min and val <= main_max:
                tick_values.append(val)

        # Ensure we have the sequence value and data extremes
        if int(seq_val) not in tick_values:
            tick_values.append(int(seq_val))
        if int(main_min) not in tick_values:
            tick_values.append(int(main_min))
        if int(main_max) not in tick_values:
            tick_values.append(int(main_max))

        tick_values = sorted(set(tick_values))

        for val in tick_values:
            pos = transform_y(val)
            main_tick_positions.append(pos)
            main_tick_labels.append(str(val))

        # Combine all ticks
        all_positions = [direct_tick_pos] + main_tick_positions
        all_labels = [f'{direct_val:.0f}'] + main_tick_labels

        ax.set_yticks(all_positions)
        ax.set_yticklabels(all_labels)
        ax.tick_params(axis='y', which='major', labelsize=14)

    # Set x-axis to log scale with proper range and inversion
    ax.set_xscale('log')
    ax.set_xlim(1.2, 0.0008)

    # Labels and formatting with elegant styling
    ax.set_xlabel('VS Probability Threshold', fontweight='bold', fontsize=18)
    ax.set_ylabel('Diversity Score' if 'Poem' in title else '', fontweight='bold', fontsize=18)
    ax.set_title(title, fontweight='bold', pad=15, fontsize=18)

    # Elegant grid and spines
    ax.grid(True, alpha=0.15, axis='y', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines['left'].set_color('#666666')
    ax.spines['bottom'].set_color('#666666')

    # Set custom x-axis ticks with simple numeric labels and larger font size
    ax.set_xticks([1.0, 0.1, 0.01, 0.001])
    ax.set_xticklabels(['1', '0.1', '0.01', '0.001'])
    ax.tick_params(axis='x', which='major', labelsize=20)
    ax.tick_params(axis='y', which='major', labelsize=18)

    if not use_broken_axis:
        # Normal axis - calculate reasonable y-axis limits based on all data
        all_y_values = []
        if vs_standard_divs:
            all_y_values.extend([d * 100 for d in vs_standard_divs])
        if vs_multi_divs:
            all_y_values.extend([d * 100 for d in vs_multi_divs])
        if direct_diversity is not None:
            all_y_values.append(direct_diversity * 100)
        if sequence_diversity is not None:
            all_y_values.append(sequence_diversity * 100)

        if all_y_values:
            y_min = min(all_y_values) - 1
            y_max = max(all_y_values) + 1
            ax.set_ylim(y_min, y_max)

            # Add consistent y-axis tick formatting to match main creativity plot
            tick_min = int(np.floor(y_min))
            tick_max = int(np.ceil(y_max))
            
            # Custom step size for diversity tuning plots (step size 2 for g and h)
            # Check if this is subplot g or h by examining the title
            ax_title = ax.get_title()
            if 'Diversity Tuning - Poem' in ax_title or 'Diversity Tuning - Story' in ax_title:
                # Subplots g and h get 6 ticks with step size 2
                start_tick = (tick_min // 2) * 2  # Round to nearest multiple of 2
                tick_values = np.arange(start_tick, start_tick + 7*2, 2)  # 6 ticks with step size 2
            else:
                # Other plots get 6 tick labels with normal spacing
                tick_values = np.linspace(tick_min, tick_max, 6)
                tick_values = np.round(tick_values).astype(int)
            # Ensure tick values are integers and remove duplicates if any
            tick_values = np.round(tick_values).astype(int)
            tick_values = list(dict.fromkeys(tick_values))

            ax.yaxis.set_major_locator(plt.FixedLocator(tick_values))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))

def create_unified_creativity_with_diversity_tuning_figure(diversity_tuning_paths, output_dir="latex_figures"):
    """Create a unified 3x3 figure with creativity analysis and diversity tuning across tasks"""

    # Set up elegant styling
    plt.style.use('default')
    plt.rcParams.update(RC_PARAMS)

    # Load data from LaTeX tables and experiment directories
    poem_data = parse_latex_table_data("latex/results/poem.tex")
    joke_data = parse_latex_table_data("latex/results/joke.tex")
    story_data = parse_latex_table_data("latex/results/story.tex")

    # Load experiment data for scatter plot and cognitive burden analysis
    poem_results, results_by_size = load_experiment_data()

    # Create figure with better proportions for 3 rows (larger and more spaced)
    fig = plt.figure(figsize=(15, 18))
    gs = gridspec.GridSpec(4, 3, height_ratios=[0.15, 1, 1, 1], width_ratios=[1, 1, 1],
                          hspace=0.74, wspace=0.5, left=0.08, right=0.95, top=0.95, bottom=0.05)

    # Color setup
    method_names = ["Direct", "CoT", "Sequence", "Multi-turn", "VS-Standard", "VS-CoT", "VS-Multi"]

    # Row 1: Bar charts for each task (poem, story, joke)
    suffix_title = "($\\uparrow$)"
    tasks = [(f"Poem {suffix_title}", poem_data), (f"Story {suffix_title}", story_data), (f"Joke {suffix_title}", joke_data)]

    # Perform statistical tests for each task
    all_significance_results = {}
    for task_name, task_data in tasks:
        clean_task_name = task_name.replace(suffix_title, "")
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

        # Create elegant bars
        x_pos = np.arange(len(method_names))

        for i, (method, avg, std) in enumerate(zip(method_names, method_averages, method_stds)):
            ax.bar(i, avg, yerr=std,
                   color=COLORS[method], edgecolor=EDGE_COLORS[method],
                   linewidth=1.2, width=0.7, alpha=0.9,
                   error_kw={'elinewidth': 1.5, 'capsize': 3, 'capthick': 1.5, 'alpha': 0.8})

        # Formatting
        ax.set_title(f'{task_name}', fontweight='bold', pad=15, fontsize=18)
        ax.set_ylabel('Diversity Score' if col_idx == 0 else '', fontweight='bold', fontsize=18)
        ax.set_xticks(x_pos)

        # Add statistical significance markers
        clean_task_name = task_name.replace(suffix_title, "")
        significance_results = all_significance_results.get(clean_task_name, {})

        method_labels = []
        for i, method in enumerate(method_names):
            method_labels.append(method)
            if method in significance_results and significance_results[method]:
                y_pos = method_averages[i] + method_stds[i] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05
                # Get significance marker from statistical test
                marker = significance_results[method]
                ax.text(i, y_pos, marker, ha='center', va='bottom',
                       fontsize=16, fontweight='bold', color='red')

        ax.set_xticklabels(method_labels, rotation=45, ha='right')

        # Grid and styling
        ax.grid(True, alpha=0.15, axis='y', linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines['left'].set_color('#666666')
        ax.spines['bottom'].set_color('#666666')

        # Set y-axis limits
        if max(method_averages) > 0:
            min_with_error = min([avg - std for avg, std in zip(method_averages, method_stds) if avg > 0])
            max_with_error = max([avg + std for avg, std in zip(method_averages, method_stds)])

            range_val = max_with_error - min_with_error
            y_min = max(0, min_with_error - range_val * 0.1)
            y_max = max_with_error + range_val * 0.15

            if col_idx == 2:  # Joke subplot (subplot c) - use step size 3 with 6 ticks
                ax.set_ylim(12, 74)
                # Add consistent y-axis tick formatting for joke subplot (6 ticks with step size 3)
                tick_values = np.arange(20, 20 + 6*10, 10)  # 6 ticks with step size 3
                ax.yaxis.set_major_locator(plt.FixedLocator(tick_values))
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))
            elif col_idx == 1:  # Story subplot
                ax.set_ylim(17, 44)
                # Add consistent y-axis tick formatting for story subplot (6 ticks)
                tick_values = np.linspace(20, 40, 6)
                ax.yaxis.set_major_locator(plt.FixedLocator(tick_values))
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))
            else:
                ax.set_ylim(y_min, y_max)
                # Add consistent y-axis tick formatting for poem subplot (6 ticks with step size 4)
                tick_min = int(np.floor(y_min))
                # Create 6 ticks with step size 4 starting from a round number
                start_tick = (tick_min // 5) * 5  # Round to nearest multiple of 4
                tick_values = np.arange(start_tick, start_tick + 7*5, 5)  # 6 ticks with step size 4
                ax.yaxis.set_major_locator(plt.FixedLocator(tick_values))
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))

        # Add value labels
        for i, (avg, std) in enumerate(zip(method_averages, method_stds)):
            if avg > 0:
                y_pos = avg + std + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
                ax.text(i, y_pos, f'{avg:.1f}', ha='center', va='bottom',
                       fontsize=13, fontweight='600', alpha=0.9)

        # Add subplot labels a, b, c
        subplot_labels = ['a', 'b', 'c']
        ax.text(-0.15, 1.05, f'{subplot_labels[col_idx]}', transform=ax.transAxes,
                fontsize=SUBPLOT_LABEL_SIZE, fontweight='bold', ha='left', va='bottom')

    # Row 2, Col 1: Scatter plot
    ax_scatter = fig.add_subplot(gs[2, 0])

    # Calculate method averages for scatter plot from poem data
    method_scatter_data = {}

    for method in method_names:
        diversity_values = []
        quality_values = []

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

    # Plot scatter points
    for method in method_names:
        if method in method_scatter_data:
            data = method_scatter_data[method]

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
                            color=COLORS[method], marker=marker, s=size, alpha=alpha,
                             zorder=5, edgecolors=EDGE_COLORS[method], linewidth=linewidth)

            # Add labels with offset adjustments
            x_offset = 0
            ha_align = 'center'
            if method == 'Sequence':
                # x_offset = -0.65
                x_offset = -1.3
            elif method == 'Direct':
                # x_offset = 0.5
                x_offset = 1.5
            elif method == 'VS-Standard':
                # x_offset = 0.75
                x_offset = 1.2
            elif method == "Multi-turn":
                # x_offset = 0.55
                x_offset = 1.1

            ax_scatter.text(data['diversity'] + x_offset, data['quality'] - 0.5, method,
                           ha=ha_align, va='top', fontsize=15, fontweight='600',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))

    # Scatter plot formatting
    y_values = [data['quality'] for data in method_scatter_data.values()]
    if y_values:
        y_min = min(y_values) - 2
        y_max = max(y_values) + 1.3
        ax_scatter.set_ylim(y_min, y_max)
    x_values = [data['diversity'] for data in method_scatter_data.values()]
    if x_values:
        x_min = min(x_values) - 0.5
        x_max = max(x_values) + 1.3
        ax_scatter.set_xlim(x_min, x_max)

    ax_scatter.set_xlabel('Diversity Score', fontweight='bold', fontsize=18)
    ax_scatter.set_ylabel('Quality Score', fontweight='bold', fontsize=18)
    ax_scatter.set_title('Diversity vs. Quality (Poem)', fontweight='bold', pad=15, fontsize=18)
    ax_scatter.grid(True, alpha=0.15, linestyle='-', linewidth=0.5)
    ax_scatter.set_axisbelow(True)
    ax_scatter.spines['left'].set_color('#666666')
    ax_scatter.spines['bottom'].set_color('#666666')

    # Add Pareto optimal arrow
    ax_scatter.annotate('', xy=(0.98, 0.98), xytext=(0.80, 0.80),
                       xycoords='axes fraction', textcoords='axes fraction',
                       arrowprops=dict(arrowstyle='->', lw=3, color='red', alpha=0.7))
    ax_scatter.text(0.70, 0.95, 'Pareto optimal', transform=ax_scatter.transAxes,
                   fontsize=15, color='red', fontweight='bold', ha='center')

    ax_scatter.text(-0.15, 1.05, 'd', transform=ax_scatter.transAxes,
                   fontsize=SUBPLOT_LABEL_SIZE, fontweight='bold', ha='left', va='bottom')

    # Row 2, Cols 2-3: Cognitive burden analysis
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

    # Import seaborn for colors
    import seaborn as sns
    color = sns.color_palette("Paired")
    size_colors = {'small': color[8], 'large': color[9]}
    # Different purple scheme for plot f (quality burden analysis) - more distinct purples
    size_colors_f = {'small': '#E6E6FA', 'large': '#4B0082'}  # Lavender and indigo
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

    # Add value labels
    X_SMALL_SUBSET = -0.08
    X_LARGE_SUBSET = 0.03
    for i, (small_val, large_val) in enumerate(zip(small_diversity_changes, large_diversity_changes)):
        sign = '+' if small_val >= 0 else ''
        ax_burden_div.text(i - width/2 + X_SMALL_SUBSET, small_val + (0.2 if small_val > 0 else -0.2),
                            f'{sign}{small_val:.1f}', ha='center', va='bottom' if small_val > 0 else 'top',
                            fontsize=11, fontweight='600')
        sign = '+' if large_val >= 0 else ''
        ax_burden_div.text(i + width/2 + X_LARGE_SUBSET, large_val + (0.2 if large_val > 0 else -0.2),
                            f'{sign}{large_val:.1f}', ha='center', va='bottom' if large_val > 0 else 'top',
                            fontsize=11, fontweight='600')

    ax_burden_div.axhline(y=0, color='#666666', linestyle='-', alpha=0.8, linewidth=1)
    ax_burden_div.set_ylabel('$\\Delta$ Diversity (vs. Direct)', fontweight='bold', fontsize=18)
    ax_burden_div.set_title('Emergent Trend: $\\Delta$ in Diversity', fontweight='bold', pad=15, fontsize=18)
    ax_burden_div.set_xticks(x_methods)
    ax_burden_div.set_xticklabels(methods_subset, rotation=45, ha='right')
    ax_burden_div.grid(True, alpha=0.15, axis='y', linestyle='-', linewidth=0.5)
    ax_burden_div.set_axisbelow(True)
    ax_burden_div.spines['left'].set_color('#666666')
    ax_burden_div.spines['bottom'].set_color('#666666')

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
                      color=size_colors_f['small'], alpha=0.9,
                      edgecolor='#DDA0DD', linewidth=0, label=size_labels['small'])
    ax_burden_qual.bar(x_methods + width/2, large_quality_changes, width,
                      color=size_colors_f['large'], alpha=0.7,
                      edgecolor='#2E0854', linewidth=0, label=size_labels['large'])

    # Add value labels
    for i, (small_val, large_val) in enumerate(zip(small_quality_changes, large_quality_changes)):
        sign = '+' if small_val > 0 else ''
        ax_burden_qual.text(i - width/2 + X_SMALL_SUBSET, small_val + (0.2 if small_val > 0 else -0.2),
                            f'{sign}{small_val:.1f}', ha='center', va='bottom' if small_val > 0 else 'top',
                            fontsize=11, fontweight='600')
        sign = '+' if large_val > 0 else ''
        ax_burden_qual.text(i + width/2 + X_LARGE_SUBSET, large_val + (0.2 if large_val > 0 else -0.2),
                            f'{sign}{large_val:.1f}', ha='center', va='bottom' if large_val > 0 else 'top',
                            fontsize=11, fontweight='600')

    ax_burden_qual.axhline(y=0, color='#666666', linestyle='-', alpha=0.8, linewidth=1)
    ax_burden_qual.set_ylabel('$\\Delta$ Quality (vs. Direct)', fontweight='bold', fontsize=18)
    ax_burden_qual.set_title('Cognitive Burden: $\\Delta$ in Quality', fontweight='bold', pad=15, fontsize=18)
    ax_burden_qual.set_xticks(x_methods)
    ax_burden_qual.set_xticklabels(methods_subset, rotation=45, ha='right')
    ax_burden_qual.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))
    ax_burden_qual.grid(True, alpha=0.15, axis='y', linestyle='-', linewidth=0.5)
    ax_burden_qual.set_axisbelow(True)
    ax_burden_qual.spines['left'].set_color('#666666')
    ax_burden_qual.spines['bottom'].set_color('#666666')

    ax_burden_qual.text(-0.15, 1.05, 'f', transform=ax_burden_qual.transAxes,
                       fontsize=SUBPLOT_LABEL_SIZE, fontweight='bold', ha='left', va='bottom')

    # Set y-axis limits with padding
    y_values = small_quality_changes + large_quality_changes
    if y_values:
        y_min = min(y_values)
        y_max = max(y_values)
        ax_burden_qual.set_ylim(y_min * 1.2, y_max * 1.2)

    # Row 3: Diversity tuning plots for poem, story, joke
    diversity_tasks = [("poem", "Poem"), ("book", "Story"), ("joke", "Joke")]
    prob_values = [-1, 0.9, 0.5, 0.2, 0.05, 0.005, 0.001]
    # Only use gemini-2.5-flash for diversity tuning plots
    model = "google_gemini-2.5-flash"

    for col_idx, (task_key, task_title) in enumerate(diversity_tasks):
        ax = fig.add_subplot(gs[3, col_idx])

        # Find the appropriate diversity tuning data path
        base_path = None
        if task_key in diversity_tuning_paths:
            base_path = diversity_tuning_paths[task_key]

        if base_path and os.path.exists(base_path):
            # Plot only gemini-2.5-flash results
            # Use consistent title formatting (separate model name as annotation)
            title_text = f'Diversity Tuning - {task_title}'
            plot_single_model_diversity_tuning(ax, base_path, model, task_key,
                                             prob_values, title_text,
                                             COLORS, EDGE_COLORS)
            # Add model name as separate annotation text (not bolded)
            ax.text(0.5, 1.02, '(Gemini-2.5-Flash)', transform=ax.transAxes,
                   ha='center', va='top', fontsize=17, style='italic', alpha=0.8)
        else:
            print(f"No data available under {base_path}")
            # No data available - create empty plot with message
            ax.text(0.5, 0.5, f'No data available\nfor {task_title}',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=16, fontweight='bold')
            # Use consistent title formatting
            title_text = f'Diversity Tuning - {task_title}'
            ax.set_title(title_text, fontweight='bold', pad=15, fontsize=18)
            # Add model name as separate annotation text (not bolded)
            ax.text(0.5, 0.95, '(Gemini-2.5-Flash)', transform=ax.transAxes,
                   ha='center', va='top', fontsize=14, style='italic', alpha=0.8)

        # Add subplot labels g, h, i
        subplot_labels = ['g', 'h', 'i']
        ax.text(-0.15, 1.05, f'{subplot_labels[col_idx]}', transform=ax.transAxes,
                fontsize=SUBPLOT_LABEL_SIZE, fontweight='bold', ha='left', va='bottom')

    # Legends
    # Legend 1: Methods legend above bar charts
    method_patches = []
    for method in method_names:
        patch = Patch(color=COLORS[method], label=method)
        method_patches.append(patch)

    legend1 = fig.legend(handles=method_patches,
                        loc='upper center', bbox_to_anchor=(0.5, 0.885),
                        fontsize=18, title_fontsize=20, ncol=7,
                        frameon=True, fancybox=False, shadow=False)
    legend1.get_frame().set_linewidth(0.0)

    # Legend 2: Model sizes legend (positioned above cognitive burden plots e-f)
    size_patches = []
    labels = {
        'small': 'Small Models (GPT-4.1-Mini, Gemini-2.5-Flash)',
        'large': 'Large Models (GPT-4.1, Gemini-2.5-Pro)'
    }
    alphas = {'small': 0.7, 'large': 0.7}
    for size, color in size_colors.items():
        patch = Patch(color=color, alpha=alphas[size], label=labels[size])
        size_patches.append(patch)

    legend2 = fig.legend(handles=size_patches,
                        loc='center', bbox_to_anchor=(0.68, 0.56),
                        fontsize=14, ncol=2)
    legend2.get_frame().set_linewidth(0.0)

    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(f'{output_dir}/unified_creativity_w_diversity_tuning.png',
               dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(f'{output_dir}/unified_creativity_w_diversity_tuning.pdf',
               bbox_inches='tight', facecolor='white')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Create unified creativity analysis with diversity tuning')
    parser.add_argument('--poem-path', default="ablation_data/poem_experiments_diversity_tuning/",
                       help='Path to poem diversity tuning data (e.g., ablation_data/poem_experiments_diversity_tuning/)')
    parser.add_argument('--story-path', default="ablation_data/story_diversity_tuning/",
                       help='Path to story diversity tuning data (e.g., ablation_data/story_diversity_tuning/)')
    parser.add_argument('--joke-path', default="ablation_data/joke_diversity_tuning/",
                       help='Path to joke diversity tuning data (e.g., ablation_data/joke_diversity_tuning/)')

    args = parser.parse_args()

    # Set up diversity tuning paths
    diversity_tuning_paths = {}
    if args.poem_path:
        diversity_tuning_paths["poem"] = args.poem_path
    if args.story_path:
        diversity_tuning_paths["book"] = args.story_path
    if args.joke_path:
        diversity_tuning_paths["joke"] = args.joke_path

    # Create the unified figure
    create_unified_creativity_with_diversity_tuning_figure(diversity_tuning_paths)

    print("\n" + "="*80)
    print("UNIFIED CREATIVITY WITH DIVERSITY TUNING ANALYSIS COMPLETE")
    print("="*80)
    print("✓ Generated unified creativity analysis figure with diversity tuning")
    print(f"📁 Saved to: latex_figures/unified_creativity_w_diversity_tuning.{{png,pdf}}")

if __name__ == "__main__":
    main()