#!/usr/bin/env python3

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch


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
                    diversity = data['overall_metrics']['avg_diversity']
                    diversities.append(diversity)
                    actual_prob_values.append(prob_val)
                    print(f"  ✓ Loaded {method} prob={prob_val}: diversity={diversity:.4f}")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  ✗ Error loading {file_path}: {e}")
        else:
            print(f"  ✗ File not found: {file_path}")

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
                return data['overall_metrics']['avg_diversity']
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error loading {baseline_path}: {e}")
            return None
    else:
        print(f"Baseline file not found: {baseline_path}")
        return None


def print_diversity_results(base_path, task="joke"):
    """Print diversity results in formatted console output"""
    prob_values = [-1, 0.9, 0.5, 0.2, 0.05, 0.005, 0.001]
    models = ["openai_gpt-4.1", "google_gemini-2.5-flash"]
    model_names = ["GPT-4.1", "Gemini 2.5 Flash"]

    for i, (model, model_name) in enumerate(zip(models, model_names)):
        print(f"\n{model_name}")

        # Load baseline data
        direct_diversity = load_baseline_data(base_path, model, task, "direct")
        sequence_diversity = load_baseline_data(base_path, model, task, "sequence")

        if direct_diversity is not None:
            print(f"'Direct': {direct_diversity:.4f},")
        else:
            print("'Direct': Not found,")

        if sequence_diversity is not None:
            print(f"'Sequence': {sequence_diversity:.4f},")
        else:
            print("'Sequence': Not found,")

        # Load VS-Standard data (suppress debug output)
        print_orig = print
        def silent_print(*args, **kwargs): pass
        import builtins
        builtins.print = silent_print

        vs_standard_probs, vs_standard_divs = load_diversity_data(
            base_path, model, task, "vs_standard", prob_values
        )

        # Load VS-Multi data
        vs_multi_probs, vs_multi_divs = load_diversity_data(
            base_path, model, task, "vs_multi", prob_values
        )

        # Restore original print function
        builtins.print = print_orig

        for j, (prob, div) in enumerate(zip(vs_standard_probs, vs_standard_divs)):
            prob_str = "p=-1" if prob == -1 else f"p={prob}"
            comma = "," if j < len(vs_standard_probs) - 1 else ""
            print_orig(f"'VS-Standard ({prob_str})': {div:.4f}{comma}")

        for j, (prob, div) in enumerate(zip(vs_multi_probs, vs_multi_divs)):
            prob_str = "p=-1" if prob == -1 else f"p={prob}"
            comma = "," if j < len(vs_multi_probs) - 1 else ""
            print_orig(f"'VS-Multi ({prob_str})': {div:.4f}{comma}")


def plot_diversity_tuning(base_path, task="joke"):
    """Create 1x2 plot for both models with elegant styling"""

    # Style configuration matching latex/plot_unify_creativity.py
    RC_PARAMS = {
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 15,
        'xtick.labelsize': 15,
        'ytick.labelsize': 18,
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
    }

    COLORS = {
        'Direct': '#B8E0F5',
        'Sequence': '#6BB6FF',
        'VS-Standard': '#FFCCCB',
        'VS-Multi': '#FF6B6B'
    }

    EDGE_COLORS = {
        'Direct': '#4A90E2',
        'Sequence': '#4A90E2',
        'VS-Standard': '#FF6B6B',
        'VS-Multi': '#FF6B6B'
    }

    # Apply styling
    plt.style.use('default')
    plt.rcParams.update(RC_PARAMS)

    # Probability tuning values (from 10^0 to 10^-3)
    prob_values = [-1, 0.9, 0.5, 0.2, 0.05, 0.005, 0.001]

    # Models to compare
    models = ["openai_gpt-4.1", "google_gemini-2.5-flash"]
    model_names = ["GPT-4.1", "Gemini 2.5 Flash"]

    # Create figure with gridspec for legend
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[0.15, 1], hspace=0.3, wspace=0.3)

    # Create subplots
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])

    # Plot each model
    plot_single_model(ax1, base_path, models[0], task, prob_values, model_names[0], COLORS, EDGE_COLORS)
    plot_single_model(ax2, base_path, models[1], task, prob_values, model_names[1], COLORS, EDGE_COLORS)

    # Create legend above plots (only for line plots, not baselines)
    method_patches = [
        Patch(color=COLORS['VS-Standard'], label='VS-Standard'),
        Patch(color=COLORS['VS-Multi'], label='VS-Multi')
    ]

    legend = fig.legend(handles=method_patches,
                       loc='upper center', bbox_to_anchor=(0.5, 0.80),
                       fontsize=16, ncol=2,
                       frameon=False, columnspacing=3.0)
    legend.get_frame().set_linewidth(0.0)

    plt.savefig(f"{task}_diversity_tuning_comparison.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{task}_diversity_tuning_comparison.pdf", bbox_inches='tight', facecolor='white')


def plot_single_model(ax, base_path, model, task, prob_values, title, colors, edge_colors):
    """Plot diversity tuning for a single model with elegant styling and broken axis for jokes"""

    # Load data for VS-Standard (structure_with_prob) and VS-Multi (VS-Multi (vs_multi))
    vs_standard_probs, vs_standard_divs = load_diversity_data(
        base_path, model, task, "vs_standard", prob_values
    )
    vs_multi_probs, vs_multi_divs = load_diversity_data(
        base_path, model, task, "vs_multi", prob_values
    )

    # Debug: Print what data was loaded
    print(f"\nDebugging for {model} {task}:")
    print(f"VS-Standard: Found {len(vs_standard_probs)} points - {vs_standard_probs}")
    print(f"VS-Standard divs: {[f'{d:.4f}' for d in vs_standard_divs] if vs_standard_divs else 'None'}")
    print(f"VS-Multi: Found {len(vs_multi_probs)} points - {vs_multi_probs}")
    print(f"VS-Multi divs: {[f'{d:.4f}' for d in vs_multi_divs] if vs_multi_divs else 'None'}")

    # Load baseline data
    direct_diversity = load_baseline_data(base_path, model, task, "direct")
    sequence_diversity = load_baseline_data(base_path, model, task, "sequence")

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

    # Check if we need a broken axis - only for cases with large gaps
    use_broken_axis = False
    if (task == "joke" and direct_diversity is not None and
        sequence_diversity is not None):
        gap_size = abs((sequence_diversity - direct_diversity) * 100)
        # Only use broken axis if gap is very large (>10 points)
        if gap_size > 10:
            use_broken_axis = True
            print(f"Using broken axis for {task} - gap size: {gap_size:.1f}")

    # For poem or other tasks, always use normal axis
    if task != "joke":
        use_broken_axis = False

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

    # Plot lines (multiply by 100 for percentage) with elegant styling
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
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))

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
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))

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

        # Custom y-axis ticks for broken axis with clear labels before and after break
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

        # Make y-tick labels more visible
        ax.tick_params(axis='y', which='major', labelsize=18, colors='black')

    # Set x-axis to log scale with proper range and inversion
    ax.set_xscale('log')
    # Set limits with small buffer on both sides
    ax.set_xlim(1.2, 0.0008)

    # Labels and formatting with elegant styling (aligned with main script)
    ax.set_xlabel('VS Probability Threshold', fontweight='bold', fontsize=18)
    ax.set_ylabel('Diversity Score' if 'GPT-4.1' in title else '', fontweight='bold', fontsize=18)
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

    if use_broken_axis:
        # Ensure x-axis spine is visible for broken axis
        ax.spines['bottom'].set_visible(True)
        ax.spines['bottom'].set_color('#666666')
    else:
        # Normal axis - ensure proper y-axis limits
        # Calculate reasonable y-axis limits based on all data
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


def main():
    parser = argparse.ArgumentParser(description='Plot diversity tuning results')
    # parser.add_argument('data_path',
    #                    help='Path to ablation data directory (e.g., ablation_data/joke_diversity_tuning/, ablation_data/poem_experiments_diversity_tuning/, or ablation_data/story_diversity_tuning/)')

    # args = parser.parse_args()

    path = {
        "joke": "ablation_data/joke_diversity_tuning/",
        "poem": "ablation_data/poem_experiments_diversity_tuning/",
        "book": "ablation_data/story_diversity_tuning/",
    }
    for task in path:
        print_diversity_results(path[task], task)
        plot_diversity_tuning(path[task], task)

    # Print console output with diversity values
    print_diversity_results(path[task], task)

    # Create 2x1 plot comparing both models
    plot_diversity_tuning(path[task], task)

    print(f"\nPlot saved as {task}_diversity_tuning_comparison.png and .pdf")


if __name__ == "__main__":
    main()