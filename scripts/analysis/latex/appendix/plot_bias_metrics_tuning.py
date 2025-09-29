#!/usr/bin/env python3

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch


def load_metrics_data(base_path, model, task, method, prob_tuning_values):
    """Load metrics data for a specific method and prob_tuning values"""
    kl_divergences = []
    precisions = []
    coverage_ns = []
    actual_prob_values = []
    # print("Base path: ", base_path)
    
    for prob_val in prob_tuning_values:
        if prob_val == -1:
            # prob_tuning=-1 corresponds to no probability tuning
            prob_str = "prob_tuning=-1"
        else:
            prob_str = f"prob_tuning={prob_val}"
        
        if "gpt" in model:
            prob_def = "prob_def=explicit"
        elif "gemini" in model:
            if method == "vs_standard":
                prob_def = "prob_def=explicit"
            elif method == "vs_multi":
                prob_def = "prob_def=confidence"
            else:
                raise ValueError(f"Unknown method: {method}")
        else:
            raise ValueError(f"Unknown model: {model}")

        file_path = os.path.join(
            base_path,
            f"{model}_{task}",
            "evaluation",
            f"{method} [strict] (samples=20) ({prob_def}) ({prob_str})",
            "response_count_results.json"
        )
        
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    overall_metrics = data['overall_metrics']
                    kl_divergences.append(overall_metrics['average_kl_divergence'])
                    precisions.append(overall_metrics['average_precision'])
                    coverage_ns.append(overall_metrics['average_unique_recall_rate'])
                    actual_prob_values.append(prob_val)
                    print(f"  ✓ Loaded {method} prob={prob_val}: KL={overall_metrics['average_kl_divergence']:.4f}, Precision={overall_metrics['average_precision']:.4f}, Coverage-n={overall_metrics['average_unique_recall_rate']:.4f}")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  ✗ Error loading {file_path}: {e}")
        else:
            print(f"  ✗ File not found: {file_path}")

    return actual_prob_values, kl_divergences, precisions, coverage_ns


def load_baseline_data(base_path, model, task, baseline_type):
    """Load baseline metrics data"""
    if baseline_type == "direct":
        baseline_path = os.path.join(
            base_path,
            f"{model}_{task}",
            "evaluation",
            "direct (samples=20)",
            "response_count_results.json"
        )
    elif baseline_type == "sequence":
        baseline_path = os.path.join(
            base_path,
            f"{model}_{task}",
            "evaluation",
            "sequence [strict] (samples=20)",
            "response_count_results.json"
        )
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")

    if os.path.exists(baseline_path):
        try:
            with open(baseline_path, 'r') as f:
                data = json.load(f)
                overall_metrics = data['overall_metrics']
                return {
                    'kl_divergence': overall_metrics['average_kl_divergence'],
                    'precision': overall_metrics['average_precision'],
                    'coverage_n': overall_metrics['average_unique_recall_rate']
                }
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error loading {baseline_path}: {e}")
            return None
    else:
        print(f"Baseline file not found: {baseline_path}")
        return None


def plot_metrics_tuning(base_path, task="state_name"):
    """For each metric, draw a 1x2 plot (one row, two columns: one per model)"""

    # Style configuration matching latex/plot_unify_creativity.py
    RC_PARAMS = {
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'font.size': 11,
        'axes.labelsize': 15,
        'axes.titlesize': 18,
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

    # Probability tuning values
    prob_values = [-1, 0.9, 0.5, 0.1, 0.05, 0.01]

    # Models to compare
    models = ["gpt-4.1", "google_gemini-2.5-flash"]
    model_names = ["GPT-4.1", "Gemini-2.5-Flash"]

    # Metrics to plot
    metrics = ['kl_divergence', 'coverage_n']
    metric_titles = ['KL Divergence ($\\downarrow$)', 'Coverage-N ($\\uparrow$)']
    metric_ylabels = ['KL Divergence', 'Coverage-N']

    for i, metric in enumerate(metrics):
        # For each metric, create a 1x2 plot (one row, two columns) - more square aspect ratio
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        for j, (model, model_name) in enumerate(zip(models, model_names)):
            ax = axes[j]
            plot_single_metric(
                ax, base_path, model, task, prob_values,
                model_name, metric, metric_titles[i], metric_ylabels[i],
                COLORS, EDGE_COLORS, show_legend=(j == 0)
            )
            ax.set_title(f"{model_name}", fontsize=15, fontweight='bold')
        # Create legend above plots
        method_patches = [
            Patch(color=COLORS['VS-Standard'], label='VS-Standard'),
            Patch(color=COLORS['VS-Multi'], label='VS-Multi')
        ]
        legend = fig.legend(
            handles=method_patches,
            loc='upper center', bbox_to_anchor=(0.5, 1.05),
            fontsize=16, ncol=2,
            frameon=False, columnspacing=3.0
        )
        legend.get_frame().set_linewidth(0.0)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.subplots_adjust(top=0.85)
        # Save with metric name in filename
        plt.savefig(f"{task}_metrics_tuning_{metric}.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(f"{task}_metrics_tuning_{metric}.pdf", bbox_inches='tight', facecolor='white')
        plt.close(fig)


def plot_single_metric(ax, base_path, model, task, prob_values, title, metric, metric_title, ylabel, colors, edge_colors, show_legend):
    """Plot a single metric for a single model"""

    # Load data for VS-Standard (structure_with_prob) and VS-Multi (VS-Multi (vs_multi))
    vs_standard_probs, vs_standard_kl, vs_standard_prec, vs_standard_cov = load_metrics_data(
        base_path, model, task, "vs_standard", prob_values
    )
    vs_multi_probs, vs_multi_kl, vs_multi_prec, vs_multi_cov = load_metrics_data(
        base_path, model, task, "vs_multi", prob_values
    )

    # Select the appropriate metric data
    if metric == 'kl_divergence':
        vs_standard_data = vs_standard_kl
        vs_multi_data = vs_multi_kl
    elif metric == 'precision':
        vs_standard_data = vs_standard_prec
        vs_multi_data = vs_multi_prec
    elif metric == 'coverage_n':
        vs_standard_data = vs_standard_cov
        vs_multi_data = vs_multi_cov

    # Debug: Print what data was loaded
    print(f"\nDebugging for {model} {task} {metric}:")
    print(f"VS-Standard: Found {len(vs_standard_probs)} points - {vs_standard_probs}")
    print(f"VS-Standard data: {[f'{d:.4f}' for d in vs_standard_data] if vs_standard_data else 'None'}")
    print(f"VS-Multi: Found {len(vs_multi_probs)} points - {vs_multi_probs}")
    print(f"VS-Multi data: {[f'{d:.4f}' for d in vs_multi_data] if vs_multi_data else 'None'}")

    # Load baseline data
    # direct_data = load_baseline_data(base_path, model, task, "direct")
    sequence_data = load_baseline_data(base_path, model, task, "sequence")

    # Convert prob values to x-axis values
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

    # Plot lines with elegant styling
    if vs_standard_probs and vs_standard_data:
        ax.plot(vs_standard_x, vs_standard_data, 'o-',
                linewidth=2, markersize=6,
                color=colors['VS-Standard'], markeredgecolor=edge_colors['VS-Standard'],
                markeredgewidth=1.2, alpha=0.9, label='VS-Standard' if show_legend else "")

    if vs_multi_probs and vs_multi_data:
        ax.plot(vs_multi_x, vs_multi_data, 's-',
                linewidth=2, markersize=6,
                color=colors['VS-Multi'], markeredgecolor=edge_colors['VS-Multi'],
                markeredgewidth=1.2, alpha=0.9, label='VS-Multi' if show_legend else "")

    # # Plot baseline horizontal lines with annotations
    # if direct_data is not None:
    #     y_pos = direct_data[metric]
    #     ax.axhline(y=y_pos, color=colors['Direct'], linestyle='--',
    #               linewidth=2, alpha=0.8)
    #     # Add annotation on the right side
    #     ax.text(0.0009, y_pos, 'Direct',
    #             verticalalignment='bottom', horizontalalignment='left',
    #             fontsize=14, fontweight='bold', color=colors['Direct'],
    #             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))

    if sequence_data is not None:
        y_pos = sequence_data[metric]
        ax.axhline(y=y_pos, color=colors['Sequence'], linestyle='--',
                  linewidth=2, alpha=0.8)
        # Add annotation on the right side
        ax.text(0.007, y_pos, 'Sequence',
                verticalalignment='bottom', horizontalalignment='left',
                fontsize=14, fontweight='bold', color=colors['Sequence'],
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))

    # Set x-axis to log scale with proper range and inversion
    ax.set_xscale('log')
    # Set limits to match the tick positions exactly for full-width baseline
    ax.set_xlim(1.2, 0.007)

    # Labels and formatting with elegant styling
    ax.set_xlabel('VS Probability Threshold', fontweight='bold')
    ax.set_ylabel(ylabel if 'GPT-4.1' in title else '', fontweight='bold')
    ax.set_title(f"{title} - {metric_title}", fontweight='bold', pad=15, fontsize=16)

    # Elegant grid and spines
    ax.grid(True, alpha=0.15, axis='y', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines['left'].set_color('#666666')
    ax.spines['bottom'].set_color('#666666')

    # Set custom x-axis ticks with simple numeric labels
    ax.set_xticks([1.0, 0.1, 0.01])
    ax.set_xticklabels(['1', '0.1', '0.01'])

    # Set reasonable y-axis limits based on data
    all_y_values = []
    if vs_standard_data:
        all_y_values.extend(vs_standard_data)
    if vs_multi_data:
        all_y_values.extend(vs_multi_data)
    # if direct_data is not None:
    #     all_y_values.append(direct_data[metric])
    if sequence_data is not None:
        all_y_values.append(sequence_data[metric])

    if all_y_values:
        y_min = min(all_y_values) - 0.05 * (max(all_y_values) - min(all_y_values))
        y_max = max(all_y_values) + 0.05 * (max(all_y_values) - min(all_y_values))
        ax.set_ylim(y_min, y_max)


def main():
    data_path = "ablation_data/bias_experiments_prob_tuning"

    if "bias" in data_path:
        task = "state_name"
    else:
        raise ValueError("Please specify task and data path")

    # Create 3x2 plot comparing both models for three metrics
    plot_metrics_tuning(data_path, task)

    print(f"Plot saved as {task}_metrics_tuning_comparison.png and .pdf")


if __name__ == "__main__":
    main()
