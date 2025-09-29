import json
import os
from pathlib import Path
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from matplotlib import gridspec
# plt.style.use('seaborn-v0_8')

# Maps canonical method keys to (display name, matching substring in dir name)
METHOD_MAP = {
    "direct": ("Direct", "direct"),
    "direct_cot": ("CoT", "cot"),
    "sequence": ("Sequence", "sequence"),
    "multi_turn": ("Multi-turn", "multi_turn"),
    "vs_standard": ("VS-Standard", "vs_standard"),
    "vs_cot": ("VS-CoT", "vs_cot"),
    "vs_multi": ("VS-Multi", "vs_multi"),
}


# COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
colors = {
    'direct': '#E8F4FD',   # Very light blue
    'cot':      '#B8E0F5',   # Light blue
    'multi_turn': '#7CC7EA',   # Medium blue
    'sequence':      '#D6E4FF',   # Absolute blue (strong blue, e.g. #1976D2 or #1565C0, but here a lighter blue for "absolute")
    'vs_standard':         '#FFCCCB',   # Light red
    'vs_cot':    '#FF9999',   # Medium red
    'vs_multi':  '#FF6B6B',   # Absolute red
}
# Use different edge colors for baselines and VS methods
edge_colors = {
    'direct': '#4A90E2',
    'cot': '#4A90E2',
    'sequence': '#4A90E2',
    'multi_turn': '#4A90E2',
    'vs_standard': '#FF6B6B',
    'vs_cot': '#FF6B6B',
    'vs_multi': '#FF6B6B'
}

# def bar_plot_all_methods_with_ttest_box(metrics_values, all_model_names, plot_metrics, metric_labels, baseline_methods, vs_methods, all_methods):
#     """
#     For each VS method, draw a bar plot with all methods, and include a text box at the top left summarizing t-test results (p-value, t-value, significance) for that VS method vs each baseline (reported individually).
#     Draw the bar plot with error bars and put the value on top of the error bar.
#     Ensure error bars do not exceed the proper range (e.g., for non-negative metrics, error bars do not go below zero).
#     Show the value as mean±std.
#     """
#     import numpy as np
#     from scipy.stats import ttest_ind

#     # Only do t-tests for these metrics
#     ttest_metrics = ["kl_divergence", "unique_recall_rate"]
#     y_label_map = {
#         "kl_divergence": "KL Divergence",
#         "unique_recall_rate": "Unique Recall",
#         "precision": "Precision"
#     }

#     # Define which metrics are non-negative (so error bars should not go below zero)
#     non_negative_metrics = ["precision", "unique_recall_rate"]

#     for metric in plot_metrics:
#         display_names = [METHOD_MAP[m][0] for m in all_methods]
#         means = []
#         stds = []
#         error_bars = []  # Will be a list of [lower, upper] for asymmetric error bars

#         # Collect means and stds for all methods, and adjust error bars as needed
#         for method in all_methods:
#             vals = []
#             for model_name in all_model_names:
#                 if model_name in metrics_values:
#                     for method_name, method_data in metrics_values[model_name].items():
#                         if METHOD_MAP[method][1] in method_name:
#                             vals.extend(method_data[metric])
#             mean_val = np.mean(vals) if vals else np.nan
#             std_val = np.std(vals) if vals else np.nan
#             stds.append(std_val)

#             # Adjust error bars so they do not exceed the proper range
#             if not np.isnan(mean_val) and not np.isnan(std_val):
#                 if metric in non_negative_metrics:
#                     # Lower error bar cannot go below zero
#                     lower = min(std_val, mean_val)
#                     upper = std_val
#                 else:
#                     lower = std_val
#                     upper = std_val
#             else:
#                 lower = std_val
#                 upper = std_val

#             means.append(mean_val)
#             error_bars.append([lower, upper])

#         # Convert error_bars to shape (2, N) for matplotlib's yerr
#         yerr = np.array(error_bars).T  # shape (2, N)

#         # For metrics that require t-tests, draw a separate plot for each VS method
#         if metric in ttest_metrics:
#             for vs in vs_methods:
#                 # Prepare t-test results for the box, one line per baseline
#                 box_lines = [f"Statistical Tests: {METHOD_MAP[vs][0]}"]
#                 for baseline in baseline_methods:
#                     # Gather baseline values
#                     vals_b = []
#                     for model_name in all_model_names:
#                         if model_name in metrics_values:
#                             for method_name, method_data in metrics_values[model_name].items():
#                                 if METHOD_MAP[baseline][1] in method_name:
#                                     vals_b.extend(method_data[metric])
#                     # Gather VS method values
#                     vals_v = []
#                     for model_name in all_model_names:
#                         if model_name in metrics_values:
#                             for method_name, method_data in metrics_values[model_name].items():
#                                 if METHOD_MAP[vs][1] in method_name:
#                                     vals_v.extend(method_data[metric])
#                     # Prepare t-test result for this baseline
#                     if vals_b and vals_v:
#                         t_stat, p_val = ttest_ind(vals_b, vals_v, equal_var=False)
#                         if p_val < 0.001:
#                             sig = '***'
#                         elif p_val < 0.01:
#                             sig = '**'
#                         elif p_val < 0.05:
#                             sig = '*'
#                         else:
#                             sig = 'ns'
#                         box_lines.append(f"vs {METHOD_MAP[baseline][0]}: p={p_val:.4g} (t={t_stat:.2f}) {sig}")
#                     else:
#                         box_lines.append(f"vs {METHOD_MAP[baseline][0]}: Not enough data")
#                 box_text = '\n'.join(box_lines)

#                 # Plot
#                 fig, ax = plt.subplots(figsize=(10, 6))
#                 x = np.arange(len(all_methods))
#                 bars = ax.bar(display_names, means, yerr=yerr, capsize=5, color=[colors[m] for m in all_methods], alpha=0.8, edgecolor=[edge_colors[m] for m in all_methods], linewidth=1)

#                 # Add hatches to VS methods (last 3 bars)
#                 for i, bar in enumerate(bars[-3:], start=len(bars)-3):
#                     bar.set_hatch('///')

#                 # Add value labels on bars
#                 for bar, mean, std in zip(bars, means, stds):
#                     height = bar.get_height()
#                     std_val = std if not np.isnan(std) else 0.0
#                     ax.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.05,
#                             f'{height:.2f}±{std_val:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

#                 ax.set_xlabel('Methods', fontsize=16, fontweight='bold')
#                 ax.set_ylabel(metric_labels[metric], fontsize=16, fontweight='bold')
#                 ax.set_title(f'{metric_labels[metric]} - Average Across All Models', fontsize=18, fontweight='bold', pad=20)
#                 ax.grid(True, alpha=0.3, axis='y')
#                 ax.tick_params(axis='x', labelsize=14)
#                 ax.tick_params(axis='y', labelsize=14)
#                 plt.xticks(rotation=0)
#                 # Set y-axis limits to provide some margin above the highest bar + error bar
#                 max_height = max([mean + err[1] if not np.isnan(mean) and not np.isnan(err[1]) else 0 for mean, err in zip(means, error_bars)])
#                 ax.set_ylim(0, max_height * 1.25 if max_height > 0 else 1)

#                 # Highlight best performing method
#                 if metric == 'kl_divergence':  # Lower is better
#                     best_idx = np.nanargmin(means)
#                 else:  # Higher is better
#                     best_idx = np.nanargmax(means)
#                 bars[best_idx].set_edgecolor('red')
#                 bars[best_idx].set_linewidth(3)

#                 # Add p-test results annotation in top left or right
#                 if metric == 'kl_divergence':
#                     ax.text(0.98, 0.98, box_text, transform=ax.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right', multialignment='left', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8), fontweight='bold')
#                 else:
#                     ax.text(0.02, 0.98, box_text, transform=ax.transAxes, fontsize=11, verticalalignment='top', horizontalalignment='left', multialignment='left', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8), fontweight='bold')

#                 plt.tight_layout()
#                 # plt.savefig(f"bias_metrics_{metric}_{METHOD_MAP[vs][0]}_vs_all_baselines.png", dpi=300, bbox_inches='tight')
#                 plt.savefig(f"{metric}_{vs}_vs_all_baselines.pdf", bbox_inches='tight')
#                 plt.close()
#                 print(f"✓ Saved {metric_labels[metric]} method average plot for {METHOD_MAP[vs][0]}")
#         else:
#             # For metrics that do not require t-tests, just plot the bar chart with error bars
#             # box_text = "No t-test performed for this metric."
#             fig, ax = plt.subplots(figsize=(10, 6))
#             x = np.arange(len(all_methods))
#             bars = ax.bar(display_names, means, yerr=yerr, capsize=5, color=COLORS[:len(all_methods)], alpha=0.8, edgecolor='black', linewidth=1)

#             # Add hatches to VS methods (last 3 bars)
#             for i, bar in enumerate(bars[-3:], start=len(bars)-3):
#                 bar.set_hatch('///')

#             # Add value labels on bars
#             for bar, mean, std in zip(bars, means, stds):
#                 height = bar.get_height()
#                 std_val = std if not np.isnan(std) else 0.0
#                 ax.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.005,
#                         f'{height:.2f}±{std_val:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

#             ax.set_xlabel('Methods', fontsize=16, fontweight='bold')
#             ax.set_ylabel(metric_labels[metric], fontsize=16, fontweight='bold')
#             ax.set_title(f'{metric_labels[metric]} - Average Across All Models', fontsize=18, fontweight='bold', pad=20)
#             ax.grid(True, alpha=0.3, axis='y')
#             ax.tick_params(axis='x', labelsize=14)
#             ax.tick_params(axis='y', labelsize=14)
#             plt.xticks(rotation=0)

#             # Highlight best performing method
#             if metric == 'kl_divergence':  # Lower is better
#                 best_idx = np.nanargmin(means)
#             else:  # Higher is better
#                 best_idx = np.nanargmax(means)
#             bars[best_idx].set_edgecolor('red')
#             bars[best_idx].set_linewidth(3)

#             # Add annotation box
#             # ax.text(0.02, 0.98, box_text, transform=ax.transAxes, fontsize=14, verticalalignment='top', horizontalalignment='left', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8), fontweight='bold')

#             plt.tight_layout()
#             # plt.savefig(f"{metric}_no_ttest.png", dpi=300, bbox_inches='tight')
#             plt.savefig(f"{metric}_no_ttest.pdf", bbox_inches='tight')
#             plt.close()
#             print(f"✓ Saved {metric_labels[metric]} method average plot (no t-test)")


# def bar_plot_all_methods(metrics_values, all_model_names, plot_metrics, metric_labels, all_methods, vs_methods):
#     """
#     For each metric in plot_metrics, draw a grouped bar plot.
#     Each group is a sub-group in metrics_values (e.g., dataset/task).
#     For each sub-group, draw bars for all methods in all_methods (different color per method).
#     Legend at the bottom. Each plot is saved as a separate PDF using plt.savefig.
#     """
#     # Determine sub-groups (keys in metrics_values, e.g., "LCB", "GSM8K")
#     sub_groups = list(metrics_values.keys())
#     n_metrics = len(plot_metrics)
#     n_subgroups = len(sub_groups)
#     n_methods = len(all_methods)
#     print(all_methods)

#     for metric in plot_metrics:
#         # For each sub-group, collect mean and std for each method (averaged across models)
#         means = np.zeros((n_subgroups, n_methods))
#         stds = np.zeros((n_subgroups, n_methods))

#         for i, sub_group in enumerate(sub_groups):
#             sub_metrics = metrics_values[sub_group]
#             for j, method in enumerate(all_methods):
#                 # Only draw the bar for methods in all_methods
#                 vals = []
#                 if method in sub_metrics:
#                     for model_name, model_metrics in sub_metrics[method].items():
#                         vals.extend(model_metrics.get(metric, []))
#                 means[i, j] = np.mean(vals) if vals else np.nan
#                 stds[i, j] = np.std(vals) if vals else np.nan

#         # Plotting
#         fig, ax = plt.subplots(figsize=(max(8, 2.5*n_subgroups), 6))
#         bar_width = 0.8 / n_methods
#         x = np.arange(n_subgroups)

#         # Draw bars for each method in each sub-group (only for methods in all_methods)
#         bars = []
#         legend_handles = []
#         legend_labels = []
#         for j, method in enumerate(all_methods):
#             bar = ax.bar(
#                 x + j * bar_width - (bar_width * (n_methods-1)/2),
#                 means[:, j],
#                 bar_width,
#                 label=METHOD_MAP[method][0],
#                 color=COLORS[j % len(COLORS)],
#                 edgecolor='black',
#                 linewidth=1,
#                 alpha=0.85
#             )
#             # If method is in vs_methods, add hatch to all bars in this group
#             if method in vs_methods:
#                 for b in bar:
#                     b.set_hatch('///')
#             bars.append(bar)
#             # Only add to legend if not already present (avoid accidental duplicates)
#             legend_handles.append(bar)
#             legend_labels.append(METHOD_MAP[method][0])
#             # print(method)
#             # print(METHOD_MAP[method][0])

#         # Add value labels on bars
#         for j, bar_group in enumerate(bars):
#             for i, bar in enumerate(bar_group):
#                 height = bar.get_height()
#                 # std_val = stds[i, j] if not np.isnan(stds[i, j]) else 0.0
#                 if not np.isnan(height):
#                     ax.text(
#                         bar.get_x() + bar.get_width()/2., height + 0.005,
#                         f'{height:.2f}',
#                         ha='center', va='bottom', fontsize=11, fontweight='bold'
#                     )

#         # Set x-ticks and labels
#         ax.set_xticks(x)
#         ax.set_xticklabels(sub_groups, fontsize=14, fontweight='bold')
#         ax.set_xlabel('', fontsize=16, fontweight='bold')
#         ax.set_ylabel(metric_labels[metric], fontsize=16, fontweight='bold')
#         ax.set_title(f'{metric_labels[metric]} - Averaged Across Models', fontsize=18, fontweight='bold', pad=20)
#         ax.grid(True, alpha=0.3, axis='y')
#         ax.tick_params(axis='y', labelsize=14)
#         plt.xticks(rotation=0)

#         # Only show legend for methods in all_methods (should be n_methods entries)
#         ax.legend(
#             [h[0] for h in legend_handles],  # bar is a BarContainer, so h[0] is the patch
#             legend_labels,
#             loc='upper center',
#             bbox_to_anchor=(0.5, -0.13),
#             ncol=min(n_methods, 4),
#             fontsize=13,
#             frameon=False
#         )

#         plt.tight_layout(rect=[0, 0.05, 1, 1])
#         plt.savefig(f"{metric}_grouped_barplot.pdf", bbox_inches='tight')
#         plt.close(fig)
#         print(f"✓ Saved {metric_labels[metric]} grouped bar plot to {metric}_grouped_barplot.pdf.")

    
def bar_plot_all_methods(metrics_values, all_model_names, plot_metrics, metric_labels, all_methods, vs_methods):
    """
    Create a single figure with subplots for all metrics, sharing a common legend.
    Each subplot shows grouped bar plots for all methods across sub-groups.
    """
    # Determine sub-groups (keys in metrics_values, e.g., "LCB", "GSM8K")
    sub_groups = list(metrics_values.keys())
    n_metrics = len(plot_metrics)
    n_subgroups = len(sub_groups)
    n_methods = len(all_methods)
    print(all_methods)

    # Create a single figure with subplots
    fig, axes = plt.subplots(1, n_metrics, figsize=(8*n_metrics, 8))
    if n_metrics == 1:
        axes = [axes]
    plt.style.use('default')  # Start with clean slate
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
    
    # Store legend handles and labels (will be the same for all subplots)
    legend_handles = []
    legend_labels = []

    # Process each metric
    for metric_idx, metric in enumerate(plot_metrics):
        ax = axes[metric_idx]
        
        # For each sub-group, collect mean and std for each method (averaged across models)
        means = np.zeros((n_subgroups, n_methods))
        stds = np.zeros((n_subgroups, n_methods))

        for i, sub_group in enumerate(sub_groups):
            sub_metrics = metrics_values[sub_group]
            for j, method in enumerate(all_methods):
                # Only draw the bar for methods in all_methods
                vals = []
                if method in sub_metrics:
                    for model_name, model_metrics in sub_metrics[method].items():
                        vals.extend(model_metrics.get(metric, []))
                means[i, j] = np.mean(vals) if vals else np.nan
                stds[i, j] = np.std(vals) if vals else np.nan

        # Plotting
        bar_width = 0.8 / n_methods
        x = np.arange(n_subgroups)

        # Draw bars for each method in each sub-group (only for methods in all_methods)
        bars = []
        for j, method in enumerate(all_methods):
            bar = ax.bar(
                x + j * bar_width - (bar_width * (n_methods-1)/2),
                means[:, j],
                bar_width,
                label=METHOD_MAP[method][0],
                color=colors[method],
                edgecolor=edge_colors[method],
                linewidth=1,
                alpha=0.85
            )
            # # If method is in vs_methods, add hatch to all bars in this group
            # if method in vs_methods:
            #     for b in bar:
            #         b.set_hatch('///')
            bars.append(bar)
            
            # Store legend handles and labels only once (from first subplot)
            if metric_idx == 0:
                legend_handles.append(bar)
                legend_labels.append(METHOD_MAP[method][0])

        # Add value labels on bars
        for j, bar_group in enumerate(bars):
            for i, bar in enumerate(bar_group):
                height = bar.get_height()
                if not np.isnan(height):
                    ax.text(
                        bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=12, fontweight='bold'
                    )

        # Set x-ticks and labels
        ax.set_xticks(x)
        ax.set_xticklabels(sub_groups, fontsize=32)
        # ax.set_xlabel('', fontsize=24, fontweight='bold')
        # ax.set_ylabel('', fontsize=24, fontweight='bold')
        ax.set_title(f'{metric_labels[metric]}', fontsize=36, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='y', labelsize=24)
        ax.tick_params(axis='x', labelsize=24)
        ax.set_ylim(0, max(means[:, j]) + max(stds[:, j]) + 0.15)
        plt.setp(ax.get_xticklabels(), rotation=0)

    # Add shared legend at the top
    fig.legend(
        [h[0] for h in legend_handles],  # bar is a BarContainer, so h[0] is the patch
        legend_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.05),
        ncol=min(n_methods, 6),
        fontsize=28,
        frameon=False
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig("latex/qualitative_tasks/synthetic_data_metrics_grouped_barplot.pdf", bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved combined grouped bar plot to synthetic_data_metrics_grouped_barplot.pdf.")


def draw_combined_metrics_plot(
    metrics_values, 
    all_model_names, 
    cosine_similarity_dict=None, 
    save_path=None
):
    """
    Draws a 2x2 grid of subplots:
    (a) IR Rate (incorrect answer rate)
    (b) Distinct-N 
    (c) Semantic Diversity
    (d) Cosine Similarity histogram + KDE for direct, sequence, vs-standard
    Each subplot is labeled (a), (b), (c), (d).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    # Methods and labels
    methods = ["direct", "sequence", "multi_turn", "vs_standard", "vs_cot", "vs_multi"]
    method_to_labels = {
        "direct": "Direct",
        "sequence": "Sequence",
        "multi_turn": "Multi-turn",
        "vs_standard": "VS-Standard",
        "vs_cot": "VS-CoT",
        "vs_multi": "VS-Multi"
    }
    
    # Colors aligned with method types
    colors = {
        'direct': '#E8F4FD',      # Very light blue (baseline)
        'cot': '#B8E0F5',         # Light blue (baseline)
        'sequence': '#7CC7EA',    # Medium blue (baseline)
        'multi_turn': '#4A90E2',  # Distinct blue (baseline)
        'vs_standard': '#FFCCCB', # light red
        'vs_cot': '#FF9999',      # medium red
        'vs_multi': '#FF6B6B'     # distinct red
    }
    edge_colors = {
        'direct': '#4A90E2',
        'cot': '#4A90E2', 
        'sequence': '#4A90E2',
        'multi_turn': '#4A90E2',
        'vs_standard': '#FF6B6B',
        'vs_cot': '#FF6B6B',
        'vs_multi': '#FF6B6B'
    }

    # Prepare data for each metric
    def get_means(metric_dict, metric_key):
        means = []
        for m in methods:
            if m in metric_dict:
                # Calculate mean across all models for this method
                all_values = []
                for model_name in all_model_names:
                    if model_name in metric_dict[m]:
                        # Handle both list and single value cases
                        metric_data = metric_dict[m][model_name].get(metric_key, [])
                        if isinstance(metric_data, list):
                            all_values.extend(metric_data)
                        else:
                            all_values.append(metric_data)
                means.append(np.mean(all_values) if all_values else 0)
            else:
                means.append(0)
        return means

    def get_means_std(metric_dict, metric_key):
        means = []
        stds = []
        for m in methods:
            if m in metric_dict:
                # Calculate mean and std across all models for this method
                all_values = []
                for model_name in all_model_names:
                    if model_name in metric_dict[m]:
                        # Handle both list and single value cases
                        metric_data = metric_dict[m][model_name].get(metric_key, [])
                        if isinstance(metric_data, list):
                            all_values.extend(metric_data)
                        else:
                            all_values.append(metric_data)
                if all_values:
                    means.append(np.mean(all_values))
                    stds.append(np.std(all_values))
                else:
                    means.append(0)
                    stds.append(0)
            else:
                means.append(0)
                stds.append(0)
        return means, stds

    # Get data for each metric
    ir_means = get_means(metrics_values, 'avg_ir_rate')
    distinct_means, distinct_stds = get_means_std(metrics_values, 'avg_distinct_n')
    diversity_means, diversity_stds = get_means_std(metrics_values, 'avg_diversity')

    # For cosine similarity, only plot for direct, sequence, vs_standard
    cos_methods = ["direct", "sequence", "vs_standard"]
    cos_labels = [method_to_labels[m] for m in cos_methods]
    if cosine_similarity_dict is not None:
        cos_data = [cosine_similarity_dict.get(m, []) for m in cos_methods]
    else:
        cos_data = [[0], [0], [0]]

    # Plotting
    plt.style.use('default')  # Start with clean slate
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'font.size': 24,
        'axes.labelsize': 28,
        'axes.titlesize': 32,
        'xtick.labelsize': 28,
        'ytick.labelsize': 28,
        'legend.fontsize': 28,
        'axes.linewidth': 0.8,
        'axes.edgecolor': '#666666',
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    plt.subplots_adjust(wspace=0.3, hspace=0.2)

    # Place the legend at the upper center of the whole figure
    handles = [plt.Rectangle((0,0),1,1, facecolor=colors[m], edgecolor=edge_colors[m]) for m in methods]
    labels = [method_to_labels[m] for m in methods]
    fig.legend(
        handles, labels,
        loc='upper center', bbox_to_anchor=(0.5, 1.05),
        fontsize=28, frameon=False, ncol=len(methods)
    )

    # (a) IR Rate (Incorrect Answer Rate)
    ax = axes[0, 0]
    bars = ax.bar(
        range(len(methods)), ir_means, 
        color=[colors[m] for m in methods], 
        edgecolor=[edge_colors[m] for m in methods], 
        alpha=0.9
    )
    ax.set_ylabel('Rate', fontweight='bold')
    ax.set_title('IR Rate ($\\uparrow$)', fontweight='bold', pad=20)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([method_to_labels[m] for m in methods], rotation=30, ha='right')
    ax.tick_params(axis='y', labelsize=24)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.2f}', 
                ha='center', va='bottom', fontweight='bold')
    ax.text(-0.1, 1.18, "a", transform=ax.transAxes, fontsize=44, fontweight='bold', va='top')

    # (b) Distinct-N
    ax = axes[0, 1]
    bars = ax.bar(
        range(len(methods)), distinct_means,
        color=[colors[m] for m in methods],
        edgecolor=[edge_colors[m] for m in methods],
        alpha=0.9
    )
    ax.set_ylabel('Distinct-N', fontweight='bold')
    ax.set_title('Distinct-N ($\\uparrow$)', fontweight='bold', pad=20)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([method_to_labels[m] for m in methods], rotation=30, ha='right')
    ax.tick_params(axis='y', labelsize=24)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_ylim(0, 1)
    for bar, mean in zip(bars, distinct_means):
        ax.text(
            bar.get_x() + bar.get_width()/2., 
            mean + 0.01, 
            f'{mean:.2f}',
            ha='center', va='bottom', fontweight='bold'
        )
    ax.text(-0.1, 1.18, "b", transform=ax.transAxes, fontsize=44, fontweight='bold', va='top')

    # (c) Semantic Diversity
    ax = axes[1, 0]
    bars = ax.bar(
        range(len(methods)), diversity_means,
        color=[colors[m] for m in methods],
        edgecolor=[edge_colors[m] for m in methods],
        alpha=0.9
    )
    ax.set_ylabel('Semantic Diversity', fontweight='bold')
    ax.set_title('Semantic Diversity ($\\uparrow$)', fontweight='bold', pad=20)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([method_to_labels[m] for m in methods], rotation=30, ha='right')
    ax.tick_params(axis='y', labelsize=24)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_ylim(0, 0.4)
    for bar, mean in zip(bars, diversity_means):
        ax.text(
            bar.get_x() + bar.get_width()/2., 
            mean + 0.001, 
            f'{mean:.2f}',
            ha='center', va='bottom', fontweight='bold'
        )
    ax.text(-0.1, 1.18, "c", transform=ax.transAxes, fontsize=44, fontweight='bold', va='top')

    # (d) Cosine Similarity (histogram + KDE)
    ax = axes[1, 1]
    idxs = [cos_methods.index(m) for m in ['direct', 'sequence', 'vs_standard']]
    data = [cos_data[i] for i in idxs]
    labels = [cos_labels[i] for i in idxs]
    bar_colors = ['#D5D1D1', '#7FBDDA', '#FF6B6B']  # Gray, Blue, Red
    kde_colors = ['gray', 'royalblue', 'deeppink']
    
    ax.hist(data, bins=50, alpha=0.5, color=bar_colors, label=labels, density=True, histtype='stepfilled', linewidth=2)
    for d, c in zip(data, kde_colors):
        try: 
            sns.kdeplot(d, color=c, lw=2, ax=ax)
        except: 
            pass
    
    ax.set_xlabel('Cosine Similarity', fontweight='bold')
    ax.set_ylabel('Density', fontweight='bold')
    ax.set_title('Cosine Similarity (Pairwise) ($\\downarrow$)', fontweight='bold', pad=20)
    ax.tick_params(axis='y', labelsize=24)
    ax.tick_params(axis='x', labelsize=24)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xlim(0.1, 1)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend(frameon=False, reverse=True, loc='upper right', fontsize=24, bbox_to_anchor=(1.02, 1.02))
    ax.text(-0.1, 1.18, "d", transform=ax.transAxes, fontsize=44, fontweight='bold', va='top')

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.savefig("latex/qualitative_tasks/synthetic_data_combined_metrics.pdf", bbox_inches='tight')
    plt.close()
    print(f"✓ Saved combined metrics plot to synthetic_data_combined_metrics.pdf")


def debug_print_data_structure(metrics_values, all_model_names):
    """Debug function to print the data structure"""
    print("\n=== Debug: Data Structure ===")
    for method in ["direct", "sequence", "vs_standard"]:
        if method in metrics_values:
            print(f"\nMethod: {method}")
            for model_name in all_model_names:
                if model_name in metrics_values[method]:
                    print(f"  Model: {model_name}")
                    for metric in ["avg_ir_rate", "avg_distinct_n", "avg_diversity"]:
                        data = metrics_values[method][model_name].get(metric, "NOT_FOUND")
                        print(f"    {metric}: {data} (type: {type(data)})")
        else:
            print(f"\nMethod: {method} - NOT_FOUND")


def read_metrics_values(folder, task_name, all_model_names, all_metrics):
    """
    Reads metric values from result files in the specified folder and task,
    organizing them into the global metrics_values dictionary.
    """
    metrics_values = {}
    for model_dir in os.listdir(folder):
        # Only process directories matching the task name
        if not model_dir.endswith(f"_{task_name}"):
            continue
        model_name = model_dir.replace(f"_{task_name}", "")
        if model_name not in all_model_names:
            continue

        evaluation_dir = Path(folder) / model_dir / "evaluation"
        if not evaluation_dir.exists():
            print(f"Warning: No evaluation directory found for {model_name}")
            continue

        # Iterate through all method directories within the evaluation directory
        for method_dir in evaluation_dir.iterdir():
            if not method_dir.is_dir():
                continue

            method_name = method_dir.name
            method_name = method_name.split()[0]
            # List of possible result files to look for
            results_files = [
                method_dir / "diversity_results.json",
                method_dir / "ngram_results.json",
                method_dir / "synthetic_data_quality_results.json"
            ]

            for rf in results_files:
                if rf.exists():
                    with open(rf, "r") as f:
                        data = json.load(f)
                    aggregate_metrics = data.get("overall_metrics", {})

                    # Map method_name to canonical key if possible
                    for canonical_key, (display_name, match_substr) in METHOD_MAP.items():
                        if method_name == match_substr:
                            method_name = canonical_key
                            break
                    # Initialize data structure for this model-method combination if needed
                    if method_name not in metrics_values:
                        metrics_values[method_name] = {}
                    if model_name not in metrics_values[method_name]:
                        metrics_values[method_name][model_name] = {metric: [] for metric in all_metrics}

                    for metric in all_metrics:
                        if metric in aggregate_metrics:
                            metrics_values[method_name][model_name][metric].append(aggregate_metrics[metric])
                else:
                    print(f"Warning: No results file found for {model_name} - {method_name}")
    
    return metrics_values


def main():
    folder_1 = "method_results_lcb"
    task_name_1 = "livecodebench"
    folder_2 = "method_results_gsm8k"
    task_name_2 = "gsm8k"

    all_model_names = [
        # "gpt-4.1-mini",
        "gpt-4.1",
        # "gemini-2.5-flash",
        # "gemini-2.5-pro",
        # "meta-llama_Llama-3.1-70B-Instruct",
        # "deepseek-r1",
        # "o3",
        # "claude-4-sonnet",
    ]
    
    # Define the four metrics we want to analyze
    metrics = ["avg_diversity", "avg_distinct_n", "avg_ir_rate"]
    
    # Only keep these metrics for plotting
    plot_metrics = ["avg_diversity", "avg_distinct_n", "avg_ir_rate"]
    metric_labels = {
        "avg_diversity": "Semantic Diversity ↑",
        "avg_distinct_n": "Distinct-N ↑",
        "avg_ir_rate": "IR Rate ↑"
    }

    # Group methods
    all_methods = [
        "direct",
        # "direct_cot",
        "sequence",
        "multi_turn",
        "vs_standard",
        "vs_cot",
        "vs_multi"
    ]

    # Collect all data
    lcb_metrics_values = read_metrics_values(folder_1, task_name_1, all_model_names, metrics)
    # print(lcb_metrics_values)
    gsm8k_metrics_values = read_metrics_values(folder_2, task_name_2, all_model_names, metrics)
    # print(gsm8k_metrics_values)
    metrics_values = {
        "LCB": lcb_metrics_values,
        "GSM8K": gsm8k_metrics_values
    }
    # print(metrics_values)
        
    # Load cosine similarity data if available
    cosine_similarity_dict = None
    with open("latex/qualitative_tasks/synthetic_data_gsm8k_similarity.json", "r") as f:
        similarity_data = json.load(f)
    # Map the similarity data to method names
    cosine_similarity_dict = {}
    if "sim_direct" in similarity_data:
        cosine_similarity_dict["direct"] = similarity_data["sim_direct"]
    if "sim_sequence" in similarity_data:
        cosine_similarity_dict["sequence"] = similarity_data["sim_sequence"]
    if "sim_vs" in similarity_data:
        cosine_similarity_dict["vs_standard"] = similarity_data["sim_vs"]
        
    print("✓ Loaded cosine similarity data")
    print(f"Available methods: {list(cosine_similarity_dict.keys())}")
        
    # baseline_methods = ["direct", "direct_cot", "sequence", "multi_turn"]
    vs_methods = ["vs_standard", "vs_cot", "vs_multi"]
    
    # Create the VS-Multi (vs_multi) metrics plot
    draw_combined_metrics_plot(
        metrics_values["GSM8K"],  # Use GSM8K data for the combined plot
        all_model_names,
        cosine_similarity_dict,
        save_path="latex/qualitative_tasks/synthetic_data_combined_metrics.pdf"
    )
    
    # # bar_plot_all_methods_with_ttest_box(metrics_values, all_model_names, plot_metrics, metric_labels, baseline_methods, vs_methods, all_methods)
    # bar_plot_all_methods(metrics_values, all_model_names, plot_metrics, metric_labels, all_methods, vs_methods)

    # debug_print_data_structure(metrics_values, all_model_names)


if __name__ == "__main__":
    main() 