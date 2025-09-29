import json
import os
from pathlib import Path
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

# Maps canonical method keys to (display name, matching substring in dir name)
METHOD_MAP = {
    "direct": ("Direct", "direct"),
    "direct_cot": ("CoT", "cot"),
    "sequence": ("Sequence", "sequence"),
    "multi_turn": ("Multi-turn", "multi_turn"),
    "vs_standard": ("VS-Standard", "vs_standard"),
    "vs_cot": ("VS-CoT", "vs_cot"),
    "vs_combined": ("VS-Combined", "vs_multi"),
}


# def method_bar_plot(metrics_values, all_model_names, plot_metrics, metric_labels, all_methods):
#     """
#     Draw bar graph for all models and all methods, with each method in a different color. No error bars.
#     """
#     import matplotlib.cm as cm

#     for metric in plot_metrics:
#         display_names = [METHOD_MAP[m][0] for m in all_methods]
#         means = []
#         for method in all_methods:
#             vals = []
#             for model_name in all_model_names:
#                 if model_name in metrics_values:
#                     for method_name, method_data in metrics_values[model_name].items():
#                         if METHOD_MAP[method][1] in method_name:
#                             vals.extend(method_data[metric])
#             print(vals)
#             mean_val = np.mean(vals) if vals else np.nan
#             means.append(mean_val)

#         # Assign a different color to each method
#         cmap = cm.get_cmap('tab10')
#         colors = [cmap(i % 10) for i in range(len(all_methods))]

#         fig, ax = plt.subplots(figsize=(10, 5))
#         x = np.arange(len(all_methods))
#         # Plot each bar individually to assign colors
#         for i, (mean, color) in enumerate(zip(means, colors)):
#             ax.bar(
#                 display_names[i],
#                 mean,
#                 color=color,
#                 label=display_names[i]
#             )
#         ax.set_title(f"{metric_labels[metric]} (all models aggregated)")
#         ax.set_ylabel('Mean')
#         ax.set_ylim(0, 0.8)
#         # Optionally add a legend if you want to show method names/colors
#         # ax.legend()
#         plt.tight_layout()
#         plt.show()

                        

# def ttest_vs_vs_baseline(metrics_values, all_model_names, plot_metrics):
#     """
#     Calculate t-tests to see if any of the baseline methods are statistically significant compared to each VS method for each metric.
#     """
#     baseline_methods = ["direct", "direct_cot", "sequence", "multi_turn"]
#     vs_methods = ["vs_standard", "vs_cot", "vs_VS-Multi (vs_multi)"]
#     for metric in plot_metrics:
#         print(f"\n=== T-TESTS for {metric} ===")
#         for v in vs_methods:
#             for b in baseline_methods:
#                 vals_v = []
#                 vals_b = []
#                 for model_name in all_model_names:
#                     if model_name in metrics_values:
#                         for method_name, method_data in metrics_values[model_name].items():
#                             if METHOD_MAP[v][1] in method_name:
#                                 vals_v.extend(method_data[metric])
#                             if METHOD_MAP[b][1] in method_name:
#                                 vals_b.extend(method_data[metric])
#                 if vals_b and vals_v:
#                     t_stat, p_val = ttest_ind(vals_b, vals_v, equal_var=False)
                    
#                     # Determine significance level
#                     if p_val < 0.001:
#                         sig = '***'
#                     elif p_val < 0.01:
#                         sig = '**'
#                     elif p_val < 0.05:
#                         sig = '*'
#                     else:
#                         sig = 'ns'
                    
#                     print(f"{METHOD_MAP[b][0]} vs {METHOD_MAP[v][0]}: p={p_val:.4g} (t={t_stat:.2f}) {sig}")
#                 else:
#                     print(f"{METHOD_MAP[b][0]} vs {METHOD_MAP[v][0]}: Not enough data")


# def bar_plot_with_ttest(metrics_values, all_model_names, plot_metrics, metric_labels, baseline_methods, vs_methods):
#     """
#     For each baseline method, draw a bar plot comparing it to the three VS methods for each metric.
#     Annotate each VS bar with the t-test significance against the baseline.
#     """
#     import matplotlib.cm as cm
#     for metric in plot_metrics:
#         for baseline in baseline_methods:
#             methods_to_plot = [baseline] + vs_methods
#             display_names = [METHOD_MAP[m][0] for m in methods_to_plot]
#             means = []
#             sigs = [None]  # First is baseline, no sig
#             # Collect means and t-test results
#             for i, method in enumerate(methods_to_plot):
#                 vals = []
#                 for model_name in all_model_names:
#                     if model_name in metrics_values:
#                         for method_name, method_data in metrics_values[model_name].items():
#                             if METHOD_MAP[method][1] in method_name:
#                                 vals.extend(method_data[metric])
#                 mean_val = np.mean(vals) if vals else np.nan
#                 means.append(mean_val)
#                 # For VS methods, compute t-test vs baseline
#                 if i > 0:
#                     vals_b = []
#                     vals_v = []
#                     # Baseline values
#                     for model_name in all_model_names:
#                         if model_name in metrics_values:
#                             for method_name, method_data in metrics_values[model_name].items():
#                                 if METHOD_MAP[baseline][1] in method_name:
#                                     vals_b.extend(method_data[metric])
#                                 if METHOD_MAP[method][1] in method_name:
#                                     vals_v.extend(method_data[metric])
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
#                         sigs.append(sig)
#                     else:
#                         sigs.append('n/a')
#             # Plot
#             cmap = cm.get_cmap('tab10')
#             colors = [cmap(i % 10) for i in range(len(methods_to_plot))]
#             fig, ax = plt.subplots(figsize=(8, 5))
#             x = np.arange(len(methods_to_plot))
#             bars = ax.bar(x, means, color=colors)
#             ax.set_xticks(x)
#             ax.set_xticklabels(display_names)
#             ax.set_title(f"{metric_labels[metric]}: {METHOD_MAP[baseline][0]} vs VS methods")
#             ax.set_ylabel('Mean')
#             ax.set_ylim(0, 0.8)
#             # Annotate significance
#             for i, bar in enumerate(bars):
#                 height = bar.get_height()
#                 if i > 0:
#                     ax.text(bar.get_x() + bar.get_width()/2., height + 0.02, sigs[i], ha='center', va='bottom', fontsize=12)
#             plt.tight_layout()
#             plt.show()


def bar_plot_all_methods_with_ttest_box(metrics_values, all_model_names, plot_metrics, metric_labels, baseline_methods, vs_methods, all_methods):
    """
    For each baseline method, draw a bar plot with all methods, and include a text box at the top left summarizing t-test results (p-value, t-value, significance) for that baseline vs each VS method.
    """
    import matplotlib.cm as cm
    for metric in plot_metrics:
        for baseline in baseline_methods:
            display_names = [METHOD_MAP[m][0] for m in all_methods]
            means = []
            stds = []
            # Collect means and stds for all methods
            for method in all_methods:
                vals = []
                for model_name in all_model_names:
                    if model_name in metrics_values:
                        for method_name, method_data in metrics_values[model_name].items():
                            if METHOD_MAP[method][1] in method_name:
                                vals.extend(method_data[metric])
                mean_val = np.mean(vals) if vals else np.nan
                std_val = np.std(vals) if vals else np.nan
                means.append(mean_val)
                stds.append(std_val)
            # Prepare t-test results for the box
            box_lines = [f"Statistical Tests ({METHOD_MAP[baseline][0]}):"]
            for vs in vs_methods:
                vals_b = []
                vals_v = []
                for model_name in all_model_names:
                    if model_name in metrics_values:
                        for method_name, method_data in metrics_values[model_name].items():
                            if METHOD_MAP[baseline][1] in method_name:
                                vals_b.extend(method_data[metric])
                            if METHOD_MAP[vs][1] in method_name:
                                vals_v.extend(method_data[metric])
                if vals_b and vals_v:
                    t_stat, p_val = ttest_ind(vals_b, vals_v, equal_var=False)
                    if p_val < 0.001:
                        sig = '***'
                    elif p_val < 0.01:
                        sig = '**'
                    elif p_val < 0.05:
                        sig = '*'
                    else:
                        sig = 'ns'
                    box_lines.append(f"vs {METHOD_MAP[vs][0]}: p={p_val:.4g} (t={t_stat:.2f}) {sig}")
                else:
                    box_lines.append(f"vs {METHOD_MAP[vs][0]}: Not enough data")
            box_text = '\n'.join(box_lines)
            # Plot
            cmap = cm.get_cmap('tab10')
            colors = [cmap(i % 10) for i in range(len(all_methods))]
            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(len(all_methods))
            bars = ax.bar(x, means, color=colors)
            ax.set_xticks(x)
            ax.set_xticklabels(display_names, rotation=20)
            ax.set_title(f"{metric_labels[metric]} - Average Across All Models")
            ax.set_ylabel('Mean')
            ax.set_ylim(0, 0.7)
            # Add value labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, f"{height:.2f}", ha='center', va='bottom', fontsize=10)
            # Place the box at the top right of the image (figure area), left-aligned
            fig.text(0.97, 0.90, box_text, fontsize=11, verticalalignment='top', horizontalalignment='right',
                     multialignment='left',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            plt.subplots_adjust(right=0.80)
            plt.tight_layout()
            plt.savefig(f"factuality_metrics_{metric}_{METHOD_MAP[baseline][0]}.png")
            plt.show()


def plot_factual_method_averages(metrics_values, all_model_names, plot_metrics, metric_labels, all_methods):
    """
    Create bar charts showing average performance across all models for each method, with error bars, color, hatching, and significance annotation box.
    For the statistical test, use the best baseline (from direct, direct_cot, sequence, multi_turn) and compare each VS method to it.
    """
    import numpy as np
    from scipy.stats import ttest_ind

    baseline_methods = ["direct", "direct_cot", "sequence", "multi_turn"]
    vs_methods = ["vs_standard", "vs_cot", "vs_combined"]

    for metric in plot_metrics:
        display_names = [METHOD_MAP[m][0] for m in all_methods]
        means = []
        stds = []
        error_bars = []
        # Collect means and stds for all methods
        for method in all_methods:
            vals = []
            for model_name in all_model_names:
                if model_name in metrics_values:
                    for method_name, method_data in metrics_values[model_name].items():
                        if METHOD_MAP[method][1] in method_name:
                            vals.extend(method_data[metric])
            mean_val = np.mean(vals) if vals else np.nan
            std_val = np.std(vals) if vals else np.nan
            means.append(mean_val)
            stds.append(std_val)
            # Error bars: lower cannot go below zero
            lower = min(std_val, mean_val) if not np.isnan(mean_val) and not np.isnan(std_val) else std_val
            upper = std_val
            error_bars.append([lower, upper])
        yerr = np.array(error_bars).T

        # Find the best baseline method for this metric
        baseline_means = [means[all_methods.index(b)] for b in baseline_methods]
        if metric == 'first_response_accuracy' or metric == 'pass_at_k_accuracy':
            # Higher is better
            best_baseline_idx = np.nanargmax(baseline_means)
        else:
            best_baseline_idx = np.nanargmin(baseline_means)
        best_baseline_method = baseline_methods[best_baseline_idx]
        best_baseline_label = METHOD_MAP[best_baseline_method][0]
        best_baseline_data = []
        for model_name in all_model_names:
            if model_name in metrics_values:
                for method_name, method_data in metrics_values[model_name].items():
                    if METHOD_MAP[best_baseline_method][1] in method_name:
                        best_baseline_data.extend(method_data[metric])

        # For each VS method, compare to best baseline
        box_lines = [f"Statistical Tests (Best Baseline: {best_baseline_label})"]
        for vs in vs_methods:
            vs_data = []
            for model_name in all_model_names:
                if model_name in metrics_values:
                    for method_name, method_data in metrics_values[model_name].items():
                        if METHOD_MAP[vs][1] in method_name:
                            vs_data.extend(method_data[metric])
            if len(best_baseline_data) > 1 and len(vs_data) > 1:
                t_stat, p_val = ttest_ind(vs_data, best_baseline_data, equal_var=False)
                if p_val < 0.001:
                    sig_level = "***"
                elif p_val < 0.01:
                    sig_level = "**"
                elif p_val < 0.05:
                    sig_level = "*"
                else:
                    sig_level = "ns"
                box_lines.append(f"{METHOD_MAP[vs][0]} vs {best_baseline_label}: p={p_val:.4f} {sig_level}")
            else:
                box_lines.append(f"{METHOD_MAP[vs][0]} vs {best_baseline_label}: insufficient data")
        box_text = '\n'.join(box_lines)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(display_names, means, yerr=yerr, capsize=5, color=COLORS[:len(all_methods)], alpha=0.8, edgecolor='black', linewidth=1)
        # Add hatches to VS methods (last 3 bars)
        for i, bar in enumerate(bars[-3:], start=len(bars)-3):
            bar.set_hatch('///')
        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            std_val = std if not np.isnan(std) else 0.0
            ax.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.005,
                    f'{height:.2f}±{std_val:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax.set_xlabel('Methods', fontsize=16, fontweight='bold')
        ax.set_ylabel(metric_labels[metric], fontsize=16, fontweight='bold')
        ax.set_title(f'{metric_labels[metric]} - Average Across All Models', fontsize=18, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        plt.xticks(rotation=0)
        # Set y-axis limits to provide some margin above the highest bar + error bar
        max_height = max([mean + err[1] if not np.isnan(mean) and not np.isnan(err[1]) else 0 for mean, err in zip(means, error_bars)])
        ax.set_ylim(0, max_height * 1.25 if max_height > 0 else 1)
        # Highlight best performing method
        if metric == 'first_response_accuracy' or metric == 'pass_at_k_accuracy':
            best_idx = np.nanargmax(means)
        else:
            best_idx = np.nanargmin(means)
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(3)
        # Add annotation box
        ax.text(0.02, 0.98, box_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='left', multialignment='left', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8), fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{metric}_method_average.pdf", bbox_inches='tight')
        plt.close()
        print(f"✓ Saved {metric_labels[metric]} method average plot (factual)")


def generate_latex_factual_table(metrics_values, all_model_names, all_methods, metric_labels):
    """
    Generate a LaTeX table summarizing Top@1 and Pass@K accuracy (mean±std) for each method, and p-values (t-test vs. CoT) for VS variants.
    """
    import numpy as np
    from scipy.stats import ttest_ind

    # Table order and display names
    table_methods = [
        "direct",
        "direct_cot",
        "sequence",
        "multi_turn",
        "vs_standard",
        "vs_cot",
        "vs_combined"
    ]
    display_names = [METHOD_MAP[m][0] for m in table_methods]
    metrics = ["first_response_accuracy", "pass_at_k_accuracy"]
    metric_short = ["Top@1 Accuracy", "Pass@K Accuracy"]
    # Find which baseline is best (for bolding)
    baseline_methods = ["direct", "direct_cot", "sequence", "multi_turn"]
    baseline_means = []
    for m in baseline_methods:
        vals = []
        for model_name in all_model_names:
            if model_name in metrics_values:
                for method_name, method_data in metrics_values[model_name].items():
                    if METHOD_MAP[m][1] in method_name:
                        vals.extend(method_data[metrics[0]])
        mean_val = np.mean(vals) if vals else np.nan
        baseline_means.append(mean_val)
    best_baseline_idx = np.nanargmax(baseline_means)
    best_baseline = baseline_methods[best_baseline_idx]

    # Gather means and stds for all methods and metrics
    means = {m: {} for m in table_methods}
    stds = {m: {} for m in table_methods}
    for m in table_methods:
        for i, metric in enumerate(metrics):
            vals = []
            for model_name in all_model_names:
                if model_name in metrics_values:
                    for method_name, method_data in metrics_values[model_name].items():
                        if METHOD_MAP[m][1] in method_name:
                            vals.extend(method_data[metric])
            means[m][metric] = np.mean(vals) if vals else np.nan
            stds[m][metric] = np.std(vals) if vals else np.nan

    # Compute p-values for VS methods vs CoT (direct_cot)
    pvals = {m: {metric: '--' for metric in metrics} for m in table_methods}
    cot_data = {metric: [] for metric in metrics}
    for i, metric in enumerate(metrics):
        for model_name in all_model_names:
            if model_name in metrics_values:
                for method_name, method_data in metrics_values[model_name].items():
                    if METHOD_MAP['direct_cot'][1] in method_name:
                        cot_data[metric].extend(method_data[metric])
    for m in ["vs_standard", "vs_cot", "vs_combined"]:
        for i, metric in enumerate(metrics):
            vs_data = []
            for model_name in all_model_names:
                if model_name in metrics_values:
                    for method_name, method_data in metrics_values[model_name].items():
                        if METHOD_MAP[m][1] in method_name:
                            vs_data.extend(method_data[metric])
            if len(vs_data) > 1 and len(cot_data[metric]) > 1:
                t_stat, p_val = ttest_ind(vs_data, cot_data[metric], equal_var=False)
                pvals[m][metric] = f"{p_val:.2f}"
            else:
                pvals[m][metric] = '--'

    # Build LaTeX table
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lcc|cc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Top@1 Accuracy & Pass@K Accuracy & Top@1 $p$-value (vs. CoT) & Pass@K $p$-value (vs. CoT) \\")
    lines.append(r"\midrule")
    for m, disp in zip(table_methods, display_names):
        row = []
        # Bold the best baseline row
        is_bold = (m == best_baseline)
        for i, metric in enumerate(metrics):
            mean = means[m][metric]
            std = stds[m][metric]
            cell = f"{mean:.2f}$_{{\pm{std:.2f}}}$" if not np.isnan(mean) and not np.isnan(std) else "--"
            if is_bold:
                cell = f"\\textbf{{{cell}}}"
            row.append(cell)
        # p-values: only for VS methods
        if m in ["vs_standard", "vs_cot", "vs_combined"]:
            p1 = pvals[m][metrics[0]]
            p2 = pvals[m][metrics[1]]
        else:
            p1 = p2 = '--'
        if is_bold:
            disp = f"\\textbf{{{disp}}}"
            p1 = p2 = '--'
        lines.append(f"{disp} & {row[0]} & {row[1]} & {p1} & {p2} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\vspace{-0.5em}")
    lines.append(r"\caption{Top@1 and Pass@K accuracy ($x_{\pm y}$) for each method, and $p$-values (t-test vs. CoT) for VS variants.}")
    lines.append(r"\label{tab:compact_all_in_one}")
    lines.append(r"\end{table}")
    latex_table = '\n'.join(lines)
    print(latex_table)


def main():
    folder = "method_results_simple_qa"
    task_name = "simple_qa"

    all_model_names = [
        "gpt-4.1-mini",
        "gpt-4.1",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "meta-llama_Llama-3.1-70B-Instruct",
        "qwen3-23b",
        "deepseek-r1",
        "o3",
        "claude-4-sonnet",
    ]
    
    # Define the four metrics we want to analyze
    metrics = ["first_response_accuracy", "pass_at_k_accuracy"]
    
    # Only keep these metrics for plotting
    plot_metrics = ["first_response_accuracy", "pass_at_k_accuracy"]
    metric_labels = {
        "first_response_accuracy": "Top@1 Accuracy ↑",
        "pass_at_k_accuracy": "Pass@K Accuracy ↑"
    }

    # Group methods
    all_methods = [
        "direct",
        "direct_cot",
        "sequence",
        "multi_turn",
        "vs_standard",
        "vs_cot",
        "vs_combined"
    ]

    # Collect all data
    metrics_values = {}
    
    # Iterate through all model directories
    for model_dir in os.listdir(folder):
        if not model_dir.endswith(f"_{task_name}"):
            continue
        model_name = model_dir.replace(f"_{task_name}", "")
        evaluation_dir = Path(folder) / model_dir / "evaluation"
        if not evaluation_dir.exists():
            print(f"Warning: No evaluation directory found for {model_name}")
            continue
        # Iterate through all method directories
        for method_dir in evaluation_dir.iterdir():
            if not method_dir.is_dir():
                continue
            method_name = method_dir.name
            results_file = method_dir / "factuality_results.json"
            if not results_file.exists():
                print(f"Warning: No results file found for {model_name} - {method_name}")
                continue
            try:
                with open(results_file, "r") as f:
                    data = json.load(f)
                aggregate_metrics = data.get("overall_metrics", {})

                if method_name in METHOD_MAP.values():
                    method_name = METHOD_MAP[method_name][0]

                # Initialize data structure for this model-method combination
                if model_name not in metrics_values:
                    metrics_values[model_name] = {}
                if method_name not in metrics_values[model_name]:
                    metrics_values[model_name][method_name] = {metric: [] for metric in metrics}
                # Collect metric values from all prompts
                for metric in metrics:
                    if metric in aggregate_metrics:
                        metrics_values[model_name][method_name][metric].append(aggregate_metrics[metric])
            except Exception as e:
                print(f"Error reading {results_file}: {e}")

    # print(metrics_values)

    # method_with_error_bars(metrics_values, all_model_names, plot_metrics, metric_labels, all_methods)
    # method_bar_plot(metrics_values, all_model_names, plot_metrics, metric_labels, all_methods)

    # ttest_vs_vs_baseline(metrics_values, all_model_names, plot_metrics)

    baseline_methods = ["direct", "direct_cot", "sequence", "multi_turn"]
    vs_methods = ["vs_standard", "vs_cot", "vs_combined"]
    all_methods = [
        "direct",
        "direct_cot",
        "sequence",
        "multi_turn",
        "vs_standard",
        "vs_cot",
        "vs_combined"
    ]
    # bar_plot_all_methods_with_ttest_box(metrics_values, all_model_names, plot_metrics, metric_labels, baseline_methods, vs_methods, all_methods)

    # plot_factual_method_averages(metrics_values, all_model_names, plot_metrics, metric_labels, all_methods)
    
    generate_latex_factual_table(metrics_values, all_model_names, all_methods, metric_labels)


if __name__ == "__main__":
    main() 