import json
import os
from pathlib import Path
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

# Maps canonical method keys to (display name, matching substring in dir name)
METHOD_MAP = {
    "direct": ("Direct", "direct"),
    "direct_cot": ("Direct CoT", "direct_cot"),
    "sequence": ("Sequence", "sequence"),
    "multi_turn": ("Multi-turn", "multi_turn"),
    "vs_standard": ("VS Standard", "structure_with_prob"),
    "vs_cot": ("VS CoT", "chain_of_thought"),
    "vs_combined": ("VS Combined", "combined"),
}


def plot_error_bars_all_models_methods(metrics_values, all_model_names, plot_metrics, metric_labels, all_methods):
    """
    Draw error bars for each model-method pair for each metric.
    Group by model, color by method.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    n_methods = len(all_methods)
    n_models = len(all_model_names)
    colors = plt.cm.get_cmap('tab10', n_methods)

    for metric in plot_metrics:
        fig, ax = plt.subplots(figsize=(2.5 * n_models, 6))
        bar_width = 0.8 / n_methods
        x = np.arange(n_models)
        for i, method in enumerate(all_methods):
            match_substr = METHOD_MAP[method][1]
            means = []
            stds = []
            for model_name in all_model_names:
                vals = []
                if model_name in metrics_values:
                    for method_name, method_data in metrics_values[model_name].items():
                        if match_substr in method_name:
                            vals.extend(method_data[metric])
                mean_val = np.mean(vals) if vals else np.nan
                std_val = np.std(vals) if vals else np.nan
                # Ensure error bars don't go below 0 for non-negative metrics
                if not np.isnan(mean_val) and not np.isnan(std_val):
                    if metric in ["precision", "unique_recall_rate"]:
                        lower_error = min(std_val, mean_val)
                        upper_error = std_val
                    else:
                        lower_error = std_val
                        upper_error = std_val
                else:
                    lower_error = std_val
                    upper_error = std_val
                means.append(mean_val)
                stds.append([lower_error, upper_error])
                print(f"{model_name} {METHOD_MAP[method][0]}: {mean_val:.2f}$_{{\\pm {std_val:.2f}}}$") # 4.14$_{\pm 0.44}$
            ax.bar(x + i * bar_width, means, yerr=np.array(stds).T, width=bar_width, label=METHOD_MAP[method][0], color=colors(i))
        ax.set_xticks(x + bar_width * (n_methods - 1) / 2)
        ax.set_xticklabels(all_model_names, rotation=30, ha='right')
        ax.set_title(f"{metric_labels[metric]} (per model, grouped by method)")
        ax.set_ylabel('Mean ± Std')
        ax.legend(title='Method')
        plt.tight_layout()
        plt.show()
                        

def ttest_vs_vs_baseline(metrics_values, all_model_names, plot_metrics):
    """
    Calculate t-tests between each VS method and each baseline method for each metric.
    """
    baseline_methods = ["direct", "direct_cot", "sequence", "multi_turn"]
    vs_methods = ["vs_standard", "vs_cot", "vs_combined"]
    for metric in plot_metrics:
        print(f"\n=== T-TESTS for {metric} ===")
        for b in baseline_methods:
            for v in vs_methods:
                vals_b = []
                vals_v = []
                for model_name in all_model_names:
                    if model_name in metrics_values:
                        for method_name, method_data in metrics_values[model_name].items():
                            if METHOD_MAP[b][1] in method_name:
                                vals_b.extend(method_data[metric])
                            if METHOD_MAP[v][1] in method_name:
                                vals_v.extend(method_data[metric])
                if vals_b and vals_v:
                    t_stat, p_val = ttest_ind(vals_b, vals_v, equal_var=False)
                    
                    # Determine significance level
                    if p_val < 0.001:
                        sig = '***'
                    elif p_val < 0.01:
                        sig = '**'
                    elif p_val < 0.05:
                        sig = '*'
                    else:
                        sig = 'ns'
                    
                    print(f"{METHOD_MAP[v][0]} vs {METHOD_MAP[b][0]}: p={p_val:.4g} (t={t_stat:.2f}) {sig}")
                else:
                    print(f"{METHOD_MAP[v][0]} vs {METHOD_MAP[b][0]}: Not enough data")


def bar_plot_all_methods(metrics_values, all_model_names, plot_metrics, metric_labels, all_methods):
    for metric in plot_metrics:
        display_names = [METHOD_MAP[m][0] for m in all_methods]
        means = []
        stds = []
        for method in all_methods:
            vals = []
            for model_name in all_model_names:
                if model_name in metrics_values:
                    for method_name, method_data in metrics_values[model_name].items():
                        if METHOD_MAP[method][1] in method_name:
                            vals.extend(method_data[metric])
            mean_val = np.mean(vals) if vals else np.nan
            std_val = np.std(vals) if vals else np.nan
            
            # Ensure error bars don't go below 0 for non-negative metrics
            if not np.isnan(mean_val) and not np.isnan(std_val):
                # For metrics that should be non-negative (precision, unique_recall_rate)
                if metric in ["precision", "unique_recall_rate"]:
                    # Cap the lower error bar at 0
                    lower_error = min(std_val, mean_val)
                    upper_error = std_val
                else:
                    # For other metrics (like kl_divergence), allow full error bars
                    lower_error = std_val
                    upper_error = std_val
            else:
                lower_error = std_val
                upper_error = std_val
            
            means.append(mean_val)
            stds.append([lower_error, upper_error])  # Use asymmetric error bars
        
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(all_methods))
        ax.bar(display_names, means, yerr=np.array(stds).T, capsize=5)
        ax.set_title(f"{metric_labels[metric]} (all models aggregated)")
        ax.set_ylabel('Mean ± Std')
        # 3. Calculate t-test for all three baseline, direct, sequence, multi-turn and annotate
        annotate_ttest_baselines(metrics_values, all_model_names, [metric], all_methods, ax)
        plt.tight_layout()
        plt.show()


def bar_plot_all_methods_with_ttest_box(metrics_values, all_model_names, plot_metrics, metric_labels, baseline_methods, vs_methods, all_methods):
    """
    For each VS method, draw a bar plot with all methods, and include a text box at the top left summarizing t-test results (p-value, t-value, significance) for that VS method vs each baseline (reported individually).
    Draw the bar plot with error bars and put the value on top of the error bar.
    Ensure error bars do not exceed the proper range (e.g., for non-negative metrics, error bars do not go below zero).
    """
    import matplotlib.cm as cm

    # Only do t-tests for these metrics
    ttest_metrics = ["kl_divergence", "unique_recall_rate"]
    y_label_map = {
        "kl_divergence": "KL Divergence",
        "unique_recall_rate": "Unique Recall",
        "precision": "Precision"
    }

    # Define which metrics are non-negative (so error bars should not go below zero)
    non_negative_metrics = ["precision", "unique_recall_rate"]

    for metric in plot_metrics:
        display_names = [METHOD_MAP[m][0] for m in all_methods]
        means = []
        error_bars = []  # Will be a list of [lower, upper] for asymmetric error bars

        # Collect means and stds for all methods, and adjust error bars as needed
        for method in all_methods:
            vals = []
            for model_name in all_model_names:
                if model_name in metrics_values:
                    for method_name, method_data in metrics_values[model_name].items():
                        if METHOD_MAP[method][1] in method_name:
                            vals.extend(method_data[metric])
            mean_val = np.mean(vals) if vals else np.nan
            std_val = np.std(vals) if vals else np.nan

            # Adjust error bars so they do not exceed the proper range
            if not np.isnan(mean_val) and not np.isnan(std_val):
                if metric in non_negative_metrics:
                    # Lower error bar cannot go below zero
                    lower = min(std_val, mean_val)
                    upper = std_val
                else:
                    lower = std_val
                    upper = std_val
            else:
                lower = std_val
                upper = std_val

            means.append(mean_val)
            error_bars.append([lower, upper])

        # Convert error_bars to shape (2, N) for matplotlib's yerr
        yerr = np.array(error_bars).T  # shape (2, N)

        # For metrics that require t-tests, draw a separate plot for each VS method
        if metric in ttest_metrics:
            for vs in vs_methods:
                # Prepare t-test results for the box, one line per baseline
                box_lines = [f"Statistical Tests: {METHOD_MAP[vs][0]} vs Baselines"]
                for baseline in baseline_methods:
                    # Gather baseline values
                    vals_b = []
                    for model_name in all_model_names:
                        if model_name in metrics_values:
                            for method_name, method_data in metrics_values[model_name].items():
                                if METHOD_MAP[baseline][1] in method_name:
                                    vals_b.extend(method_data[metric])
                    # Gather VS method values
                    vals_v = []
                    for model_name in all_model_names:
                        if model_name in metrics_values:
                            for method_name, method_data in metrics_values[model_name].items():
                                if METHOD_MAP[vs][1] in method_name:
                                    vals_v.extend(method_data[metric])
                    # Prepare t-test result for this baseline
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
                        box_lines.append(f"vs {METHOD_MAP[baseline][0]}: p={p_val:.4g} (t={t_stat:.2f}) {sig}")
                    else:
                        box_lines.append(f"vs {METHOD_MAP[baseline][0]}: Not enough data")
                box_text = '\n'.join(box_lines)

                # Plot
                cmap = cm.get_cmap('tab10')
                colors = [cmap(i % 10) for i in range(len(all_methods))]
                fig, ax = plt.subplots(figsize=(10, 5))
                x = np.arange(len(all_methods))
                # Draw bar plot with error bars (asymmetric)
                bars = ax.bar(x, means, color=colors, yerr=yerr, capsize=5)
                ax.set_xticks(x)
                ax.set_xticklabels(display_names, rotation=20)
                ax.set_title(f"{metric_labels[metric]} - Average Across All Models")
                ax.set_ylabel(f'Average {y_label_map[metric]}')
                # Add value labels on top of the error bar
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    upper_error = yerr[1, i] if not np.isnan(yerr[1, i]) else 0.0
                    label_y = height + upper_error + 0.01  # 0.01 offset above error bar
                    ax.text(bar.get_x() + bar.get_width()/2., label_y, f"{height:.2f}", ha='center', va='bottom', fontsize=10)
                # Place the box at the top right of the image (figure area), left-aligned
                if metric in ['kl_divergence']:
                    fig.text(0.97, 0.90, box_text, fontsize=11, verticalalignment='top', horizontalalignment='right',
                             multialignment='left',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
                else:
                    fig.text(0.08, 0.90, box_text, fontsize=9, verticalalignment='top', horizontalalignment='left',
                             multialignment='left',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
                plt.subplots_adjust(right=0.80)
                plt.tight_layout()
                plt.savefig(f"bias_metrics_{metric}_{METHOD_MAP[vs][0]}_vs_all_baselines.png")
                plt.show()
        else:
            # For metrics that do not require t-tests, just plot the bar chart with error bars
            box_text = "No t-test performed for this metric."
            cmap = cm.get_cmap('tab10')
            colors = [cmap(i % 10) for i in range(len(all_methods))]
            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(len(all_methods))
            bars = ax.bar(x, means, color=colors, yerr=yerr, capsize=5)
            ax.set_xticks(x)
            ax.set_xticklabels(display_names, rotation=20)
            ax.set_title(f"{metric_labels[metric]} - Average Across All Models")
            ax.set_ylabel(f'Average {y_label_map[metric]}')
            # Add value labels on top of the error bar
            for i, bar in enumerate(bars):
                height = bar.get_height()
                upper_error = yerr[1, i] if not np.isnan(yerr[1, i]) else 0.0
                label_y = height + upper_error + 0.01  # 0.01 offset above error bar
                ax.text(bar.get_x() + bar.get_width()/2., label_y, f"{height:.2f}", ha='center', va='bottom', fontsize=10)
            # # Place the box at the top right of the image (figure area), left-aligned
            # fig.text(0.97, 0.90, box_text, fontsize=11, verticalalignment='top', horizontalalignment='right',
            #          multialignment='left',
            #          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            plt.subplots_adjust(right=0.80)
            plt.tight_layout()
            plt.savefig(f"bias_metrics_{metric}_no_ttest.png")
            plt.show()



def main():
    folder = "method_results_bias"
    task_name = "state_name"

    all_model_names = [
        "gpt-4.1-mini",
        "gpt-4.1",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "meta-llama_Llama-3.1-70B-Instruct",
        "deepseek-r1",
        "o3",
        "claude-4-sonnet",
    ]
    
    # Define the four metrics we want to analyze
    metrics = ["kl_divergence", "chi_square", "precision", "unique_recall_rate"]
    
    # Only keep these metrics for plotting
    plot_metrics = ["kl_divergence", "unique_recall_rate", "precision"]
    metric_labels = {
        "kl_divergence": "KL Divergence ↓",
        "unique_recall_rate": "Unique Recall ↑",
        "precision": "Precision ↑"
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
            results_file = method_dir / "response_count_results.json"
            if not results_file.exists():
                print(f"Warning: No results file found for {model_name} - {method_name}")
                continue
            try:
                with open(results_file, "r") as f:
                    data = json.load(f)
                aggregate_metrics = data.get("overall_metrics", {})
                per_prompt_stats = aggregate_metrics.get("per_prompt_stats", {})

                if method_name in METHOD_MAP.values():
                    method_name = METHOD_MAP[method_name][0]

                # Initialize data structure for this model-method combination
                if model_name not in metrics_values:
                    metrics_values[model_name] = {}
                if method_name not in metrics_values[model_name]:
                    metrics_values[model_name][method_name] = {metric: [] for metric in metrics}
                # Collect metric values from all prompts
                for prompt_stats in per_prompt_stats.values():
                    for metric in metrics:
                        if metric in prompt_stats:
                            metrics_values[model_name][method_name][metric].append(prompt_stats[metric])
            except Exception as e:
                print(f"Error reading {results_file}: {e}")

    # print(metrics_values)

    # 1. Draw error bar graph for all models and all methods
    # plot_error_bars_all_models_methods(metrics_values, all_model_names, plot_metrics, metric_labels, all_methods)

    # 2. Draw error bar graph for only the methods over all the models
    #    and annotate t-test for baselines
    # We'll modify plot_error_bars_methods_overall to return the axes for annotation
    # bar_plot_all_methods(metrics_values, all_model_names, plot_metrics, metric_labels, all_methods)
    # ttest_vs_vs_baseline(metrics_values, all_model_names, plot_metrics)

    baseline_methods = ["direct", "direct_cot", "sequence", "multi_turn"]
    vs_methods = ["vs_standard", "vs_cot", "vs_combined"]
    bar_plot_all_methods_with_ttest_box(metrics_values, all_model_names, plot_metrics, metric_labels, baseline_methods, vs_methods, all_methods)


if __name__ == "__main__":
    main() 