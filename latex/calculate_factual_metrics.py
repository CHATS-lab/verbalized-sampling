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


def method_bar_plot(metrics_values, all_model_names, plot_metrics, metric_labels, all_methods):
    """
    Draw bar graph for all models and all methods, with each method in a different color. No error bars.
    """
    import matplotlib.cm as cm

    for metric in plot_metrics:
        display_names = [METHOD_MAP[m][0] for m in all_methods]
        means = []
        for method in all_methods:
            vals = []
            for model_name in all_model_names:
                if model_name in metrics_values:
                    for method_name, method_data in metrics_values[model_name].items():
                        if METHOD_MAP[method][1] in method_name:
                            vals.extend(method_data[metric])
            print(vals)
            mean_val = np.mean(vals) if vals else np.nan
            means.append(mean_val)

        # Assign a different color to each method
        cmap = cm.get_cmap('tab10')
        colors = [cmap(i % 10) for i in range(len(all_methods))]

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(all_methods))
        # Plot each bar individually to assign colors
        for i, (mean, color) in enumerate(zip(means, colors)):
            ax.bar(
                display_names[i],
                mean,
                color=color,
                label=display_names[i]
            )
        ax.set_title(f"{metric_labels[metric]} (all models aggregated)")
        ax.set_ylabel('Mean')
        ax.set_ylim(0, 0.8)
        # Optionally add a legend if you want to show method names/colors
        # ax.legend()
        plt.tight_layout()
        plt.show()

                        

def ttest_vs_vs_baseline(metrics_values, all_model_names, plot_metrics):
    """
    Calculate t-tests to see if any of the baseline methods are statistically significant compared to each VS method for each metric.
    """
    baseline_methods = ["direct", "direct_cot", "sequence", "multi_turn"]
    vs_methods = ["vs_standard", "vs_cot", "vs_combined"]
    for metric in plot_metrics:
        print(f"\n=== T-TESTS for {metric} ===")
        for v in vs_methods:
            for b in baseline_methods:
                vals_v = []
                vals_b = []
                for model_name in all_model_names:
                    if model_name in metrics_values:
                        for method_name, method_data in metrics_values[model_name].items():
                            if METHOD_MAP[v][1] in method_name:
                                vals_v.extend(method_data[metric])
                            if METHOD_MAP[b][1] in method_name:
                                vals_b.extend(method_data[metric])
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
                    
                    print(f"{METHOD_MAP[b][0]} vs {METHOD_MAP[v][0]}: p={p_val:.4g} (t={t_stat:.2f}) {sig}")
                else:
                    print(f"{METHOD_MAP[b][0]} vs {METHOD_MAP[v][0]}: Not enough data")

def bar_plot_with_ttest(metrics_values, all_model_names, plot_metrics, metric_labels, baseline_methods, vs_methods):
    """
    For each baseline method, draw a bar plot comparing it to the three VS methods for each metric.
    Annotate each VS bar with the t-test significance against the baseline.
    """
    import matplotlib.cm as cm
    for metric in plot_metrics:
        for baseline in baseline_methods:
            methods_to_plot = [baseline] + vs_methods
            display_names = [METHOD_MAP[m][0] for m in methods_to_plot]
            means = []
            sigs = [None]  # First is baseline, no sig
            # Collect means and t-test results
            for i, method in enumerate(methods_to_plot):
                vals = []
                for model_name in all_model_names:
                    if model_name in metrics_values:
                        for method_name, method_data in metrics_values[model_name].items():
                            if METHOD_MAP[method][1] in method_name:
                                vals.extend(method_data[metric])
                mean_val = np.mean(vals) if vals else np.nan
                means.append(mean_val)
                # For VS methods, compute t-test vs baseline
                if i > 0:
                    vals_b = []
                    vals_v = []
                    # Baseline values
                    for model_name in all_model_names:
                        if model_name in metrics_values:
                            for method_name, method_data in metrics_values[model_name].items():
                                if METHOD_MAP[baseline][1] in method_name:
                                    vals_b.extend(method_data[metric])
                                if METHOD_MAP[method][1] in method_name:
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
                        sigs.append(sig)
                    else:
                        sigs.append('n/a')
            # Plot
            cmap = cm.get_cmap('tab10')
            colors = [cmap(i % 10) for i in range(len(methods_to_plot))]
            fig, ax = plt.subplots(figsize=(8, 5))
            x = np.arange(len(methods_to_plot))
            bars = ax.bar(x, means, color=colors)
            ax.set_xticks(x)
            ax.set_xticklabels(display_names)
            ax.set_title(f"{metric_labels[metric]}: {METHOD_MAP[baseline][0]} vs VS methods")
            ax.set_ylabel('Mean')
            ax.set_ylim(0, 0.8)
            # Annotate significance
            for i, bar in enumerate(bars):
                height = bar.get_height()
                if i > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02, sigs[i], ha='center', va='bottom', fontsize=12)
            plt.tight_layout()
            plt.show()

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


def main():
    folder = "method_results_simple_qa"
    task_name = "simple_qa"

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
    bar_plot_all_methods_with_ttest_box(metrics_values, all_model_names, plot_metrics, metric_labels, baseline_methods, vs_methods, all_methods)

    

if __name__ == "__main__":
    main() 