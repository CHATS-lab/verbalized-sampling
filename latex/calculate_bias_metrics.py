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
    plot_error_bars_all_models_methods(metrics_values, all_model_names, plot_metrics, metric_labels, all_methods)

    # 2. Draw error bar graph for only the methods over all the models
    #    and annotate t-test for baselines
    # We'll modify plot_error_bars_methods_overall to return the axes for annotation
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

    ttest_vs_vs_baseline(metrics_values, all_model_names, plot_metrics)

if __name__ == "__main__":
    main() 