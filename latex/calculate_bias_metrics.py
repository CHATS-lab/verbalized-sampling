import json
import os
from pathlib import Path
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

# Maps canonical method keys to (display name, matching substring in dir name)
METHOD_MAP = {
    "direct": ("Direct", "direct"),
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
                means.append(np.mean(vals) if vals else np.nan)
                stds.append(np.std(vals) if vals else np.nan)
            ax.bar(x + i * bar_width, means, yerr=stds, width=bar_width, label=METHOD_MAP[method][0], color=colors(i))
        ax.set_xticks(x + bar_width * (n_methods - 1) / 2)
        ax.set_xticklabels(all_model_names, rotation=30, ha='right')
        ax.set_title(f"{metric_labels[metric]} (per model, grouped by method)")
        ax.set_ylabel('Mean ± Std')
        ax.legend(title='Method')
        plt.tight_layout()
        plt.show()

def plot_error_bars_methods_overall(metrics_values, all_model_names, plot_metrics, metric_labels, all_methods):
    """
    Draw error bars for each method, aggregated over all models, for each metric.
    """
    import matplotlib.pyplot as plt
    import numpy as np

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
            means.append(np.mean(vals) if vals else np.nan)
            stds.append(np.std(vals) if vals else np.nan)
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(all_methods))
        ax.bar(display_names, means, yerr=stds, capsize=5)
        ax.set_title(f"{metric_labels[metric]} (all models aggregated)")
        ax.set_ylabel('Mean ± Std')
        plt.tight_layout()
        plt.show()

def annotate_ttest_baselines(metrics_values, all_model_names, plot_metrics, all_methods, ax):
    """
    Calculate t-tests between VS methods and baseline methods for each metric.
    Annotate the graph with significance.
    """
    from scipy.stats import ttest_ind
    import numpy as np

    baseline_methods = ["direct", "sequence", "multi_turn"]
    vs_methods = ["vs_standard", "vs_cot", "vs_combined"]
    
    for metric_idx, metric in enumerate(plot_metrics):
        print(f"\n=== T-TESTS for {metric} ===")
        for baseline_method in baseline_methods:
            for vs_method in vs_methods:
                vals_baseline = []
                vals_vs = []
                for model_name in all_model_names:
                    if model_name in metrics_values:
                        for method_name, method_data in metrics_values[model_name].items():
                            if baseline_method.lower() in method_name.lower():
                                vals_baseline.extend(method_data[metric])
                            if vs_method.lower() in method_name.lower():
                                vals_vs.extend(method_data[metric])
                
                if vals_baseline and vals_vs:
                    t_stat, p_val = ttest_ind(vals_baseline, vals_vs, equal_var=False)
                    
                    # Determine significance level
                    if p_val < 0.001:
                        sig = '***'
                    elif p_val < 0.01:
                        sig = '**'
                    elif p_val < 0.05:
                        sig = '*'
                    else:
                        sig = 'ns'
                    
                    # Print results
                    print(f"{METHOD_MAP[vs_method][0]} vs {METHOD_MAP[baseline_method][0]}: p={p_val:.4g} (t={t_stat:.2f}) {sig}")
                    
                    # Calculate position for annotation
                    baseline_idx = baseline_methods.index(baseline_method)
                    vs_idx = vs_methods.index(vs_method)
                    x_pos = baseline_idx + (vs_idx * len(baseline_methods))
                    
                    # Calculate y position above the bars
                    baseline_mean = np.mean(vals_baseline)
                    vs_mean = np.mean(vals_vs)
                    baseline_std = np.std(vals_baseline)
                    vs_std = np.std(vals_vs)
                    y_pos = max(baseline_mean, vs_mean) + max(baseline_std, vs_std) + 0.05
                    
                    # Annotate with t-statistic and p-value
                    annotation_text = f"t={t_stat:.2f}\np={p_val:.3f}\n{sig}"
                    ax.annotate(annotation_text, 
                               xy=(x_pos, y_pos),
                               xytext=(0, 10),
                               textcoords='offset points',
                               ha='center', va='bottom',
                               color='red', fontsize=8,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                               arrowprops=dict(arrowstyle='->', color='red', lw=1))
                        

def ttest_vs_vs_baseline(metrics_values, all_model_names, plot_metrics):
    """
    Calculate t-tests between each VS method and each baseline method for each metric.
    """
    baseline_methods = ["direct", "sequence", "multi_turn"]
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
            means.append(np.mean(vals) if vals else np.nan)
            stds.append(np.std(vals) if vals else np.nan)
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(all_methods))
        ax.bar(display_names, means, yerr=stds, capsize=5)
        ax.set_title(f"{metric_labels[metric]} (all models aggregated)")
        ax.set_ylabel('Mean ± Std')
        # 3. Calculate t-test for all three baseline, direct, sequence, multi-turn and annotate
        annotate_ttest_baselines(metrics_values, all_model_names, [metric], all_methods, ax)
        plt.tight_layout()
        plt.show()

    ttest_vs_vs_baseline(metrics_values, all_model_names, plot_metrics)

if __name__ == "__main__":
    main() 