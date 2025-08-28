import os
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def draw_bar_chart(data):
    """
    Draw grouped bar charts for 3 metrics (KL divergence, unique recall rate, precision)
    for each of 5 prompt methods, with different colors and value labels.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Metrics and their display names
    metrics = [
        ("kl_divergence", "KL Divergence"),
        ("unique_recall_rate", "Coverage-N"),
        ("precision", "Precision")
    ]
    metric_keys = [m[0] for m in metrics]
    metric_names = [m[1] for m in metrics]

    # Methods and their display names
    method_names = {
        "direct": "Direct",
        "implicit_prob": "Implicit Prob",
        "explicit_prob": "Explicit Prob",
        "nll": "NLL",
        "perplexity": "Perplexity"
    }
    methods = list(method_names.keys())

    # Prepare data: for each metric, get the value for each method
    values = []
    for metric_key in metric_keys:
        metric_vals = []
        for method in methods:
            val = data[method].get(f"average_{metric_key}", None)
            # fallback for "unique_recall_rate" (no "average_" prefix in some files)
            if val is None and metric_key == "unique_recall_rate":
                val = data[method].get("average_unique_recall_rate", None)
            metric_vals.append(val)
        values.append(metric_vals)  # shape: [metric][method]

    # Bar chart parameters
    n_metrics = len(metrics)
    n_methods = len(methods)
    bar_width = 0.15
    x = np.arange(n_metrics)  # the label locations

    # Set color palette
    colors = plt.get_cmap("tab10").colors[:n_methods]

    plt.figure(figsize=(12, 7))
    for i, method in enumerate(methods):
        # For each method, plot a bar for each metric
        offsets = x + (i - n_methods/2) * bar_width + bar_width/2
        vals = [values[m][i] for m in range(n_metrics)]
        bars = plt.bar(offsets, vals, width=bar_width, label=method_names[method], color=colors[i])
        # Add value labels on top
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f"{height:.2f}",
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.xticks(x, metric_names, fontsize=16, fontweight='bold')
    plt.xlabel("", fontsize=14)
    plt.ylabel("", fontsize=14)
    plt.title("", fontsize=16)
    plt.legend(title="Method", fontsize=11)
    plt.tight_layout()
    # plt.show()
    plt.savefig("qualitative_tasks/comparison_of_different_prompt_variants.pdf", dpi=300, bbox_inches='tight')


def main():
    folder_path = "ablation_prompt_bias/gpt-4.1_state_name/evaluation"

    direct_path = os.path.join(folder_path, "direct/response_count_results.json")
    implicit_prob_path = os.path.join(folder_path, "imp_prob/response_count_results.json")
    explicit_prob_path = os.path.join(folder_path, "exp_prob/response_count_results.json")
    nll_path = os.path.join(folder_path, "nll/response_count_results.json")
    perplexity_path = os.path.join(folder_path, "perplexity/response_count_results.json")

    direct_data = json.load(open(direct_path))
    implicit_prob_data = json.load(open(implicit_prob_path))
    explicit_prob_data = json.load(open(explicit_prob_path))
    nll_data = json.load(open(nll_path))
    perplexity_data = json.load(open(perplexity_path))

    all_data = {
        "direct": direct_data['overall_metrics'],
        "implicit_prob": implicit_prob_data['overall_metrics'],
        "explicit_prob": explicit_prob_data['overall_metrics'],
        "nll": nll_data['overall_metrics'],
        "perplexity": perplexity_data['overall_metrics']
    }

    # print(direct_data)
    draw_bar_chart(all_data)

if __name__ == "__main__":
    main()