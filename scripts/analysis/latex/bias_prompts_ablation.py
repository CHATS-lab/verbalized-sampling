import os
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tabulate import tabulate

# Set up plotting style
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 20,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'axes.linewidth': 0.8,
    'axes.edgecolor': '#666666',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white'
})

# def draw_bar_chart(data):
#     """
#     Draw grouped bar charts for 3 metrics (KL divergence, unique recall rate, precision)
#     for each of 5 prompt methods, with different colors and value labels.
#     """
#     import matplotlib.pyplot as plt
#     import numpy as np

#     # Metrics and their display names
#     metrics = [
#         ("kl_divergence", "KL Divergence"),
#         ("unique_recall_rate", "Coverage-N"),
#         ("precision", "Precision")
#     ]
#     metric_keys = [m[0] for m in metrics]
#     metric_names = [m[1] for m in metrics]

#     # Methods and their display names
#     method_names = {
#         "direct": "Direct",
#         "implicit_prob": "Implicit Prob",
#         "explicit_prob": "Explicit Prob",
#         "nll": "NLL",
#         "perplexity": "Perplexity"
#     }
#     methods = list(method_names.keys())

#     # Prepare data: for each metric, get the value for each method
#     values = []
#     for metric_key in metric_keys:
#         metric_vals = []
#         for method in methods:
#             val = data[method].get(f"average_{metric_key}", None)
#             # fallback for "unique_recall_rate" (no "average_" prefix in some files)
#             if val is None and metric_key == "unique_recall_rate":
#                 val = data[method].get("average_unique_recall_rate", None)
#             metric_vals.append(val)
#         values.append(metric_vals)  # shape: [metric][method]

#     # Bar chart parameters
#     n_metrics = len(metrics)
#     n_methods = len(methods)
#     bar_width = 0.15
#     x = np.arange(n_metrics)  # the label locations

#     # Set color palette
#     colors = plt.get_cmap("tab10").colors[:n_methods]

#     plt.figure(figsize=(12, 7))
#     for i, method in enumerate(methods):
#         # For each method, plot a bar for each metric
#         offsets = x + (i - n_methods/2) * bar_width + bar_width/2
#         vals = [values[m][i] for m in range(n_metrics)]
#         bars = plt.bar(offsets, vals, width=bar_width, label=method_names[method], color=colors[i])
#         # Add value labels on top
#         for bar in bars:
#             height = bar.get_height()
#             plt.annotate(f"{height:.2f}",
#                          xy=(bar.get_x() + bar.get_width() / 2, height),
#                          xytext=(0, 3),  # 3 points vertical offset
#                          textcoords="offset points",
#                          ha='center', va='bottom', fontsize=12, fontweight='bold')

#     plt.xticks(x, metric_names, fontsize=16, fontweight='bold')
#     plt.xlabel("", fontsize=14)
#     plt.ylabel("", fontsize=14)
#     plt.title("", fontsize=16)
#     plt.legend(title="Method", fontsize=11)
#     plt.tight_layout()
#     # plt.show()
#     plt.savefig("qualitative_tasks/comparison_of_different_prompt_variants.pdf", dpi=300, bbox_inches='tight')


def create_line_chart_plot(all_data):
    """
    Create a line chart showing performance across probability definitions for each metric and model.
    Layout: 3x2 grid (3 metrics x 2 models) with lines for VS-Standard and VS-Multi.
    """
    
    # Define metrics and their properties
    metrics = {
        "average_kl_divergence": {
            "name": "KL Divergence",
            "direction": "$\\downarrow$",  # Lower is better
            "color": "#FF6B6B"  # Red for lower-is-better
        },
        "average_unique_recall_rate": {
            "name": "Coverage-N", 
            "direction": "$\\uparrow$",  # Higher is better
            "color": "#4ECDC4"  # Teal for higher-is-better
        },
        "average_precision": {
            "name": "Precision",
            "direction": "$\\uparrow$",  # Higher is better  
            "color": "#45B7D1"  # Blue for higher-is-better
        }
    }
    
    # Method display names and colors
    method_names = {
        'vs_standard': 'VS-Standard',
        'vs_multi': 'VS-Multi'
    }
    
    method_colors = {
        'vs_standard': '#4A90E2',  # Red
        'vs_multi': '#FF6B6B'  # Blue
    }
    
    method_markers = {
        'vs_standard': 'o',  # Circle
        'vs_multi': 's'  # Square
    }
    
    # Probability definition display names and order
    prob_names = {
        'implicit': 'Implicit',
        'explicit': 'Explicit', 
        'relative': 'Relative',
        'percentage': 'Percentage',
        'confidence': 'Confidence',
        'nll': 'NLL',
        'perplexity': 'Perplexity'
    }
    
    # Order for x-axis
    probability_definitions = ['implicit', 'explicit', 'relative', 'percentage', 'confidence', 'nll', 'perplexity']
    methods = ['vs_standard', 'vs_multi']
    models = ['gpt-4.1', 'gemini-2.5-flash']
    model_names = {
        'gpt-4.1': 'GPT-4.1',
        'gemini-2.5-flash': 'Gemini 2.5 Flash'
    }

    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'font.size': 16,
        'axes.titlesize': 24,
        'xtick.labelsize': 20,
        'ytick.labelsize': 18,
        'axes.labelsize': 22,  # Set y label size to 20 (applies to both x and y labels)
        'legend.fontsize': 28,
        'axes.linewidth': 0.8,
        'axes.edgecolor': '#666666',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white'
    })
    
    # Create figure with subplots (2 rows, 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(25, 13))
    
    # Process each metric and model combination
    for metric_idx, (metric_key, metric_info) in enumerate(metrics.items()):
        for model_idx, model in enumerate(models):
            ax = axes[model_idx, metric_idx]
            
            # Plot lines for each method
            for method in methods:
                # Extract data for this method-model-metric combination
                values = []
                for prob_def in probability_definitions:
                    if (method in all_data and 
                        model in all_data[method] and 
                        prob_def in all_data[method][model]):
                        value = all_data[method][model][prob_def].get(metric_key, None)
                        values.append(value if value is not None else 0)
                    else:
                        values.append(0)
                
                # Plot line with markers
                x_positions = range(len(probability_definitions))
                ax.plot(x_positions, values, 
                       marker=method_markers[method],
                       linewidth=3,
                       markersize=8,
                       color=method_colors[method],
                       label=method_names[method],
                       alpha=0.8)
                
                # # Add value labels on markers
                # for i, value in enumerate(values):
                #     if value > 0:
                #         ax.annotate(f'{value:.3f}', 
                #                    (i, value),
                #                    textcoords="offset points",
                #                    xytext=(0, 10),
                #                    ha='center', va='bottom',
                #                    fontsize=10, fontweight='bold',
                #                    color=method_colors[method])
            
            # Customize subplot
            ax.set_xlabel('', fontweight='bold')
            ax.set_ylabel(metric_info['name'], fontweight='bold')
            ax.set_title(f'{metric_info["name"]} ({metric_info["direction"]}) - {model_names[model]}', 
                        fontweight='bold', pad=15)
            ax.set_xticks(range(len(probability_definitions)))
            ax.set_xticklabels([prob_names[prob] for prob in probability_definitions], rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            # ax.legend(loc='upper right', fontsize=12)
            
            # Set y-axis limits based on metric type
            if metric_key == "average_kl_divergence":
                ax.set_ylim(0.9, 1.2)  # KL Divergence range 0-2
            if metric_key == "average_unique_recall_rate":
                ax.set_ylim(0.3, 0.6)
            if metric_key == "average_precision":
                ax.set_ylim(0.9, 1.0)
            
            # # Add horizontal reference line for better comparison
            # if metric_key == "average_kl_divergence":
            #     # For KL divergence, add a reference line at a reasonable value
            #     ax.axhline(y=0.05, color='gray', linestyle='--', alpha=0.5, label='Reference')
            # else:
            #     # For other metrics, add reference line at 0.5
            #     ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Reference')
    
    # Add subplot labels
    subplot_labels = ['a', 'b', 'c', 'd', 'e', 'f']
    for i, ax in enumerate(axes.flat):
        ax.text(-0.17, 1.2, subplot_labels[i], transform=ax.transAxes, 
                fontsize=32, fontweight='bold', va='top')
    
    # Create a legend for the whole figure at the top center
    # Get handles and labels from one of the axes (they are the same for all)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(method_names), frameon=False, bbox_to_anchor=(0.5, 1.01))
    
    # plt.tight_layout()
    plt.subplots_adjust(top=0.88, hspace=0.50, wspace=0.30)

    # Save the plot
    output_path = "latex/qualitative_tasks/bias_prompts_ablation_line_chart.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved bias prompts ablation line chart to: {output_path}")


def print_numerical_results(all_data):
    """
    Print the average_kl_divergence, average_unique_recall_rate, and average_precision
    for each method, model, and probability_definition in all_data as a user-friendly table.
    """
    from tabulate import tabulate

    metrics = ["average_kl_divergence", "average_unique_recall_rate", "average_precision"]
    metric_names = {
        "average_kl_divergence": "KL Divergence",
        "average_unique_recall_rate": "Unique Recall Rate",
        "average_precision": "Precision"
    }

    # Collect all rows
    rows = []
    for method in all_data:
        for model in all_data[method]:
            for prob_def in all_data[method][model]:
                results = all_data[method][model][prob_def]
                row = [
                    method,
                    model,
                    prob_def,
                ]
                for metric in metrics:
                    value = results.get(metric, None)
                    if value is not None:
                        # Format floats to 3 decimals
                        value = f"{value:.3f}"
                    else:
                        value = "-"
                    row.append(value)
                rows.append(row)

    headers = ["Method", "Model", "Probability Definition"] + [metric_names[m] for m in metrics]
    print(tabulate(rows, headers=headers, tablefmt="grid", stralign="center", numalign="center"))


def main():
    folder_path = "ablation_bias_task"

    methods = ['vs_standard', 'vs_multi'] #  'vs_multi'
    models = ['gpt-4.1', 'gemini-2.5-flash'] # 'gemini-2.5-flash'
    probability_definitions = ['implicit', 'explicit', 'relative', 'percentage', 'confidence', 'nll', 'perplexity']
    detailed_methods = {
        'vs_standard': 'structure_with_prob [strict] (samples=20)',
        'vs_multi': 'combined [strict] (samples=20)'
    }

    all_data = {}
    for method in methods:
        for model in models:
            for probability_definition in probability_definitions:
                path = os.path.join(folder_path, method, model, probability_definition, 'evaluation', detailed_methods[method], 'response_count_results.json')
                data = json.load(open(path))

                if method not in all_data:  
                    all_data[method] = {}
                if model not in all_data[method]:
                    all_data[method][model] = {}
                if probability_definition not in all_data[method][model]:
                    all_data[method][model][probability_definition] = {}

                all_data[method][model][probability_definition] = data['overall_metrics']
    
    # print(all_data)
    print_numerical_results(all_data)
    
    
    # Create line chart visualization
    create_line_chart_plot(all_data)


if __name__ == "__main__":
    main()