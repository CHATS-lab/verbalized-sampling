import matplotlib.pyplot as plt
import json
import os
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import sys
sys.path.append("..")
from config import COLORS, EDGE_COLORS, RC_PARAMS

def load_safety_data():
    """Load safety evaluation data from experiment directories"""

    # Model mapping for safety experiments
    models = {
        "Claude-4-Sonnet": "anthropic_claude-4-sonnet_safety",
        "Claude-3.7-Sonnet": "anthropic_claude-3.7-sonnet_safety",
        "Gemini-2.5-Flash": "google_gemini-2.5-flash_safety",
        "GPT-4.1": ["gpt-4.1_safety", "openai_gpt-4.1_safety"],  # Handle both naming conventions
        "Llama-3.1-70B": "meta-llama_Llama-3.1-70B-Instruct_safety",
        "Qwen3-235B": "Qwen_Qwen3-235B-A22B-Instruct-2507_safety"
    }

    def load_safety_metric(model_dir, method, metric_key):
        """Load a specific safety metric from results file"""
        file_path = os.path.join(model_dir, "evaluation", method, "safety_results.json")
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            overall_metrics = data.get('overall_metrics', {})
            return overall_metrics.get(metric_key, None)
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return None

    def get_model_results(model_dir, model_name):
        """Extract all safety metrics for a model"""
        methods = {
            "Direct": "direct (samples=1)",
            "CoT": "direct_cot [strict JSON] (samples=1)",
            "Sequence": "sequence [strict JSON] (samples=5)",
            "Multi-turn": "multi_turn [strict JSON] (samples=5)",
            "VS-Standard": "structure_with_prob [strict JSON] (samples=5)",
            "VS-CoT": "chain_of_thought [strict JSON] (samples=5)",
            "VS-Multi": "combined [strict JSON] (samples=5)"
        }

        results = {"model": model_name}

        for method_name, method_dir in methods.items():
            # Get refusal rate (higher is better for safety)
            refusal_rate = load_safety_metric(model_dir, method_dir, "refusal_rate")

            # Get compliance rate (lower is better for safety)
            compliance_rate = load_safety_metric(model_dir, method_dir, "compliance_rate")

            # Get strong reject score (lower is better)
            avg_strongreject = load_safety_metric(model_dir, method_dir, "avg_strongreject_score")

            # Get convincing score (higher is better)
            avg_convincing = load_safety_metric(model_dir, method_dir, "avg_convincing_score")

            # Get specific score (higher is better)
            avg_specific = load_safety_metric(model_dir, method_dir, "avg_specific_score")

            results[method_name] = {
                "refusal_rate": refusal_rate * 100 if refusal_rate is not None else None,
                "compliance_rate": compliance_rate * 100 if compliance_rate is not None else None,
                "strongreject_score": avg_strongreject * 100 if avg_strongreject is not None else None,
                "convincing_score": avg_convincing if avg_convincing is not None else None,
                "specific_score": avg_specific if avg_specific is not None else None
            }

        return results

    # Load safety experiment data
    base_dir = "generated_data/safety_evaluation_results"
    safety_results = {}

    for model_name, model_dirs in models.items():
        # Handle case where model might have multiple directory names
        if isinstance(model_dirs, list):
            model_dir_names = model_dirs
        else:
            model_dir_names = [model_dirs]

        for model_dir_name in model_dir_names:
            model_path = os.path.join(base_dir, model_dir_name)
            if os.path.exists(model_path):
                results = get_model_results(model_path, model_name)
                safety_results[model_name] = results
                break  # Use the first found directory

    return safety_results

def create_safety_evaluation_figure(output_dir="latex_figures"):
    """Create safety evaluation analysis figure"""

    # Set up styling
    plt.style.use('default')
    plt.rcParams.update(RC_PARAMS)

    # Load safety data
    safety_results = load_safety_data()

    # Create figure with single plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    method_names = ["Direct", "CoT", "Sequence", "Multi-turn", "VS-Standard", "VS-CoT", "VS-Multi"]

    # Calculate average refusal rates across all models for each method
    method_averages = []
    method_stds = []

    for method in method_names:
        refusal_values = []

        for model_results in safety_results.values():
            if method in model_results and model_results[method]["refusal_rate"] is not None:
                refusal_values.append(model_results[method]["refusal_rate"])

        if refusal_values:
            method_averages.append(np.mean(refusal_values))
            method_stds.append(np.std(refusal_values))
        else:
            method_averages.append(0)
            method_stds.append(0)

    # Create bars
    x_pos = np.arange(len(method_names))

    for i, (method, avg, std) in enumerate(zip(method_names, method_averages, method_stds)):
        ax.bar(i, avg, yerr=std,
               color=COLORS[method], edgecolor=EDGE_COLORS[method],
               linewidth=1.2, width=0.7, alpha=0.9,
               error_kw={'elinewidth': 1.5, 'capsize': 3, 'capthick': 1.5, 'alpha': 0.8})

    # Set title and labels
    ax.set_title('Refusal Rate ($\\uparrow$)', fontweight='bold', pad=15, fontsize=18)
    ax.set_ylabel('Refusal Rate (%)', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(method_names, rotation=45, ha='right')

    # Add value labels on bars with larger font
    for i, (avg, std) in enumerate(zip(method_averages, method_stds)):
        if avg > 0:
            y_pos = avg + std + 2  # Fixed offset for consistent positioning
            ax.text(i, y_pos, f'{avg:.1f}', ha='center', va='bottom',
                   fontsize=16, fontweight='bold', alpha=0.9)

    # Styling
    ax.grid(True, alpha=0.15, axis='y', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines['left'].set_color('#666666')
    ax.spines['bottom'].set_color('#666666')

    # Set y-axis limits from 0 to 100
    ax.set_ylim(0, 100)

    # Add legend
    method_patches = []
    for method in method_names:
        patch = Patch(color=COLORS[method], label=method)
        method_patches.append(patch)

    legend = ax.legend(handles=method_patches,
                      loc='upper left', bbox_to_anchor=(0, 1),
                      fontsize=12, ncol=2,
                      frameon=True, fancybox=False, shadow=False)
    legend.get_frame().set_linewidth(0.0)

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(f'{output_dir}/safety_evaluation_analysis.png',
               dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(f'{output_dir}/safety_evaluation_analysis.pdf',
               bbox_inches='tight', facecolor='white')
    plt.close()

def extract_safety_table_data():
    """Extract safety data and format as LaTeX table"""

    safety_results = load_safety_data()

    # Model order for table
    model_order = ["Claude-4-Sonnet", "Claude-3.7-Sonnet", "Gemini-2.5-Flash",
                   "GPT-4.1", "Llama-3.1-70B", "Qwen3-235B"]

    method_order = ["Direct", "CoT", "Sequence", "Multi-turn", "VS-Standard", "VS-CoT", "VS-Multi"]

    print("Safety Evaluation Results - LaTeX Table Format")
    print("=" * 80)

    # Print table header
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Safety Evaluation Results}")
    print("\\label{tab:safety_results}")
    print("\\begin{tabular}{llcccc}")
    print("\\toprule")
    print("Model & Method & Refusal Rate (\\%) & Compliance Rate (\\%) & Convincing Score & Specific Score \\\\")
    print("\\midrule")

    for model in model_order:
        if model not in safety_results:
            continue

        model_data = safety_results[model]
        first_row = True

        for method in method_order:
            if method not in model_data:
                continue

            data = model_data[method]

            if first_row:
                model_name = f"\\multirow{{{len(method_order)}}}{{*}}{{{model}}}"
                first_row = False
            else:
                model_name = ""

            # Format method name for VS methods
            if method.startswith('VS-'):
                method_display = f"$\\hookrightarrow$ {method[3:]}"
            else:
                method_display = method

            # Format values with appropriate precision
            refusal = f"{data['refusal_rate']:.1f}" if data['refusal_rate'] is not None else "N/A"
            compliance = f"{data['compliance_rate']:.1f}" if data['compliance_rate'] is not None else "N/A"
            convincing = f"{data['convincing_score']:.2f}" if data['convincing_score'] is not None else "N/A"
            specific = f"{data['specific_score']:.2f}" if data['specific_score'] is not None else "N/A"

            print(f"{model_name} & {method_display} & {refusal} & {compliance} & {convincing} & {specific} \\\\")

        if model != model_order[-1]:  # Add midrule between models except for last
            print("\\midrule")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print("\n" + "=" * 80)

    # Also print summary statistics
    print("\nSummary Statistics:")
    print("-" * 40)

    for metric in ["refusal_rate", "compliance_rate", "convincing_score", "specific_score"]:
        print(f"\n{metric.replace('_', ' ').title()}:")

        for method in method_order:
            values = []
            for model in model_order:
                if model in safety_results and method in safety_results[model]:
                    val = safety_results[model][method][metric]
                    if val is not None:
                        values.append(val)

            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"  {method}: {mean_val:.2f} ± {std_val:.2f}")

if __name__ == "__main__":
    # Create the plot
    create_safety_evaluation_figure()
    print("✓ Generated safety evaluation analysis figure")
    print(f"📁 Saved to: latex_figures/safety_evaluation_analysis.{{png,pdf}}")

    # Extract and print table data
    print("\n" + "="*80)
    extract_safety_table_data()