import json
import os
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns  # noqa: F401  (style only)
from matplotlib import gridspec
from matplotlib import font_manager as fm

# ----------------------------
# Font setup: News Gothic MT + fallback
# ----------------------------
# NOTE: DejaVu Sans is kept as fallback to avoid "glyph missing" warnings (e.g., ↑).
font_path = "/Users/jiayizx/Downloads/NewsGothicMT.ttf"
font_name = None
try:
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        font_name = fm.FontProperties(fname=font_path).get_name()
except Exception as e:
    print(f"Warning: could not add font at {font_path}: {e}")

plt.rcParams["font.size"] = 12
plt.rcParams["font.family"] = [font_name]
plt.rcParams["axes.unicode_minus"] = False  # safer for some unicode glyphs
print(font_name)

# ----------------------------
# Method mapping (dir substring -> display name)
# ----------------------------
METHOD_MAP = {
    "direct": ("Direct", "direct"),
    "direct_cot": ("CoT", "direct_cot"),
    "sequence": ("Sequence", "sequence"),
    "multi_turn": ("Multi-turn", "multi_turn"),
    "vs_standard": ("VS-Standard", "structure_with_prob"),
    "vs_cot": ("VS-CoT", "chain_of_thought"),
    "vs_combined": ("VS-Multi", "combined"),
}

# Useful ordered lists
DISPLAY_METHODS = ["Direct", "CoT", "Sequence", "Multi-turn", "VS-Standard", "VS-CoT", "VS-Multi"]
BASELINE_METHODS = ["Direct", "CoT", "Sequence", "Multi-turn"]
VS_METHODS = ["VS-Standard", "VS-CoT", "VS-Multi"]


def method_display_name_from_dir(method_dir_name: str) -> str | None:
    """
    Map a raw directory name (e.g., 'structure_with_prob') to our display name (e.g., 'VS-Standard')
    using METHOD_MAP substring matching.
    """
    method_dir_name = method_dir_name.split()[0]
    # print(method_dir_name)
    for _, (display, sub) in METHOD_MAP.items():
        if sub.lower() == method_dir_name.lower():
            # print(f"Found {sub} in {method_dir_name}")
            return display
    # also allow exact matches on display names (if the directory is already named so)
    if method_dir_name in DISPLAY_METHODS:
        return method_dir_name
    return None


def aggregate_metrics_over_prompts(per_prompt_stats: dict, metric_keys: list[str]) -> dict:
    """
    Given per-prompt stats (a dict of {prompt_id: {metric: value}}),
    return {metric: [values...]} lists across prompts (skipping missing).
    """
    out = {m: [] for m in metric_keys}
    for stats_d in per_prompt_stats.values():
        if not isinstance(stats_d, dict):
            continue
        for m in metric_keys:
            if m in stats_d and stats_d[m] is not None:
                out[m].append(stats_d[m])
    return out


def plot_method_averages(all_results, task_type, output_dir):
    """Create bar charts showing average performance across all models for each method"""
    
    # Create task-specific subdirectory
    task_output_dir = os.path.join(output_dir, task_type, "method_averages")
    os.makedirs(task_output_dir, exist_ok=True)
    
    # Use clean modern style
    plt.style.use('default')  # Start with clean slate
    plt.rcParams.update({
        'font.family': 'sans-serif',
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
    
    # Colors aligned with method types
    colors = {
        'direct': '#FFE5E5',      # Very light red
        'cot': '#FFCCCB',         # Light red  
        'sequence': '#FF9999',     # Medium red
        'multi_turn': '#FF6B6B',   # Distinct red
        'vs_standard': '#E8F4FD',  # Very light blue
        'vs_cot': '#B8E0F5',       # Light blue
        'vs_multi': '#7CC7EA'      # Medium blue
    }
    edge_colors = {
        'direct': '#FF6B6B',
        'cot': '#FF6B6B', 
        'sequence': '#FF6B6B',
        'multi_turn': '#FF6B6B',
        'vs_standard': '#4A90E2',
        'vs_cot': '#4A90E2',
        'vs_multi': '#4A90E2'
    }
    
    method_names = ["Direct", "CoT", "Sequence", "Multi-turn", "VS-Standard", "VS-CoT", "VS-Multi"]
    
    # Calculate averages and std across all models for each method
    method_stats = {}
    
    for method in method_names:
        method_stats[method] = {
            'kl_divergence': [], 'unique_recall_rate': [], 'precision': []
        }
    
    # Collect data from all models
    for model_name, results in all_results.items():
        for method in method_names:
            if results.get(method):
                data = results[method]
                for metric in ['kl_divergence', 'unique_recall_rate', 'precision']:
                    if data[metric] is not None:
                        method_stats[method][metric].append(data[metric])
    
    # Calculate means and stds
    method_means = {}
    method_stds = {}
    
    for method in method_names:
        method_means[method] = {}
        method_stds[method] = {}
        for metric in ['kl_divergence', 'unique_recall_rate', 'precision']:
            values = method_stats[method][metric]
            if values:
                method_means[method][metric] = np.mean(values)
                method_stds[method][metric] = np.std(values)
            else:
                method_means[method][metric] = 0
                method_stds[method][metric] = 0
    
    # Find best VS method for each metric
    vs_methods = ["VS-Standard", "VS-CoT", "VS-Multi"]
    baseline_methods = ["Direct", "CoT", "Sequence", "Multi-turn"]
    
    metrics = [
        ('kl_divergence', 'KL Divergence', 'Lower is Better'),
        ('unique_recall_rate', 'Coverage-N', 'Higher is Better'),
        ('precision', 'Precision', 'Higher is Better')
    ]
    
    for metric_key, metric_title, direction in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        means = [method_means[method][metric_key] for method in method_names]
        stds = [method_stds[method][metric_key] for method in method_names]
        
        # Ensure all methods have valid colors
        bar_colors = []
        bar_edge_colors = []
        for method in method_names:
            method_key = method.lower().replace('-', '_').replace(' ', '_')
            if method_key in colors:
                bar_colors.append(colors[method_key])
            else:
                print(f"Warning: Missing color for method {method} (key: {method_key})")
                bar_colors.append('#CCCCCC')  # Default gray color
            
            if method_key in edge_colors:
                bar_edge_colors.append(edge_colors[method_key])
            else:
                print(f"Warning: Missing edge color for method {method} (key: {method_key})")
                bar_edge_colors.append('#999999')  # Default gray edge color
        
        # Create bars with proper colors and edge colors
        bars = ax.bar(
            method_names, means, yerr=stds, capsize=5,
            color=bar_colors, alpha=0.8, edgecolor=bar_edge_colors,
            error_kw={'markeredgewidth': 1}
        )
        
        # Labels on bars
        for bar, mean, std in zip(bars, means, stds):
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + std + (0.01 if std > 0 else 0.005) * (max(means) if max(means) > 0 else 1.0),
                f"{mean:.2f}±{std:.2f}",
                ha='center', va='bottom', fontsize=13, fontweight='bold'
            )
        
        # Find best VS method for this metric
        best_vs_method = "VS-Standard"
        best_vs_data = method_stats[best_vs_method][metric_key]
        
        # Perform t-tests against baseline methods
        p_values = {}
        for baseline_method in baseline_methods:
            baseline_data = method_stats[baseline_method][metric_key]
            if len(baseline_data) > 1 and len(best_vs_data) > 1:
                # Perform two-sample t-test
                t_stat, p_val = stats.ttest_ind(best_vs_data, baseline_data, equal_var=False)
                p_values[baseline_method] = p_val
            else:
                p_values[baseline_method] = None
        
        # Add p-test results annotation (for diversity only to avoid clutter)
        if metric_key in ['kl_divergence', 'unique_recall_rate']:
            lines = [f"VS-Standard $p$-values:"]
            # For marking significance on the bars
            significance_marks = {}
            for baseline_method in baseline_methods:
                p = p_values[baseline_method]
                if p is None:
                    lines.append(f"{baseline_method}: insufficient data")
                    significance_marks[baseline_method] = ""
                else:
                    # Fix: If p is an array (e.g., numpy array), get scalar value
                    if hasattr(p, "__len__") and not isinstance(p, str):
                        # If p is a numpy array or similar, take the first element
                        p_scalar = float(np.asarray(p).flatten()[0])
                    else:
                        p_scalar = float(p)
                    # Always add the significance mark (***, **, *, or empty) to the error bar, even if not significant
                    if p_scalar < 0.001:
                        lines.append(f"{baseline_method}: {p_scalar:.4f} (p < 0.001) ***")
                        significance_marks[baseline_method] = "***"
                    elif p_scalar < 0.01:
                        lines.append(f"{baseline_method}: {p_scalar:.4f} (p < 0.01) **")
                        significance_marks[baseline_method] = "**"
                    elif p_scalar < 0.05:
                        lines.append(f"{baseline_method}: {p_scalar:.4f} (p < 0.05) *")
                        significance_marks[baseline_method] = "*"
                    else:
                        lines.append(f"{baseline_method}: {p_scalar:.4f} (p ≥ 0.05)")
                        significance_marks[baseline_method] = ""
            # Add significance marks (e.g., ***) to the top of the error bar for each baseline method
            for idx, method in enumerate(method_names):
                if method in baseline_methods:
                    # Get bar
                    bar = bars[idx]
                    # Place the mark at the top of the error bar
                    mean = means[idx]
                    std = stds[idx]
                    mark = significance_marks.get(method, "")
                    # Always show the mark, even if empty (for alignment)
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        mean + std + (0.05 if std > 0 else 0.03) * (max(means) if max(means) > 0 else 1.0),
                        mark,
                        ha='center', va='bottom', fontsize=18, fontweight='bold', color='red'
                    )
        
        # Highlight best performing method
        if metric_key == 'kl_divergence':  # Lower is better
            best_idx = np.argmin(means)
        else:  # Higher is better
            best_idx = np.argmax(means)
        
        # bars[best_idx].set_edgecolor('red')
        # bars[best_idx].set_linewidth(3)
        
        ax.set_ylabel(metric_title, fontsize=16, fontweight='bold')
        ax.set_title(f"", fontsize=22, fontweight='bold', pad=16)
        ax.grid(True, alpha=0.3, axis='y')
        # ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=14)
        
        # Color the x labels according to edge_colors
        xtick_labels = ax.get_xticklabels()
        for label in xtick_labels:
            method = label.get_text()
            method_key = method.lower().replace('-', '_').replace(' ', '_')
            color = edge_colors.get(method_key, "#000000")
            label.set_color(color)
        
        plt.xticks(rotation=0, fontsize=13, fontweight='bold')
        
        ymax = (max(means) if len(means) else 1.0)
        plt.ylim(0, ymax * 1.35 if ymax > 0 else 1.0)
        
        plt.tight_layout()
        
        # Save both PNG and PDF
        # out_png = os.path.join(task_output_dir, f"method_average_{metric_key}.png")
        out_pdf = os.path.join(task_output_dir, f"method_average_{metric_key}.pdf")
        # plt.savefig(out_png, dpi=300, bbox_inches='tight')
        plt.savefig(out_pdf, bbox_inches='tight')
        plt.close()
        
        # print(f"✓ Saved '{metric_title}' plots to:\n  - {out_png}\n  - {out_pdf}")
        print(f"✓ Saved '{metric_title}' plots to:\n  - {out_pdf}")
        print(f"  Best VS method: {best_vs_method}")


def output_latex_table(all_model_method_metric_values):
    """
    Print the mean and std for each method and each model, grouped by the model.
    """
    import numpy as np

    metric_keys = ["kl_divergence", "unique_recall_rate", "precision"]
    for model_name, methods_dict in all_model_method_metric_values.items():
        print(f"Model: {model_name}")
        # Print header
        print("Method & KL Divergence & Coverage-N & Precision \\\\")
        for m in DISPLAY_METHODS:
            if m in methods_dict:
                row = [m]
                for metric in metric_keys:
                    values = methods_dict[m].get(metric, [])
                    if values:
                        mean = np.mean(values)
                        std = np.std(values)
                        row.append(f"{mean:.2f}$_{{\\pm {std:.2f}}}$")
                    else:
                        row.append("-")
                print(" & ".join(row) + " \\\\")
            else:
                print(f"{m} & - & - & - \\\\")
        print("-" * 40)


def main():
    folder = "method_results_bias"
    task_name = "state_name"
    output_dir = "latex"

    # These are the metrics present in your JSONs (per your code comments)
    metric_keys = ["kl_divergence", "precision", "unique_recall_rate"]

    # Only plot a subset (labels encode direction: ↓ lower better, ↑ higher better)
    plot_metric_keys = ["kl_divergence", "unique_recall_rate", "precision"]
    metric_labels = {
        "kl_divergence": "KL Divergence",
        "unique_recall_rate": "Coverage-N",
        "precision": "Precision",
    }
    metric_directions = {
        "kl_divergence": "lower",
        "unique_recall_rate": "higher",
        "precision": "higher",
    }
    all_models = ["gpt-4.1-mini", "gpt-4.1", "gemini-2.5-flash", "gemini-2.5-pro", "qwen3-235b", "claude-4-sonnet", "deepseek-r1", "o3"]

    # Collect data:
    # all_values[model_name][display_method][metric] = list of values (across prompts)
    all_values: dict = {}

    base_path = Path(folder)
    if not base_path.exists():
        print(f"Error: folder '{folder}' not found.")
        return

    for model_dir in os.listdir(folder):
        if not model_dir.endswith(f"_{task_name}"):
            continue
        model_name = model_dir.replace(f"_{task_name}", "")
        evaluation_dir = base_path / model_dir / "evaluation"
        if not evaluation_dir.exists():
            print(f"Warning: No evaluation directory found for {model_name}")
            continue

        for method_dir in evaluation_dir.iterdir():
            if not method_dir.is_dir():
                continue

            display_method = method_display_name_from_dir(method_dir.name)
            if display_method is None:
                # skip unrecognized method directories
                print(f"Note: Skipping unrecognized method dir '{method_dir.name}'")
                continue

            results_file = method_dir / "response_count_results.json"
            if not results_file.exists():
                print(f"Warning: No results file found for {model_name} - {method_dir.name}")
                continue

            # Load and process results for this method/model
            try:
                with open(results_file, "r") as f:
                    data = json.load(f)
                aggregate_metrics = data.get("overall_metrics", {})
                per_prompt_stats = aggregate_metrics.get("per_prompt_stats", {})
                per_prompt_values = aggregate_metrics_over_prompts(per_prompt_stats, metric_keys)
            except Exception as e:
                print(f"Error reading {results_file}: {e}")
                continue

            # Initialize nested dicts as needed
            if model_name in all_models:
                if model_name not in all_values:
                    all_values[model_name] = {}
                if display_method not in all_values[model_name]:
                    all_values[model_name][display_method] = {mk: [] for mk in metric_keys}

                # Add per-prompt values for each metric
                for mk in metric_keys:
                    all_values[model_name][display_method][mk].extend(per_prompt_values[mk])

    # Plot method averages across models (using the subset you want to show)
    plot_method_averages(
        all_results=all_values,
        task_type=task_name,
        output_dir=output_dir,
        # metric_keys=plot_metric_keys,
        # metric_labels=metric_labels,
        # metric_directions=metric_directions,
    )
    # output_latex_table(all_values)


if __name__ == "__main__":
    main()
