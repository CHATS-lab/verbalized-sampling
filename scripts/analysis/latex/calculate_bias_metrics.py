import json
import os
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns  # noqa: F401  (style only)
from matplotlib import gridspec
from matplotlib import font_manager as fm
from scipy.stats import ttest_ind

# ----------------------------
# Font setup: News Gothic MT + fallback
# ----------------------------
# NOTE: DejaVu Sans is kept as fallback to avoid "glyph missing" warnings (e.g., ↑).
# font_path = "/Users/jiayizx/Downloads/NewsGothicMT.ttf"
# font_name = None
# try:
#     if os.path.exists(font_path):
#         fm.fontManager.addfont(font_path)
#         font_name = fm.FontProperties(fname=font_path).get_name()
# except Exception as e:
#     print(f"Warning: could not add font at {font_path}: {e}")

# plt.rcParams["font.size"] = 12
# plt.rcParams["font.family"] = [font_name]
# plt.rcParams["axes.unicode_minus"] = False  # safer for some unicode glyphs
# print(font_name)

# ----------------------------
# Method mapping (dir substring -> display name)
# ----------------------------
METHOD_MAP = {
    "direct": ("Direct", "direct"),
    "direct_cot": ("CoT", "direct_cot"),
    "sequence": ("Sequence", "sequence"),
    "multi_turn": ("Multi-turn", "multi_turn"),
    "vs_standard": ("VS-Standard", "vs_standard"),
    "vs_cot": ("VS-CoT", "vs_cot"),
    "vs_combined": ("VS-Multi", "vs_multi"),
}

# Useful ordered lists
DISPLAY_METHODS = ["Direct", "CoT", "Sequence", "Multi-turn", "VS-Standard", "VS-CoT", "VS-Multi"]
BASELINE_METHODS = ["Direct", "CoT", "Sequence", "Multi-turn"]
VS_METHODS = ["VS-Standard", "VS-CoT", "VS-Multi"]


def method_display_name_from_dir(method_dir_name: str) -> str | None:
    """
    Map a raw directory name (e.g., 'vs_standard') to our display name (e.g., 'VS-Standard')
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


def perform_statistical_tests(all_results, task_type, metric='diversity'):
    """Perform t-tests comparing baselines against VS-Standard for a given metric"""
    print(f"\n=== Statistical Tests for {task_type} - {metric} ===")
    
    # Collect individual model data for VS-Standard
    vs_standard_values = []
    for model_name, model_results in all_results.items():
        if 'VS-Standard' in model_results and model_results['VS-Standard'].get(metric):
            # Add all values from this model for VS-Standard
            vs_standard_values.extend(model_results['VS-Standard'][metric])
    
    if not vs_standard_values:
        print(f"No VS-Standard data found for {task_type} - {metric}")
        return {}
    
    baseline_methods = ['Direct', 'CoT', 'Sequence', 'Multi-turn']
    p_values = {}
    
    for method in baseline_methods:
        # Collect individual model data for this baseline method
        baseline_values = []
        for model_name, model_results in all_results.items():
            if method in model_results and model_results[method].get(metric):
                # Add all values from this model for the baseline method
                baseline_values.extend(model_results[method][metric])
        
        if len(baseline_values) < 2 or len(vs_standard_values) < 2:
            print(f"Insufficient data for {method} vs VS-Standard comparison")
            p_values[method] = None
            continue
            
        # Perform two-sample t-test (one-tailed: VS-Standard > baseline for higher-is-better metrics)
        # For KL divergence (lower is better), we test VS-Standard < baseline
        if metric == 'kl_divergence':
            t_stat, p_val = ttest_ind(vs_standard_values, baseline_values, alternative='less')
        else:
            t_stat, p_val = ttest_ind(vs_standard_values, baseline_values, alternative='greater')
        
        vs_mean = np.mean(vs_standard_values)
        baseline_mean = np.mean(baseline_values)
        
        p_values[method] = p_val
        
        significance_marker = "**" if p_val < 0.05 else ""
        print(f"{method}{significance_marker}: VS-Standard ({vs_mean:.2f}) vs {method} ({baseline_mean:.2f}), t={t_stat:.3f}, p={p_val:.4f}")
    
    return p_values


def plot_method_averages(all_results, task_type, output_dir):
    """Create bar charts showing average performance across all models for each method"""
    
    # Create task-specific subdirectory
    task_output_dir = os.path.join(output_dir, task_type, "method_averages")
    os.makedirs(task_output_dir, exist_ok=True)
    
    # Use clean modern style
    plt.style.use('default')  # Start with clean slate
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'font.size': 20,
        'axes.labelsize': 28,
        'axes.titlesize': 30,
        'xtick.labelsize': 22,
        'ytick.labelsize': 22,
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
                f"{mean:.2f}",
                ha='center', va='bottom', fontweight='bold'
            )
        
        # Find best VS method for this metric
        best_vs_method = "VS-Standard"
        best_vs_data = method_stats[best_vs_method][metric_key]
        
        # Perform t-tests against baseline methods using perform_statistical_tests
        p_values = perform_statistical_tests(all_results, task_type, metric_key)
        
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
                        ha='center', va='bottom', fontsize=28, fontweight='bold', color='red'
                    )
        
        # Highlight best performing method
        if metric_key == 'kl_divergence':  # Lower is better
            best_idx = np.argmin(means)
        else:  # Higher is better
            best_idx = np.argmax(means)
        
        # bars[best_idx].set_edgecolor('red')
        # bars[best_idx].set_linewidth(3)
        
        ax.set_ylabel(metric_title, fontweight='bold')
        ax.set_title(f"", fontweight='bold', pad=16)
        ax.grid(True, alpha=0.3, axis='y')
        # ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y')
        
        # # Color the x labels according to edge_colors
        # xtick_labels = ax.get_xticklabels()
        # for label in xtick_labels:
        #     method = label.get_text()
        #     method_key = method.lower().replace('-', '_').replace(' ', '_')
        #     color = edge_colors.get(method_key, "#000000")
        #     label.set_color(color)
        
        plt.xticks(rotation=30)
        
        ymax = (max(means) if len(means) else 1.0)
        plt.ylim(0, ymax * 1.35 if ymax > 0 else 1.0)
        
        if metric_key == 'precision' or metric_key == 'unique_recall_rate':
            plt.title(f"{metric_title} ($\\uparrow$)", fontweight='bold', pad=16)
        else:
            plt.title(f"{metric_title} ($\\downarrow$)",fontweight='bold', pad=16)
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


def plot_combined_metrics(all_results, task_type, output_dir):
    """Create a VS-Multi (vs_multi) graph showing KL divergence and Coverage-N metrics with legend on top"""
    
    # Create task-specific subdirectory
    task_output_dir = os.path.join(output_dir, task_type, "combined_metrics")
    os.makedirs(task_output_dir, exist_ok=True)
    
    # Use clean modern style
    plt.style.use('default')  # Start with clean slate
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'font.size': 24,
        'axes.labelsize': 28,
        'axes.titlesize': 32,
        'xtick.labelsize': 28,
        'ytick.labelsize': 28,
        'legend.fontsize': 24,
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
    
    
    method_names = ["Direct", "CoT", "Sequence", "Multi-turn", "VS-Standard", "VS-CoT", "VS-Multi"]
    
    # Calculate averages and std across all models for each method
    method_stats = {}
    
    for method in method_names:
        method_stats[method] = {
            'kl_divergence': [], 'unique_recall_rate': []
        }
    
    # Collect data from all models
    for model_name, results in all_results.items():
        for method in method_names:
            if results.get(method):
                data = results[method]
                for metric in ['kl_divergence', 'unique_recall_rate']:
                    if data[metric] is not None:
                        method_stats[method][metric].append(data[metric])
    
    # Calculate means and stds
    method_means = {}
    method_stds = {}
    
    for method in method_names:
        method_means[method] = {}
        method_stds[method] = {}
        for metric in ['kl_divergence', 'unique_recall_rate']:
            values = method_stats[method][metric]
            if values:
                method_means[method][metric] = np.mean(values)
                method_stds[method][metric] = np.std(values)
            else:
                method_means[method][metric] = 0
                method_stds[method][metric] = 0
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 6))
    
    # KL Divergence (lower is better) - left subplot
    kl_means = [method_means[method]['kl_divergence'] for method in method_names]
    kl_stds = [method_stds[method]['kl_divergence'] for method in method_names]
    
    # Coverage-N (higher is better) - right subplot  
    coverage_means = [method_means[method]['unique_recall_rate'] for method in method_names]
    coverage_stds = [method_stds[method]['unique_recall_rate'] for method in method_names]
    
    # Plot KL Divergence (left subplot)
    bar_colors = []
    bar_edge_colors = []
    for method in method_names:
        method_key = method.lower().replace('-', '_').replace(' ', '_')
        if method_key in colors:
            bar_colors.append(colors[method_key])
        else:
            bar_colors.append('#CCCCCC')
        
        if method_key in edge_colors:
            bar_edge_colors.append(edge_colors[method_key])
        else:
            bar_edge_colors.append('#999999')
    
    # KL Divergence bars (left) - using same style as plot_method_averages
    bars1 = ax1.bar(
        method_names, kl_means, yerr=kl_stds, capsize=5,
        color=bar_colors, alpha=0.9, edgecolor=bar_edge_colors,
        error_kw={'markeredgewidth': 3}
    )
    
    # Add value labels on KL Divergence bars - same style as plot_method_averages
    for bar, mean, std in zip(bars1, kl_means, kl_stds):
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + std + (0.03 if std > 0 else 0.005) * (max(kl_means) if max(kl_means) > 0 else 1.0),
            # f"{mean:.2f}±{std:.2f}",
            f"{mean:.2f}",
            ha='center', va='bottom', fontweight='bold'
        )
    
    # Coverage-N bars (right) - using same style as plot_method_averages
    bars2 = ax2.bar(
        method_names, coverage_means, yerr=coverage_stds, capsize=5,
        color=bar_colors, alpha=0.9, edgecolor=bar_edge_colors,
        error_kw={'markeredgewidth': 3}
    )
    
    # Add value labels on Coverage-N bars - same style as plot_method_averages
    for bar, mean, std in zip(bars2, coverage_means, coverage_stds):
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + std + (0.03 if std > 0 else 0.005) * (max(coverage_means) if max(coverage_means) > 0 else 1.0),
            # f"{mean:.2f}±{std:.2f}",
            f"{mean:.2f}",
            ha='center', va='bottom', fontweight='bold'
        )
    
    # Customize left subplot (KL Divergence) - same style as plot_method_averages
    ax1.set_ylabel('KL Divergence', fontweight='bold')
    ax1.set_title('KL Divergence ($\\downarrow$)',fontweight='bold', pad=16)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='y', labelsize=24)
    
    # Add subplot label A for KL Divergence
    ax1.text(-0.02, 1.05, 'a', transform=ax1.transAxes, fontsize=36, fontweight='bold', 
              ha='center', va='bottom')
    
    # Customize right subplot (Coverage-N) - same style as plot_method_averages
    ax2.set_ylabel('Coverage-N', fontweight='bold')
    ax2.set_title('Coverage-N ($\\uparrow$)', fontweight='bold', pad=16)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='y', labelsize=24)
    
    # Add subplot label B for Coverage-N (move more to the right and up)
    ax2.text(-0.02, 1.05, 'b', transform=ax2.transAxes, fontsize=36, fontweight='bold', 
              ha='center', va='bottom')
    
    # Set x-axis labels for both subplots (compact) with correct alignment
    for ax in [ax1, ax2]:
        ax.tick_params(axis='x', color='black', labelsize=24)
        # Fix: Set rotation and alignment for x-tick labels individually to avoid matplotlib error
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_horizontalalignment('right')
    
    
    # Set y-axis limits - same logic as plot_method_averages
    ymax1 = (max(kl_means) if len(kl_means) else 1.0)
    ymax2 = (max(coverage_means) if len(coverage_means) else 1.0)
    ax1.set_ylim(0, ymax1 * 1.5 if ymax1 > 0 else 1.0)
    ax2.set_ylim(0, ymax2 * 1.5 if ymax2 > 0 else 1.0)
    
    # Perform statistical tests and add significance markers - same as plot_method_averages
    baseline_methods = ["Direct", "CoT", "Sequence", "Multi-turn"]
    vs_methods = ["VS-Standard", "VS-CoT", "VS-Multi"]
    
    # For KL Divergence (left subplot)
    best_vs_method = "VS-Standard"
    best_vs_data = method_stats[best_vs_method]['kl_divergence']
    
    # Perform t-tests against baseline methods for KL Divergence using perform_statistical_tests
    p_values_kl = perform_statistical_tests(all_results, task_type, 'kl_divergence')
    
    # Add significance marks for KL Divergence
    significance_marks_kl = {}
    for baseline_method in baseline_methods:
        p = p_values_kl[baseline_method]
        if p is None:
            significance_marks_kl[baseline_method] = ""
        else:
            # Fix: If p is an array (e.g., numpy array), get scalar value
            if hasattr(p, "__len__") and not isinstance(p, str):
                # If p is a numpy array or similar, take the first element
                p_scalar = float(np.asarray(p).flatten()[0])
            else:
                p_scalar = float(p)
            # Always add the significance mark (***, **, *, or empty) to the error bar, even if not significant
            if p_scalar < 0.001:
                significance_marks_kl[baseline_method] = "***"
            elif p_scalar < 0.01:
                significance_marks_kl[baseline_method] = "**"
            elif p_scalar < 0.05:
                significance_marks_kl[baseline_method] = "*"
            else:
                significance_marks_kl[baseline_method] = ""
    
    # Add significance marks to KL Divergence bars (higher placement)
    for idx, method in enumerate(method_names):
        if method in baseline_methods:
            # Get bar
            bar = bars1[idx]
            # Place the mark higher above the error bar
            mean = kl_means[idx]
            std = kl_stds[idx]
            mark = significance_marks_kl.get(method, "")
            # Always show the mark, even if empty (for alignment)
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                mean + std + (0.1 if std > 0 else 0.08) * (max(kl_means) if max(kl_means) > 0 else 1.0),
                mark,
                ha='center', va='bottom', fontsize=28, fontweight='bold', color='red'
            )
    
    # For Coverage-N (right subplot)
    best_vs_data_coverage = method_stats[best_vs_method]['unique_recall_rate']
    
    # Perform t-tests against baseline methods for Coverage-N using perform_statistical_tests
    p_values_coverage = perform_statistical_tests(all_results, task_type, 'unique_recall_rate')
    
    # Add significance marks for Coverage-N
    significance_marks_coverage = {}
    for baseline_method in baseline_methods:
        p = p_values_coverage[baseline_method]
        if p is None:
            significance_marks_coverage[baseline_method] = ""
        else:
            # Fix: If p is an array (e.g., numpy array), get scalar value
            if hasattr(p, "__len__") and not isinstance(p, str):
                # If p is a numpy array or similar, take the first element
                p_scalar = float(np.asarray(p).flatten()[0])
            else:
                p_scalar = float(p)
            # Always add the significance mark (***, **, *, or empty) to the error bar, even if not significant
            if p_scalar < 0.001:
                significance_marks_coverage[baseline_method] = "***"
            elif p_scalar < 0.01:
                significance_marks_coverage[baseline_method] = "**"
            elif p_scalar < 0.05:
                significance_marks_coverage[baseline_method] = "*"
            else:
                significance_marks_coverage[baseline_method] = ""
    
    # Add significance marks to Coverage-N bars
    for idx, method in enumerate(method_names):
        if method in baseline_methods:
            # Get bar
            bar = bars2[idx]
            # Place the mark at the top of the error bar
            mean = coverage_means[idx]
            std = coverage_stds[idx]
            mark = significance_marks_coverage.get(method, "")
            # Always show the mark, even if empty (for alignment)
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                mean + std + (0.1 if std > 0 else 0.08) * (max(coverage_means) if max(coverage_means) > 0 else 1.0),
                mark,
                ha='center', va='bottom', fontsize=28, fontweight='bold', color='red'
            )
    
    # Add legend at the top of the figure
    # Compact legend for all baseline and VS methods
    method_labels = ["Direct", "CoT", "Sequence", "Multi-turn", "VS-Standard", "VS-CoT", "VS-Multi"]
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor=colors[m.lower().replace("-", "_").replace(" ", "_")], 
                      edgecolor=edge_colors[m.lower().replace("-", "_").replace(" ", "_")], label=m)
        for m in method_labels
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.15), 
               ncol=7, frameon=False)
    
    plt.tight_layout()
    
    # Adjust layout to make room for legend
    plt.subplots_adjust(top=0.92)
    
    # Save both PNG and PDF
    out_pdf = os.path.join(task_output_dir, f"combined_kl_coverage.pdf")
    plt.savefig(out_pdf, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"✓ Saved combined KL Divergence and Coverage-N plot to:\n  - {out_pdf}")


def output_latex_table(all_model_method_metric_values, average_across_models=False):
    """
    Print the mean and std for each method and each model, grouped by the model,
    in a LaTeX table format matching the requested style.
    Also calculates the average for each method across all models.
    """
    import numpy as np

    # Order and display names for methods
    method_rows = [
        ("Direct", None, "Direct"),
        ("CoT", None, "CoT"),
        ("Sequence", None, "Sequence"),
        ("Multi-turn", None, "Multi-turn"),
        (None, r"\textbf{Verbalized Sampling:}", None),
        ("VS-Standard", r"$\hookrightarrow$ Standard", "VS-Standard"),
        ("VS-CoT", r"$\hookrightarrow$ CoT", "VS-CoT"),
        ("VS-Multi", r"$\hookrightarrow$ Multi-turn", "VS-Multi"),
    ]
    metric_keys = ["kl_divergence", "unique_recall_rate", "precision"]

    # For method averages across all models
    # method_agg[dict_key][metric] = list of all values across all models
    method_agg = {row[2]: {metric: [] for metric in metric_keys} for row in method_rows if row[2] is not None}

    for model_name, methods_dict in all_model_method_metric_values.items():
        print(f"Model: {model_name}")
        print("Method & KL Divergence & Coverage-N & Precision \\\\")
        
        # Gather all means for each metric for this model, for best/second-best marking
        metric_means = {metric: [] for metric in metric_keys}
        for method_key, display_label, dict_key in method_rows:
            if method_key is None:
                continue
            metrics = methods_dict.get(dict_key, {})
            for i, metric in enumerate(metric_keys):
                values = metrics.get(metric, [])
                if values:
                    mean = np.mean(values)
                    metric_means[metric].append((mean, method_key))
                else:
                    metric_means[metric].append((None, method_key))
            # Aggregate for method averages
            if dict_key is not None:
                for metric in metric_keys:
                    values = metrics.get(metric, [])
                    if values:
                        method_agg[dict_key][metric].extend(values)
        
        # For each metric, determine best and second best (handle direction)
        best_methods = {}
        second_methods = {}
        for i, metric in enumerate(metric_keys):
            vals = [(mean, mkey) for mean, mkey in metric_means[metric] if mean is not None]
            if not vals:
                continue
            # Direction: kl_divergence is lower better, others higher better
            reverse = metric != "kl_divergence"
            sorted_vals = sorted(vals, key=lambda x: x[0], reverse=reverse)
            best_methods[metric] = sorted_vals[0][1]
            if len(sorted_vals) > 1:
                second_methods[metric] = sorted_vals[1][1]
            else:
                second_methods[metric] = None

        for idx, (method_key, display_label, dict_key) in enumerate(method_rows):
            if method_key is None:
                # Section header row (e.g., Verbalized Sampling)
                print(f"& {display_label} \\\\")
                continue
            metrics = methods_dict.get(dict_key, {})
            row = []
            for i, metric in enumerate(metric_keys):
                values = metrics.get(metric, [])
                if values:
                    mean = np.mean(values)
                    std = np.std(values)
                    cell = f"{mean:.2f}$_{{\\pm {std:.2f}}}$"
                    # Mark best/second best
                    if method_key == best_methods.get(metric):
                        cell = f"\\bestcell{{{cell}}}"
                    elif method_key == second_methods.get(metric):
                        cell = f"\\secondcell{{{cell}}}"
                    row.append(cell)
                else:
                    row.append("-")
            # For VS rows, indent with & and use display_label
            if display_label is not None:
                print(f"& {display_label}  & " + " & ".join(row) + r" \\")
            else:
                print(f"& {method_key:<16} & " + " & ".join(row) + r" \\")
        print("-" * 40)

    if average_across_models:
        # Now print the average across all models for each method
        print("Average across all models:")
        print("Method & KL Divergence & Coverage-N & Precision \\\\")
        # For best/second-best marking across methods (for each metric)
        avg_metric_means = {metric: [] for metric in metric_keys}
        for method_key, display_label, dict_key in method_rows:
            if method_key is None:
                continue
            row = []
            for metric in metric_keys:
                values = method_agg[dict_key][metric]
                if values:
                    mean = np.mean(values)
                    std = np.std(values)
                    avg_metric_means[metric].append((mean, method_key))
                else:
                    avg_metric_means[metric].append((None, method_key))
        # Determine best/second-best for averages
        avg_best_methods = {}
        avg_second_methods = {}
        for metric in metric_keys:
            vals = [(mean, mkey) for mean, mkey in avg_metric_means[metric] if mean is not None]
            if not vals:
                continue
            reverse = metric != "kl_divergence"
            sorted_vals = sorted(vals, key=lambda x: x[0], reverse=reverse)
            avg_best_methods[metric] = sorted_vals[0][1]
            if len(sorted_vals) > 1:
                avg_second_methods[metric] = sorted_vals[1][1]
            else:
                avg_second_methods[metric] = None

        for method_key, display_label, dict_key in method_rows:
            if method_key is None:
                print(f"& {display_label} \\\\")
                continue
            row = []
            for metric in metric_keys:
                values = method_agg[dict_key][metric]
                if values:
                    mean = np.mean(values)
                    std = np.std(values)
                    cell = f"{mean:.2f}$_{{\\pm {std:.2f}}}$"
                    if method_key == avg_best_methods.get(metric):
                        cell = f"\\bestcell{{{cell}}}"
                    elif method_key == avg_second_methods.get(metric):
                        cell = f"\\secondcell{{{cell}}}"
                    row.append(cell)
                else:
                    row.append("-")
            if display_label is not None:
                print(f"& {display_label}  & " + " & ".join(row) + r" \\")
            else:
                print(f"& {method_key:<16} & " + " & ".join(row) + r" \\")
        print("-" * 40)


def main():
    # folder = "method_results_bias"
    folder = "generated_data/openended_qa_general"
    # folder = "openended_qa_coverageqa"
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
        # if not model_dir.endswith(f"_{task_name}"):
        #     continue
        # model_name = model_dir.replace(f"_{task_name}", "")
        model_name = model_dir
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

    # print(all_values)
    # Plot method averages across models (using the subset you want to show)
    plot_method_averages(
        all_results=all_values,
        task_type=task_name,
        output_dir=output_dir,
        # metric_keys=plot_metric_keys,
        # metric_labels=metric_labels,
        # metric_directions=metric_directions,
    )
    
    # Plot VS-Multi (vs_multi) KL divergence and Coverage-N metrics
    plot_combined_metrics(
        all_results=all_values,
        task_type=task_name,
        output_dir=output_dir
    )
    
    # output_latex_table(all_values, average_across_models=True)


if __name__ == "__main__":
    main()
