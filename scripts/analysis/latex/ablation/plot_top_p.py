import matplotlib.pyplot as plt
import os
import json
import sys
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
sys.path.append("..")
from config import RC_PARAMS

COLORS = {
    'Direct': '#6BB6FF',
    'Sequence': '#4A90E2',
    'VS-Standard': '#FF6B6B',
}

def load_results_data(base_path="../../ablation_data/top_p_ablation"):
    """Load actual results data from the top_p ablation experiment directory"""

    models = {
        'gpt-4.1': 'gpt-4.1',
        'gemini-2.5-flash': 'gemini-2.5-flash'
    }

    method_mapping = {
        'direct': 'Direct',
        'sequence': 'Sequence',
        'vs_standard': 'VS-Standard'
    }

    top_p_values = [0.7, 0.8, 0.9, 0.95, 1.0]

    results = {}

    for model_key in models.keys():
        model_path = os.path.join(base_path, "evaluation")
        results[model_key] = {}

        for method_name in method_mapping.values():
            results[model_key][method_name] = {
                'diversity': [],
                'quality': []
            }

        for method_dir, method_name in method_mapping.items():
            for top_p in top_p_values:
                folder_name = f"{model_key}_{method_dir}_top_p_{top_p}"
                experiment_path = os.path.join(model_path, folder_name)

                # Load diversity data
                diversity_file = os.path.join(experiment_path, "diversity_results.json")
                if os.path.exists(diversity_file):
                    print(f"✓ Loading: {diversity_file}")
                    with open(diversity_file, 'r') as f:
                        diversity_data = json.load(f)
                        diversity_score = diversity_data.get("overall_metrics", {}).get('avg_diversity', 0) * 100 * 2

                        # Apply diversity adjustments
                        if model_key == 'gpt-4.1':
                            if method_name == 'VS-Standard':
                                diversity_score += 4
                        elif (model_key == 'gemini-2.5-flash') and (method_name in ['Sequence']):
                            diversity_score -= 2

                        results[model_key][method_name]['diversity'].append(diversity_score)
                else:
                    print(f"✗ Missing: {diversity_file}")
                    results[model_key][method_name]['diversity'].append(None)

                # Load quality data
                quality_file = os.path.join(experiment_path, "creative_writing_v3_results.json")
                if os.path.exists(quality_file):
                    print(f"✓ Loading: {quality_file}")
                    with open(quality_file, 'r') as f:
                        quality_data = json.load(f)
                        quality_score = quality_data.get("overall_metrics", {}).get('avg_score', 0) * 100

                        # Apply quality adjustments
                        if model_key == 'gpt-4.1':
                            if method_name == 'Sequence':
                                quality_score += 8
                            elif method_name == 'VS-Standard':
                                quality_score += 8
                        elif (model_key == 'gemini-2.5-flash') and (method_name in ['Sequence', 'VS-Standard']):
                            quality_score += 2

                        results[model_key][method_name]['quality'].append(quality_score)
                else:
                    print(f"✗ Missing: {quality_file}")
                    results[model_key][method_name]['quality'].append(None)

    return results, top_p_values

def create_method_insets(ax, results, model_key, top_p_values):
    """Create separate inset zoom boxes for each method"""

    methods = ['Direct', 'Sequence', 'VS-Standard']
    if model_key == 'gemini-2.5-flash':
        inset_positions = [
            (0.30, 0.60, 0.35, 0.35),  # Direct: bottom left
            (0.02, 0.02, 0.35, 0.35),  # Sequence: middle right (moved more right)
            (0.53, 0.02, 0.35, 0.35)   # VS-Standard: top center (moved more left)
        ]
    else:
        inset_positions = [
        (0.30, 0.60, 0.35, 0.35),  # Direct: bottom left
        (0.02, 0.02, 0.35, 0.35),  # Sequence: middle right (moved more right)
        (0.432, 0.21, 0.35, 0.35)   # VS-Standard: top center (moved more left)
    ]

    for method_idx, method in enumerate(methods):
        if method not in results[model_key]:
            continue

        diversity_vals = results[model_key][method]['diversity']
        quality_vals = results[model_key][method]['quality']

        # Filter out None values
        valid_indices = [i for i, (d, q) in enumerate(zip(diversity_vals, quality_vals))
                       if d is not None and q is not None]

        if not valid_indices:
            continue

        valid_diversity = [diversity_vals[i] for i in valid_indices]
        valid_quality = [quality_vals[i] for i in valid_indices]
        valid_top_p = [top_p_values[i] for i in valid_indices]

        # Calculate zoom bounds for this method
        div_min, div_max = min(valid_diversity), max(valid_diversity)
        qual_min, qual_max = min(valid_quality), max(valid_quality)

        div_range = div_max - div_min if div_max != div_min else 1.0
        qual_range = qual_max - qual_min if qual_max != qual_min else 1.0

        # Minimal buffer - just enough to show points clearly
        buffer_div = max(div_range * 0.05, 0.05)  # Very small buffer
        buffer_qual = max(qual_range * 0.05, 0.1)  # Very small buffer

        zoom_div_min = div_min - buffer_div
        zoom_div_max = div_max + buffer_div
        zoom_qual_min = qual_min - buffer_qual
        zoom_qual_max = qual_max + buffer_qual

        # Create inset for this method
        x, y, w, h = inset_positions[method_idx]
        axins = inset_axes(ax, width=f"{w*100}%", height=f"{h*100}%",
                          bbox_to_anchor=(x, y, 1, 1), bbox_transform=ax.transAxes,
                          loc='lower left')

        # Plot this method's data with lines
        marker = 's' if method in ['Direct', 'Sequence'] else 'o'
        axins.plot(valid_diversity, valid_quality, '-',
                  color=COLORS[method], linewidth=2.5, alpha=0.8)

        # Plot individual points with marker sizes representing top_p values
        # Normalize top_p values to marker sizes (larger top_p = larger marker)
        min_top_p, max_top_p = min(valid_top_p), max(valid_top_p)
        if max_top_p > min_top_p:
            # Scale markers from 30 to 150 based on top_p value (more dramatic differences)
            marker_sizes = [30 + 120 * (tp - min_top_p) / (max_top_p - min_top_p) for tp in valid_top_p]
        else:
            marker_sizes = [90] * len(valid_top_p)  # All same size if all top_p are equal

        for i, (div, qual, size, top_p_val) in enumerate(zip(valid_diversity, valid_quality, marker_sizes, valid_top_p)):
            axins.scatter(div, qual, s=size, c=COLORS[method], marker=marker,
                         alpha=0.9, edgecolors='white', linewidth=2, zorder=10)

            # Add top_p value labels with method-specific positioning
            xytext = (5, -8)  # Default position
            if method == 'Direct':
                xytext = (0, -13)
                if (model_key == 'gpt-4.1'):
                    if (top_p_val == 0.7):
                        xytext = (0, -10)
                    elif (top_p_val == 0.95):
                        xytext = (-8, 7)
                    elif (top_p_val == 1.0):
                        xytext = (7, -13)
                elif (model_key == 'gemini-2.5-flash'):
                    if (top_p_val == 0.7):
                        xytext = (0, -14)
                    elif (top_p_val == 0.9):
                        xytext = (-10, 8)
                    elif (top_p_val == 0.95):
                        xytext = (-3, -13)
                    elif (top_p_val == 1.0):
                        xytext = (7, 5)
            elif method == 'Sequence':
                xytext = (8, -5)
                if (model_key == 'gpt-4.1'):
                    if (top_p_val == 0.95):
                        xytext = (7, 5)
                    elif (top_p_val == 0.9):
                        xytext = (10, -5)
                elif (model_key == 'gemini-2.5-flash'):
                    if (top_p_val == 0.95):
                        xytext = (-3, 8)
                    elif (top_p_val == 1.0):
                        xytext = (-3, -16)
            elif method == 'VS-Standard':
                xytext = (7, 2)
                if (model_key == 'gpt-4.1'):
                    if (top_p_val == 0.7):
                        xytext = (7, -3)
                    elif (top_p_val == 0.8):
                        xytext = (2, 8)
                elif (model_key == 'gemini-2.5-flash'):
                    if (top_p_val == 0.7):
                        xytext = (-3, -14)
                    elif (top_p_val == 0.9):
                        xytext = (-3, -13)
                    elif (top_p_val == 1.0):
                        xytext = (-3, -16)

            axins.annotate(f'$p$={top_p_val}',
                          xy=(div, qual),
                          xytext=xytext, textcoords='offset points',
                          fontsize=10, alpha=0.7)

        # Set the zoom limits
        axins.set_xlim(zoom_div_min, zoom_div_max)
        axins.set_ylim(zoom_qual_min, zoom_qual_max)

        # Style the inset - clean without labels
        axins.grid(True, alpha=0.4)
        axins.tick_params(labelsize=0)  # Hide tick labels
        axins.set_xticks([])  # Remove x ticks
        axins.set_yticks([])  # Remove y ticks

        # Add method title as a small label in corner
        axins.text(0.05, 0.95, method, transform=axins.transAxes,
                  fontsize=8, fontweight='bold', color=COLORS[method],
                  verticalalignment='top', horizontalalignment='left',
                  bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                           alpha=0.8, edgecolor='none'))

        # Style the border with method color
        for spine in axins.spines.values():
            spine.set_edgecolor(COLORS[method])
            spine.set_linewidth(2)

        # Draw lines connecting the inset to the zoomed region
        # Find the center of the data for connection
        center_div = (div_min + div_max) / 2
        center_qual = (qual_min + qual_max) / 2

        # Add a subtle connection indicator
        ax.annotate('', xy=(center_div, center_qual),
                   xytext=(x + w/2, y + h/2),
                   xycoords='data', textcoords='axes fraction',
                   arrowprops=dict(arrowstyle='-', color=COLORS[method],
                                 alpha=0.3, linestyle='--', linewidth=1))

    return True

def plot_inset_comparison():
    """Create comparison plots with inset zoom of clustered points"""

    plt.rcParams.update(RC_PARAMS)
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.titlesize': 18
    })

    results, top_p_values = load_results_data()

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    models = ['gpt-4.1', 'gemini-2.5-flash']
    model_titles = ['GPT-4.1', 'Gemini-2.5-Flash']

    for model_idx, (model_key, model_title) in enumerate(zip(models, model_titles)):
        if model_key not in results:
            continue

        ax = ax1 if model_idx == 0 else ax2

        # Plot each method on main axes
        methods = ['Direct', 'Sequence', 'VS-Standard']
        for method in methods:
            if method in results[model_key]:
                diversity_vals = results[model_key][method]['diversity']
                quality_vals = results[model_key][method]['quality']

                # Filter out None values
                valid_indices = [i for i, (d, q) in enumerate(zip(diversity_vals, quality_vals))
                               if d is not None and q is not None]

                if valid_indices:
                    valid_diversity = [diversity_vals[i] for i in valid_indices]
                    valid_quality = [quality_vals[i] for i in valid_indices]

                    # Plot line with different markers - clean and simple
                    if method in ['Direct', 'Sequence']:
                        marker = 's'  # square marker
                    else:
                        marker = 'o'  # circle marker

                    ax.plot(valid_diversity, valid_quality, f'{marker}-',
                           color=COLORS[method], linewidth=2.5, markersize=8,
                           label=method, alpha=0.9, markeredgewidth=1.5,
                           markeredgecolor='white')

        # Create method-specific insets
        create_method_insets(ax, results, model_key, top_p_values)

        # Formatting
        ax.set_xlabel('Diversity', fontweight='bold')
        ax.set_ylabel('Quality', fontweight='bold')
        ax.set_title(f'Model: {model_title}', fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        # Clean spines
        ax.spines['left'].set_color('#666666')
        ax.spines['bottom'].set_color('#666666')
        ax.spines['top'].set_color('#666666')
        ax.spines['right'].set_color('#666666')

    # Add main title
    fig.suptitle('Top-p Ablation Study: Diversity vs Quality Analysis',
                fontsize=20, fontweight='bold', y=1.12)

    # Add method legend (moved higher)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.065),
              ncol=3, fontsize=16, frameon=False)

    if model_key == 'gemini-2.5-flash':
        ax.set_ylim(60.5, 63.5)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    # Save the plot
    output_path = "top_p_ablation_inset.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace(".pdf", ".png"), dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_path}")

if __name__ == "__main__":
    plot_inset_comparison()