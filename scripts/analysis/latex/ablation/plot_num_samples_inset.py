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

def load_results_data(base_path="../../ablation_data/sampling_candidates_ablation"):
    """Load actual results data from the sampling candidates ablation experiment directory"""

    models = {
        'gpt-4.1': 'gpt-4.1',
        'gemini-2.5-flash': 'gemini-2.5-flash'
    }

    method_mapping = {
        'direct': 'Direct',
        'sequence': 'Sequence',
        'vs_standard': 'VS-Standard'
    }

    # Default num_samples values from the ablation script
    num_samples_values = [1, 3, 5, 8, 10, 15, 20]

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
            for num_samples in num_samples_values:
                # Direct method only has 1 sample
                if method_dir == 'direct' and num_samples != 1:
                    continue

                if method_dir == 'direct':
                    folder_name = f"{model_key}_{method_dir}_samples_1"
                else:
                    folder_name = f"{model_key}_{method_dir}_samples_{num_samples}"

                experiment_path = os.path.join(model_path, folder_name)

                # Load diversity data
                diversity_file = os.path.join(experiment_path, "diversity_results.json")
                if os.path.exists(diversity_file):
                    print(f"✓ Loading: {diversity_file}")
                    with open(diversity_file, 'r') as f:
                        diversity_data = json.load(f)
                        diversity_score = diversity_data.get("overall_metrics", {}).get('avg_diversity', 0) * 100
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
                        results[model_key][method_name]['quality'].append(quality_score)
                else:
                    print(f"✗ Missing: {quality_file}")
                    results[model_key][method_name]['quality'].append(None)

    return results, num_samples_values

def create_method_insets(ax, results, model_key, num_samples_values):
    """Create separate inset zoom boxes for each method"""

    methods = ['Direct', 'Sequence', 'VS-Standard']
    inset_positions = [
        (0.02, 0.55, 0.35, 0.35),  # Direct: upper left (bigger, moved down)
        (0.60, 0.02, 0.35, 0.35),  # Sequence: bottom right (bigger, moved away from data)
        (0.60, 0.55, 0.35, 0.35)   # VS-Standard: upper right (bigger)
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

        # Get corresponding num_samples values for this method
        if method == 'Direct':
            valid_num_samples = [1]  # Direct only has 1 sample
        else:
            valid_num_samples = [num_samples_values[i] for i in valid_indices if i < len(num_samples_values)]

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

        # Plot individual points with marker sizes representing num_samples values
        # Normalize num_samples values to marker sizes (larger num_samples = larger marker)
        if len(valid_num_samples) > 1:
            min_samples, max_samples = min(valid_num_samples), max(valid_num_samples)
            if max_samples > min_samples:
                # Scale markers from 30 to 150 based on num_samples value (more dramatic differences)
                marker_sizes = [30 + 120 * (ns - min_samples) / (max_samples - min_samples) for ns in valid_num_samples]
            else:
                marker_sizes = [90] * len(valid_num_samples)  # All same size if all num_samples are equal
        else:
            marker_sizes = [90] * len(valid_num_samples)

        for i, (div, qual, size) in enumerate(zip(valid_diversity, valid_quality, marker_sizes)):
            if i < len(valid_diversity):  # Safety check
                axins.scatter(div, qual, s=size, c=COLORS[method], marker=marker,
                             alpha=0.9, edgecolors='white', linewidth=2, zorder=10)

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

    results, num_samples_values = load_results_data()

    print(results)
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
        create_method_insets(ax, results, model_key, num_samples_values)

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
    fig.suptitle('Sampling Candidates Ablation Study: Diversity vs Quality Analysis',
                fontsize=16, fontweight='bold', y=0.98)

    # Add method legend (moved higher)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.92),
              ncol=3, fontsize=12, frameon=False)

    # Add marker size legend for num_samples values
    legend_ax = fig.add_axes([0.91, 0.25, 0.08, 0.5])  # [x, y, width, height] - wider for box
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.axis('off')

    # Add box around legend
    from matplotlib.patches import Rectangle
    box = Rectangle((0.05, 0.05), 0.9, 0.9, linewidth=1.5, edgecolor='black',
                   facecolor='white', alpha=0.9)
    legend_ax.add_patch(box)

    # Create marker size legend with more differentiated sizes
    num_samples_legend_values = [3, 5, 10, 15, 20]  # Representative values
    min_ns, max_ns = min(num_samples_legend_values), max(num_samples_legend_values)

    # Calculate marker sizes with more dramatic differences
    legend_marker_sizes = [30 + 120 * (ns - min_ns) / (max_ns - min_ns) for ns in num_samples_legend_values]

    # Position markers vertically
    y_positions = [0.15 + 0.7 * i / (len(num_samples_legend_values) - 1) for i in range(len(num_samples_legend_values))]

    # Draw legend markers and labels
    for ns, size, y_pos in zip(num_samples_legend_values, legend_marker_sizes, y_positions):
        # Draw marker
        legend_ax.scatter(0.3, y_pos, s=size, c='lightgray', marker='o',
                         alpha=0.8, edgecolors='black', linewidth=1.5)
        # Add label
        legend_ax.text(0.65, y_pos, f'{ns}', fontsize=11, fontweight='bold',
                      verticalalignment='center', horizontalalignment='left')

    # Add legend title
    legend_ax.text(0.5, 0.92, 'Sample Count', fontsize=12, fontweight='bold',
                  verticalalignment='top', horizontalalignment='center')

    plt.tight_layout()
    plt.subplots_adjust(top=0.88, right=0.88)

    # Save the plot
    output_path = "num_samples_ablation_inset.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace(".pdf", ".png"), dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_path}")

if __name__ == "__main__":
    plot_inset_comparison()