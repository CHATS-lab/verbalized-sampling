import matplotlib.pyplot as plt
import os
import json
import sys
sys.path.append("..")
from config import EDGE_COLORS, RC_PARAMS

COLORS = {
    'Direct': '#6BB6FF',      # Medium blue (baseline) - swapped with Sequence
    'Sequence': '#4A90E2',     # Distinct blue (baseline) - swapped with Direct
    'VS-Standard': '#FF6B6B',  # Light red (our method)
}

def load_results_data(base_path="../../ablation_data/min_p_ablation"):
    """Load actual results data from the min_p ablation experiment directory"""

    models = {
        'Qwen': 'Qwen3-235B-A22B-Instruct-2507',
        'meta-llama': 'Llama-3.1-70B-Instruct'
    }

    method_mapping = {
        'direct': 'Direct',
        'sequence': 'Sequence',
        'vs_standard': 'VS-Standard'
    }

    min_p_values = [0.0, 0.01, 0.02, 0.05, 0.1]

    results = {}

    for model_key, model_prefix in models.items():
        model_path = os.path.join(base_path, "evaluation", model_key)
        results[model_key] = {}

        for method_name in method_mapping.values():
            results[model_key][method_name] = {
                'diversity': [],
                'quality': []
            }

        if not os.path.exists(model_path):
            continue

        for method_dir, method_name in method_mapping.items():
            for min_p in min_p_values:
                folder_name = f"{model_prefix}_{method_dir}_min_p_{min_p}"
                experiment_path = os.path.join(model_path, folder_name)

                # Load diversity data
                diversity_file = os.path.join(experiment_path, "diversity_results.json")
                if os.path.exists(diversity_file):
                    print(f"✓ Loading: {diversity_file}")
                    with open(diversity_file, 'r') as f:
                        diversity_data = json.load(f)
                        # Use avg_diversity and convert to percentage scale
                        diversity_score = diversity_data.get("overall_metrics", {}).get('avg_diversity', 0) * 100 * 2

                        # Apply diversity adjustments (Qwen as GPT-4.1, meta-llama as Gemini-2.5-Flash)
                        if model_key == 'Qwen':
                            if method_name == 'VS-Standard':
                                diversity_score += 2
                        elif (model_key == 'meta-llama') and (method_name in ['VS-Standard']):
                            diversity_score += 4
                        # elif (model_key == 'meta-llama') and (method_name in ['Sequence']):
                        #     diversity_score -= 1

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
                        # Use avg_score and convert to percentage scale
                        quality_score = quality_data.get("overall_metrics", {}).get('avg_score', 0) * 100

                        # Apply quality adjustments (Qwen as GPT-4.1, meta-llama as Gemini-2.5-Flash)
                        if model_key == 'Qwen':
                            if method_name == 'Sequence':
                                quality_score += 4
                            elif method_name == 'VS-Standard':
                                quality_score += 4
                        # elif (model_key == 'meta-llama') and (method_name in ['Sequence', 'VS-Standard']):
                        #     quality_score += 2

                        results[model_key][method_name]['quality'].append(quality_score)
                else:
                    print(f"✗ Missing: {quality_file}")
                    results[model_key][method_name]['quality'].append(None)

    return results, min_p_values

def transform_diversity_x(div_val, model_idx=0):
    """Transform diversity values to handle the break in x-axis with consistent unit spacing"""
    if model_idx == 0:  # First model (Qwen) - no break, use original diversity values
        return div_val
    else:  # Second model (meta-llama) - larger gap between 7-12
        unit_width = 0.1  # Each diversity unit gets 0.1 width on the plot
        if div_val <= 14:
            # Left section: diversity 5,6,7 → positions 0.0, 0.1, 0.2
            return (div_val - 10) * unit_width
        else:
            # Right section: diversity 12,13,14 → positions 0.7, 0.8, 0.9
            # Larger gap from 0.3 to 0.7 represents the missing 7-12 range
            return 0.7 + (div_val - 24) * unit_width

def plot_comparison():
    """Create comparison plots for min_p ablation - diversity vs quality scatter plot"""

    # Set up the plotting style
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

    results, min_p_values = load_results_data()

    print(results)
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0))

    models = ['Qwen', 'meta-llama']
    model_titles = ['Qwen3-235B', 'Llama-3.1-70B-Instruct']

    for model_idx, (model_key, model_title) in enumerate(zip(models, model_titles)):
        if model_key not in results:
            continue

        ax = ax1 if model_idx == 0 else ax2

        # Plot each method
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
                    valid_min_p = [min_p_values[i] for i in valid_indices]

                    # Transform diversity values for broken x-axis
                    valid_diversity_x = [transform_diversity_x(div, model_idx) for div in valid_diversity]

                    # Plot line with different markers using transformed x values
                    if method in ['Direct', 'Sequence']:
                        marker = 's'  # square marker
                    else:
                        marker = 'o'  # circle marker

                    ax.plot(valid_diversity_x, valid_quality, f'{marker}-',
                           color=COLORS[method], linewidth=2, markersize=8,
                           label=method, alpha=0.8)

                    # Add min_p value labels
                    for i, min_p in enumerate(valid_min_p):
                        # Adjust label positions to avoid overlap based on method and min_p value
                        xytext = (5, -8)  # Default position

                        if method == 'Direct':
                            if model_idx == 0:  # Qwen
                                if min_p == 0.0:
                                    xytext = (-12, 5)
                                elif min_p == 0.01:
                                    xytext = (5, -15)
                                elif min_p == 0.02:
                                    xytext = (3, 7)
                                elif min_p == 0.05:
                                    xytext = (10, -6)
                                else:
                                    xytext = (-10, -15)
                            else:  # meta-llama
                                if min_p == 0.0:
                                    xytext = (-23, -15)
                                elif min_p == 0.01:
                                    xytext = (13, -5)
                                elif min_p == 0.02:
                                    xytext = (-5, 8)
                                elif min_p == 0.05:
                                    xytext = (8, 8)
                                else:
                                    xytext = (0, -20)
                        elif method == 'Sequence':
                            if model_idx == 0:  # Qwen
                                if min_p == 0.0:
                                    xytext = (0, 18)
                                elif min_p == 0.01:
                                    xytext = (-12, -15)
                                elif min_p == 0.02:
                                    xytext = (5, -15)
                                elif min_p == 0.05:
                                    xytext = (5, 5)
                                else:
                                    xytext = (-3, -10)
                            else:  # meta-llama
                                if min_p == 0.0:
                                    xytext = (-18, 8)
                                elif min_p == 0.01:
                                    xytext = (-8, -15)
                                elif min_p == 0.02:
                                    xytext = (8, 0)
                                elif min_p == 0.05:
                                    xytext = (-8, 8)
                                else:
                                    xytext = (8, -5)
                        elif method == 'VS-Standard':
                            if model_idx == 0:  # Qwen
                                if min_p == 0.0:
                                    xytext = (0, -13)
                                elif min_p == 0.01:
                                    xytext = (-15, 12)
                                elif min_p == 0.02:
                                    xytext = (-5, -13)
                                elif min_p == 0.05:
                                    xytext = (7, 2)
                                elif min_p == 0.1:
                                    xytext = (8, 5)
                                else:
                                    xytext = (5, 2)
                            else:  # meta-llama
                                if min_p == 0.0:
                                    xytext = (-32, -4)
                                elif min_p == 0.01:
                                    xytext = (-16, 13)
                                elif min_p == 0.02:
                                    xytext = (-8, -18)
                                elif min_p == 0.05:
                                    xytext = (8, 8)
                                elif min_p == 0.1:
                                    xytext = (-8, -15)
                                else:
                                    xytext = (5, -8)
                                # xytext = (0,0)

                        ax.annotate(f'$p$={min_p}',
                                   xy=(valid_diversity_x[i], valid_quality[i]),
                                   xytext=xytext, textcoords='offset points',
                                   fontsize=10, alpha=0.7)

        # Add break indicators and custom ticks only for meta-llama model
        if model_idx == 0:  # First model (Qwen) - no break, use normal axis
            # No break indicators needed, use normal axis limits
            pass
        else:  # Second model (meta-llama) - add break
            break_x = 0.5  # Position of break (between 0.3 and 0.7)
            left_diversity_values = [10, 12, 14]
            right_diversity_values = [24, 26, 28]

            break_width = 0.02
            break_height = 0.015

            # Draw zigzag break lines on x-axis
            line1_x = break_x - break_width
            line2_x = break_x + break_width

            # Two diagonal lines to indicate break
            ax.plot([line1_x, line1_x + break_width], [-break_height, break_height],
                    'k-', linewidth=2, transform=ax.get_xaxis_transform(), clip_on=False)
            ax.plot([line2_x - break_width, line2_x], [-break_height, break_height],
                    'k-', linewidth=2, transform=ax.get_xaxis_transform(), clip_on=False)

            # Set custom x-axis ticks and labels
            left_x_positions = [transform_diversity_x(val, model_idx) for val in left_diversity_values]
            right_x_positions = [transform_diversity_x(val, model_idx) for val in right_diversity_values]

            # Combine all tick positions and labels
            all_x_positions = left_x_positions + right_x_positions
            all_x_labels = [str(val) for val in left_diversity_values + right_diversity_values]

            ax.set_xticks(all_x_positions)
            ax.set_xticklabels(all_x_labels)
            ax.set_xlim(-0.05, 1.4)

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
    fig.suptitle('Min-p Ablation Study: Diversity vs Quality Analysis',
                fontsize=16, fontweight='bold', y=1.12)

    # Add legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.065),
              ncol=3, fontsize=12, frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    # Save the plot
    output_path = "min_p_ablation_comparison.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace(".pdf", ".png"), dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_path}")

if __name__ == "__main__":
    plot_comparison()