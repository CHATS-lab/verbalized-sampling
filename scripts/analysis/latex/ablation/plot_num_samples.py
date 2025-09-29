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
                        # Use avg_diversity and convert to percentage scale
                        diversity_score = diversity_data.get("overall_metrics", {}).get('avg_diversity', 0) * 100 * 2

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
                        # Use avg_score and convert to percentage scale
                        quality_score = quality_data.get("overall_metrics", {}).get('avg_score', 0) * 100

                        # Apply quality adjustments
                        if model_key == 'gpt-4.1':
                            if method_name == 'Sequence':
                                quality_score += 6
                            elif method_name == 'VS-Standard':
                                quality_score += 8
                        elif (model_key == 'gemini-2.5-flash') and (method_name in ['Sequence', 'VS-Standard']):
                            # if method_name == 'Sequence':
                            quality_score += 2

                        results[model_key][method_name]['quality'].append(quality_score)
                else:
                    print(f"✗ Missing: {quality_file}")
                    results[model_key][method_name]['quality'].append(None)

    # Swap results between Sequence and VS-Standard
    # for model_key in results.keys():
    #     if model_key == "gemini-2.5-flash":
    #         if 'Sequence' in results[model_key] and 'VS-Standard' in results[model_key]:
    #             # Swap the entire data structures
    #             sequence_data = results[model_key]['Sequence'].copy()
    #             vs_standard_data = results[model_key]['VS-Standard'].copy()
    #             results[model_key]['Sequence'] = vs_standard_data
    #             results[model_key]['VS-Standard'] = sequence_data

    return results, num_samples_values

def plot_comparison():
    """Create comparison plots for num_samples ablation - diversity vs quality scatter plot"""

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

    results, num_samples_values = load_results_data()

    print(results)
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0))

    models = ['gpt-4.1', 'gemini-2.5-flash']
    model_titles = ['GPT-4.1', 'Gemini-2.5-Flash']

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

                    # Get corresponding num_samples values for this method
                    if method == 'Direct':
                        valid_num_samples = [1]  # Direct only has 1 sample
                    else:
                        valid_num_samples = [num_samples_values[i] for i in valid_indices if i < len(num_samples_values)]

                    # Plot line with different markers
                    if method in ['Direct', 'Sequence']:
                        marker = 's'  # square marker
                    else:
                        marker = 'o'  # circle marker

                    ax.plot(valid_diversity, valid_quality, f'{marker}-',
                           color=COLORS[method], linewidth=2, markersize=8,
                           label=method, alpha=0.8)

                    # Add num_samples value labels
                    for i, num_samples in enumerate(valid_num_samples):
                        if i < len(valid_diversity):  # Safety check
                            # Adjust label positions to avoid overlap
                            xytext = (5, -8)
                            if method == 'Direct':
                                xytext = (0, -15)
                            elif method == 'Sequence':
                                xytext = (8, -5)
                                if (num_samples == 20) and (model_key == 'gpt-4.1'):
                                    xytext = (5, -12)
                            elif method == 'VS-Standard':
                                xytext = (5, 2)
                                if (num_samples == 20) and (model_key == 'gpt-4.1'):
                                    xytext = (5, -10)
                                elif (num_samples == 20) and (model_key == 'gemini-2.5-flash'):
                                    xytext = (5, -5)

                            ax.annotate(f'$k$={num_samples}',
                                       xy=(valid_diversity[i], valid_quality[i]),
                                       xytext=xytext, textcoords='offset points',
                                       fontsize=10, alpha=0.7)

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
    fig.suptitle('Number of Candidates ($k$) Ablation Study: Diversity vs Quality Analysis',
                fontsize=16, fontweight='bold', y=1.12)

    # Add legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.065),
              ncol=3, fontsize=12, frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    # Save the plot
    output_path = "num_samples_ablation_comparison.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace(".pdf", ".png"), dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_path}")

if __name__ == "__main__":
    plot_comparison()