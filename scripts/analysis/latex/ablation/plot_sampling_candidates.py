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
        'gemini-2.5-flash': 'gemini-2.5-flash',
        'gpt-4.1': 'gpt-4.1'
    }

    method_mapping = {
        'direct': 'Direct',
        'sequence': 'Sequence',
        'vs_standard': 'VS-Standard'
    }

    # Note: Direct method uses samples=1, others use variable sample counts
    sample_counts = [1, 3, 5, 10, 15, 20]

    results = {}

    for model_key in models.keys():
        model_path = os.path.join(base_path, "evaluation")
        results[model_key] = {}

        for method_name in method_mapping.values():
            results[model_key][method_name] = {
                'diversity': []
            }

        for method_dir, method_name in method_mapping.items():
            for samples in sample_counts:
                if method_dir == 'direct' and samples != 1:
                    # Direct method only has samples=1
                    results[model_key][method_name]['diversity'].append(None)
                    continue
                elif method_dir == 'direct':
                    folder_name = f"{model_key}_{method_dir}_samples_{samples}"
                else:
                    folder_name = f"{model_key}_{method_dir}_samples_{samples}"

                experiment_path = os.path.join(model_path, folder_name)

                # Load diversity data
                diversity_file = os.path.join(experiment_path, "diversity_results.json")
                if os.path.exists(diversity_file):
                    print(f"✓ Loading: {diversity_file}")
                    with open(diversity_file, 'r') as f:
                        diversity_data = json.load(f)
                        # Use avg_diversity and convert to percentage scale
                        diversity_score = diversity_data.get("overall_metrics", {}).get('avg_diversity', 0) * 100
                        results[model_key][method_name]['diversity'].append(diversity_score)
                else:
                    print(f"✗ Missing: {diversity_file}")
                    results[model_key][method_name]['diversity'].append(None)

    return results, sample_counts

def plot_comparison():
    """Create comparison plots for sampling candidates ablation"""

    # Set up the plotting style
    plt.rcParams.update(RC_PARAMS)

    results, sample_counts = load_results_data()

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Sampling Candidates (c) Ablation Study Results - Diversity', fontsize=16, fontweight='bold')

    models = ['gemini-2.5-flash', 'gpt-4.1']
    model_titles = ['Gemini 2.5 Flash', 'GPT-4.1']

    for model_idx, (model_key, model_title) in enumerate(zip(models, model_titles)):
        if model_key not in results:
            continue

        # Diversity plot
        ax = axes[model_idx]
        ax.set_title(f'{model_title} - Diversity', fontweight='bold')
        ax.set_xlabel('Number of Candidates (c)')
        ax.set_ylabel('Diversity Score (%)')
        ax.grid(True, alpha=0.3)

        for method in ['Direct', 'Sequence', 'VS-Standard']:
            if method in results[model_key]:
                diversity_scores = results[model_key][method]['diversity']

                # Filter out None values and get corresponding sample counts
                valid_data = [(sample_counts[i], diversity_scores[i])
                             for i in range(len(sample_counts))
                             if diversity_scores[i] is not None]

                if valid_data:
                    valid_samples, valid_diversity = zip(*valid_data)

                    # Plot diversity
                    ax.plot(valid_samples, valid_diversity,
                              color=COLORS[method],
                              marker='o',
                              linewidth=2,
                              markersize=6,
                              label=method,
                              markeredgecolor=EDGE_COLORS.get(method, 'black'),
                              markeredgewidth=1)

        # Add legend
        ax.legend(loc='best')

        # Set x-axis ticks
        ax.set_xticks(sample_counts)

        # Set x-axis to log scale for better visualization
        ax.set_xscale('log')

    plt.tight_layout()

    # Save the plot
    output_path = "sampling_candidates_ablation_comparison.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_path}")

if __name__ == "__main__":
    plot_comparison()