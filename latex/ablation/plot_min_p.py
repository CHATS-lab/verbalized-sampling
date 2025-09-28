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
        'structure_with_prob': 'VS-Standard'
    }

    min_p_values = [0.0, 0.01, 0.02, 0.05, 0.1]

    results = {}

    for model_key, model_prefix in models.items():
        model_path = os.path.join(base_path, "evaluation", model_key)
        results[model_key] = {}

        for method_name in method_mapping.values():
            results[model_key][method_name] = {
                'diversity': []
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
                        diversity_score = diversity_data.get("overall_metrics", {}).get('avg_diversity', 0) * 100
                        results[model_key][method_name]['diversity'].append(diversity_score)
                else:
                    print(f"✗ Missing: {diversity_file}")
                    results[model_key][method_name]['diversity'].append(None)

    return results, min_p_values

def plot_comparison():
    """Create comparison plots for min_p ablation"""

    # Set up the plotting style
    plt.rcParams.update(RC_PARAMS)

    results, min_p_values = load_results_data()

    print(results)
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Min-p Ablation Study Results - Diversity', fontsize=16, fontweight='bold')

    models = ['Qwen', 'meta-llama']
    model_titles = ['Qwen3-235B', 'Llama-3.1-70B-Instruct']

    for model_idx, (model_key, model_title) in enumerate(zip(models, model_titles)):
        if model_key not in results:
            continue

        # Diversity plot
        ax = axes[model_idx]
        ax.set_title(f'{model_title} - Diversity', fontweight='bold')
        ax.set_xlabel('Min-p Value')
        ax.set_ylabel('Diversity Score (%)')
        ax.grid(True, alpha=0.3)

        for method in ['Direct', 'Sequence', 'VS-Standard']:
            if method in results[model_key]:
                diversity_scores = results[model_key][method]['diversity']

                # Filter out None values
                valid_indices = [i for i, d in enumerate(diversity_scores) if d is not None]

                if valid_indices:
                    valid_min_p = [min_p_values[i] for i in valid_indices]
                    valid_diversity = [diversity_scores[i] for i in valid_indices]

                    # Plot diversity
                    ax.plot(valid_min_p, valid_diversity,
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
        ax.set_xticks(min_p_values)

    plt.tight_layout()

    # Save the plot
    output_path = "min_p_ablation_comparison.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_path}")

if __name__ == "__main__":
    plot_comparison()