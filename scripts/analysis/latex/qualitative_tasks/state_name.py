import numpy as np
import json


def draw_distribution_comparison(direct_results, vs_results, direct_coverage_n, vs_coverage_n):
    """
    Draws a bar plot comparing the response distributions for direct and verbalized sampling,
    using the same color and style for both bars, and adds coverage n as a legend entry
    similar to the KL divergence box in the reference code.

    This version plots counts (not probabilities) on the y-axis.

    Bars are ordered: first, all states that appear in direct, sorted by direct count descending;
    then, any remaining states that appear only in vs, sorted by vs count descending.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # States present in direct, sorted by direct count descending
    direct_sorted = sorted(direct_results.items(), key=lambda x: -x[1])
    direct_states_sorted = [state for state, count in direct_sorted]

    # States present in vs but not in direct
    vs_only_states = set(vs_results.keys()) - set(direct_results.keys())
    vs_only_sorted = sorted(
        [(state, vs_results[state]) for state in vs_only_states],
        key=lambda x: -x[1]
    )
    vs_only_states_sorted = [state for state, count in vs_only_sorted]

    # Final order: direct (by direct count desc), then vs-only (by vs count desc)
    all_states_ordered = direct_states_sorted + vs_only_states_sorted

    # Get counts for each state in the new order
    direct_counts = np.array([direct_results.get(state, 0) for state in all_states_ordered])
    vs_counts = np.array([vs_results.get(state, 0) for state in all_states_ordered])

    x = np.arange(len(all_states_ordered))
    width = 0.4

    fig, ax = plt.subplots(figsize=(18, 7))

    # Use same color and style for both bars, but offset for clarity
    bars1 = ax.bar(x - width/2, direct_counts, width, label='Direct Sampling', color='#FC8EAC', alpha=0.8, edgecolor='#FC8EAC', linewidth=1)
    bars2 = ax.bar(x + width/2, vs_counts, width, label='Verbalized Sampling', color='#A4C8E1', alpha=0.8, edgecolor='#A4C8E1', linewidth=1)

    # X-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(all_states_ordered, rotation=45, fontsize=10)
    ax.set_xlabel('US State', fontsize=13, fontweight='bold')
    ax.set_ylabel('Count', fontsize=13, fontweight='bold')
    ax.set_title('', fontsize=15, fontweight='bold', pad=20)

    # Add value labels on bars (show count)
    def add_value_labels(bars, counts):
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{int(count)}', ha='center', va='bottom', fontsize=8)

    add_value_labels(bars1, direct_counts)
    add_value_labels(bars2, vs_counts)

    # Add coverage n as a statistics box (like KL divergence in reference)
    stats_text = f'Coverage-N:\nDirect Sampling: {direct_coverage_n:.2f}\nVerbalized Sampling: {vs_coverage_n:.2f}'
    ax.text(0.865, 0.84, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    ax.legend(fontsize=12, loc='upper right')
    ax.set_ylim(0, max(direct_counts.max(), vs_counts.max()) * 1.15)

    plt.tight_layout()
    # plt.savefig("qualitative_tasks/state_name_distribution_comparison.png", bbox_inches='tight')
    # plt.show()


def main():
    direct_results_file = "method_results_state_name/gpt-4.1_state_name/evaluation/direct (samples=1)/response_count_results.json"
    vs_results_file = "method_results_state_name/gpt-4.1_state_name/evaluation/vs_multi [strict] (samples=20)/response_count_results.json"

    with open(direct_results_file, "r") as f:
        direct_results = json.load(f)
    with open(vs_results_file, "r") as f:
        vs_results = json.load(f)
    
    prompt = "Name a US State. Only provide the answer without explanation or punctuation."
    direct_results = direct_results["overall_metrics"]["per_prompt_stats"][prompt]
    vs_results = vs_results["overall_metrics"]["per_prompt_stats"][prompt]

    direct_response_distribution = direct_results["response_distribution"]
    vs_response_distribution = vs_results["response_distribution"]

    direct_coverage_n = direct_results["unique_recall_rate"]
    vs_coverage_n = vs_results["unique_recall_rate"]

    # Print both direct and vs response distribution like a sheet

    # Collect all unique states from both distributions
    all_states = set(direct_response_distribution.keys()) | set(vs_response_distribution.keys())
    all_states = sorted(all_states)

    # Print header
    print(f"{'State':<20} {'Direct':>10} {'VS':>10}")
    print("-" * 42)
    for state in all_states:
        direct_count = direct_response_distribution.get(state, 0)
        vs_count = vs_response_distribution.get(state, 0)
        print(f"{state:<20} {direct_count:>10} {vs_count:>10}")

    # print(direct_results)
    # print(vs_results)
    draw_distribution_comparison(direct_response_distribution, vs_response_distribution, direct_coverage_n, vs_coverage_n)
    

if __name__ == "__main__":
    main()