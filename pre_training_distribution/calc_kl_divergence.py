import json
import math

state_name_abbreviations = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY"
}

def calc_kl_divergence(gt_data, data):
    """
    Calculate the KL divergence D_KL(gt_data || data)
    Both gt_data and data are dicts mapping terms to probabilities.
    Returns a non-negative value (or 0 if identical).
    This version only considers keys present in both gt_data and data,
    and adds a small epsilon to avoid log(0) and renormalizes probabilities.
    """
    epsilon = 1e-10
    # Only use keys present in both gt_data and data
    matched_keys = set(gt_data.keys()) & set(data.keys())
    if not matched_keys:
        return 0.0  # No overlap, KL is 0 by convention

    # Build probability vectors with epsilon smoothing
    p_vec = []
    q_vec = []
    for key in matched_keys:
        p = gt_data.get(key, 0.0)
        q = data.get(key, 0.0)
        p_vec.append(p + epsilon)
        q_vec.append(q + epsilon)
    # Renormalize so they sum to 1
    p_sum = sum(p_vec)
    q_sum = sum(q_vec)
    p_vec = [x / p_sum for x in p_vec]
    q_vec = [x / q_sum for x in q_vec]
    # Calculate KL divergence: KL(P || Q) = sum(P * log(P / Q))
    kl_div = 0.0
    for p, q in zip(p_vec, q_vec):
        kl_div += p * math.log(p / q)
    return kl_div


def calc_kl_divergence_uniform(data):
    """
    Calculate the KL divergence D_KL(data || uniform)
    data is a dict mapping terms to probabilities.
    """
    n = len(data)
    uniform_prob = 1.0 / n
    kl_div = 0.0
    for key in data:
        p = data[key]
        if p > 0:
            kl_div += p * math.log(p / uniform_prob)
    return kl_div

def read_data(path):
    result = {}
    probability_sum = 0.0
    with open(path, "r") as f:
        data = json.load(f)
        responses = data["responses"] if "responses" in data else data
        for resp in responses:
            result[resp["text"]] = resp["probability"]
            probability_sum += resp["probability"]
    if abs(probability_sum - 1.0) > 1e-8:
        print(f"Probability sum is not 1.0 for {path}")
        print(f"Probability sum: {probability_sum}")
        # Manually renormalize so the probabilities sum to 1
        for key in result:
            result[key] = result[key] / probability_sum
    return result


import matplotlib.pyplot as plt
import numpy as np

def plot_histograms(gt_data, direct_data, sequence_data, vs_data, model_name):
    """
    Draw histograms of gt_data vs gpt_4_1_data and gt_data vs claude_4_sonnet_data.
    Also draw the straight line of uniform distribution.
    The count is prob * 100.
    Sorts states by direct, then gt.
    """
    # Sort by direct_data (descending), then gt_data (descending), then alphabetically
    all_states = list(set(gt_data.keys()) | set(direct_data.keys()) | set(sequence_data.keys()) | set(vs_data.keys()))
    def sort_key(state):
        # Use negative for descending sort
        return (
            -direct_data.get(state, 0.0),
            -gt_data.get(state, 0.0),
            state
        )
    all_states = sorted(all_states, key=sort_key)
    n = len(all_states)

    gt_counts = [gt_data.get(state, 0.0) for state in all_states]
    uniform_count = [1.0 / n] * n

    direct_counts = [direct_data.get(state, 0.0) for state in all_states]
    # sequence_counts = [sequence_data.get(state, 0.0) for state in all_states]
    vs_counts = [vs_data.get(state, 0.0) for state in all_states]

    x = np.arange(n)

    # Plot gt_data vs gpt_4_1_data
    plt.figure(figsize=(18, 6))
    # Plot bars side by side with no overlap
    total_bars = 3
    bar_width = 0.25  # fixed width for clarity
    plt.bar(x-bar_width, direct_counts, bar_width, label='Direct', color='Orange')
    plt.bar(x, gt_counts, bar_width, label='Ground Truth', color='Blue')
    plt.bar(x + bar_width, vs_counts, bar_width, label='VS', color='red')
    plt.plot(x, uniform_count, 'k--', label='Uniform/Sequence', linewidth=2)
    plt.xticks(x, all_states, rotation=90)
    plt.ylabel('Count (probability * 100)')
    plt.title(f'Ground Truth vs {model_name} State Distribution')
    plt.ylim(0, 0.15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'pre_training_distribution/gt_vs_{model_name}.png')
    plt.show()


def plot_comprehensive_comparison(gt_data, direct_gpt_data, direct_claude_data, 
                                 sequence_gpt_data, sequence_claude_data,
                                 vs_gpt_data, vs_claude_data):
    """
    Create a comprehensive comparison plot with all GT vs method comparisons.
    Shows 3 methods (Direct, Sequence, VS) x 2 models (GPT-4.1, Claude-4-Sonnet) in a 6x1 grid.
    Includes uniform distribution line and uses different colors for bars.
    """
    plt.style.use('default')  # Start with clean slate
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'font.size': 28,
        'axes.labelsize': 28,
        'axes.titlesize': 30,
        'xtick.labelsize': 16,
        'ytick.labelsize': 20,
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

    # Get all states and sort them
    all_states = list(set(gt_data.keys()) | set(direct_gpt_data.keys()) | set(direct_claude_data.keys()) |
                      set(sequence_gpt_data.keys()) | set(sequence_claude_data.keys()) |
                      set(vs_gpt_data.keys()) | set(vs_claude_data.keys()))
    all_states = sorted(all_states)

    n = len(all_states)
    
    # Prepare data
    gt_counts = [gt_data.get(state, 0.0) for state in all_states]
    direct_gpt_counts = [direct_gpt_data.get(state, 0.0) for state in all_states]
    direct_claude_counts = [direct_claude_data.get(state, 0.0) for state in all_states]
    sequence_gpt_counts = [sequence_gpt_data.get(state, 0.0) for state in all_states]
    sequence_claude_counts = [sequence_claude_data.get(state, 0.0) for state in all_states]
    vs_gpt_counts = [vs_gpt_data.get(state, 0.0) for state in all_states]
    vs_claude_counts = [vs_claude_data.get(state, 0.0) for state in all_states]
    uniform_count = [1.0 / n] * n

    bar_width = 0.3
    x = np.arange(n)

    # Create 6x1 subplot grid
    fig, axes = plt.subplots(6, 1, figsize=(18, 36))
    
    # Define colors for consistency
    gt_color = '#57b4e9'
    direct_color = '#e79f00'
    sequence_color = '#d65e00'
    vs_color = '#b31529'
    uniform_color = 'black'
    
    # 6x1 layout: Direct-GPT, Direct-Claude, Sequence-GPT, Sequence-Claude, VS-GPT, VS-Claude
    ax1, ax2, ax3, ax4, ax5, ax6 = axes[0], axes[1], axes[2], axes[3], axes[4], axes[5]
    
    # GPT-4.1 Direct
    ax1.text(-0.1, 1.07, "(a)", transform=ax1.transAxes, fontsize=32, fontweight='bold', va='top', ha='left')
    ax1.bar(x - bar_width, gt_counts, bar_width, label='Ground Truth', color=gt_color, alpha=0.8)
    ax1.bar(x, direct_gpt_counts, bar_width, label='Direct', color=direct_color, alpha=0.8)
    ax1.plot(x, uniform_count, 'k--', label='Uniform', linewidth=2, color=uniform_color)
    ax1.set_xticks(x)
    ax1.set_xticklabels([state_name_abbreviations[state] for state in all_states], rotation=60, ha='center')
    ax1.set_ylabel('Probability')
    ax1.set_title('Ground Truth vs Direct - GPT-4.1', fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.set_ylim(0, max(max(gt_counts), max(direct_gpt_counts), max(uniform_count)) * 1.1)
    
    # Claude-4-Sonnet Direct
    ax2.text(-0.1, 1.07, "(b)", transform=ax2.transAxes, fontsize=32, fontweight='bold', va='top', ha='left')
    ax2.bar(x - bar_width, gt_counts, bar_width, label='Ground Truth', color=gt_color, alpha=0.8)
    ax2.bar(x, direct_claude_counts, bar_width, label='Direct', color=direct_color, alpha=0.8)
    ax2.plot(x, uniform_count, 'k--', label='Uniform', linewidth=2, color=uniform_color)
    ax2.set_xticks(x)
    ax2.set_xticklabels([state_name_abbreviations[state] for state in all_states], rotation=60, ha='right')
    ax2.set_ylabel('Probability')
    ax2.set_title('Ground Truth vs Direct - Claude-4-Sonnet', fontweight='bold')
    # ax2.legend()
    ax2.set_ylim(0, max(max(gt_counts), max(direct_claude_counts), max(uniform_count)) * 1.1)
    
    # GPT-4.1 Sequence
    ax3.text(-0.1, 1.07, "(c)", transform=ax3.transAxes, fontsize=32, fontweight='bold', va='top', ha='left')
    ax3.bar(x - bar_width, gt_counts, bar_width, label='Ground Truth', color=gt_color, alpha=0.8)
    ax3.bar(x, sequence_gpt_counts, bar_width, label='Sequence', color=sequence_color, alpha=0.8)
    ax3.plot(x, uniform_count, 'k--', label='Uniform', linewidth=2, color=uniform_color)
    ax3.set_xticks(x)
    ax3.set_xticklabels([state_name_abbreviations[state] for state in all_states], rotation=60, ha='right')
    ax3.set_ylabel('Probability')
    ax3.set_title('Ground Truth vs Sequence - GPT-4.1', fontweight='bold')
    ax3.legend(loc='upper left')
    ax3.set_ylim(0, max(max(gt_counts), max(sequence_gpt_counts), max(uniform_count)) * 1.1)
    
    # Claude-4-Sonnet Sequence
    ax4.text(-0.1, 1.07, "(d)", transform=ax4.transAxes, fontsize=32, fontweight='bold', va='top', ha='left')
    ax4.bar(x - bar_width, gt_counts, bar_width, label='Ground Truth', color=gt_color, alpha=0.8)
    ax4.bar(x, sequence_claude_counts, bar_width, label='Sequence', color=sequence_color, alpha=0.8)
    ax4.plot(x, uniform_count, 'k--', label='Uniform', linewidth=2, color=uniform_color)
    ax4.set_xticks(x)
    ax4.set_xticklabels([state_name_abbreviations[state] for state in all_states], rotation=60, ha='right')
    ax4.set_ylabel('Probability')
    ax4.set_title('Ground Truth vs Sequence - Claude-4-Sonnet', fontweight='bold')
    # ax4.legend()
    ax4.set_ylim(0, max(max(gt_counts), max(sequence_claude_counts), max(uniform_count)) * 1.1)
    
    # GPT-4.1 VS
    ax5.text(-0.1, 1.07, "(e)", transform=ax5.transAxes, fontsize=32, fontweight='bold', va='top', ha='left')
    ax5.bar(x - bar_width, gt_counts, bar_width, label='Ground Truth', color=gt_color, alpha=0.8)
    ax5.bar(x, vs_gpt_counts, bar_width, label='VS-Standard', color=vs_color, alpha=0.8)
    ax5.plot(x, uniform_count, 'k--', label='Uniform', linewidth=2, color=uniform_color)
    ax5.set_xticks(x)
    ax5.set_xticklabels([state_name_abbreviations[state] for state in all_states], rotation=60, ha='right')
    ax5.set_ylabel('Probability')
    ax5.set_title('Ground Truth vs VS-Standard - GPT-4.1', fontweight='bold')
    ax5.legend(loc='upper left')
    ax5.set_ylim(0, max(max(gt_counts), max(vs_gpt_counts), max(uniform_count)) * 1.1)
    
    # Claude-4-Sonnet VS
    ax6.text(-0.1, 1.07, "(f)", transform=ax6.transAxes, fontsize=32, fontweight='bold', va='top', ha='left')
    ax6.bar(x - bar_width, gt_counts, bar_width, label='Ground Truth', color=gt_color, alpha=0.8)
    ax6.bar(x, vs_claude_counts, bar_width, label='VS-Standard', color=vs_color, alpha=0.8)
    ax6.plot(x, uniform_count, 'k--', label='Uniform', linewidth=2, color=uniform_color)
    ax6.set_xticks(x)
    ax6.set_xticklabels([state_name_abbreviations[state] for state in all_states], rotation=60, ha='right')
    ax6.set_ylabel('Probability')
    ax6.set_title('Ground Truth vs VS-Standard - Claude-4-Sonnet', fontweight='bold')
    # ax6.legend()
    ax6.set_ylim(0, max(max(gt_counts), max(vs_claude_counts), max(uniform_count)) * 1.1)
    
    plt.tight_layout()
    # plt.savefig('pre_training_distribution/comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('pre_training_distribution/pre_training_distribution_comparison.pdf', bbox_inches='tight')
    # plt.show()


def calc_kl_divergence_vs(gt_data, direct_data, sequence_data, vs_data):
    gt_direct_kl = calc_kl_divergence(gt_data, direct_data)
    gt_sequence_kl = calc_kl_divergence(gt_data, sequence_data)
    gt_vs_kl = calc_kl_divergence(gt_data, vs_data)
    
    uniform_direct_kl = calc_kl_divergence_uniform(direct_data)
    uniform_sequence_kl = calc_kl_divergence_uniform(sequence_data)
    uniform_vs_kl = calc_kl_divergence_uniform(vs_data)

    print(f"GT vs Direct: {gt_direct_kl}, GT vs Sequence: {gt_sequence_kl}, GT vs VS: {gt_vs_kl}")
    print(f"Uniform vs Direct: {uniform_direct_kl}, Uniform vs Sequence: {uniform_sequence_kl}, Uniform vs VS: {uniform_vs_kl}")
    
    # return gt_direct_kl, gt_sequence_kl, gt_vs_kl, uniform_direct_kl, uniform_sequence_kl, uniform_vs_kl


def main():
    gt_path = "pre_training_distribution/state_name_distribution.json"


    with open(gt_path, "r") as f:
        gt_data = json.load(f)
    vs_gpt_4_1_data = read_data("pre_training_distribution/vs_gpt-4.1.json")
    vs_claude_4_sonnet_data = read_data("pre_training_distribution/vs_claude-4-sonnet.json")
    sequence_gpt_4_1_data = read_data("pre_training_distribution/sequence_gpt-4.1.json")
    sequence_claude_4_sonnet_data = read_data("pre_training_distribution/sequence_claude-4-sonnet.json")
    
    with open("pre_training_distribution/direct_gpt-4.1.json", "r") as f:
        direct_gpt_4_1_data = json.load(f)
    with open("pre_training_distribution/direct_claude-4-sonnet.json", "r") as f:
        direct_claude_4_sonnet_data = json.load(f)

    # Build a new dict instead of mutating during iteration; convert percentage to probability
    gt_data = {state_name: info["percentage"] for state_name, info in gt_data.items()}
    direct_gpt_4_1_data = {state_name: info["percentage"] for state_name, info in direct_gpt_4_1_data.items()}
    direct_claude_4_sonnet_data = {state_name: info["percentage"] for state_name, info in direct_claude_4_sonnet_data.items()}
    # print(gt_data)


    # plot_histograms(gt_data, direct_gpt_4_1_data, sequence_gpt_4_1_data, vs_gpt_4_1_data, "gpt-4.1")
    # plot_histograms(gt_data, direct_claude_4_sonnet_data, sequence_claude_4_sonnet_data, vs_claude_4_sonnet_data, "claude-4-sonnet")
    
    # Create comprehensive comparison plot with all methods and models
    plot_comprehensive_comparison(gt_data, direct_gpt_4_1_data, direct_claude_4_sonnet_data,
                                 sequence_gpt_4_1_data, sequence_claude_4_sonnet_data,
                                 vs_gpt_4_1_data, vs_claude_4_sonnet_data)

    # calc_kl_divergence_vs(gt_data, direct_gpt_4_1_data, sequence_gpt_4_1_data, vs_gpt_4_1_data)
    # calc_kl_divergence_vs(gt_data, direct_claude_4_sonnet_data, sequence_claude_4_sonnet_data, vs_claude_4_sonnet_data)



if __name__ == "__main__":
    main()