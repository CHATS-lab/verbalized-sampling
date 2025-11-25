import json
import math
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
from collections import defaultdict


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


def plot_histograms(gt_data, direct_data, sequence_data, vs_data, model_name):
    """
    Draw histograms of gt_data vs gpt_4_1_data and gt_data vs claude_4_sonnet_data.
    Also draw the straight line of uniform distribution.
    The count is prob * 100.
    Sorts states by direct, then gt.
    Uses a broken y-axis to better show the distribution.
    """
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'font.size': 19,
        'axes.labelsize': 22,
        'axes.titlesize': 25,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 20,
        'axes.linewidth': 0.8,
        'axes.edgecolor': '#666666',
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'lines.linewidth': 2.0,
        'lines.markersize': 8,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white'
    })

    all_states = list(set(gt_data.keys()) | set(direct_data.keys()) | set(sequence_data.keys()) | set(vs_data.keys()))
    def sort_key(state):
        return (-gt_data.get(state, 0.0), state)
    all_states = sorted(all_states, key=sort_key)
    n = len(all_states)

    gt_color = '#e79f00'
    direct_color = '#7CC7EA'
    sequence_color = '#4A90E2'
    vs_color = '#cd1c18'
    uniform_color = 'black'

    gt_counts = [gt_data.get(state, 0.0) for state in all_states]
    uniform_count = [1.0 / n] * n

    direct_counts = [direct_data.get(state, 0.0) for state in all_states]
    vs_counts = [vs_data.get(state, 0.0) for state in all_states]

    x = np.arange(n)
    max_val = max(max(direct_counts), max(gt_counts), max(vs_counts))
    
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(22, 6), 
                                             gridspec_kw={'height_ratios': [0.6, 5], 'hspace': 0.08})
    
    bar_width = 0.25
    
    ax_top.bar(x - bar_width, direct_counts, bar_width, label='Direct', color=direct_color, alpha=0.8)
    ax_top.bar(x, gt_counts, bar_width, label='Pretraining Distribution\n("Ground Truth")', color=gt_color, alpha=0.8)
    ax_top.bar(x + bar_width, vs_counts, bar_width, label='VS-Standard', color=vs_color, alpha=0.8)
    
    ax_bottom.bar(x - bar_width, direct_counts, bar_width, label='Direct', color=direct_color, alpha=0.8)
    ax_bottom.bar(x, gt_counts, bar_width, label='Pretraining Distribution\n("Ground Truth")', color=gt_color, alpha=0.8)
    ax_bottom.bar(x + bar_width, vs_counts, bar_width, label='VS-Standard', color=vs_color, alpha=0.8)
    
    ax_bottom.plot(x, uniform_count, linestyle='--', color=uniform_color, linewidth=2)
    ax_bottom.text(
        x[-1] - 2.5, uniform_count[-1] + 0.002, 'Uniform',
        color=uniform_color, fontsize=20, ha='left', va='bottom'
    )
    
    break_point = 0.16
    ax_top.set_ylim(break_point, max_val + 0.01)
    ax_bottom.set_ylim(0, break_point)
    
    ax_top.spines[['bottom', 'top', 'right']].set_visible(False)
    ax_bottom.spines[['top', 'right']].set_visible(False)
    
    # Remove x-axis ticks and labels from top subplot
    ax_top.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, labeltop=False)
    ax_bottom.xaxis.tick_bottom()
    
    d = 0.012
    kwargs = dict(transform=ax_top.transAxes, color='#666666', clip_on=False, linewidth=1.2)
    ax_top.plot((-d, +d), (-d*0.5, +d*0.5), **kwargs)
    kwargs.update(transform=ax_bottom.transAxes)
    ax_bottom.plot((-d, +d), (1 - d*0.3, 1 + d*0.3), **kwargs)
    
    left_lim = -0.8 + bar_width
    right_lim = n - 0.2 - bar_width
    ax_top.set_xlim(left_lim, right_lim)
    ax_bottom.set_xlim(left_lim, right_lim)
    
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(all_states, rotation=60, ha='right')

    # abbrev_labels = [state_name_abbreviations.get(state, state) for state in all_states]
    # ax_bottom.set_xticklabels(abbrev_labels, rotation=60, ha='right')
    
    ax_top.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax_bottom.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    fig.text(0.085, 0.5, 'Probability', va='center', rotation='vertical', fontsize=22)
    
    ax_bottom.set_yticks([0, 0.03, 0.06, 0.09, 0.12, 0.15])
    max_tick = np.ceil(max_val * 100) / 100
    ax_top.set_yticks([max_tick])
    
    if model_name == "claude-4-sonnet":
        model_name = "Claude-4-Sonnet"
    elif model_name == "gpt-4.1":
        model_name = "GPT-4.1"
    
    fig.suptitle(f'US State Name Generation ({model_name})', y=0.98)
    
    legend_handles = [
        Patch(facecolor=direct_color, alpha=0.8, label='Direct'),
        Patch(facecolor=gt_color, alpha=0.8, label='Pretraining Distribution\n("Reference")'),
        Patch(facecolor=vs_color, alpha=0.8, label='VS-Standard')
    ]

    leg = fig.legend(
        handles=legend_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.96),
        ncol=3,
        frameon=False
    )
    for text in leg.get_texts():
        text.set_multialignment('center')

    gt_direct_kl = calc_kl_divergence(gt_data, direct_data)
    gt_vs_kl = calc_kl_divergence(gt_data, vs_data)
    gt_uniform_kl = calc_kl_divergence_uniform(gt_data)
    
    kl_text = (
        f"KL from Pretraining Distribution:\n"
        f"Direct: {gt_direct_kl:.2f}\n"
        f"VS-Standard: {gt_vs_kl:.2f}\n"
        f"Uniform: {gt_uniform_kl:.2f}"
    )
    ax_bottom.text(0.75, 0.96, kl_text,
        transform=ax_bottom.transAxes,
        va='top', ha='left', fontweight='bold',
        bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', linewidth=1.2)
    ) # 0.96 for claude, 1.2 for gpt

    plt.tight_layout(rect=[0.05, 0, 1, 0.94])
    plt.savefig(f'analyse/pre_training_distribution/gt_vs_{model_name}.png', bbox_inches='tight')
    with PdfPages(f'analyse/pre_training_distribution/gt_vs_{model_name}.pdf') as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    


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


# def calc_kl_divergence(gt_data, data):
#     """
#     Calculate the KL divergence D_KL(gt_data || data)
#     Both gt_data and data are dicts mapping terms to probabilities.
#     Returns a non-negative value (or 0 if identical).
#     This version considers all keys present in either gt_data or data,
#     sets missing probabilities to 0, adds a small epsilon to avoid log(0),
#     and renormalizes probabilities. Lowercases all keys for matching.
#     """
#     # Lowercase all keys in both gt_data and data
#     gt_data_lower = {k.lower(): v for k, v in gt_data.items()}
#     data_lower = {k.lower(): v for k, v in data.items()}
#     epsilon = 1e-10
#     # Use all keys present in either gt_data or data
#     all_keys = set(gt_data_lower.keys()) | set(data_lower.keys())
#     if not all_keys:
#         return 0.0  # No keys, KL is 0 by convention

#     # Build probability vectors with epsilon smoothing
#     p_vec = []
#     q_vec = []
#     for key in all_keys:
#         p = gt_data_lower.get(key, 0.0)
#         q = data_lower.get(key, 0.0)
#         p_vec.append(p + epsilon)
#         q_vec.append(q + epsilon)
#     # Renormalize so they sum to 1
#     p_sum = sum(p_vec)
#     q_sum = sum(q_vec)
#     p_vec = [x / p_sum for x in p_vec]
#     q_vec = [x / q_sum for x in q_vec]
#     # Calculate KL divergence: KL(P || Q) = sum(P * log(P / Q))
#     kl_div = 0.0
#     for p, q in zip(p_vec, q_vec):
#         kl_div += p * math.log(p / q)
#     return kl_div


# def calc_kl_divergence_uniform(data):
#     """
#     Calculate the KL divergence D_KL(data || uniform)
#     data is a dict mapping terms to probabilities.
#     """
#     n = len(data)
#     uniform_prob = 1.0 / n
#     kl_div = 0.0
#     for key in data:
#         p = data[key]
#         if p > 0:
#             kl_div += p * math.log(p / uniform_prob)
#     return kl_div


def calc_kl_divergence_vs(gt_data, data_1, data_2, data_3):
    gt_data_1_kl = calc_kl_divergence(gt_data, data_1)
    gt_data_2_kl = calc_kl_divergence(gt_data, data_2)
    gt_data_3_kl = calc_kl_divergence(gt_data, data_3)
    
    uniform_data_1_kl = calc_kl_divergence_uniform(data_1)
    uniform_data_2_kl = calc_kl_divergence_uniform(data_2)
    uniform_data_3_kl = calc_kl_divergence_uniform(data_3)

    # Print a nice table for KL divergences
    print("\nKL Divergence Table")
    print("+----------------------+--------------------+")
    print("|      Comparison      |     KL Value       |")
    print("+----------------------+--------------------+")
    print(f"| GT vs Data 1         | {gt_data_1_kl:18.6f} |")
    print(f"| GT vs Data 2         | {gt_data_2_kl:18.6f} |")
    print(f"| GT vs Data 3         | {gt_data_3_kl:18.6f} |")
    print("+----------------------+--------------------+")
    print(f"| Uniform vs Data 1    | {uniform_data_1_kl:18.6f} |")
    print(f"| Uniform vs Data 2    | {uniform_data_2_kl:18.6f} |")
    print(f"| Uniform vs Data 3    | {uniform_data_3_kl:18.6f} |")
    print("+----------------------+--------------------+\n")


def read_VS_data(path):
    full_result = {}
    probability_sum = 0.0
    with open(path, "r") as f:
        data = json.load(f)
    
    for key, responses in data.items():
        probability_sum = 0.0
        temp_results = {}
        for resp in responses:
            temp_results[resp["text"]] = resp["probability"]
            probability_sum += resp["probability"]
        if abs(probability_sum - 1.0) > 1e-8:
            # Manually renormalize so the probabilities sum to 1
            for state in temp_results:
                temp_results[state] = temp_results[state] / probability_sum
        full_result[key] = temp_results

    # print("VS: ", full_result)
    
    # For each state, compute the average probability across all keys
    # (i.e., across all prompts or response sets)
    # This will create a new dict: {state: average_probability}

    state_totals = defaultdict(float)
    state_counts = defaultdict(int)

    for key, value in full_result.items():
        for state, prob in value.items():
            state_totals[state] += prob
            state_counts[state] += 1

    avg_state_probs = {}
    for state in state_totals:
        avg_state_probs[state] = state_totals[state] / state_counts[state] if state_counts[state] > 0 else 0.0

    result = avg_state_probs
    # print(result)

    return result


def read_sequence_data(path):
    with open(path, "r") as f:
        data = json.load(f)
    
    results = {}
    for key, responses in data.items():
        for resp in responses:
            results[resp["text"]] = resp["probability"]

    return results


def read_direct_data(path):
    with open(path, "r") as f:
        data = json.load(f)

    all_results = defaultdict(int)
    total_counts = 0
    for response_dict in data:
        for state, count in response_dict.items():
            all_results[state] += count
            total_counts += count
    for state in all_results:
        all_results[state] = all_results[state] / total_counts

    # print("Direct: ", all_results)

    # state_probs = defaultdict(list)
    # for response_dict in data:
    #     total_counts = sum(response_dict.values())
    #     if total_counts == 0:
    #         continue
    #     for state, count in response_dict.items():
    #         prob = count / total_counts
    #         state_probs[state].append(prob)

    # Now, average the probabilities for each state
    # avg_results = {}
    # for state, probs in state_probs.items():
    #     avg_results[state] = sum(probs) / len(probs) if probs else 0.0

    # all_results = avg_results

    # print(all_results)
    return all_results


def export_to_excel(gt_data, direct_data, vs_data, sequence_data, filename="state_comparison.xlsx"):
    # Get all states present in any of the dicts
    all_states = set(gt_data.keys()) | set(direct_data.keys()) | set(vs_data.keys()) | set(sequence_data.keys())
    # Build rows
    rows = []
    for state in all_states:
        gt = gt_data.get(state, 0.0)
        direct = direct_data.get(state, 0.0)
        vs = vs_data.get(state, 0.0)
        sequence = sequence_data.get(state, 0.0)
        rows.append({
            "State": state,
            "Direct": direct,
            "Pretraining": gt,
            "VS": vs,
            "Sequence": sequence
        })
    # First, sort by Direct probability descending, then by Pretraining Distribution descending
    rows_sorted = sorted(
        rows,
        key=lambda x: (x["Direct"], x["Pretraining"]),
        reverse=True
    )
    df = pd.DataFrame(rows_sorted)
    # Format probabilities to three decimals for output
    for col in ["Pretraining", "Direct", "VS", "Sequence"]:
        df[col] = df[col].apply(lambda x: round(x, 3))
    df.to_excel(filename, index=False)
    print(f"Excel file saved to {filename}")


def main():
    gt_path = "analyse/pre_training_distribution/state_name_distribution.json"
    with open(gt_path, "r") as f:
        gt_data = json.load(f)
    gt_data = {state_name: info["percentage"] for state_name, info in gt_data.items()}

    vs_gpt_4_1_data = read_VS_data("analyse/pre_training_distribution/vs_gpt-4.1.json")
    vs_claude_4_sonnet_data = read_VS_data("analyse/pre_training_distribution/vs_claude-4-sonnet.json")
    vs_cot_gpt_4_1_data = read_VS_data("analyse/pre_training_distribution/vs_cot_gpt-4.1.json")
    vs_cot_claude_4_sonnet_data = read_VS_data("analyse/pre_training_distribution/vs_cot_claude-4-sonnet.json")
    vs_multi_turn_gpt_4_1_data = read_VS_data("analyse/pre_training_distribution/vs_multi_gpt-4.1.json")
    vs_multi_turn_claude_4_sonnet_data = read_VS_data("analyse/pre_training_distribution/vs_multi_claude-4-sonnet.json")
    # sequence_gpt_4_1_data = read_sequence_data("analyse/pre_training_distribution/sequence_gpt-4.1.json")
    # sequence_claude_4_sonnet_data = read_sequence_data("analyse/pre_training_distribution/sequence_claude-4-sonnet.json")
    direct_gpt_4_1_data = read_direct_data("analyse/pre_training_distribution/direct_gpt-4.1.json")
    direct_claude_4_sonnet_data = read_direct_data("analyse/pre_training_distribution/direct_claude-4-sonnet.json")

    # INSERT_YOUR_CODE
    # print("Ground-truth: ", {k: f"{v:.2f}" for k, v in gt_data.items()})
    # print("Claude-4-Sonnet Direct: ", {k: f"{v:.2f}" for k, v in direct_claude_4_sonnet_data.items()})
    # print("Claude-4-Sonnet Sequence: ", {k: f"{v:.2f}" for k, v in sequence_claude_4_sonnet_data.items()})
    # print("Claude-4-Sonnet VS: ", {k: f"{v:.2f}" for k, v in vs_claude_4_sonnet_data.items()})
    # print("GPT-4.1 Direct: ", direct_gpt_4_1_data)
    # print("GPT-4.1 Sequence: ", sequence_gpt_4_1_data)
    # print("GPT-4.1 VS: ", vs_gpt_4_1_data)

    # # Example usage for Claude-4-Sonnet
    # export_to_excel(
    #     gt_data,
    #     direct_claude_4_sonnet_data,
    #     vs_claude_4_sonnet_data,
    #     sequence_claude_4_sonnet_data,
    #     filename="state_comparison_claude_4_sonnet.xlsx"
    # )

    calc_kl_divergence_vs(gt_data, vs_claude_4_sonnet_data, vs_cot_claude_4_sonnet_data, vs_multi_turn_claude_4_sonnet_data)
    calc_kl_divergence_vs(gt_data, vs_gpt_4_1_data, vs_cot_gpt_4_1_data, vs_multi_turn_gpt_4_1_data)

    # plot_histograms(gt_data, vs_gpt_4_1_data, vs_cot_gpt_4_1_data, vs_multi_turn_gpt_4_1_data, "gpt-4.1")
    # plot_histograms(gt_data, vs_claude_4_sonnet_data, vs_cot_claude_4_sonnet_data, vs_multi_turn_claude_4_sonnet_data, "claude-4-sonnet")

    # plot_histograms(gt_data, direct_gpt_4_1_data, sequence_gpt_4_1_data, vs_gpt_4_1_data, "gpt-4.1")
    # plot_histograms(gt_data, direct_claude_4_sonnet_data, sequence_claude_4_sonnet_data, vs_claude_4_sonnet_data, "claude-4-sonnet")

    # calc_kl_divergence_vs(gt_data, direct_gpt_4_1_data, sequence_gpt_4_1_data, vs_gpt_4_1_data)
    # calc_kl_divergence_vs(gt_data, direct_claude_4_sonnet_data, sequence_claude_4_sonnet_data, vs_claude_4_sonnet_data)



if __name__ == "__main__":
    main()