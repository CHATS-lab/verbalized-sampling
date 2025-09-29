import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path

METHOD_MAP = {
    "direct": ("Direct", "direct"),
    "direct_cot": ("Direct_CoT", "direct_cot"),
    "sequence": ("Sequence", "sequence"),
    "multi_turn": ("Multi_turn", "multi_turn"),
    "vs_standard": ("Structure_with_prob", "vs_standard"),
    "vs_cot": ("Chain_of_thought", "vs_cot"),
    "vs_multi": ("Combined", "vs_multi"),
}

def load_metric_from_file(file_path, metric_key):
    """Load a specific metric from a results file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        result = data.get('overall_metrics', {}).get(metric_key, None)
        return result
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Error loading {metric_key} from {file_path}: {e}")
        return None


def extract_creative_data(base_dir, task_name):
    # Define methods to look for
    model_list = [
        "openai_gpt-4.1-mini",
        "openai_gpt-4.1",
        "google_gemini-2.5-flash",
        "google_gemini-2.5-pro",
        "meta-llama_Llama-3.1-70B-Instruct",
        "deepseek_deepseek-r1-0528",
        "openai_o3",
        "anthropic_claude-4-sonnet",
    ]
    baseline_method = ["direct", "direct_cot", "sequence", "multi_turn"]
    if task_name == "joke":
        baseline_method = ["direct", "sequence", "multi_turn"]
    verbalized_methods = ["vs_standard", "vs_cot", "vs_multi"]
    metrics = ["avg_diversity"]
    
    # Collect all data
    metrics_values = {}
    for model_dir in os.listdir(base_dir):
        model_name = model_dir
        if model_name not in model_list:
            continue
        evaluation_dir = Path(base_dir) / model_dir / f"{model_name}_{task_name}" / "evaluation"
        # print(evaluation_dir)
        if not evaluation_dir.exists():
            print(f"Warning: No evaluation directory found for {model_name}")
            continue
        # Iterate through all method directories
        for method_dir in evaluation_dir.iterdir():
            if not method_dir.is_dir():
                continue
            method_name = method_dir.name
            method_name = method_name.split(' ')[0]

            results_file = method_dir / "diversity_results.json"
            if not results_file.exists():
                print(f"Warning: No results file found for {model_name} - {method_name}")
                continue
            try:
                with open(results_file, "r") as f:
                    data = json.load(f)
                overall_metrics = data.get("overall_metrics", {})
     
                # Find the matching method in METHOD_MAP
                mapped_method_name = None
                for key, (display_name, method_value) in METHOD_MAP.items():
                    if method_value == method_name:
                        mapped_method_name = display_name
                        break
                
                if mapped_method_name is None:
                    print(f"Warning: {method_name} not found in METHOD_MAP")
                    continue
                
                method_name = mapped_method_name

                # Initialize data structure for this model-method combination
                if model_name not in metrics_values:
                    metrics_values[model_name] = {}
                if method_name not in metrics_values[model_name]:
                    metrics_values[model_name][method_name] = {metric: [] for metric in metrics}
                # Collect metric values from all prompts
                for metric in metrics:
                    if metric in overall_metrics:
                        metrics_values[model_name][method_name][metric].append(overall_metrics[metric])
            except Exception as e:
                print(f"Error reading {results_file}: {e}")

    # print(metrics_values)
    return metrics_values


def extract_creative_task_data(exp_path, task_name):
    """Extract poem data from poem_experiments_final folder - focusing on pairwise semantic diversity"""
    poem_baseline = []
    poem_verbalized = []
    
    poem_dir = Path(exp_path)
    if not poem_dir.exists():
        print(f"{exp_path} folder not found")
        return {f'{task_name}_baseline': 0, f'{task_name}_verbalized': 0}
    
    print(f"Extracting creative task data from {exp_path}...")
    
    metrics_values = extract_creative_data(poem_dir, task_name)

    model_list = [
        "openai_gpt-4.1-mini",
        "openai_gpt-4.1",
        "google_gemini-2.5-flash",
        "google_gemini-2.5-pro",
        "meta-llama_Llama-3.1-70B-Instruct",
        "deepseek_deepseek-r1-0528",
        "openai_o3",
        "anthropic_claude-4-sonnet",
    ]
    baseline_methods = ["direct", "direct_cot", "sequence", "multi_turn"]
    if task_name == "joke":
        baseline_methods = ["direct", "sequence", "multi_turn"]
    vs_methods = ["vs_standard", "vs_cot", "vs_multi"]
    metrics = ["avg_diversity"]
    baseline_method_means = []
    vs_method_means = []

    for method in baseline_methods:
        vals = []
        for model_name in model_list:
            if model_name in metrics_values:
                for method_name, method_data in metrics_values[model_name].items():
                    if METHOD_MAP[method][1] in method_name.lower():
                        vals.extend(method_data["avg_diversity"])
        mean_val = np.mean(vals) if vals else np.nan
        baseline_method_means.append(mean_val)

    for method in vs_methods:
        vals = []
        for model_name in model_list:
            if model_name in metrics_values:
                for method_name, method_data in metrics_values[model_name].items():
                    if METHOD_MAP[method][1] in method_name.lower():
                        vals.extend(method_data["avg_diversity"])
        mean_val = np.mean(vals) if vals else np.nan
        vs_method_means.append(mean_val)

    return {
        f'{task_name}_direct': baseline_method_means[0],
        f'{task_name}_baseline': np.max(baseline_method_means) if baseline_method_means else 0,
        f'{task_name}_verbalized': np.max(vs_method_means) if vs_method_means else 0
    }


def extract_bias_data():
    """Extract bias task data from method_results_bias folder"""
    baseline_values = []
    vs_values = []
    metrics = ["unique_recall_rate"]  # Define metrics at the beginning
    
    bias_dir = Path("method_results_bias")
    task_name = "state_name"
    if not bias_dir.exists():
        print("method_results_bias folder not found")
        return 0
    
    print("Extracting bias task data from method_results_bias...")

    model_list = [
        "gpt-4.1-mini",
        "gpt-4.1",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "meta-llama_Llama-3.1-70B-Instruct",
        "deepseek-r1",
        "o3",
        "claude-4-sonnet",
    ]
    baseline_methods = ["direct", "direct_cot", "sequence", "multi_turn"]
    vs_methods = ["vs_standard", "vs_cot", "vs_multi"]  # Use the keys, not the values
    
    # Collect all data
    metrics_values = {}
    for model_dir in os.listdir(bias_dir):
        if not model_dir.endswith(f"_{task_name}"):
            continue
        model_name = model_dir.replace(f"_{task_name}", "")
        evaluation_dir = Path(bias_dir) / model_dir / "evaluation"
        if not evaluation_dir.exists():
            print(f"Warning: No evaluation directory found for {model_name}")
            continue
        # Iterate through all method directories
        for method_dir in evaluation_dir.iterdir():
            if not method_dir.is_dir():
                continue
            method_name = method_dir.name
            method_name = method_name.split(' ')[0]

            results_file = method_dir / "response_count_results.json"
            if not results_file.exists():
                print(f"Warning: No results file found for {model_name} - {method_name}")
                continue
            try:
                with open(results_file, "r") as f:
                    data = json.load(f)
                aggregate_metrics = data.get("overall_metrics", {})
                per_prompt_stats = aggregate_metrics.get("per_prompt_stats", {})

                # Find the matching method in METHOD_MAP
                mapped_method_name = None
                for key, (display_name, method_value) in METHOD_MAP.items():
                    if method_value == method_name:
                        mapped_method_name = display_name
                        break
                
                if mapped_method_name is None:
                    print(f"Warning: {method_name} not found in METHOD_MAP")
                    continue
                
                method_name = mapped_method_name

                # Initialize data structure for this model-method combination
                if model_name not in metrics_values:
                    metrics_values[model_name] = {}
                if method_name not in metrics_values[model_name]:
                    metrics_values[model_name][method_name] = {metric: [] for metric in metrics}
                # Collect metric values from all prompts
                for prompt_stats in per_prompt_stats.values():
                    for metric in metrics:
                        if metric in prompt_stats:
                            metrics_values[model_name][method_name][metric].append(prompt_stats[metric])
            except Exception as e:
                print(f"Error reading {results_file}: {e}")

    # First, average the unique_recall_rate across models for each method (baseline and vs)
    baseline_method_means = []
    vs_method_means = []

    for method in baseline_methods:
        vals = []
        for model_name in model_list:
            if model_name in metrics_values:
                for method_name, method_data in metrics_values[model_name].items():
                    if METHOD_MAP[method][1] in method_name.lower():
                        vals.extend(method_data["unique_recall_rate"])
        mean_val = np.mean(vals) if vals else np.nan
        baseline_method_means.append(mean_val)

    for method in vs_methods:
        vals = []
        for model_name in model_list:
            if model_name in metrics_values:
                for method_name, method_data in metrics_values[model_name].items():
                    if METHOD_MAP[method][1] in method_name.lower():
                        vals.extend(method_data["unique_recall_rate"])
        mean_val = np.mean(vals) if vals else np.nan
        vs_method_means.append(mean_val)

    return {
        'bias_direct': baseline_method_means[0],
        'baseline': max(baseline_method_means) if baseline_method_means else 0,
        'vs': max(vs_method_means) if vs_method_means else 0
    }

def extract_dialogue_data():
    """Extract dialogue simulation data from latex_table_results.txt as fallback"""
    baseline_values = {"l1_distance": [], "ks_value": []}
    vs_values = {"l1_distance": [], "ks_value": []}
    fine_tuned_values = {"l1_distance": [], "ks_value": []}
    sequence_values = {"l1_distance": [], "ks_value": []}

    dialogue_dir = Path("dialogue_simulation_final/exp_results")
    if not dialogue_dir.exists():
        print("dialogue_simulation_final folder not found")
        return {'dialogue_baseline': 0, 'dialogue_verbalized': 0}
    
    print("Extracting dialogue simulation data from dialogue_simulation_final...")
    baseline_dir = dialogue_dir / "baseline" / "gpt-4.1"
    sequence_dir = dialogue_dir / "sampling" / "sequence" / "gpt-4.1"
    vs_dir = dialogue_dir / "sampling" / "random_selection" / "gpt-4.1"
    fine_tuned_dir = dialogue_dir / "fine_tuning" / "gpt-4.1" / "llama3.1_8b_sft_w_promp_5epochs"
    model_list = ["gpt-4.1-mini", "gpt-4.1", "gemini-2.5-flash", "gemini-2-5-pro", "claude-4-sonnet", "meta-llama_Llama-3.1-70b-Instruct", "deepseek-r1", "o3"]

    for model in model_list:
        # print(f"Processing {model}...")
        baseline_file = baseline_dir / model / "analysis_total_results.json"
        vs_file = vs_dir / model / "analysis_total_results.json"
        sequence_file = sequence_dir / model / "analysis_total_results.json"

        if baseline_file.exists():
            with open(baseline_file, "r") as f:
                baseline_json = json.load(f)
                baseline_l1_distance = baseline_json["donation"]["amount"]["l1_distance"]["l1_distance"]["intended_donation_amount"]["0"]
                baseline_ks_value = baseline_json["donation"]["amount"]["ks_test"]["ks_statistic"]["intended_donation_amount"]
                baseline_values["l1_distance"].append(baseline_l1_distance)
                baseline_values["ks_value"].append(baseline_ks_value)
        if vs_file.exists():
            with open(vs_file, "r") as f:
                vs_json = json.load(f)
                vs_l1_distance = vs_json["donation"]["amount"]["l1_distance"]["l1_distance"]["intended_donation_amount"]["0"]
                vs_ks_value = vs_json["donation"]["amount"]["ks_test"]["ks_statistic"]["intended_donation_amount"]
                vs_values["l1_distance"].append(vs_l1_distance)
                vs_values["ks_value"].append(vs_ks_value)
        if sequence_file.exists():
            with open(sequence_file, "r") as f:
                sequence_json = json.load(f)
                sequence_l1_distance = sequence_json["donation"]["amount"]["l1_distance"]["l1_distance"]["intended_donation_amount"]["0"]
                sequence_ks_value = sequence_json["donation"]["amount"]["ks_test"]["ks_statistic"]["intended_donation_amount"]
                sequence_values["l1_distance"].append(sequence_l1_distance)
                sequence_values["ks_value"].append(sequence_ks_value)
    
    if fine_tuned_dir.exists():
        with open(fine_tuned_dir / "analysis_total_results.json", "r") as f:
            fine_tuned_json = json.load(f)
            fine_tuned_l1_distance = fine_tuned_json["donation"]["amount"]["l1_distance"]["l1_distance"]["intended_donation_amount"]["0"]
            fine_tuned_ks_value = fine_tuned_json["donation"]["amount"]["ks_test"]["ks_statistic"]["intended_donation_amount"]
            fine_tuned_values["l1_distance"].append(fine_tuned_l1_distance)
            fine_tuned_values["ks_value"].append(fine_tuned_ks_value)
    return {
        'dialogue_l1_distance_baseline': np.min(baseline_values["l1_distance"]),
        'dialogue_l1_distance_verbalized': np.min(vs_values["l1_distance"]),
        'dialogue_ks_value_baseline': np.min(baseline_values["ks_value"]),
        'dialogue_ks_value_verbalized': np.min(vs_values["ks_value"]),
        'dialogue_l1_distance_fine_tuned': np.min(fine_tuned_values["l1_distance"]),
        'dialogue_ks_value_fine_tuned': np.min(fine_tuned_values["ks_value"]),
        'dialogue_l1_distance_sequence': np.min(sequence_values["l1_distance"]),
        'dialogue_ks_value_sequence': np.min(sequence_values["ks_value"])
    }

# Extract actual data from experimental results
print("Extracting performance data from experimental results...")
print("=" * 60)

# Extract joke data
joke_data = extract_creative_task_data("joke_experiments_final", "joke")
print(f"\nJoke Task:")
print(f"  Direct: {joke_data['joke_direct']:.3f}")
print(f"  Baseline: {joke_data['joke_baseline']:.2f}")
print(f"  Verbalized: {joke_data['joke_verbalized']:.2f}")

# Extract poem data
poem_data = extract_creative_task_data("poem_experiments_final", "poem")
print(f"\nPoem Task:")
print(f"  Direct: {poem_data['poem_direct']:.3f}")
print(f"  Baseline: {poem_data['poem_baseline']:.3f}")
print(f"  Verbalized: {poem_data['poem_verbalized']:.3f}")

# Extract story data
story_data = extract_creative_task_data("story_experiments_final", "book")
print(f"\nStory Task:")
print(f"  Direct: {story_data['book_direct']:.3f}")
print(f"  Baseline: {story_data['book_baseline']:.2f}")
print(f"  Verbalized: {story_data['book_verbalized']:.2f}")

# Extract bias data
bias_value = extract_bias_data()
print(f"\nBias Task:")
print(f"  Direct: {bias_value['bias_direct']:.3f}")
print(f"  Baseline: {bias_value['baseline']:.2f}")
print(f"  Verbalized: {bias_value['vs']:.2f}")

# Extract dialogue data
dialogue_data = extract_dialogue_data()
print(f"\nDialogue Simulation:")
print(f"  Baseline L1 Distance: {dialogue_data['dialogue_l1_distance_baseline']:.2f}")
print(f"  Verbalized L1 Distance: {dialogue_data['dialogue_l1_distance_verbalized']:.2f}")
print(f"  Baseline KS Value: {dialogue_data['dialogue_ks_value_baseline']:.2f}")
print(f"  Verbalized KS Value: {dialogue_data['dialogue_ks_value_verbalized']:.2f}")

# Task names
tasks = ['Joke Writing', 'Poem Writing', 'Story Writing', 'Dialogue Simulation', 'Open-ended QA']

# Prepare data for each task
# For Poem and Bias Mitigation: [Direct, Previous Best, Verbalized]
# For Dialogue Simulation: [Baseline, Fine-tuned (Previous Best), Verbalized]
# Use color-blind friendly colors (Color Universal Design: blue, orange, green)
# https://davidmathlogic.com/colorblind/

# Data extraction
joke_direct = joke_data['joke_direct']
joke_baseline = joke_data['joke_baseline']
joke_vs = joke_data['joke_verbalized']

poem_direct = poem_data['poem_direct']
poem_baseline = poem_data['poem_baseline']
poem_vs = poem_data['poem_verbalized']

story_direct = story_data['book_direct']
story_baseline = story_data['book_baseline']
story_vs = story_data['book_verbalized']

dialogue_baseline = dialogue_data['dialogue_ks_value_baseline']
dialogue_finetuned = dialogue_data['dialogue_ks_value_fine_tuned']
dialogue_sequence = dialogue_data['dialogue_ks_value_sequence']
dialogue_vs = dialogue_data['dialogue_ks_value_verbalized']

bias_direct = bias_value['bias_direct']
bias_baseline = bias_value['baseline']
bias_vs = bias_value['vs']

# Data for plotting
bar_labels = [
    ['Direct', 'Previous Best', 'Verbalized'],  # Poem
    ['Baseline', 'Previous Best', 'Verbalized'],  # Dialogue
    ['Direct', 'Previous Best', 'Verbalized']   # Bias
]
# Values for each group
bar_values = [
    [joke_direct, joke_baseline, joke_vs],
    [poem_direct, poem_baseline, poem_vs],
    [story_direct, story_baseline, story_vs],
    # [dialogue_baseline, dialogue_finetuned, dialogue_vs],
    [dialogue_baseline, dialogue_sequence, dialogue_vs],
    [bias_direct, bias_baseline, bias_vs]
]

# Color-blind friendly palette (blue, orange, green)
bar_colors = ['#DBDEE4', '#F392B0', '#80BFFD']

fig, ax = plt.subplots(figsize=(18, 10))

x = np.arange(len(tasks))
width = 0.25

# Draw bars for each group
bars = []
for i in range(3):  # 3 bars per group
    bar = ax.bar(
        x + (i - 1) * width,  # center the three bars
        [bar_values[j][i] for j in range(len(tasks))],
        width,
        label=bar_labels[0][i] if i == 0 else (bar_labels[1][i] if i == 1 else bar_labels[2][i]),
        color=bar_colors[i],
        alpha=0.85,
        edgecolor=bar_colors[i],
        linewidth=1
    )
    bars.append(bar)

# Customize the chart
ax.set_xlabel('', fontsize=18, fontweight='bold')
ax.set_ylabel('Performance Score', fontsize=20, fontweight='bold')
ax.tick_params(axis='y', which='major', labelsize=18)
ax.set_title('', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(tasks, fontsize=20, fontweight='bold')

# Custom legend: use consistent labels for all tasks
custom_labels = ['Direct', 'Best Baseline', 'Verbalized Sampling']
ax.legend(bars, custom_labels, fontsize=20, loc='upper center', bbox_to_anchor=(0.5, 1.09), ncol=3)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
def add_value_labels(bars):
    for bar in bars:
        for rect in bar:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=18, fontweight='bold')

for bar in bars:
    add_value_labels([bar])

# Add improvement percentages (Verbalized vs Previous Best)
for i in range(len(tasks)):
    prev_best = bar_values[i][1]
    verbalized = bar_values[i][2]
    if prev_best != 0:
        improvement = ((verbalized - prev_best) / prev_best) * 100
        ax.annotate(f'{improvement:+.1f}%' if improvement > 0 else f'{improvement:.1f}%',
                    xy=(x[i] + width, verbalized),
                    xytext=(8, 25),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=18, fontweight='bold',
                    color='#2E8B57')  # green for improvement

plt.tight_layout()
plt.ylim(0, max([max(vals) for vals in bar_values]) * 1.15)

# Save the plot
plt.savefig('intro_performance_comparison.pdf', bbox_inches='tight')

# Print summary statistics
print("\n" + "=" * 60)
print("PERFORMANCE SUMMARY")

# Create a summary table
summary_data = {
    'Task': tasks,
    'Direct': [bar_values[i][0] for i in range(len(tasks))],
    'Best Baseline': [bar_values[i][1] for i in range(len(tasks))],
    'Verbalized': [bar_values[i][2] for i in range(len(tasks))],
    'Improvement (%)': [
        ((bar_values[i][2] - bar_values[i][1]) / bar_values[i][1] * 100) if bar_values[i][1] != 0 else 0
        for i in range(len(tasks))
    ]
}

summary_df = pd.DataFrame(summary_data)
print("Summary Table:")
print(summary_df.to_string(index=False, float_format='%.2f'))
