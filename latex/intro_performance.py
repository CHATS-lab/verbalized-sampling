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
    "structure_with_prob": ("Structure_with_prob", "structure_with_prob"),
    "chain_of_thought": ("Chain_of_thought", "chain_of_thought"),
    "combined": ("Combined", "combined"),
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
    verbalized_methods = ["structure_with_prob", "chain_of_thought", "combined"]
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


def extract_poem_data():
    """Extract poem data from poem_experiments_final folder - focusing on pairwise semantic diversity"""
    poem_baseline = []
    poem_verbalized = []
    
    poem_dir = Path("poem_experiments_final")
    if not poem_dir.exists():
        print("poem_experiments_final folder not found")
        return {'poem_baseline': 0, 'poem_verbalized': 0}
    
    print("Extracting poem task data from poem_experiments_final...")
    
    metrics_values = extract_creative_data(poem_dir, "poem")

    model_list = [
        "openai_gpt-4.1-mini",
        "openai_gpt-4.1",
        "google_gemini-2.5-flash",
        "google_gemini-2.5-pro",
        # "meta-llama_Llama-3.1-70B-Instruct",
        "deepseek_deepseek-r1-0528",
        "openai_o3",
        "anthropic_claude-4-sonnet",
    ]
    baseline_methods = ["direct", "direct_cot", "sequence", "multi_turn"]
    vs_methods = ["structure_with_prob", "chain_of_thought", "combined"]
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
        'poem_direct': baseline_method_means[0],
        'poem_baseline': np.max(baseline_method_means) if baseline_method_means else 0,
        'poem_verbalized': np.max(vs_method_means) if vs_method_means else 0
    }

def extract_joke_data():
    """Extract joke data from joke_experiments_final folder - focusing on pairwise semantic diversity"""
    joke_baseline = []
    joke_verbalized = []
    
    joke_dir = Path("joke_experiments_final")
    if not joke_dir.exists():
        print("joke_experiments_final folder not found")
        return {'joke_baseline': 0, 'joke_verbalized': 0}
    
    print("Extracting joke task data from joke_experiments_final...")

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
    baseline_methods = ["direct", "sequence", "multi_turn"]
    vs_methods = ["structure_with_prob", "chain_of_thought", "combined"]
    metrics = ["avg_diversity"]
    
    metrics_values = extract_creative_data(joke_dir, "joke")
    
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
        print(method, mean_val)

    for method in vs_methods:
        vals = []
        for model_name in model_list:
            if model_name in metrics_values:
                for method_name, method_data in metrics_values[model_name].items():
                    if METHOD_MAP[method][1] in method_name.lower():
                        vals.extend(method_data["avg_diversity"])
        mean_val = np.mean(vals) if vals else np.nan
        vs_method_means.append(mean_val)
        print(method, mean_val)
        
    return {
        'joke_direct': baseline_method_means[0],
        'joke_baseline': np.max(baseline_method_means) if baseline_method_means else 0,
        'joke_verbalized': np.max(vs_method_means) if vs_method_means else 0
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
    vs_methods = ["structure_with_prob", "chain_of_thought", "combined"]  # Use the keys, not the values
    
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

    dialogue_dir = Path("dialogue_simulation_final/exp_results")
    if not dialogue_dir.exists():
        print("dialogue_simulation_final folder not found")
        return {'dialogue_baseline': 0, 'dialogue_verbalized': 0}
    
    print("Extracting dialogue simulation data from dialogue_simulation_final...")
    baseline_dir = dialogue_dir / "baseline" / "gpt-4.1"
    vs_dir = dialogue_dir / "sampling" / "random_selection" / "gpt-4.1"
    model_list = ["gpt-4.1-mini", "gpt-4.1", "gemini-2.5-flash", "gemini-2-5-pro", "claude-4-sonnet", "meta-llama_Llama-3.1-70b-Instruct", "deepseek-r1", "o3"]

    for model in model_list:
        # print(f"Processing {model}...")
        baseline_file = baseline_dir / model / "analysis_total_results.json"
        vs_file = vs_dir / model / "analysis_total_results.json"

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

    return {
        'dialogue_l1_distance_baseline': np.mean(baseline_values["l1_distance"]),
        'dialogue_l1_distance_verbalized': np.mean(vs_values["l1_distance"]),
        'dialogue_ks_value_baseline': np.mean(baseline_values["ks_value"]),
        'dialogue_ks_value_verbalized': np.mean(vs_values["ks_value"])
    }

# Extract actual data from experimental results
print("Extracting performance data from experimental results...")
print("=" * 60)

# Extract poem data
poem_data = extract_poem_data()
print(f"\nPoem Task:")
print(f"  Direct: {poem_data['poem_direct']:.3f}")
print(f"  Baseline: {poem_data['poem_baseline']:.3f}")
print(f"  Verbalized: {poem_data['poem_verbalized']:.3f}")
if poem_data['poem_baseline'] > 0:
    improvement = ((poem_data['poem_verbalized'] - poem_data['poem_baseline']) / poem_data['poem_baseline'] * 100)
    print(f"  Improvement: {improvement:+.1f}%")

# Extract joke data
joke_data = extract_joke_data()
print(f"\nJoke Task:")
print(f"  Direct: {joke_data['joke_direct']:.3f}")
print(f"  Baseline: {joke_data['joke_baseline']:.2f}")
print(f"  Verbalized: {joke_data['joke_verbalized']:.2f}")
if joke_data['joke_baseline'] > 0:
    improvement = ((joke_data['joke_verbalized'] - joke_data['joke_baseline']) / joke_data['joke_baseline'] * 100)
    print(f"  Improvement: {improvement:+.1f}%")

# Extract bias data
bias_value = extract_bias_data()
print(f"\nBias Task:")
print(f"  Direct: {bias_value['bias_direct']:.3f}")
print(f"  Baseline: {bias_value['baseline']:.2f}")
print(f"  Verbalized: {bias_value['vs']:.2f}")
if bias_value['baseline'] > 0:
    improvement = ((bias_value['vs'] - bias_value['baseline']) / bias_value['baseline'] * 100)
    print(f"  Improvement: {improvement:+.1f}%")

# Extract dialogue data
dialogue_data = extract_dialogue_data()
print(f"\nDialogue Simulation:")
print(f"  Baseline L1 Distance: {dialogue_data['dialogue_l1_distance_baseline']:.2f}")
print(f"  Verbalized L1 Distance: {dialogue_data['dialogue_l1_distance_verbalized']:.2f}")
print(f"  Baseline KS Value: {dialogue_data['dialogue_ks_value_baseline']:.2f}")
print(f"  Verbalized KS Value: {dialogue_data['dialogue_ks_value_verbalized']:.2f}")
improvement_l1_distance = ((dialogue_data['dialogue_l1_distance_verbalized'] - dialogue_data['dialogue_l1_distance_baseline']) / dialogue_data['dialogue_l1_distance_baseline'] * 100)
improvement_ks_value = ((dialogue_data['dialogue_ks_value_verbalized'] - dialogue_data['dialogue_ks_value_baseline']) / dialogue_data['dialogue_ks_value_baseline'] * 100)
print(f"  Improvement L1 Distance: {improvement_l1_distance:+.1f}%")
print(f"  Improvement KS Value: {improvement_ks_value:+.1f}%")

# Task names
tasks = ['Poem', 'Dialogue Simulation', 'Bias Mitigation']

# Use extracted data
# baseline_avg = [
#     poem_data['poem_baseline'],
#     # joke_data['joke_baseline'], 
#     dialogue_data['dialogue_ks_value_baseline'],
#     bias_value['baseline']
# ]
baseline_avg = [
    poem_data['poem_direct'],
    # joke_data['joke_direct'], 
    dialogue_data['dialogue_ks_value_baseline'],
    bias_value['bias_direct']
]

verbalized_avg = [
    poem_data['poem_verbalized'],
    # joke_data['joke_verbalized'],
    dialogue_data['dialogue_ks_value_verbalized'],
    bias_value['vs']
]

# Create the grouped bar chart
fig, ax = plt.subplots(figsize=(12, 8))

# Set up the bar positions
x = np.arange(len(tasks))
width = 0.35

# Create bars
bars1 = ax.bar(x - width/2, baseline_avg, width, label='Direct Sampling Prompting', 
               color='#D41159', alpha=0.8, edgecolor='#D41159', linewidth=1)
bars2 = ax.bar(x + width/2, verbalized_avg, width, label='Verbalized Sampling Prompting', 
               color='#1A85FF', alpha=0.8, edgecolor='#1A85FF', linewidth=1)

# Add a line in the baseline bar to indicate the direct sampling value (for Poem, Joke, and Bias tasks)
# Assume poem_data['poem_direct'], joke_data['joke_direct'], and bias_value['bias_direct'] are available
# direct_sampling_values = [
#     poem_data.get('poem_direct', None),
#     joke_data.get('joke_direct', None),
#     bias_value.get('bias_direct', None)
# ]

# # Skip the dialogue simulation (index 1) when drawing direct sampling lines
# for i, direct_val in enumerate(direct_sampling_values):
#     if i == 1:
#         continue  # skip dialogue simulation
#     if direct_val is not None:
#         # Draw a horizontal line on the baseline bar for direct sampling
#         ax.hlines(direct_val, x[i] - width, x[i], 
#                   colors='black', linestyles='dashed', linewidth=2, label='Direct Sampling' if i == 0 else None)
#         # Optionally, annotate the value
#         ax.annotate(f'{direct_val:.3f}',
#                     xy=(x[i] - width/2, direct_val),
#                     xytext=(0, 10),
#                     textcoords="offset points",
#                     ha='center', va='top',
#                     fontsize=9, fontweight='bold', color='black')

# Customize the chart
ax.set_xlabel('', fontsize=14, fontweight='bold')
ax.set_ylabel('Performance Score', fontsize=14, fontweight='bold')
ax.set_title('', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(tasks, fontsize=12, fontweight='bold')
# Add 'Direct Sampling' to legend only once if present
handles, labels = ax.get_legend_handles_labels()
if any(l == 'Direct Sampling' for l in labels):
    ax.legend(fontsize=12, loc='upper left')
else:
    ax.legend(fontsize=12, loc='upper left')
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold')

add_value_labels(bars1)
add_value_labels(bars2)

# Add improvement percentages
for i, (baseline, verbalized) in enumerate(zip(baseline_avg, verbalized_avg)):
    if baseline != 0:  # Avoid division by zero
        improvement = ((verbalized - baseline) / baseline) * 100
        ax.annotate(f'{improvement:+.1f}%' if improvement > 0 else f'{improvement:.1f}%',
                    xy=(x[i] + width/2, verbalized),
                    xytext=(0, 15),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold',
                    color='green')

# Customize the plot
plt.tight_layout()
plt.ylim(0, max(max(baseline_avg), max(verbalized_avg)) * 1.15)

# Display the plot
# plt.show()
# plt.savefig('intro_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('intro_performance_comparison.pdf', bbox_inches='tight')

# Print summary statistics
print("\n" + "=" * 60)
print("PERFORMANCE SUMMARY")
print("=" * 60)
for i, task in enumerate(tasks):
    baseline = baseline_avg[i]
    verbalized = verbalized_avg[i]
    improvement = ((verbalized - baseline) / baseline) * 100 if baseline != 0 else 0
    print(f"{task}:")
    print(f"  Baseline: {baseline:.2f}")
    print(f"  Verbalized: {verbalized:.2f}")
    print(f"  Improvement: {improvement:+.1f}%")
    print()

# Create a summary table
summary_data = {
    'Task': tasks,
    'Baseline': baseline_avg,
    'Verbalized': verbalized_avg,
    'Improvement (%)': [(v-b)/b*100 if b != 0 else 0 for b, v in zip(baseline_avg, verbalized_avg)]
}

summary_df = pd.DataFrame(summary_data)
print("Summary Table:")
print(summary_df.to_string(index=False, float_format='%.2f'))
