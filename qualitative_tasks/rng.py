import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from openai import OpenAI
import json
import os
from tqdm import tqdm

roll_dice_prompt = "Roll a fair six-sided dice. Return ONLY the integer result (1-6), with no explanation or extra text."
direct_sampling_system_prompt = "Generate a response to the input prompt. Output ONLY the response, with no explanations or extra text."

def get_verbalized_sampling_system_prompt(num_samples):
    return f"""
Generate {num_samples} responses to the input prompt.

Return the responses in JSON format with keys: "responses" (list of dicts with 'text' and 'probability'). Each dictionary must include:
- 'text': the response string only.
- 'probability': a score from 0.0 to 1.0 representing how likely each response would be (1.0 = very likely, 0.0 = very unlikely).

Give ONLY the JSON object, with no explanations or extra text.
"""
structured_response_list_with_prob_schema = {
    "type": "json_schema",  # Required for OpenRouter
    "json_schema": {
        "name": "structured_with_prob_schema",
        "schema": {
            "type": "object",
            "properties": {
                "responses": {
                    "type": "array",
                    "description": "A list of dicts, each with a 'text' and 'probability' field, representing possible responses to the input prompt and corresponding probabilities of each response.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "The text of the response."
                            },
                            "probability": {
                                "type": "number",
                                "description": "How likely each response would be (value between 0 and 1)"
                            }
                        },
                        "required": ["text", "probability"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["responses"],
            "additionalProperties": False
        },
        "strict": True
    }
}


def _parse_response_with_schema(response):
    """
    Parses a response string (JSON) with a schema containing a 'responses' field.
    Returns a list of dicts with 'response' and 'probability' keys.
    """
    try:
        if isinstance(response, str):
            parsed = json.loads(response)
            
            # Handle double-escaped JSON strings (i.e., string inside a string)
            if isinstance(parsed, str):
                parsed = json.loads(parsed)
            
            # Handle different schema types
            if "responses" in parsed:
                responses = parsed["responses"]
                if isinstance(responses, list):
                    result = []
                    for resp in responses:
                        if isinstance(resp, dict) and "text" in resp and "probability" in resp:
                            result.append({
                                "text": resp["text"],
                                "probability": resp["probability"]
                            })
                    return result
        # If not a string or doesn't match expected schema, return as is
        return response
    except Exception as e:
        print(f"Error parsing response with schema: {e}")
        return [{"text": str(response), "probability": 1.0}]


def model_generate(num_samples, model_name, config, verbalized=False):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if verbalized:
        messages = [
            {"role": "system", "content": get_verbalized_sampling_system_prompt(num_samples)},
            {"role": "user", "content": roll_dice_prompt}
        ]
        completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                **config,
                response_format=structured_response_list_with_prob_schema
        )
        response = completion.choices[0].message.content
        parsed_response = _parse_response_with_schema(response)
        # print(f"Structured Output Response:\n" + "\n".join(str(resp) for resp in parsed_response))
        return parsed_response
    else:
        messages = [
            {"role": "system", "content": direct_sampling_system_prompt},
            {"role": "user", "content": roll_dice_prompt}
        ]
        completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                **config,
        )
        response = completion.choices[0].message.content
        return response

def vs_combined(num_samples, model_name, config, n_samples_per_turn):
    vs_results = []
    num_turns = num_samples // n_samples_per_turn
    last_turn_samples = num_samples % n_samples_per_turn
    for _ in tqdm(range(num_turns), desc="Verbalized sampling"):
        result = model_generate(n_samples_per_turn, model_name, config, verbalized=True)
        vs_results.extend([resp["text"] for resp in result])
    if last_turn_samples > 0:
        result = model_generate(last_turn_samples, model_name, config, verbalized=True)
        vs_results.extend([resp["text"] for resp in result])
    return vs_results


def roll_dice(num_samples, model_name, config, verbalized=False, n_samples_per_turn=1):
    direct_results = []
    vs_results = []
    if not verbalized:
        for _ in tqdm(range(num_samples), desc="Direct sampling"):
            single_result = model_generate(n_samples_per_turn, model_name, config, verbalized=verbalized)
            direct_results.append(single_result)
        return np.array(direct_results, dtype=int)
    else:
        vs_results = vs_combined(num_samples, model_name, config, n_samples_per_turn)
        return np.array(vs_results, dtype=int)


def compute_distribution(samples):
    counts = np.bincount(samples, minlength=7)[1:]  # ignore index 0
    probs = counts / counts.sum()
    return probs


def plot_distribution_comparison(direct_samples, sequence_samples, vs_standard_samples, n_rolls=10):
    """
    Plot the distribution comparison for dice rolls 1-6 using seaborn and matplotlib.
    Also computes and displays KL divergence from the uniform distribution for each method.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import entropy
    import pandas as pd

    # Set seaborn style
    sns.set_style("white")
    sns.set_palette("husl")

    # Dice values
    dice_values = [1, 2, 3, 4, 5, 6]

    # Compute counts for each method
    direct_counts = np.bincount(direct_samples, minlength=7)[1:]
    sequence_counts = np.bincount(sequence_samples, minlength=7)[1:]
    vs_standard_counts = np.bincount(vs_standard_samples, minlength=7)[1:]
    uniform_counts = np.ones(6) / 6 * n_rolls

    # Print counts for debugging
    print("Direct counts: ", direct_counts)
    print("Sequence counts: ", sequence_counts)
    print("Verbalized counts: ", vs_standard_counts)
    print("Uniform counts: ", uniform_counts)

    # Compute probabilities for KL divergence
    direct_probs = direct_counts / direct_counts.sum()
    sequence_probs = sequence_counts / sequence_counts.sum()
    vs_standard_probs = vs_standard_counts / vs_standard_counts.sum()
    uniform_probs = np.ones(6) / 6

    # Compute KL divergence
    kl_direct = entropy(direct_probs, uniform_probs)
    kl_sequence = entropy(sequence_probs, uniform_probs)
    kl_verbalized = entropy(vs_standard_probs, uniform_probs)

    # Prepare data for plotting
    data = []
    for i, dice in enumerate(dice_values):
        data.append({'Dice': dice, 'Count': direct_counts[i], 'Method': 'Direct Sampling'})
        data.append({'Dice': dice, 'Count': sequence_counts[i], 'Method': 'Sequence Sampling'})
        data.append({'Dice': dice, 'Count': vs_standard_counts[i], 'Method': 'Verbalized Sampling'})
    df = pd.DataFrame(data)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 7))

    x_positions = np.arange(len(dice_values))
    width = 0.25

    # Plot bars for each method, side by side
    bars1 = ax.bar(x_positions-width, direct_counts, width, label='Direct Sampling',
                   color='#FC8EAC', alpha=0.5, edgecolor='#FC8EAC', linewidth=1)
    bars2 = ax.bar(x_positions, sequence_counts, width, label='Sequence Sampling',
                   color='#84b067', alpha=0.5, edgecolor='#84b067', linewidth=1)
    bars3 = ax.bar(x_positions+width, vs_standard_counts, width, label='Verbalized Sampling',
                   color='#678dc6', alpha=0.5, edgecolor='#678dc6', linewidth=1)

    # Set x-axis ticks and labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(dice_values)
    ax.set_xlim(-0.5, len(dice_values) - 0.5)

    # Add horizontal reference line for uniform distribution
    uniform_value = uniform_counts[0]
    ax.axhline(y=uniform_value, color='red', linestyle='--', linewidth=2,
               label=f'Uniform Distribution ({uniform_value:.1f})', alpha=0.8)
    ax.text(len(uniform_counts)-0.2, uniform_value + 0.2, f'{uniform_value:.1f}',
            color='red', fontsize=15, fontweight='bold', va='bottom', ha='right', alpha=0.9)

    # Add value labels on bars
    def add_value_labels(bars, bar_text_color):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom', fontsize=13, fontweight='bold', color=bar_text_color)

    add_value_labels(bars1, '#FC8EAC')
    add_value_labels(bars2, '#84b067')
    add_value_labels(bars3, '#678dc6')

    # Customize the plot
    ax.set_xlabel('Dice Roll Value', fontsize=15, fontweight='bold')
    ax.set_ylabel('Count', fontsize=15, fontweight='bold')
    ax.set_title('Dice Roll Distribution Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=13, loc='upper right')
    ax.set_ylim(0, max(direct_counts.max(), sequence_counts.max(), vs_standard_counts.max(), uniform_value) * 1.15)

    # Add statistics box
    stats_text = (
        f'KL Divergence from Uniform:\n'
        f'Direct: {kl_direct:.4f}\n'
        f'Sequence: {kl_sequence:.4f}\n'
        f'Verbalized: {kl_verbalized:.4f}'
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    plt.savefig("qualitative_tasks/rng_distribution_comparison.pdf", bbox_inches='tight')
    # plt.show()

def read_response_file(file_path):
    text_responses = []
    with open(file_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                response = json.loads(line)
                # Each resp should have a "responses" field which is a list of dicts with "text"
                for resp in response.get("responses", []):
                    try:
                        # Sometimes resp["text"] may not be a direct integer (e.g., '{"roll": 1}')
                        # Try to extract an integer between 1 and 6
                        if isinstance(resp["text"], int):
                            integer_response = resp["text"]
                        elif isinstance(resp["text"], str):
                            # Try to parse as int directly
                            try:
                                integer_response = int(resp["text"])
                            except ValueError:
                                # Try to parse if it's a JSON string like '{"roll": 1}'
                                try:
                                    possible_json = json.loads(resp["text"])
                                    if isinstance(possible_json, dict):
                                        # Look for a value that is an int between 1 and 6
                                        for v in possible_json.values():
                                            if isinstance(v, int) and 1 <= v <= 6:
                                                integer_response = v
                                                break
                                        else:
                                            continue  # No valid int found, skip
                                    else:
                                        continue  # Not a dict, skip
                                except Exception:
                                    continue  # Not valid JSON, skip
                        else:
                            continue  # Not a string or int, skip

                        if isinstance(integer_response, int) and 1 <= integer_response <= 6:
                            text_responses.append(integer_response)
                    except Exception:
                        continue
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in {file_path}: {e}")
    return text_responses

def calculate_kl_divergence_empirical(samples):
    """
    Calculate the KL divergence between the empirical distribution of dice rolls
    and the uniform distribution for a fair 6-faced dice (assuming 500 rolls).

    Args:
        samples (list or array-like): List of dice roll results (integers 1-6).

    Returns:
        float: The KL divergence D_KL(empirical || uniform).
    """
    # Count occurrences for faces 1-6
    counts = np.bincount(samples, minlength=7)[1:]  # ignore index 0
    empirical_probs = counts / counts.sum()
    uniform_probs = np.ones(6) / 6
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-12
    empirical_probs = np.clip(empirical_probs, epsilon, 1)
    uniform_probs = np.clip(uniform_probs, epsilon, 1)
    
    # Calculate KL divergence using scipy's entropy function
    # D_KL(P||Q) = sum(P * log(P/Q))
    kl_div = np.sum(empirical_probs * np.log(empirical_probs / uniform_probs))
    
    return kl_div

def calculate_kl_divergence():
    models = [
        "gpt-4.1-mini",
        "gpt-4.1",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "claude-4-sonnet",
        "deepseek-r1",
        "o3",
    ]
    direct_kl_results = []
    cot_kl_results = []
    multi_turn_kl_results = []
    sequence_kl_results = []
    vs_standard_kl_results = []
    vs_cot_kl_results = []
    vs_multi_kl_results = []
    for model in models:
        direct_file = f"method_results_rng/{model}_rand_num/generation/direct (samples=1)/responses.jsonl"
        cot_file = f"method_results_rng/{model}_rand_num/generation/direct_cot (samples=1)/responses.jsonl"
        multi_turn_file = f"method_results_rng/{model}_rand_num/generation/multi_turn (samples=5)/responses.jsonl"
        sequence_file = f"method_results_rng/{model}_rand_num/generation/sequence [strict] (samples=5)/responses.jsonl"
        vs_standard_file = f"method_results_rng/{model}_rand_num/generation/structure_with_prob [strict] (samples=5)/responses.jsonl"
        vs_cot_file = f"method_results_rng/{model}_rand_num/generation/chain_of_thought [strict] (samples=5)/responses.jsonl"
        vs_multi_file = f"method_results_rng/{model}_rand_num/generation/combined [strict] (samples=5)/responses.jsonl"
        
        direct_samples = read_response_file(direct_file)
        cot_samples = read_response_file(cot_file)
        multi_turn_samples = read_response_file(multi_turn_file)
        sequence_samples = read_response_file(sequence_file)
        vs_standard_samples = read_response_file(vs_standard_file)
        vs_cot_samples = read_response_file(vs_cot_file)
        vs_multi_samples = read_response_file(vs_multi_file)

        direct_kl_results.append(calculate_kl_divergence_empirical(direct_samples))
        cot_kl_results.append(calculate_kl_divergence_empirical(cot_samples))
        multi_turn_kl_results.append(calculate_kl_divergence_empirical(multi_turn_samples))
        sequence_kl_results.append(calculate_kl_divergence_empirical(sequence_samples))
        vs_standard_kl_results.append(calculate_kl_divergence_empirical(vs_standard_samples))
        vs_cot_kl_results.append(calculate_kl_divergence_empirical(vs_cot_samples))
        vs_multi_kl_results.append(calculate_kl_divergence_empirical(vs_multi_samples))
    
    print("Direct sampling KL divergence: %.3f" % np.mean(direct_kl_results))
    print("Direct sampling with COT KL divergence: %.3f" % np.mean(cot_kl_results))
    print("Multi-turn sampling KL divergence: %.3f" % np.mean(multi_turn_kl_results))
    print("Sequence sampling KL divergence: %.3f" % np.mean(sequence_kl_results))
    print("Verbalized sampling with standard schema KL divergence: %.3f" % np.mean(vs_standard_kl_results))
    print("Verbalized sampling with COT schema KL divergence: %.3f" % np.mean(vs_cot_kl_results))
    print("Verbalized sampling with multi-turn schema KL divergence: %.3f" % np.mean(vs_multi_kl_results))


def main():
    # np.random.seed(42)
    # n_rolls = 600
    # model_name = "gpt-4.1"
    # n_samples_per_turn = 5
    # config = {
    #     "temperature": 0.7,
    #     "top_p": 1.0,
    # }

    # if not os.path.exists("qualitative_tasks/rng_direct_samples.json"):
    #     direct_samples = roll_dice(n_rolls, model_name, config, verbalized=False)
    #     with open("qualitative_tasks/rng_direct_samples.json", "w") as f:
    #         json.dump(direct_samples.tolist(), f)
    # else:
    #     with open("qualitative_tasks/rng_direct_samples.json", "r") as f:
    #         direct_samples = np.array(json.load(f), dtype=int)

    # if not os.path.exists("qualitative_tasks/rng_vs_samples.json"):
    #     verbalized_samples = roll_dice(n_rolls, model_name, config, verbalized=True, n_samples_per_turn=n_samples_per_turn)
    #     with open("qualitative_tasks/rng_vs_samples.json", "w") as f:
    #         json.dump(verbalized_samples.tolist(), f)
    # else:
    #     with open("qualitative_tasks/rng_vs_samples.json", "r") as f:
    #         verbalized_samples = np.array(json.load(f), dtype=int)

    # direct_probs = compute_distribution(direct_samples)
    # verbalized_probs = compute_distribution(verbalized_samples)
    # uniform_probs = np.ones(6) / 6

    # print("Direct sampling distribution:", direct_probs)
    # print("Verbalized sampling distribution:", verbalized_probs)
    # print("Uniform distribution:", uniform_probs)

    # calculate_kl_divergence()
    model = "gemini-2.5-pro"
    direct_file = f"method_results_rng/{model}_rand_num/generation/direct (samples=1)/responses.jsonl"
    sequence_file = f"method_results_rng/{model}_rand_num/generation/sequence [strict] (samples=5)/responses.jsonl"
    vs_standard_file = f"method_results_rng/{model}_rand_num/generation/structure_with_prob [strict] (samples=5)/responses.jsonl"

    direct_samples = read_response_file(direct_file)
    sequence_samples = read_response_file(sequence_file)
    vs_standard_samples = read_response_file(vs_standard_file)
    n_rolls = 600

    # Plot distribution comparison
    plot_distribution_comparison(direct_samples, sequence_samples, vs_standard_samples, n_rolls)

if __name__ == "__main__":
    main()