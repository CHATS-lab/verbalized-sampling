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


def plot_distribution_comparison(direct_probs, verbalized_probs, uniform_probs, n_rolls=10):
    """
    Plot the distribution comparison for dice rolls 1-6 using seaborn
    """
    import seaborn as sns
    
    # Set seaborn style
    sns.set_style("white")
    sns.set_palette("husl")
    
    # Convert probabilities to counts
    direct_counts = direct_probs * n_rolls
    verbalized_counts = verbalized_probs * n_rolls
    uniform_counts = uniform_probs * n_rolls
    
    # Prepare data for seaborn
    dice_values = [1, 2, 3, 4, 5, 6]
    data = []
    for i, dice in enumerate(dice_values):
        data.append({'Dice': dice, 'Count': direct_counts[i], 'Method': 'Direct Sampling'})
        data.append({'Dice': dice, 'Count': verbalized_counts[i], 'Method': 'Verbalized Sampling'})
    
    import pandas as pd
    df = pd.DataFrame(data)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create overlapping bars using matplotlib directly for more control
    x_positions = np.arange(len(dice_values))
    width = 1.0  # Full width bars with no gaps
    
    # Plot bars with overlap
    bars1 = ax.bar(x_positions, direct_counts, width, label='Direct Sampling', 
                   color='#FC8EAC', alpha=0.8, edgecolor='#FC8EAC', linewidth=1)
    bars2 = ax.bar(x_positions, verbalized_counts, width, label='Verbalized Sampling', 
                   color='#A4C8E1', alpha=0.8, edgecolor='#A4C8E1', linewidth=1)
    
    # Set x-axis ticks to be continuous with no gaps
    ax.set_xticks(x_positions)
    ax.set_xticklabels(dice_values)
    ax.set_xlim(-0.5, len(dice_values) - 0.5)  # Remove gaps at edges
    
    # Add horizontal reference line for uniform distribution
    uniform_value = uniform_counts[0]
    ax.axhline(y=uniform_value, color='red', linestyle='--', linewidth=2, 
               label=f'Uniform Distribution ({n_rolls//6})', alpha=0.8)
    # Show the value at y-axis
    ax.text(len(uniform_counts)-0.2, uniform_value + 0.2, f'{uniform_value:.1f}', 
            color='red', fontsize=11, fontweight='bold', va='bottom', ha='right', alpha=0.9)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{int(height)}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    
    # Customize the plot
    ax.set_xlabel('Dice Roll Value', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='upper right')
    ax.set_ylim(0, max(max(direct_counts), max(verbalized_counts)) * 1.15)
    
    # Add statistics box
    kl_direct = entropy(direct_probs, uniform_probs)
    kl_verbalized = entropy(verbalized_probs, uniform_probs)
    stats_text = f'KL Divergence from Uniform:\nDirect: {kl_direct:.4f}\nVerbalized: {kl_verbalized:.4f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig("qualitative_tasks/rng_distribution_comparison.png", bbox_inches='tight')
    # plt.show()

def main():
    np.random.seed(42)
    n_rolls = 600
    model_name = "gpt-4.1"
    n_samples_per_turn = 5
    config = {
        "temperature": 0.7,
        "top_p": 1.0,
    }

    if not os.path.exists("qualitative_tasks/direct_samples.json"):
        direct_samples = roll_dice(n_rolls, model_name, config, verbalized=False)
        with open("qualitative_tasks/direct_samples.json", "w") as f:
            json.dump(direct_samples.tolist(), f)
    else:
        with open("qualitative_tasks/direct_samples.json", "r") as f:
            direct_samples = np.array(json.load(f), dtype=int)

    if not os.path.exists("qualitative_tasks/verbalized_samples.json"):
        verbalized_samples = roll_dice(n_rolls, model_name, config, verbalized=True, n_samples_per_turn=n_samples_per_turn)
        with open("qualitative_tasks/verbalized_samples.json", "w") as f:
            json.dump(verbalized_samples.tolist(), f)
    else:
        with open("qualitative_tasks/verbalized_samples.json", "r") as f:
            verbalized_samples = np.array(json.load(f), dtype=int)

    direct_probs = compute_distribution(direct_samples)
    verbalized_probs = compute_distribution(verbalized_samples)
    uniform_probs = np.ones(6) / 6

    print("Direct sampling distribution:", direct_probs)
    print("Verbalized sampling distribution:", verbalized_probs)
    print("Uniform distribution:", uniform_probs)

    # Plot distribution comparison
    plot_distribution_comparison(direct_probs, verbalized_probs, uniform_probs, n_rolls)

if __name__ == "__main__":
    main()