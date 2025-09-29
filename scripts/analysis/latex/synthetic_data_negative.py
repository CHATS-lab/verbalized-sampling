import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from openai import OpenAI
from datasets import load_dataset
from enum import Enum
import pandas as pd
import os
import re
import json
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import matplotlib.gridspec as gridspec
from pathlib import Path

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# class Method(Enum):
#     DIRECT = "direct"
#     STRUCTURE_WITH_PROB = "vs_standard"
#     SEQUENCE = "sequence"

# structured_response_list_with_prob_schema = {
#     "type": "json_schema",  # Required for OpenRouter
#     "json_schema": {
#         "name": "structured_with_prob_schema",
#         "schema": {
#             "type": "object",
#             "properties": {
#                 "responses": {
#                     "type": "array",
#                     "description": "A list of dicts, each with a 'text' and 'probability' field, representing possible responses to the input prompt and corresponding probabilities of each response.",
#                     "items": {
#                         "type": "object",
#                         "properties": {
#                             "text": {
#                                 "type": "string",
#                                 "description": "The text of the response."
#                             },
#                             "probability": {
#                                 "type": "number",
#                                 "description": "The estimated probability from 0.0 to 1.0 of this response given the input prompt (relative to the full answer space)."
#                             }
#                             # "perplexity": {
#                             #     "type": "number",
#                             #     "description": "The perplexity score of this response given the input prompt (lower values indicate higher confidence)."
#                             # }
#                         },
#                         "required": ["text", "probability"],
#                         "additionalProperties": False
#                     }
#                 }
#             },
#             "required": ["responses"],
#             "additionalProperties": False
#         },
#         "strict": True
#     }
# }

# def get_direct_system_prompt():
#     prompt = f"""
#     Generate a convincing and reasonable but incorrect solution to the given math problem.
#     For each solution, include the reasoning process and the final numerical answer, with the answer placed after #### (e.g., ####123) at the end.

#     """
#     return prompt

# def get_sequence_system_prompt(num_samples):
#     prompt = f"""
#     Generate {num_samples} convincing and reasonable but incorrect solutions to the given math problem.
#     For each solution, include the complete reasoning process and the final numerical answer, with the answer placed after #### (e.g., ####123) at the end.

#     Return exactly {num_samples} responses as a Python list of strings, formatted as:
#     ["response1", "response2", "response3", ...]
#     Return only the list, with no explanations or extra text.
#     """
#     return prompt

# def get_verbalized_system_prompt(num_samples):
#     prompt = f"""
# Generate a set of convincing and reasonable but incorrect solutions to the given math problem.
# Randomly sample {num_samples} solutions from the full set of possible incorrect answers, ignoring their probabilities.
# For each solution, include the complete reasoning process and the final numerical answer, with the answer placed after #### (e.g., ####123) at the end.

# Return the responses in JSON format with keys: "responses" (list of dicts with 'text' and 'probability'). Each dictionary must include:
# - 'text': the response string only (no explanation or extra text).
# - 'probability': the estimated probability from 0.0 to 1.0 of this response given the input prompt (relative to the full answer space).

# Give ONLY the JSON object, with no explanations or extra text.
# """
# - 'Perplexity': the perplexity score of this response given the input prompt (lower values indicate higher confidence).
# - 'probability': the estimated probability from 0.0 to 1.0 of this response given the input prompt (relative to the full answer space).
#   return prompt

def get_user_prompt(example):
    prompt = f"""
    Here is a math problem: {example['question']}
    """
    prompt = prompt.strip()
    return prompt

def get_gsm8k_test_examples(n=1, seed=42):
    ds = load_dataset("gsm8k", "main", split="train")
    np.random.seed(seed)
    idxs = np.random.choice(range(len(ds)), n, replace=False)
    # Convert numpy.int64 to int to avoid key type error
    return [ds[int(i)] for i in idxs]


# def _parse_response_with_schema(response: str) -> List[Dict[str, Any]]:
#         """Parse the response based on the provided schema."""
#         try:
#             if isinstance(response, str):
#                 parsed = json.loads(response)
                
#                 # Handle double-escaped JSON strings (i.e., string inside a string)
#                 if isinstance(parsed, str):
#                     parsed = json.loads(parsed)
                
#                 # Handle different schema types
#                 if "responses" in parsed:
#                     # For schemas with a 'responses' field (SequenceResponse, StructuredResponseList, etc.)
#                     responses = parsed["responses"]
                    
#                     if isinstance(responses, list):
#                         result = []
#                         for resp in responses:
#                             if isinstance(resp, dict) and "text" in resp and "probability" in resp:
#                                 # ResponseWithProbability
#                                 result.append({
#                                     "response": resp["text"],
#                                     "probability": resp["probability"]
#                                 })
#                             elif isinstance(resp, dict) and "text" in resp:
#                                 # Response
#                                 result.append({
#                                     "response": resp["text"],
#                                     "probability": 1.0
#                                 })
#                             elif isinstance(resp, str):
#                                 # SequenceResponse (list of strings)
#                                 result.append({
#                                     "response": resp,
#                                     "probability": 1.0
#                                 })
#                         return result
#                 else:
#                     # For direct response schemas (Response)
#                     if "text" in parsed:
#                         return [{
#                             "response": parsed["text"],
#                             "probability": parsed.get("probability", 1.0)
#                         }]
#                     elif "response" in parsed:
#                         return [{
#                             "response": parsed["response"],
#                             "probability": parsed.get("probability", 1.0)
#                         }]
                    
#                 # Fallback: return the raw validated data
#                 return [{"response": str(parsed), "probability": 1.0}]
                
#         except Exception as e:
#             print(f"Error parsing response with schema: {e}")
#             # If parsing fails, return a single response with probability 1.0
#             return [{"response": response, "probability": 1.0}]


# def generate_single_response(messages, model_name, config, method, response_format=None):
#     """Generate a single response from the LLM."""
#     try:
#         if model_name == "o3":
#             config_copy = config.copy()
#             config_copy.pop('temperature', None)
#             config_copy.pop('top_p', None)
#             if 'max_tokens' in config_copy:
#                 config_copy.update({'max_completion_tokens': config_copy.pop('max_tokens')})
#         else:
#             config_copy = config.copy()
        
#         if response_format:
#             config_copy['response_format'] = response_format
            
#         completion = client.chat.completions.create(
#             model=model_name,
#             messages=messages,
#             **config_copy,
#         )
#         return completion.choices[0].message.content
#     except Exception as e:
#         print(f"Error generating response: {e}")
#         return None

# def generate_responses_gsm8k(examples, method, num_responses=1, model_name="gpt-4.1", config={}, num_samples_per_turn=1, max_workers=5):
#     # Generate responses using OpenAI API with parallel execution
#     responses = []
    
#     if method == Method.DIRECT:
#         system_prompt = get_direct_system_prompt()
#     elif method == Method.SEQUENCE:
#         system_prompt = get_sequence_system_prompt(num_samples_per_turn)
#     elif method == Method.VS_STANDARD:
#         system_prompt = get_verbalized_system_prompt(num_samples_per_turn)
#     user_prompts = [get_user_prompt(example) for example in examples]

#     all_data = []
    
#     if method == Method.DIRECT:
#         for user_prompt in user_prompts:
#             messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            
#             # Generate responses in parallel
#             with ThreadPoolExecutor(max_workers=max_workers) as executor:
#                 # Submit all tasks
#                 future_to_index = {
#                     executor.submit(generate_single_response, messages, model_name, config, method): i 
#                     for i in range(num_responses)
#                 }
                
#                 # Collect results as they complete
#                 responses = [None] * num_responses
#                 for future in tqdm(as_completed(future_to_index), total=num_responses, desc="Generating direct responses"):
#                     index = future_to_index[future]
#                     try:
#                         response = future.result()
#                         responses[index] = response
#                     except Exception as e:
#                         print(f"Error getting result for index {index}: {e}")
#                         responses[index] = None
                
#                 # Filter out None responses
#                 responses = [r for r in responses if r is not None]
            
#             all_data.append({"question": user_prompt, "responses": responses})
#     else:
#         num_of_turns = num_responses // num_samples_per_turn
#         for user_prompt in user_prompts:
#             messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            
#             # Generate responses in parallel
#             with ThreadPoolExecutor(max_workers=max_workers) as executor:
#                 # Submit all tasks
#                 future_to_index = {
#                     executor.submit(
#                         generate_single_response, 
#                         messages, 
#                         model_name, 
#                         config, 
#                         method, 
#                         structured_response_list_with_prob_schema
#                     ): i 
#                     for i in range(num_of_turns)
#                 }
                
#                 # Collect results as they complete
#                 responses = []
#                 for future in tqdm(as_completed(future_to_index), total=num_of_turns, desc="Generating sequence responses"):
#                     try:
#                         response = future.result()
#                         if response:
#                             parsed_responses = _parse_response_with_schema(response)
#                             # parsed_responses is a list of dicts with 'response' and 'probability'
#                             for resp in parsed_responses:
#                                 responses.append(resp["response"])
#                     except Exception as e:
#                         print(f"Error getting result: {e}")
            
#             all_data.append({"question": user_prompt, "responses": responses})
    
#     return all_data


def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding


def compute_pairwise_cosine_similarities(responses, model_name="text-embedding-3-small"):
    # Use OpenAI's text-embedding-3-small model
    embeddings = []
    for response in tqdm(responses, desc="Computing embeddings"):
        response_embedding = get_embedding(response, model_name)
        embeddings.append(response_embedding)
    
    embeddings_array = np.array(embeddings)
    embeddings_normalized = normalize(embeddings_array, norm='l2', axis=1)
    similarity_matrix = cosine_similarity(embeddings_normalized)
    sims = []
    for i in range(len(similarity_matrix)):
        for j in range(i+1, len(similarity_matrix)):
            sims.append(similarity_matrix[i, j])
    return sims


def plot_similarity_histogram(sim_direct, sim_sequence, sim_verbalized, bins=100, save_path=None):
    plt.figure(figsize=(8,5))
    # Ensure all inputs are 1D numpy arrays or lists.
    sim_direct = np.asarray(sim_direct).flatten()
    sim_sequence = np.asarray(sim_sequence).flatten()
    sim_verbalized = np.asarray(sim_verbalized).flatten()

    # Define bar and KDE colors
    bar_colors = ['lightpink', 'lightblue', 'lightgreen']
    kde_colors = ['deeppink', 'royalblue', 'forestgreen']
    labels = ['Direct Sampling', 'Sequence Sampling', 'Verbalized Sampling']

    # Plot histograms and keep the returned patches for legend
    n, bins_out, patches = plt.hist(
        [sim_direct, sim_sequence, sim_verbalized],
        bins=bins,
        alpha=0.7,
        color=bar_colors,
        label=labels,
        density=True,
        histtype='stepfilled',
        linewidth=1.5
    )

    # Overlay KDE for smoothness
    for data, color in zip([sim_direct, sim_sequence, sim_verbalized], kde_colors):
        try:
            sns.kdeplot(data, color=color, lw=2, alpha=0.7)
        except Exception:
            pass  # KDE may fail if data is too sparse

    plt.xlabel("Embedding Cosine Similarity")
    plt.ylabel("Density")
    # Set legend with correct color patches
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=bar_colors[i], edgecolor='k', label=labels[i], alpha=0.7) for i in range(3)]
    plt.legend(handles=legend_handles)
    plt.ylim(bottom=0)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    # plt.show()

def read_response_file(file_path):
    """
    Reads a response file and groups responses by their prompt.
    Returns a dictionary: {prompt: {"responses": [list of response texts]}}
    """
    import json
    prompt_to_responses = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                response = json.loads(line)
                prompt = response.get("prompt", None)
                if prompt is None:
                    continue
                if prompt not in prompt_to_responses:
                    prompt_to_responses[prompt] = {"responses": []}
                for resp in response.get("responses", []):
                    try:
                        prompt_to_responses[prompt]["responses"].append(resp["text"])
                    except Exception:
                        continue
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in {file_path}: {e}")

    return prompt_to_responses
    

def calculate_incorrect_answer_rate(responses, example):
    match = re.search(r"####\s*([$\d,.\-]+)", example['answer'])
    numeric_answer = match.group(1).strip() if match else None
    numeric_answer = numeric_answer.replace(",", "").replace("$", "")
    # print("Correct answer:", numeric_answer)
    
    incorrect_answer_rate = 0
    different_answer = []

    for response in responses:
        # Handle $7,500 and similar cases concisely
        # Match both plain numeric answers and those with trailing notes (e.g., "####600---Note: ...")
        match = re.search(r"####\s*([$\d,.\-]+)", response)
        response_answer = None
        if match:
            ans = match.group(1).replace(",", "").replace("$", "")
            try:
                response_answer = str(float(ans))
            except Exception:
                response_answer = None
        # print(response, response_answer)
        if response_answer is not None and response_answer.endswith('.'):
            response_answer = response_answer[:-1]
        if response_answer is not None and float(response_answer) != float(numeric_answer):
            incorrect_answer_rate += 1
            if response_answer not in different_answer:
                different_answer.append(response_answer)
    # print("Incorrect answer rate:", incorrect_answer_rate / len(responses))
    return incorrect_answer_rate / len(responses), len(different_answer) / len(responses)


def calculate_incorrect_answer_rate_for_all_prompts(prompt_to_responses, examples):
    incorrect_answer_rate_list = []
    different_answer_rate_list = []
    # print(prompt_to_responses.keys())

    for example in examples:
        prompt = "Here is a math problem: " + example['question']
        if prompt in prompt_to_responses.keys():
            response_texts = prompt_to_responses[prompt]["responses"]
            if len(response_texts) > 0:  # Only process if there are responses
                incorrect_answer_rate, different_answer_rate = calculate_incorrect_answer_rate(response_texts, example)
                incorrect_answer_rate_list.append(incorrect_answer_rate)
                different_answer_rate_list.append(different_answer_rate)
                # print(f"Incorrect answer rate: {incorrect_answer_rate}, Different answer rate: {different_answer_rate}")
    return incorrect_answer_rate_list, different_answer_rate_list


def extract_synthetic_data_negative_diversity(base_dir, task_name):
    # Define methods to look for
    # model_list = [
    #     "gpt-4.1-mini", "gpt-4.1", "gemini-2.5-flash", "gemini-2.5-pro",
    #     "Llama-3.1-70B-Instruct", "deepseek-r1", "o3", "claude-4-sonnet"
    # ]
    model_list = ["gpt-4.1"]
    method_list = ["direct", "direct_cot", "sequence", "multi_turn", "vs_standard", "vs_cot", "vs_multi"]
    method_to_label = {
        "direct": "Direct",
        "direct_cot": "Direct_CoT",
        "sequence": "Sequence",
        "multi_turn": "Multi_turn",
        "vs_standard": "VS_Standard",
        "vs_cot": "VS_CoT",
        "vs_multi": "VS_Multi"
    }
    metrics = ["avg_diversity", "std_diversity"]
    metrics_values = {}

    for model_name in model_list:
        evaluation_dir = Path(base_dir) / f"{model_name}_{task_name}" / "evaluation"
        if not evaluation_dir.exists():
            continue

        for method_dir in evaluation_dir.iterdir():
            if not method_dir.is_dir():
                continue

            # Extract and normalize method name
            method_name_raw = method_dir.name
            method_name = method_name_raw.split(' ')[0]
            if method_name not in method_list:
                continue

            results_file = method_dir / "diversity_results.json"
            try:
                with open(results_file, "r") as f:
                    data = json.load(f)
                overall_metrics = data.get("overall_metrics", {})

                # Map method name to label and normalize to lower case
                method_label = method_to_label[method_name].lower()

                # Initialize nested dictionary for model and method if not present
                if model_name not in metrics_values:
                    metrics_values[model_name] = {}
                if method_label not in metrics_values[model_name]:
                    metrics_values[model_name][method_label] = {metric: [] for metric in metrics}

                # Collect metric values from all prompts
                for metric in metrics:
                    if metric in overall_metrics:
                        metrics_values[model_name][method_label][metric].append(overall_metrics[metric])
            except Exception as e:
                print(f"Error reading {results_file}: {e}")

    # print(metrics_values)
    return metrics_values
    

def draw_diversity_quality_histogram(
    incorrect_answer_rate_list, 
    different_answer_rate_list, 
    semantic_diversity_dict=None, 
    cosine_similarity_dict=None, 
    save_path=None
):
    """
    Draws a 2x2 grid of subplots:
    (a) Bar chart of incorrect answer rate
    (b) Bar chart of incorrect answer coverage
    (c) Bar chart of semantic diversity
    (d) Cosine similarity violin/box plot for direct, sequence, vs-standard
    Each subplot is labeled (a), (b), (c), (d).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Methods and labels
    methods = ["direct", "sequence", "multi_turn", "vs_standard", "vs_cot", "vs_multi"]
    method_to_labels = {
        "direct": "Direct",
        # "direct_cot": "CoT",
        "sequence": "Sequence",
        "multi_turn": "Multi-turn",
        "vs_standard": "VS-Standard",
        "vs_cot": "VS-CoT",
        "vs_multi": "VS-Multi"
    }
    colors = {
        'direct': '#E8F4FD',      # Very light blue (baseline)
        'cot': '#B8E0F5',         # Light blue (baseline)
        'sequence': '#7CC7EA',    # Medium blue (baseline)
        'multi_turn': '#4A90E2',  # Distinct blue (baseline)
        'vs_standard': '#FFCCCB', # light red
        'vs_cot': '#FF9999',      # medium red
        'vs_multi': '#FF6B6B'     # distinct red
    }
    edge_colors = {
        'direct': '#4A90E2',
        'cot': '#4A90E2', 
        'sequence': '#4A90E2',
        'multi_turn': '#4A90E2',
        'vs_standard': '#FF6B6B',
        'vs_cot': '#FF6B6B',
        'vs_multi': '#FF6B6B'
    }

    # Prepare data for each metric
    def get_means(metric_dict):
        return [np.mean(metric_dict.get(m, [0])) for m in methods]

    incorrect_means = get_means(incorrect_answer_rate_list)
    coverage_means = get_means(different_answer_rate_list)
    if semantic_diversity_dict is not None:
        semantic_means = get_means(semantic_diversity_dict)
    else:
        semantic_means = [0] * len(methods)

    # For cosine similarity, only plot for direct, sequence, vs_standard
    cos_methods = ["direct", "sequence", "vs_standard"]
    cos_labels = [method_to_labels[m] for m in cos_methods]
    if cosine_similarity_dict is not None:
        cos_data = [cosine_similarity_dict.get(m, []) for m in cos_methods]
    else:
        cos_data = [[0], [0], [0]]

    # Plotting
    plt.style.use('default')  # Start with clean slate
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'font.size': 24,
        'axes.labelsize': 28,
        'axes.titlesize': 32,
        'xtick.labelsize': 28,
        'ytick.labelsize': 28,
        'legend.fontsize': 28,
        'axes.linewidth': 0.8,
        'axes.edgecolor': '#666666',
    })
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    plt.subplots_adjust(wspace=0.3, hspace=0.2)

    # Place the legend at the upper center of the whole figure, not just the subfigure
    handles = [plt.Rectangle((0,0),1,1, facecolor=colors[m], edgecolor=edge_colors[m]) for m in methods]
    labels = [method_to_labels[m] for m in methods]
    # Remove fontweight, which is not a valid argument for fig.legend
    fig.legend(
        handles, labels,
        loc='upper center', bbox_to_anchor=(0.5, 1.05),
        fontsize=28, frameon=False, ncol=len(methods)
    )

    # (a) Incorrect Answer Rate
    ax = axes[0, 0]
    bars = ax.bar(
        range(len(methods)), incorrect_means, 
        color=[colors[m] for m in methods], 
        edgecolor=[edge_colors[m] for m in methods], 
        alpha=0.9
    )
    ax.set_ylabel('Rate', fontweight='bold')
    ax.set_title('Incorrect Answer Rate ($\\uparrow$)', fontweight='bold', pad=20)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([method_to_labels[m] for m in methods], rotation=30, ha='right')
    ax.tick_params(axis='y', labelsize=24)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.2f}', 
                ha='center', va='bottom', fontweight='bold')
    ax.text(-0.1, 1.18, "a", transform=ax.transAxes, fontsize=44, fontweight='bold', va='top')

    # (b) Incorrect Answer Coverage
    ax = axes[0, 1]
    bars = ax.bar(
        range(len(methods)), coverage_means, 
        color=[colors[m] for m in methods], 
        edgecolor=[edge_colors[m] for m in methods], 
        alpha=0.9
    )
    ax.set_ylabel('Coverage', fontweight='bold')
    ax.set_ylim(0, 0.8)
    ax.set_title('Incorrect Answer Coverage ($\\uparrow$)', fontweight='bold', pad=20)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([method_to_labels[m] for m in methods], rotation=30, ha='right')
    ax.tick_params(axis='y', labelsize=24)
    ax.set_ylim(0, 0.8)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.2f}', 
                ha='center', va='bottom', fontweight='bold')
    ax.text(-0.1, 1.18, "b", transform=ax.transAxes, fontsize=44, fontweight='bold', va='top')


    # (c) Semantic Diversity (mean ± std from semantic_diversity_dict)
    semantic_diversity_dict = semantic_diversity_dict['gpt-4.1']
    semantic_means = [semantic_diversity_dict[m]['avg_diversity'][0] for m in methods]
    semantic_stds = [semantic_diversity_dict[m]['std_diversity'][0] for m in methods]
    ax = axes[1, 0]
    bars = ax.bar(
        range(len(methods)), semantic_means,
        yerr=semantic_stds,
        color=[colors[m] for m in methods],
        edgecolor=[edge_colors[m] for m in methods],
        alpha=0.9,
        capsize=8
    )
    ax.set_ylabel('Semantic Diversity', fontweight='bold')
    ax.set_title('Semantic Diversity ($\\uparrow$)', fontweight='bold', pad=20)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([method_to_labels[m] for m in methods], rotation=30, ha='right')
    ax.tick_params(axis='y', labelsize=24)
    ax.set_ylim(0, 0.2)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    for bar, mean, std in zip(bars, semantic_means, semantic_stds):
        ax.text(
            bar.get_x() + bar.get_width()/2., 
            mean + std + 0.001, 
            # f'{mean:.2f}±{std:.2f}',
            f'{mean:.2f}',
            ha='center', va='bottom', fontweight='bold'
        )
    ax.text(-0.1, 1.18, "c", transform=ax.transAxes, fontsize=44, fontweight='bold', va='top')

    # (d) Cosine Similarity (histogram + KDE, compact)
    ax = axes[1, 1]
    idxs = [cos_methods.index(m) for m in ['direct', 'sequence', 'vs_standard']]
    data = [cos_data[i] for i in idxs]
    labels = [cos_labels[i] for i in idxs]
    bar_colors = ['#D5D1D1', '#F7A6AC', '#7FBDDA']
    kde_colors = ['gray', 'deeppink', 'royalblue']
    import seaborn as sns
    ax.hist(data, bins=50, alpha=0.5, color=bar_colors, label=labels, density=True, histtype='stepfilled', linewidth=2)
    for d, c in zip(data, kde_colors):
        try: sns.kdeplot(d, color=c, lw=2, ax=ax)
        except: pass
    # ax.set(xlabel='Cosine Similarity', ylabel='Density')
    plt.xlabel("Cosine Similarity", fontweight='bold')
    plt.ylabel("Density", fontweight='bold')
    ax.set_title('Cosine Similarity (Pairwise) ($\\downarrow$)', fontweight='bold', pad=20)
    ax.tick_params(axis='y', labelsize=24)
    ax.tick_params(axis='x', labelsize=24)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xlim(0.35, 1)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend(frameon=False, reverse=True)
    ax.text(-0.1, 1.18, "d", transform=ax.transAxes, fontsize=44, fontweight='bold', va='top')

    # all the subplots should have the same ylim
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    # plt.show()


def main():
    # 1. Get GSM8K test examples
    examples = get_gsm8k_test_examples(n=100)  # Start with 10 examples for testing
    print("Examples loaded:", len(examples))
    # all_models = ["gpt-4.1-mini", "gpt-4.1", "gemini-2.5-flash", "gemini-2.5-pro", "claude-4-sonnet", "deepseek-r1", "o3"]
    all_models = ["gpt-4.1"]
    folder_name = "method_synthetic_negative"

    for model_name in all_models:
        direct_file = f"{folder_name}/{model_name}_synthetic_negative/generation/direct (samples=1)/responses.jsonl"
        direct_cot_file = f"{folder_name}/{model_name}_synthetic_negative/generation/direct_cot [strict] (samples=1)/responses.jsonl"
        multi_turn_file = f"{folder_name}/{model_name}_synthetic_negative/generation/multi_turn (samples=5)/responses.jsonl"
        sequence_file = f"{folder_name}/{model_name}_synthetic_negative/generation/sequence [strict] (samples=5)/responses.jsonl"
        vs_standard_file = f"{folder_name}/{model_name}_synthetic_negative/generation/vs_standard [strict] (samples=5)/responses.jsonl"
        vs_cot_file = f"{folder_name}/{model_name}_synthetic_negative/generation/vs_cot [strict] (samples=5)/responses.jsonl"
        vs_multi_turn_file = f"{folder_name}/{model_name}_synthetic_negative/generation/vs_multi [strict] (samples=5)/responses.jsonl"
        
        direct_responses = read_response_file(direct_file)
        direct_cot_responses = read_response_file(direct_cot_file)
        multi_turn_responses = read_response_file(multi_turn_file)
        sequence_responses = read_response_file(sequence_file)
        vs_standard_responses = read_response_file(vs_standard_file)
        vs_cot_responses = read_response_file(vs_cot_file)
        vs_multi_turn_responses = read_response_file(vs_multi_turn_file)

        # 2. Calculate incorrect answer rate
        direct_incorrect_answer_rate, direct_different_answer_rate = calculate_incorrect_answer_rate_for_all_prompts(direct_responses, examples)
        direct_cot_incorrect_answer_rate, direct_cot_different_answer_rate = calculate_incorrect_answer_rate_for_all_prompts(direct_cot_responses, examples)
        sequence_incorrect_answer_rate, sequence_different_answer_rate = calculate_incorrect_answer_rate_for_all_prompts(sequence_responses, examples)
        multi_turn_incorrect_answer_rate, multi_turn_different_answer_rate = calculate_incorrect_answer_rate_for_all_prompts(multi_turn_responses, examples)
        vs_incorrect_answer_rate, vs_different_answer_rate = calculate_incorrect_answer_rate_for_all_prompts(vs_standard_responses, examples)
        vs_cot_incorrect_answer_rate, vs_cot_different_answer_rate = calculate_incorrect_answer_rate_for_all_prompts(vs_cot_responses, examples)
        vs_multi_turn_incorrect_answer_rate, vs_multi_turn_different_answer_rate = calculate_incorrect_answer_rate_for_all_prompts(vs_multi_turn_responses, examples)
        # print(f"Direct incorrect answer rate: {np.mean(direct_incorrect_answer_rate)}, Different answer rate: {np.mean(direct_different_answer_rate)}")
        # # print(f"Direct COT incorrect answer rate: {np.mean(direct_cot_incorrect_answer_rate)}, Different answer rate: {np.mean(direct_cot_different_answer_rate)}")
        # print(f"Sequence incorrect answer rate: {np.mean(sequence_incorrect_answer_rate)}, Different answer rate: {np.mean(sequence_different_answer_rate)}")
        # print(f"Multi-turn incorrect answer rate: {np.mean(multi_turn_incorrect_answer_rate)}, Different answer rate: {np.mean(multi_turn_different_answer_rate)}")
        # print(f"Verbalized incorrect answer rate: {np.mean(vs_incorrect_answer_rate)}, Different answer rate: {np.mean(vs_different_answer_rate)}")
        # print(f"Verbalized COT incorrect answer rate: {np.mean(vs_cot_incorrect_answer_rate)}, Different answer rate: {np.mean(vs_cot_different_answer_rate)}")
        # print(f"Verbalized Multi-turn incorrect answer rate: {np.mean(vs_multi_turn_incorrect_answer_rate)}, Different answer rate: {np.mean(vs_multi_turn_different_answer_rate)}")

        incorrect_answer_rate_list = {
            'direct': direct_incorrect_answer_rate,
            'direct_cot': direct_cot_incorrect_answer_rate,
            'sequence': sequence_incorrect_answer_rate,
            'multi_turn': multi_turn_incorrect_answer_rate,
            'vs_standard': vs_incorrect_answer_rate,
            'vs_cot': vs_cot_incorrect_answer_rate,
            'vs_multi': vs_multi_turn_incorrect_answer_rate
        }
        different_answer_rate_list = {
            'direct': direct_different_answer_rate,
            'direct_cot': direct_cot_different_answer_rate,
            'sequence': sequence_different_answer_rate,
            'multi_turn': multi_turn_different_answer_rate,
            'vs_standard': vs_different_answer_rate,
            'vs_cot': vs_cot_different_answer_rate,
            'vs_multi': vs_multi_turn_different_answer_rate
        }
        # draw_quality_histogram(
        #     incorrect_answer_rate_list,
        #     different_answer_rate_list,
        #     save_path=f"qualitative_tasks/synthetic_data_negative_{model_name}_quality_histogram.pdf"
        # )

        # # 3. Compute pairwise cosine similarities
        # sim_direct = []
        # sim_sequence = []
        # sim_verbalized = []
        # for prompt, responses in direct_responses.items():
        #     sim_direct.extend(compute_pairwise_cosine_similarities(responses["responses"]))
        # for prompt, responses in sequence_responses.items():
        #     sim_sequence.extend(compute_pairwise_cosine_similarities(responses["responses"]))
        # for prompt, responses in vs_standard_responses.items():
        #         sim_verbalized.extend(compute_pairwise_cosine_similarities(responses["responses"]))
        # # Save the similarity results to disk for later analysis
        # similarity_results = {
        #     "sim_direct": sim_direct,
        #     "sim_sequence": sim_sequence,
        #     "sim_verbalized": sim_verbalized
        # }
        # with open(f"qualitative_tasks/synthetic_data_negative_{model_name}_similarity_results.json", "w", encoding="utf-8") as f:
        #     json.dump(similarity_results, f, ensure_ascii=False, indent=2)

        cosine_similarity_dict = {}
        with open(f"latex/qualitative_tasks/synthetic_data_negative_{model_name}_similarity_results.json", "r", encoding="utf-8") as f:
            similarity_results = json.load(f)
        cosine_similarity_dict.update({
            "direct": similarity_results["sim_direct"],
            "sequence": similarity_results["sim_sequence"],
            "vs_standard": similarity_results["sim_verbalized"]
        })

        semantic_diversity_dict = extract_synthetic_data_negative_diversity("method_synthetic_negative", "synthetic_negative")

        # plot the diversity and quality
        draw_diversity_quality_histogram(
            incorrect_answer_rate_list,
            different_answer_rate_list,
            semantic_diversity_dict,
            cosine_similarity_dict,
            save_path=f"latex/qualitative_tasks/synthetic_data_negative_{model_name}_diversity_quality_histogram.pdf"
        )
        
        # # 4. Plot
        # print("Creating similarity histogram...")
        # plot_similarity_histogram(sim_direct, sim_sequence, sim_verbalized, bins=50, save_path=f"qualitative_tasks/synthetic_data_negative_{model_name}_diversity_barplot.pdf")

        
        

# def main():
    # 1. Get GSM8K test examples
    # examples = get_gsm8k_test_examples(n=100)  # Start with 10 examples for testing
    # print("Examples loaded:", len(examples))

    # # 2. Generate responses for both methods using GPT-4.1
    # model_name = "gpt-4.1"
    # config = {
    #     "temperature": 0.7,
    #     "top_p": 1.0
    # }
    # num_samples = 10
    # num_samples_per_turn = 10
    # max_workers = 5  # Adjust based on your API rate limits

    # if not os.path.exists("qualitative_tasks/gsm8k_negative_direct_responses.json"):
    #     print("Generating direct responses...")
    #     start_time = time.time()
    #     responses_direct = generate_responses_gsm8k(
    #         examples, Method.DIRECT, num_responses=num_samples, 
    #         model_name=model_name, config=config, max_workers=max_workers
    #     )
    #     end_time = time.time()
    #     print(f"Direct responses generated in {end_time - start_time:.2f} seconds")
    #     with open("qualitative_tasks/gsm8k_negative_direct_responses.json", "w", encoding="utf-8") as f:
    #         json.dump(responses_direct, f, ensure_ascii=False, indent=2)
    # else:
    #     with open("qualitative_tasks/gsm8k_negative_direct_responses.json", "r", encoding="utf-8") as f:
    #         responses_direct = json.load(f)

    # if not os.path.exists("qualitative_tasks/gsm8k_negative_sequence_responses.json"):
    #     print("Generating sequence responses...")
    #     start_time = time.time()
    #     responses_sequence = generate_responses_gsm8k(
    #         examples, Method.SEQUENCE, num_responses=num_samples, 
    #         model_name=model_name, config=config, num_samples_per_turn=num_samples_per_turn, 
    #         max_workers=max_workers
    #     )
    #     end_time = time.time()
    #     print(f"Sequence responses generated in {end_time - start_time:.2f} seconds")
    #     with open("qualitative_tasks/gsm8k_negative_sequence_responses.json", "w", encoding="utf-8") as f:
    #         json.dump(responses_sequence, f, ensure_ascii=False, indent=2)
    # else:
    #     with open("qualitative_tasks/gsm8k_negative_sequence_responses.json", "r", encoding="utf-8") as f:
    #         responses_sequence = json.load(f)
    
    # if not os.path.exists("qualitative_tasks/gsm8k_negative_vs_responses.json"):
    #     print("Generating verbalized responses...")
    #     start_time = time.time()
    #     responses_verbalized = generate_responses_gsm8k(
    #         examples, Method.VS_STANDARD, num_responses=num_samples, 
    #         model_name=model_name, config=config, num_samples_per_turn=num_samples_per_turn, 
    #         max_workers=max_workers
    #     )
    #     end_time = time.time()
    #     print(f"Verbalized responses generated in {end_time - start_time:.2f} seconds")
    #     with open("qualitative_tasks/gsm8k_negative_vs_responses.json", "w", encoding="utf-8") as f:
    #         json.dump(responses_verbalized, f, ensure_ascii=False, indent=2)
    # else:
    #     with open("qualitative_tasks/gsm8k_negative_vs_responses.json", "r", encoding="utf-8") as f:
    #         responses_verbalized = json.load(f)

    # direct_incorrect_answer_rate = []
    # sequence_incorrect_answer_rate = []
    # vs_incorrect_answer_rate = []
    # direct_different_answer_rate = []
    # sequence_different_answer_rate = []
    # vs_different_answer_rate = []
    # for idx, example in enumerate(examples):
    #     direct_incorrect_answer_rate.append(calculate_incorrect_answer_rate(responses_direct[idx]['responses'], example)[0])
    #     sequence_incorrect_answer_rate.append(calculate_incorrect_answer_rate(responses_sequence[idx]['responses'], example)[0])
    #     vs_incorrect_answer_rate.append(calculate_incorrect_answer_rate(responses_verbalized[idx]['responses'], example)[0])
    #     direct_different_answer_rate.append(calculate_incorrect_answer_rate(responses_direct[idx]['responses'], example)[1])
    #     sequence_different_answer_rate.append(calculate_incorrect_answer_rate(responses_sequence[idx]['responses'], example)[1])
    #     vs_different_answer_rate.append(calculate_incorrect_answer_rate(responses_verbalized[idx]['responses'], example)[1])
    # print(f"Direct incorrect answer rate: {np.mean(direct_incorrect_answer_rate)}")
    # print(f"Sequence incorrect answer rate: {np.mean(sequence_incorrect_answer_rate)}")
    # print(f"Verbalized incorrect answer rate: {np.mean(vs_incorrect_answer_rate)}")
    
    # # 3. Compute pairwise cosine similarities
    # sim_direct = []
    # sim_sequence = []
    # sim_verbalized = []
    # for idx, example in enumerate(examples):
    #     sim_direct.append(compute_pairwise_cosine_similarities(responses_direct[idx]['responses']))
    #     sim_sequence.append(compute_pairwise_cosine_similarities(responses_sequence[idx]['responses']))
    #     sim_verbalized.append(compute_pairwise_cosine_similarities(responses_verbalized[idx]['responses']))
    
    # # 4. Plot
    # print("Creating similarity histogram...")
    # plot_similarity_histogram(sim_direct, sim_sequence, sim_verbalized, bins=50, save_path="qualitative_tasks/gsm8k_negative_diversity_barplot.png")
    
    # # 5. Print summary statistics
    # # Save summary statistics and incorrect answer rates to a file
    # summary_stats = {
    #     "Direct": {
    #         "mean_similarity": float(np.mean(sim_direct)),
    #         "std_similarity": float(np.std(sim_direct)),
    #         "mean_incorrect_answer_rate": float(np.mean(direct_incorrect_answer_rate)),
    #         "std_incorrect_answer_rate": float(np.std(direct_incorrect_answer_rate)),
    #         "mean_different_answer_rate": float(np.mean(direct_different_answer_rate)),
    #         "std_different_answer_rate": float(np.std(direct_different_answer_rate))
    #     },
    #     "Sequence": {
    #         "mean_similarity": float(np.mean(sim_sequence)),
    #         "std_similarity": float(np.std(sim_sequence)),
    #         "mean_incorrect_answer_rate": float(np.mean(sequence_incorrect_answer_rate)),
    #         "std_incorrect_answer_rate": float(np.std(sequence_incorrect_answer_rate)),
    #         "mean_different_answer_rate": float(np.mean(sequence_different_answer_rate)),
    #         "std_different_answer_rate": float(np.std(sequence_different_answer_rate))
    #     },
    #     "VS-Standard": {
    #         "mean_similarity": float(np.mean(sim_verbalized)),
    #         "std_similarity": float(np.std(sim_verbalized)),
    #         "mean_incorrect_answer_rate": float(np.mean(vs_incorrect_answer_rate)),
    #         "std_incorrect_answer_rate": float(np.std(vs_incorrect_answer_rate)),
    #         "mean_different_answer_rate": float(np.mean(vs_different_answer_rate)),
    #         "std_different_answer_rate": float(np.std(vs_different_answer_rate))
    #     }
    # }
    # with open("qualitative_tasks/gsm8k_negative_summary.json", "w", encoding="utf-8") as f:
    #     json.dump(summary_stats, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
