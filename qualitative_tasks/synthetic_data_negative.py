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


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Method(Enum):
    DIRECT = "direct"
    STRUCTURE_WITH_PROB = "structure_with_prob"
    SEQUENCE = "sequence"

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
                                "description": "The estimated probability from 0.0 to 1.0 of this response given the input prompt (relative to the full answer space)."
                            }
                            # "perplexity": {
                            #     "type": "number",
                            #     "description": "The perplexity score of this response given the input prompt (lower values indicate higher confidence)."
                            # }
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

def get_direct_system_prompt():
    prompt = f"""
    Generate a convincing and reasonable but incorrect solution to the given math problem.
    For each solution, include the reasoning process and the final numerical answer, with the answer placed after #### (e.g., ####123).
    """
    return prompt

def get_sequence_system_prompt(num_samples):
    prompt = f"""
    Generate {num_samples} convincing and reasonable but incorrect solutions to the given math problem.
    For each solution, include the complete reasoning process and the final numerical answer, with the answer placed after #### (e.g., ####123).

    Return exactly {num_samples} responses as a Python list of strings, formatted as:
    ["response1", "response2", "response3", ...]
    Return only the list, with no explanations or extra text.
    """
    return prompt

def get_verbalized_system_prompt(num_samples):
    prompt = f"""
Generate a set of convincing and reasonable but incorrect solutions to the given math problem.
Randomly sample {num_samples} solutions from the full set of possible incorrect answers, ignoring their probabilities.
For each solution, include the complete reasoning process and the final numerical answer, with the answer placed after #### (e.g., ####123).

Return the responses in JSON format with keys: "responses" (list of dicts with 'text' and 'probability'). Each dictionary must include:
- 'text': the response string only (no explanation or extra text).
- 'probability': the estimated probability from 0.0 to 1.0 of this response given the input prompt (relative to the full answer space).

Give ONLY the JSON object, with no explanations or extra text.
"""
# - 'Perplexity': the perplexity score of this response given the input prompt (lower values indicate higher confidence).
# - 'probability': the estimated probability from 0.0 to 1.0 of this response given the input prompt (relative to the full answer space).
    return prompt

def get_user_prompt(example):
    prompt = f"""
    Here is a math problem:
    {example['question']}
    """
    return prompt

def get_gsm8k_test_examples(n=1, seed=42):
    ds = load_dataset("gsm8k", "main", split="train")
    np.random.seed(seed)
    idxs = np.random.choice(range(len(ds)), n, replace=False)
    # Convert numpy.int64 to int to avoid key type error
    return [ds[int(i)] for i in idxs]


def _parse_response_with_schema(response: str) -> List[Dict[str, Any]]:
        """Parse the response based on the provided schema."""
        try:
            if isinstance(response, str):
                parsed = json.loads(response)
                
                # Handle double-escaped JSON strings (i.e., string inside a string)
                if isinstance(parsed, str):
                    parsed = json.loads(parsed)
                
                # Handle different schema types
                if "responses" in parsed:
                    # For schemas with a 'responses' field (SequenceResponse, StructuredResponseList, etc.)
                    responses = parsed["responses"]
                    
                    if isinstance(responses, list):
                        result = []
                        for resp in responses:
                            if isinstance(resp, dict) and "text" in resp and "probability" in resp:
                                # ResponseWithProbability
                                result.append({
                                    "response": resp["text"],
                                    "probability": resp["probability"]
                                })
                            elif isinstance(resp, dict) and "text" in resp:
                                # Response
                                result.append({
                                    "response": resp["text"],
                                    "probability": 1.0
                                })
                            elif isinstance(resp, str):
                                # SequenceResponse (list of strings)
                                result.append({
                                    "response": resp,
                                    "probability": 1.0
                                })
                        return result
                else:
                    # For direct response schemas (Response)
                    if "text" in parsed:
                        return [{
                            "response": parsed["text"],
                            "probability": parsed.get("probability", 1.0)
                        }]
                    elif "response" in parsed:
                        return [{
                            "response": parsed["response"],
                            "probability": parsed.get("probability", 1.0)
                        }]
                    
                # Fallback: return the raw validated data
                return [{"response": str(parsed), "probability": 1.0}]
                
        except Exception as e:
            print(f"Error parsing response with schema: {e}")
            # If parsing fails, return a single response with probability 1.0
            return [{"response": response, "probability": 1.0}]


def generate_responses_gsm8k(examples, method, num_responses=1, model_name="gpt-4.1", config={}, num_samples_per_turn=1):
    # Generate responses using OpenAI API directly
    responses = []
    
    if method == Method.DIRECT:
        system_prompt = get_direct_system_prompt()
    elif method == Method.SEQUENCE:
        system_prompt = get_sequence_system_prompt(num_samples_per_turn)
    elif method == Method.STRUCTURE_WITH_PROB:
        system_prompt = get_verbalized_system_prompt(num_samples_per_turn)
    user_prompts = [get_user_prompt(example) for example in examples]

    if model_name == "o3":
        config.pop('temperature', None)
        config.pop('top_p', None)
        if 'max_tokens' in config:
            config.update({'max_completion_tokens': config.pop('max_tokens')})
    
    all_data = []
    if method == Method.DIRECT:
        for user_prompt in user_prompts:
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            responses = []
            for resp in tqdm(range(num_responses), desc="Generating direct responses"):
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    **config,
                )
                response = completion.choices[0].message.content
                responses.append(response)
            all_data.append({"question": user_prompt, "responses": responses})
    else:
        num_of_turns = num_responses // num_samples_per_turn
        for user_prompt in user_prompts:
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            responses = []
            for turn in tqdm(range(num_of_turns), desc="Generating sequence responses"):
                completion = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        **config,
                        response_format=structured_response_list_with_prob_schema
                )
                response = completion.choices[0].message.content
                parsed_responses = _parse_response_with_schema(response)
                # parsed_responses is a list of dicts with 'response' and 'probability'
                for resp in parsed_responses:
                    responses.append(resp["response"])
            all_data.append({"question": user_prompt, "responses": responses})
    return all_data


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


def calculate_incorrect_answer_rate(responses, example):
    match = re.search(r"####\s*([\-]?\d[\d\.,]*)", example['answer'])
    numeric_answer = match.group(1).strip() if match else None
    # print("Correct answer:", numeric_answer)

    incorrect_answer_rate = 0
    different_answer = []
    for response in responses:
        response_answer = re.search(r"####\s*([\-]?\d[\d\.,]*)", response)
        response_answer = response_answer.group(1).strip() if response_answer else None
        if response_answer is not None and response_answer.endswith('.'):
            response_answer = response_answer[:-1]
        # print("Response answer:", response_answer)
        if float(response_answer) != float(numeric_answer):
            incorrect_answer_rate += 1
            if response_answer not in different_answer:
                different_answer.append(response_answer)
    # print("Incorrect answer rate:", incorrect_answer_rate / len(responses))
    return incorrect_answer_rate / len(responses), len(different_answer) / len(responses)


def main():
    # 1. Get GSM8K test examples
    examples = get_gsm8k_test_examples(n=5)  # Start with 10 examples for testing
    print("Examples loaded:", len(examples))
    
    # 2. Generate responses for both methods using GPT-4.1
    model_name = "gpt-4.1"
    config = {
        "temperature": 0.7,
        "top_p": 1.0
    }
    num_samples = 10
    num_samples_per_turn = 10

    if not os.path.exists("qualitative_tasks/gsm8k_negative_direct_responses.json"):
        print("Generating direct responses...")
        responses_direct = generate_responses_gsm8k(examples, Method.DIRECT, num_responses=num_samples, model_name=model_name, config=config)
        with open("qualitative_tasks/gsm8k_negative_direct_responses.json", "w", encoding="utf-8") as f:
            json.dump(responses_direct, f, ensure_ascii=False, indent=2)
    else:
        with open("qualitative_tasks/gsm8k_negative_direct_responses.json", "r", encoding="utf-8") as f:
            responses_direct = json.load(f)
    # print(responses_direct)

    if not os.path.exists("qualitative_tasks/gsm8k_negative_sequence_responses.json"):
        print("Generating sequence responses...")
        responses_sequence = generate_responses_gsm8k(examples, Method.SEQUENCE, num_responses=num_samples, model_name=model_name, config=config, num_samples_per_turn=num_samples_per_turn)
        with open("qualitative_tasks/gsm8k_negative_sequence_responses.json", "w", encoding="utf-8") as f:
            json.dump(responses_sequence, f, ensure_ascii=False, indent=2)
    else:
        with open("qualitative_tasks/gsm8k_negative_sequence_responses.json", "r", encoding="utf-8") as f:
            responses_sequence = json.load(f)
    
    if not os.path.exists("qualitative_tasks/gsm8k_negative_vs_responses.json"):
        print("Generating verbalized responses...")
        responses_verbalized = generate_responses_gsm8k(examples, Method.STRUCTURE_WITH_PROB, num_responses=num_samples, model_name=model_name, config=config, num_samples_per_turn=num_samples_per_turn)
        with open("qualitative_tasks/gsm8k_negative_vs_responses.json", "w", encoding="utf-8") as f:
            json.dump(responses_verbalized, f, ensure_ascii=False, indent=2)
    else:
        with open("qualitative_tasks/gsm8k_negative_vs_responses.json", "r", encoding="utf-8") as f:
            responses_verbalized = json.load(f)
    # print(responses_verbalized)

    direct_incorrect_answer_rate = []
    sequence_incorrect_answer_rate = []
    vs_incorrect_answer_rate = []
    direct_different_answer_rate = []
    sequence_different_answer_rate = []
    vs_different_answer_rate = []
    for idx, example in enumerate(examples):
        direct_incorrect_answer_rate.append(calculate_incorrect_answer_rate(responses_direct[idx]['responses'], example)[0])
        sequence_incorrect_answer_rate.append(calculate_incorrect_answer_rate(responses_sequence[idx]['responses'], example)[0])
        vs_incorrect_answer_rate.append(calculate_incorrect_answer_rate(responses_verbalized[idx]['responses'], example)[0])
        direct_different_answer_rate.append(calculate_incorrect_answer_rate(responses_direct[idx]['responses'], example)[1])
        sequence_different_answer_rate.append(calculate_incorrect_answer_rate(responses_sequence[idx]['responses'], example)[1])
        vs_different_answer_rate.append(calculate_incorrect_answer_rate(responses_verbalized[idx]['responses'], example)[1])
    print(f"Direct incorrect answer rate: {np.mean(direct_incorrect_answer_rate)}")
    print(f"Sequence incorrect answer rate: {np.mean(sequence_incorrect_answer_rate)}")
    print(f"Verbalized incorrect answer rate: {np.mean(vs_incorrect_answer_rate)}")
    
    # 3. Compute pairwise cosine similarities
    sim_direct = []
    sim_sequence = []
    sim_verbalized = []
    for idx, example in enumerate(examples):
        sim_direct.append(compute_pairwise_cosine_similarities(responses_direct[idx]['responses']))
        sim_sequence.append(compute_pairwise_cosine_similarities(responses_sequence[idx]['responses']))
        sim_verbalized.append(compute_pairwise_cosine_similarities(responses_verbalized[idx]['responses']))
    
    # 4. Plot
    print("Creating similarity histogram...")
    plot_similarity_histogram(sim_direct, sim_sequence, sim_verbalized, bins=50, save_path="qualitative_tasks/gsm8k_negative_diversity_barplot.png")
    
    # 5. Print summary statistics
    # Save summary statistics and incorrect answer rates to a file
    summary_stats = {
        "Direct": {
            "mean_similarity": float(np.mean(sim_direct)),
            "std_similarity": float(np.std(sim_direct)),
            "mean_incorrect_answer_rate": float(np.mean(direct_incorrect_answer_rate)),
            "std_incorrect_answer_rate": float(np.std(direct_incorrect_answer_rate)),
            "mean_different_answer_rate": float(np.mean(direct_different_answer_rate)),
            "std_different_answer_rate": float(np.std(direct_different_answer_rate))
        },
        "Sequence": {
            "mean_similarity": float(np.mean(sim_sequence)),
            "std_similarity": float(np.std(sim_sequence)),
            "mean_incorrect_answer_rate": float(np.mean(sequence_incorrect_answer_rate)),
            "std_incorrect_answer_rate": float(np.std(sequence_incorrect_answer_rate)),
            "mean_different_answer_rate": float(np.mean(sequence_different_answer_rate)),
            "std_different_answer_rate": float(np.std(sequence_different_answer_rate))
        },
        "VS-Standard": {
            "mean_similarity": float(np.mean(sim_verbalized)),
            "std_similarity": float(np.std(sim_verbalized)),
            "mean_incorrect_answer_rate": float(np.mean(vs_incorrect_answer_rate)),
            "std_incorrect_answer_rate": float(np.std(vs_incorrect_answer_rate)),
            "mean_different_answer_rate": float(np.mean(vs_different_answer_rate)),
            "std_different_answer_rate": float(np.std(vs_different_answer_rate))
        }
    }
    with open("qualitative_tasks/gsm8k_negative_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
