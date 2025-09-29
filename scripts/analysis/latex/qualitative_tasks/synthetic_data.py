import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from openai import OpenAI
from datasets import load_dataset
from enum import Enum
import pandas as pd
import os
import json
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Method(Enum):
    DIRECT = "direct"
    STRUCTURE_WITH_PROB = "vs_standard"

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

def get_direct_system_prompt():
    prompt = f"""
    Generate a response to the input prompt.
    """
    return prompt

def get_verbalized_system_prompt(num_samples):
    prompt = f"""
    Generate {num_samples} response to the input prompt.

Return the responses in JSON format with keys: "responses" (list of dicts with 'text' and 'probability'). Each dictionary must include:
- 'text': the response string only.
- 'probability': the estimated likelihood (from 0.0 to 1.0) of this response from the full answer distribution of the input prompt (not just among the {num_samples} sampled responses).

Give ONLY the JSON object, with no explanations or extra text.
    """
    return prompt

def get_user_prompt(example):
    prompt = f"""
    Provide examples of problems that might be grade school math word problems that require performing a sequence of elementary calculations using basic arithmetic operations.
A bright middle school student should be able to solve each problem. Problems require no concepts beyond the level of early Algebra. You must first specify the question then provide the brief reasoning and answer. Note how the numerical answer is provided after #### after each brief reasoning for a question. Provide your examples in the following format:

Question: [question]
Answer: [answer]

Here are some examples:
Example 1: {example[0]}
Example 2: {example[1]}
Example 3: {example[2]}

Now it's your turn. Your question should be different in content from the examples. Make sure to only provide only the question and answer. Start each response with the question.
    """
    return prompt

def get_gsm8k_test_examples(n=1, seed=42):
    ds = load_dataset("gsm8k", "main", split="train")
    np.random.seed(seed)
    idxs = np.random.choice(range(len(ds)), n, replace=False)
    # Convert numpy.int64 to int to avoid key type error
    return [ds[int(i)] for i in idxs]


def parsing_gsm8k_response(response: str):
    """
    Parses a GSM8K-style response string into a dictionary with 'question' and 'answer' fields.

    Example input:
    "Question: ...\nAnswer: ...\n#### 15"
    """
    # Ensure input is a string
    if not isinstance(response, str):
        response = str(response)

    question = None
    answer = None

    # Split into lines and strip whitespace
    lines = [line.strip() for line in response.split('\n') if line.strip()]
    for i, line in enumerate(lines):
        if line.startswith("Question:"):
            question = line[len("Question:"):].strip()
        elif line.startswith("Answer:"):
            # The answer may span multiple lines until a line starting with "####" or end of input
            answer_lines = [line[len("Answer:"):].strip()]
            for next_line in lines[i+1:]:
                if next_line.startswith("Question:") or next_line.startswith("Answer:"):
                    break
                answer_lines.append(next_line)
                if next_line.startswith("####"):
                    break
            answer = "\n".join(answer_lines).strip()
            break  # Only parse the first question/answer pair

    return {"question": question, "answer": answer}


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


def generate_responses_gsm8k(examples, method, num_responses=1, model_name="gpt-4.1", config={}, num_samples_per_turn=1):
    # Generate responses using OpenAI API directly
    responses = []
    
    if method == Method.DIRECT:
        system_prompt = get_direct_system_prompt()
    elif method == Method.VS_STANDARD:
        system_prompt = get_verbalized_system_prompt(num_samples_per_turn)
    user_prompt = get_user_prompt(examples)
    
    all_data = []
    if method == Method.DIRECT:
        for resp in tqdm(range(num_responses), desc="Generating direct responses"):
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                **config,
            )
            parsed_response = parsing_gsm8k_response(completion.choices[0].message.content)
            all_data.append(parsed_response)
    else:
        num_of_turns = num_responses // num_samples_per_turn
        for turn in tqdm(range(num_of_turns), desc="Generating verbalized responses"):
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    **config,
                    response_format=structured_response_list_with_prob_schema
            )
            response = completion.choices[0].message.content
            parsed_response = _parse_response_with_schema(response)
            for resp in parsed_response:
                gsm_parsed_response = parsing_gsm8k_response(resp['text'])
                all_data.append(gsm_parsed_response)
    return all_data

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding

def compute_pairwise_cosine_similarities(responses, model_name="text-embedding-3-small"):
    # Use OpenAI's text-embedding-3-small model
    embeddings = []
    for response in tqdm(responses, desc="Computing embeddings"):
        response_embedding = get_embedding(response['question'] + "\n" + response['answer'], model_name)
        embeddings.append(response_embedding)
    
    embeddings_array = np.array(embeddings)
    embeddings_normalized = normalize(embeddings_array, norm='l2', axis=1)
    similarity_matrix = cosine_similarity(embeddings_normalized)
    sims = []
    for i in range(len(similarity_matrix)):
        for j in range(i+1, len(similarity_matrix)):
            sims.append(similarity_matrix[i, j])
    return sims

def plot_similarity_histogram(sim_direct, sim_verbalized, bins=50, save_path=None):
    plt.figure(figsize=(8,5))
    plt.hist(sim_direct, bins=bins, alpha=0.6, color='lightpink', label='Direct Sampling', density=True)
    plt.hist(sim_verbalized, bins=bins, alpha=0.6, color='lightblue', label='Verbalized Sampling', density=True)
    plt.xlabel("Embedding Cosine Similarity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.ylim(bottom=0)
  
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def main():
    # 1. Get GSM8K test examples
    examples = get_gsm8k_test_examples(n=3)  # Start with 10 examples for testing
    print("Examples loaded:", len(examples))
    
    # 2. Generate responses for both methods using GPT-4.1
    model_name = "gpt-4.1"
    config = {
        "temperature": 0.7,
        "top_p": 1.0
    }
    num_samples = 50
    num_samples_per_turn = 10

    if not os.path.exists("qualitative_tasks/gsm8k_direct_responses.json"):
        print("Generating direct responses...")
        responses_direct = generate_responses_gsm8k(examples, Method.DIRECT, num_responses=num_samples, model_name=model_name, config=config)
        with open("qualitative_tasks/gsm8k_direct_responses.json", "w", encoding="utf-8") as f:
            json.dump(responses_direct, f, ensure_ascii=False, indent=2)
    else:
        with open("qualitative_tasks/gsm8k_direct_responses.json", "r", encoding="utf-8") as f:
            responses_direct = json.load(f)
    # print(responses_direct)
    
    if not os.path.exists("qualitative_tasks/gsm8k_verbalized_responses.json"):
        print("Generating verbalized responses...")
        responses_verbalized = generate_responses_gsm8k(examples, Method.VS_STANDARD, num_responses=num_samples, model_name=model_name, config=config, num_samples_per_turn=num_samples_per_turn)
        with open("qualitative_tasks/gsm8k_verbalized_responses.json", "w", encoding="utf-8") as f:
            json.dump(responses_verbalized, f, ensure_ascii=False, indent=2)
    else:
        with open("qualitative_tasks/gsm8k_verbalized_responses.json", "r", encoding="utf-8") as f:
            responses_verbalized = json.load(f)
    # print(responses_verbalized)
    
    # 3. Compute pairwise cosine similarities
    print("Computing similarities for direct responses...")
    sim_direct = compute_pairwise_cosine_similarities(responses_direct)
    # print(sim_direct)
    
    print("Computing similarities for verbalized responses...")
    sim_verbalized = compute_pairwise_cosine_similarities(responses_verbalized)
    
    # 4. Plot
    print("Creating similarity histogram...")
    plot_similarity_histogram(sim_direct, sim_verbalized, bins=50, save_path="qualitative_tasks/gsm8k_diversity_barplot.png")
    
    # 5. Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Direct sampling - Mean similarity: {np.mean(sim_direct):.4f}, Std: {np.std(sim_direct):.4f}")
    print(f"Verbalized sampling - Mean similarity: {np.mean(sim_verbalized):.4f}, Std: {np.std(sim_verbalized):.4f}")

if __name__ == "__main__":
    main()
