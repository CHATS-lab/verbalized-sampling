# use datasets version 2.20.0
import os
import json
import numpy as np
import pandas as pd
from datasets import load_dataset
import re
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from verbalized_sampling.llms import get_model
from verbalized_sampling.llms.vllm import VLLMOpenAI

def query_openai(model_name, messages, config):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        **config,
    )
    return response.choices[0].message.content

def query_vllm(model_name, messages, config):
    client = VLLMOpenAI(model_name=model_name, config=config)
    response = client._chat(messages)
    return response

# Check for DATASET_CACHE_DIR, set default if not present
DATASET_CACHE_DIR = os.environ.get("DATASET_CACHE_DIR", "./.cache/hf")

SYSTEM_MESSAGE_GENERIC = (
    "You are given a math competition question in the style of AMC 10, AMC 12, or AIME. "
    "Solve it and output both your reasoning process and the final answer.\n\n"
    "### Format Requirements:\n"
    "- Do not restate the question.\n"
    "- Provide the step-by-step solution in a field starting with “Reasoning:”.\n"
    "- Provide the final numerical result in a separate field starting with “Answer:”.\n\n"
    "### Constraints:\n"
    "- The reasoning should include clear intermediate steps and justifications.\n"
    "- The answer must be exact (no approximations unless explicitly required).\n\n"
    "### Output Style Example (do not copy directly):\n"
    "Reasoning: First, observe that the question reduces to solving a quadratic equation… [step-by-step reasoning continues].\n"
    "Answer: 42"
)

def get_generic_question_template_answer(question: str):
    prompt = f"Question: {question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    return prompt

def get_oaireason_question_template_answer(question: str):
    prompt = f"Question:\n{question}"
    return prompt


def generate_answer_parallel(model_name, question):
    # system = SYSTEM_MESSAGE_GENERIC
    user = get_generic_question_template_answer(question)

    messages = [
        # {"role": "system", "content": system}, 
        {"role": "user", "content": user}
    ]

    config = {
        "temperature": 0.7,
        "max_tokens": 12000
    }
    if "o3" in model_name:
        config = {
            "temperature": 0.7,
            "reasoning_effort": "high"
        }
    # model = get_model(model_name, method="direct", config=config, strict_json=False)

    max_regen = 3
    for attempt in range(max_regen):
        if model_name in ["gpt-4.1", "o3"]:
            response = query_openai(model_name, messages, config)
        else:
            response = query_vllm(model_name, messages, config)
        return response
    return None


def generate_answers_batch(model_name, questions, max_workers=16):
    """Generate answers for multiple questions in parallel using threads"""
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a separate client instance for each thread to avoid conflicts
        future_to_question = {
            executor.submit(generate_answer_parallel, model_name, question): question 
            for question in questions
        }
        
        for future in tqdm(as_completed(future_to_question), total=len(questions), desc="Generating answers"):
            question = future_to_question[future]
            try:
                answer = future.result()
                results.append((question, answer))
            except Exception as exc:
                print(f'Question {question} generated an exception: {exc}')
                results.append((question, None))
    
    return results

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
                responses = response.get("responses", [])
                if isinstance(responses, list):
                    for resp in responses:
                        try:
                            prompt_to_responses[prompt]["responses"].append(resp["text"])
                        except Exception as e:
                            print(f"Error decoding JSON in {file_path}: {e}")
                            continue
                elif isinstance(responses, dict):
                    prompt_to_responses[prompt]["responses"].append(responses["text"])
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in {file_path}: {e}")

    return prompt_to_responses             


def parse_synthetic_postive_data(raw_response):
    question = raw_response.split("Question:")[1].strip().split("Difficulty:")[0].strip()
    if "Difficulty:" not in raw_response:
        difficulty = "unknown"
    else:
        difficulty = raw_response.split("Difficulty:")[1].strip()
    return {
        "question": question,
        "difficulty": difficulty,
    }


def prepare_synthetic_positive_method_dataset(question_generate_model_name, answer_generate_model_name, max_workers=16):
    folder_path = f"method_results_amc_aime_1000/{question_generate_model_name}_amc_aime_math/generation"
    folder_path = "/root/verbalize-sampling/gemini-2.5-flash_amc_aime_math/generation"

    raw_memthod_name_list = {
        # "direct": "direct",
        "direct_cot": "direct_cot",
        # "multi_turn": "multi_turn",
        # "sequence": "sequence",
        # "vs_standard": "vs_standard",
        # "vs_cot": "vs_cot",
        # "vs_multi": "vs_multi"
    }

    os.makedirs("synthetic_amc_aime_gemini", exist_ok=True)
    for child_folder in tqdm(os.listdir(folder_path), desc="Processing synthetic positive data"):
        method_name = child_folder.split(" ")[0]
        print(f"child_folder: {child_folder}")
        if method_name not in raw_memthod_name_list.keys():
            continue
        file_path = os.path.join(folder_path, child_folder, "responses.jsonl")
        prompt_to_responses = read_response_file(file_path)
        
        train_synthetic_data = []
        
        # Collect all questions to process in parallel
        all_questions = []
        
        for prompt in tqdm(prompt_to_responses, desc=f"Preparing questions for method: {method_name}"):
            responses = prompt_to_responses[prompt]["responses"]
            for response in responses:
                parsed_data = parse_synthetic_postive_data(response)
                question = parsed_data['question']
                # print(f"Question: {question}")
                all_questions.append(question)
        
        # Process all questions in parallel
        print(f"Processing {len(all_questions)} questions in parallel with {max_workers} workers...")
        question_answer_pairs = generate_answers_batch(answer_generate_model_name, all_questions, max_workers=max_workers)
        
        # Build the final dataset
        for question, answer in tqdm(question_answer_pairs, desc=f"Building dataset for method: {method_name}"):
            if answer is not None:
                train_synthetic_data.append({
                    # "system": SYSTEM_MESSAGE_GENERIC,
                    "instruction": get_generic_question_template_answer(question),
                    "output": answer,
                })
        
        # with open(f"synthetic_amc_aime/amc_aime_training_synthetic_positive_{raw_memthod_name_list[method_name]}.json", "w", encoding="utf-8") as f:
        with open(f"synthetic_amc_aime_gemini/amc_aime_training_synthetic_positive_{raw_memthod_name_list[method_name]}.json", "w", encoding="utf-8") as f:
            json.dump(train_synthetic_data, f, indent=4, ensure_ascii=False)
        train_synthetic_data = []
        # break 



def main():
    # Load the livecodebench dataset from Hugging Face
    global DATASET_CACHE_DIR

    # prepare_synthetic_positive_method_dataset(question_generate_model_name="gpt-4.1", answer_generate_model_name="gpt-4.1", max_workers=128)
    prepare_synthetic_positive_method_dataset(question_generate_model_name="gpt-4.1", answer_generate_model_name="Qwen/Qwen3-32B", max_workers=128)



if __name__ == "__main__":
    main()
