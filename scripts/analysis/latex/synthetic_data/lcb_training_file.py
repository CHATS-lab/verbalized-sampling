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

# Check for more prompts: https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/prompts/code_generation.py
SYSTEM_MESSAGE_GENERIC = (
    "You are an expert Python programmer. You will be given a question (problem specification) "
    "and will generate a correct Python program that matches the specification and passes all tests."
)
FORMAT_MESSAGE_GENERIC = (
    "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). "
    "Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."
)

def get_generic_question_template_answer(question: str):
    prompt = f"### Question:\n{question}\n\n"
    prompt += f"### Format: {FORMAT_MESSAGE_GENERIC}\n"
    prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"### Answer: (use the provided format with backticks)\n\n"
    return prompt

def get_oaireason_question_template_answer(question: str):
    prompt = f"### Question:\n{question}\n\n"
    prompt += f"### Format: Implement a function called `main()` which orchastrates the solution by reading inputs from stdin and writing the answer to stdout. Feel free to use additional functions as necessary. Next do NOT forget to call `main` function at the end of the program otherwise you will not be awarded any points.\n"
    prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"### Answer: (use the provided format with backticks)\n\n"
    return prompt

def get_qwen_reasoning_question_template_answer(question: str):
    prompt = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n"
    prompt += f"Question: {question}\n\n"
    prompt += f"{FORMAT_MESSAGE_GENERIC}\n"
    prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    # prompt += f"### Answer: (use the provided format with backticks)\n\n"
    return prompt

def generate_answer_parallel(model_name, question):
    system = SYSTEM_MESSAGE_GENERIC
    if "o3" in model_name:
        # print(f"Using o3 model for question: {question}")
        user = get_oaireason_question_template_answer(question)
    elif "qwen" in model_name:
        user = get_qwen_reasoning_question_template_answer(question)
    else:
        # print(f"Using generic model for question: {question}")
        user = get_generic_question_template_answer(question)

    messages = [
        # {"role": "system", "content": system}, 
        {"role": "user", "content": user}
    ]

    config = {
        "temperature": 0.7,
        "max_tokens": 10000
    }
    if "o3" in model_name:
        config = {
            # "temperature": 0.7,
            "reasoning_effort": "medium"
        }

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


# def prepare_train_test_dataset(lcb_dataset):
#     rng_train = np.random.RandomState(42)
#     train_indices = rng_train.choice(len(lcb_dataset["question_content"]), size=700, replace=False)
#     test_indices = [i for i in range(len(lcb_dataset["question_content"])) if i not in train_indices]

#     output_test_data = []
#     for idx in test_indices:
#         output_test_data.append({
#             "question": lcb_dataset["question_content"][idx],
#         })
#     # Ensure output directory exists
#     os.makedirs("synthetic_lcb", exist_ok=True)
#     with open("synthetic_lcb/lcb_test.json", "w", encoding="utf-8") as f:
#         json.dump(output_test_data, f, indent=4, ensure_ascii=False)

#     output_train_data = []
#     train_questions = [lcb_dataset["question_content"][idx] for idx in train_indices]
#     train_answers = generate_answers_batch("o3", train_questions, max_workers=16)
#     for (question, answer) in train_answers:
#         output_train_data.append({
#             "system": SYSTEM_MESSAGE_GENERIC,
#             "instruction": get_generic_question_template_answer(question),
#             "output": answer
#         })

#     with open("synthetic_lcb/lcb_training_positive_700.json", "w", encoding="utf-8") as f:
#         json.dump(output_train_data, f, indent=4, ensure_ascii=False)


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


def parse_synthetic_postive_data(raw_response):
    question = raw_response.split("Question:")[1].strip().split("Difficulty:")[0].strip()
    difficulty = raw_response.split("Difficulty:")[1].strip()
    return {
        "question": question,
        "difficulty": difficulty,
    }


def prepare_synthetic_positive_method_dataset(question_generate_model_name, answer_generate_model_name, max_workers=16):
    # folder_path = f"method_results_lcb_1000/{question_generate_model_name}_livecodebench/generation"

    raw_memthod_name_list = {
        "direct": "direct",
        # "direct_cot": "direct_cot",
        # "multi_turn": "multi_turn",
        "sequence": "sequence",
        "vs_standard": "vs_standard",
        "vs_cot": "vs_cot",
        # "vs_multi": "vs_multi"
    }

    os.makedirs("synthetic_lcb", exist_ok=True)
    for child_folder in tqdm(os.listdir(folder_path), desc="Processing synthetic positive data"):
        method_name = child_folder.split(" ")[0]
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
        
        with open(f"synthetic_lcb/lcb_training_synthetic_positive_{raw_memthod_name_list[method_name]}.json", "w", encoding="utf-8") as f:
            json.dump(train_synthetic_data, f, indent=4, ensure_ascii=False)
        train_synthetic_data = []
        # break 



def main():
    # Load the livecodebench dataset from Hugging Face
    global DATASET_CACHE_DIR

    lcb_codegen = load_dataset(
        "livecodebench/code_generation_lite", 
        version_tag="release_v5",
        # trust_remote_code=True,
        cache_dir=DATASET_CACHE_DIR
    )
    print(len(lcb_codegen["test"])) # 880 questions
    # print(lcb_codegen["test"][0].keys())
    
    # prepare_train_test_dataset(lcb_codegen["test"]) # 700, 180
    # prepare_synthetic_positive_method_dataset(question_generate_model_name="gpt-4.1", answer_generate_model_name="o3", max_workers=128)
    prepare_synthetic_positive_method_dataset(question_generate_model_name="gpt-4.1", answer_generate_model_name="Qwen/Qwen3-32B", max_workers=128)




if __name__ == "__main__":
    main()
