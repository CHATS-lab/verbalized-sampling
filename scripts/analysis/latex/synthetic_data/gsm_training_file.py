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

# Check for DATASET_CACHE_DIR, set default if not present
DATASET_CACHE_DIR = os.environ.get("DATASET_CACHE_DIR", "./.cache/hf")
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def generate_answer_parallel(question, client_instance=None):
    """Thread-safe version of generate_answer for parallel processing"""
    if client_instance is None:
        client_instance = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    model = "gpt-4.1"
    system = (
        "You are an AI assistant who is an expert at solving math word problems. "
        "Think step by step then provide the numerical answer at the end after the delimiter '####', like '#### 24'."
    )
    user = f"Question: {question}\n"
    user += "Think step by step then provide the numerical answer at the end after the delimiter '####', like '#### 24'."
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    response = client_instance.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7
    )
    return response.choices[0].message.content


def generate_answers_batch(questions, max_workers=16):
    """Generate answers for multiple questions in parallel using threads"""
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a separate client instance for each thread to avoid conflicts
        future_to_question = {
            executor.submit(generate_answer_parallel, question): question 
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


def prepare_positive_dataset(dataset_train, dataset_test, label=False):
    # Sample 1000 random indices from training set and 500 from test set
    # System prompt
    system_prompt = (
        "You are an AI assistant who is an expert at solving math word problems. "
        "Think step by step then provide the numerical answer at the end after the delimiter '####', like '#### 24'."
    )
    # print(system_prompt)

    rng_train = np.random.RandomState(42)
    train_indices = rng_train.choice(len(dataset_train["question"]), size=1000, replace=False)

    train_df = pd.DataFrame({
        "question": [dataset_train["question"][i] for i in train_indices],
        "answer": [dataset_train["answer"][i] for i in train_indices]
    })

    # Prepare output data
    output_train_data = []
    for idx, row in train_df.iterrows():
        question = f"Question: {row['question']}\n"
        question += "Think step by step then provide the numerical answer at the end after the delimiter '####', like '#### 24'."

        if label:
            output_train_data.append({
                    "system": system_prompt,
                    "instruction": question,
                    "output": row["answer"],
                    "label": 1
                })
        else:
            output_train_data.append({
                "system": system_prompt,
                "instruction": question,
                "output": row["answer"]
            })
    # Ensure output directory exists
    os.makedirs("synthetic_data", exist_ok=True)
    if label:
        with open("synthetic_data/gsm8k_training_positive_1k_label.json", "w", encoding="utf-8") as f:
            json.dump(output_train_data, f, indent=4, ensure_ascii=False)
    else:
        with open("synthetic_data/gsm8k_training_positive_1k.json", "w", encoding="utf-8") as f:
            json.dump(output_train_data, f, indent=4, ensure_ascii=False)


def prepare_test_dataset(dataset_train, dataset_test):
    # Sample 1000 random indices from training set and 500 from test set
    rng_test = np.random.RandomState(40)
    test_indices = rng_test.choice(len(dataset_test["question"]), size=500, replace=False)

    test_df = pd.DataFrame({
        "question": [dataset_train["question"][i] for i in test_indices],
        "answer": [dataset_train["answer"][i] for i in test_indices]
    })

    output_test_data = []
    for idx, row in test_df.iterrows():
        output_test_data.append({
            "question": row["question"],
            "answer": row["answer"]
        })
    os.makedirs("synthetic_data", exist_ok=True)
    with open("synthetic_data/gsm8k_test.json", "w", encoding="utf-8") as f:
        json.dump(output_test_data, f, indent=4, ensure_ascii=False)


def find_corresponding_correct_answer(dataset_train, question):
    for idx, q in enumerate(dataset_train["question"]):
        if q.strip() == question.strip():
            return dataset_train["answer"][idx]
    return None


def parse_answer(response):
    match = re.search(r"####\s*([$\d,.\-]+)", response)

    response_answer = None
    if match:
        ans = match.group(1).replace(",", "").replace("$", "")
        try:
            response_answer = float(ans)
        except Exception:
            response_answer = None

    return response_answer


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


def prepare_negative_dataset(dataset_train):
    file_path = "method_synthetic_negative/gpt-4.1_synthetic_negative/generation/vs_cot [strict] (samples=5)/responses.jsonl"
    prompt_to_responses = read_response_file(file_path)

    train_negative_file = []

    # System prompt
    system_prompt = (
        "You are an AI assistant who is an expert at solving math word problems. "
        "Think step by step then provide the numerical answer at the end after the delimiter '####', like '#### 24'."
    )
    
    for prompt in prompt_to_responses:
        question = prompt.replace("Here is a math problem: ", "")

        answer = find_corresponding_correct_answer(dataset_train, question)
        correct_numerical_answer = parse_answer(answer)

        for response in prompt_to_responses[prompt]["responses"]:
            response_numerical_answer = parse_answer(response)
            if response_numerical_answer != None and correct_numerical_answer != None:
                if response_numerical_answer == correct_numerical_answer:
                    # print(f"CORRECT: {response_numerical_answer} == {correct_numerical_answer}")
                    question = f"Question: {question}\n"
                    question += "Think step by step then provide the numerical answer at the end after the delimiter '####', like '#### 24'."

                    train_negative_file.append({
                        "system": system_prompt,
                        "instruction": question,
                        "output": response,
                        "label": 1
                    })
                else:
                    # print(f"INCORRECT: {response_numerical_answer} != {correct_numerical_answer}")
                    train_negative_file.append({
                        "system": system_prompt,
                        "instruction": question,
                        "output": response,
                        "label": 0
                    })

    os.makedirs("synthetic_data", exist_ok=True)
    with open("synthetic_data/gsm8k_training_negative.json", "w", encoding="utf-8") as f:
        json.dump(train_negative_file, f, indent=4, ensure_ascii=False)               


def parse_synthetic_postive_data(raw_response):
    question = raw_response.split("Question:")[1].strip()
    return {
        "question": question,
    }


def prepare_synthetic_positive_method_dataset(dataset_train, max_workers=5):
    folder_path = "method_results_gsm8k_1000/gpt-4.1_gsm8k/generation"

    # System prompt
    system_prompt = (
        "You are an AI assistant who is an expert at solving math word problems. "
        "Think step by step then provide the numerical answer at the end after the delimiter '####', like '#### 24'."
    )

    raw_memthod_name_list = {
        # "direct": "direct",
        "direct_cot": "direct_cot",
        # "multi_turn": "multi_turn",
        # "sequence": "sequence",
        # "vs_standard": "vs_standard",
        "vs_cot": "vs_cot",
        # "vs_multi": "vs_multi"
    }

    os.makedirs("synthetic_data", exist_ok=True)
    for child_folder in tqdm(os.listdir(folder_path), desc="Processing synthetic positive data"):
        method_name = child_folder.split(" ")[0]
        if method_name not in raw_memthod_name_list.keys():
            continue
        file_path = os.path.join(folder_path, child_folder, "responses.jsonl")
        prompt_to_responses = read_response_file(file_path)
        
        train_synthetic_data = []
        
        # Collect all questions to process in parallel
        all_questions = []
        question_to_response_data = {}
        
        for prompt in tqdm(prompt_to_responses, desc=f"Preparing questions for method: {method_name}"):
            responses = prompt_to_responses[prompt]["responses"]
            for response in responses:
                parsed_data = parse_synthetic_postive_data(response)
                question = f"Question: {parsed_data['question']}\n"
                question += "Think step by step then provide the numerical answer at the end after the delimiter '####', like '#### 24'."
                
                all_questions.append(question)
                question_to_response_data[question] = {
                    "system": system_prompt,
                    "instruction": question,
                    "response": response
                }
        
        # Process all questions in parallel
        print(f"Processing {len(all_questions)} questions in parallel with {max_workers} workers...")
        question_answer_pairs = generate_answers_batch(all_questions, max_workers=max_workers)
        
        # Build the final dataset
        for question, answer in tqdm(question_answer_pairs, desc=f"Building dataset for method: {method_name}"):
            if answer is not None:
                response_data = question_to_response_data[question]
                train_synthetic_data.append({
                    "system": response_data["system"],
                    "instruction": response_data["instruction"],
                    "output": answer,
                })
        
        with open(f"synthetic_data/gsm8k_training_synthetic_positive_{raw_memthod_name_list[method_name]}.json", "w", encoding="utf-8") as f:
            json.dump(train_synthetic_data, f, indent=4, ensure_ascii=False)
        train_synthetic_data = []
        # break 


def prepare_synthetic_negative_kto_dataset(dataset_train, dataset_test):
    file_path = "method_synthetic_negative/gpt-4.1_synthetic_negative/generation/vs_cot [strict] (samples=5)/responses.jsonl"
    prompt_to_responses = read_response_file(file_path)

    train_negative_file = []

    # System prompt
    system_prompt = (
        "You are an AI assistant who is an expert at solving math word problems. "
        "Think step by step then provide the numerical answer at the end after the delimiter '####', like '#### 24'."
    )
    
    for prompt in prompt_to_responses:
        question = prompt.replace("Here is a math problem: ", "")

        answer = find_corresponding_correct_answer(dataset_train, question)
        correct_numerical_answer = parse_answer(answer)

        for response in prompt_to_responses[prompt]["responses"]:
            response_numerical_answer = parse_answer(response)
            if response_numerical_answer != None and correct_numerical_answer != None:
                if response_numerical_answer == correct_numerical_answer:
                    # print(f"CORRECT: {response_numerical_answer} == {correct_numerical_answer}")
                    question = f"Question: {question}\n"
                    question += "Think step by step then provide the numerical answer at the end after the delimiter '####', like '#### 24'."

                    train_negative_file.append({
                        "instruction": question,
                        "output": response,
                        "kto_tag": True
                    })
                else:
                    # print(f"INCORRECT: {response_numerical_answer} != {correct_numerical_answer}")
                    train_negative_file.append({
                        "instruction": question,
                        "output": response,
                        "kto_tag": False
                    })

    rng_train = np.random.RandomState(42)
    train_indices = rng_train.choice(len(dataset_train["question"]), size=1000, replace=False)

    train_df = pd.DataFrame({
        "question": [dataset_train["question"][i] for i in train_indices],
        "answer": [dataset_train["answer"][i] for i in train_indices]
    })

    for idx, row in train_df.iterrows():
        question = f"Question: {row['question']}\n"
        question += "Think step by step then provide the numerical answer at the end after the delimiter '####', like '#### 24'."

        train_negative_file.append({
            "instruction": question,
            "output": row["answer"],
            "kto_tag": True
        })

    os.makedirs("synthetic_data", exist_ok=True)
    with open("synthetic_data/gsm8k_training_negative_kto.json", "w", encoding="utf-8") as f:
        json.dump(train_negative_file, f, indent=4, ensure_ascii=False)


def main():
    # Load datasets
    dataset_test = load_dataset(
        path="openai/gsm8k",
        name="main",
        split="test",
        cache_dir=DATASET_CACHE_DIR,
    )
    dataset_train = load_dataset(
        path="openai/gsm8k",
        name="main",
        split="train",
        cache_dir=DATASET_CACHE_DIR,
    )
    
    # prepare_positive_dataset(dataset_train, dataset_test, label=False)
    # prepare_positive_dataset(dataset_train, dataset_test, label=True)
    # prepare_test_dataset(dataset_train, dataset_test)
    # prepare_negative_dataset(dataset_train)
    prepare_synthetic_positive_method_dataset(dataset_train, max_workers=5)
    # prepare_synthetic_negative_kto_dataset(dataset_train, dataset_test)


    

if __name__ == "__main__":
    main()
