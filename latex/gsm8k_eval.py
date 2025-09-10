# gsm8k_benchmark.py
"""Benchmark experiment against a dataset of GSM8K problems."""

import time
import os
import threading
import string
import pandas as pd
import requests
import json
from typing import NamedTuple

from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse
import openai

def strip_numbers_only(s: str) -> str:
    return "".join(ch for ch in s if ch in string.digits)

def check_vllm_server(base_url: str = "http://localhost:8000") -> bool:
    """Check if vLLM server is running and accessible."""
    try:
        endpoint = {
            "chat/completions": "v1/chat/completions",
            "models": "v1/models",
            "completions": "v1/completions",
        }
        for endpoint, url in endpoint.items():
            response = requests.get(f"{base_url}/{url}", timeout=10)
            if response.status_code == 200:
                return True
        return False
    except requests.exceptions.RequestException:
        return False

class SimpleVLLMClient:
    """Simple client for vLLM using OpenAI format"""
    
    def __init__(self, base_url: str, model_name: str, temperature: float = 0.7, top_p: float = 1.0):
        self.client = openai.OpenAI(
            api_key="EMPTY",
            base_url=f"{base_url}/v1",
        )
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
    
    def chat(self, messages: list) -> str:
        """Send chat completion request"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=512,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Chat completion failed: {e}")

def process_problem(args_tuple) -> tuple[bool, str]:
    """Process a single GSM8K problem"""
    row, client, problem_idx = args_tuple
    
    question = f"Question: {row['question']}\nThink step by step then provide the numerical answer at the end after the delimiter '####', like '#### 24'."
    
    messages = [
        {"role": "system", "content": "You are an AI assistant who is an expert at solving math word problems. Think step by step then provide the numerical answer at the end after the delimiter '####', like '#### 24'."},
        {"role": "user", "content": question}
    ]

    try:
        response = client.chat(messages)
        
        if "####" not in response:
            return False, f"No delimiter found: {response[:100]}..."
            
        extracted = response.split("####")[1].strip()
        gt_answer = row["answer"].split("####")[1].strip()
        
        extracted_num = strip_numbers_only(extracted)
        gt_num = strip_numbers_only(gt_answer)
        
        if extracted_num and gt_num:
            correct = float(extracted_num) == float(gt_num)
            status = "✓" if correct else f"✗ got {extracted.strip()}, expected {gt_answer}"
            return correct, status
        else:
            return False, f"Parse error: '{extracted}' vs '{gt_answer}'"
            
    except Exception as e:
        return False, f"Error: {e}"

def main():
    parser = argparse.ArgumentParser(description="GSM8K evaluation with vLLM")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct", help="Model name")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of problems to evaluate")
    parser.add_argument("--seed", type=int, default=40, help="Random seed")
    parser.add_argument("--vllm_base_url", default="http://localhost:8000", help="vLLM server base URL")
    parser.add_argument("--num_threads", type=int, default=4, help="Number of concurrent threads")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling")
    
    args = parser.parse_args()
    start_time = time.time()
    
    print(f"🚀 GSM8K Evaluation: {args.model} | {args.num_samples} samples")
    print(f"Server: {args.vllm_base_url}")
    
    # Check if server is running
    print("🔍 Checking vLLM server...")
    if not check_vllm_server(args.vllm_base_url):
        print("❌ Error: vLLM server is not running!")
        print("\n📋 To start the server, run this command in another terminal:")
        print(f"python -m vllm.entrypoints.openai.api_server \\")
        print(f"    --model {args.model} \\")
        print(f"    --port 8000 \\")
        print(f"    --host 0.0.0.0")
        return 1
    
    print("✅ Server is running!")
    
    # Create client
    client = SimpleVLLMClient(
        base_url=args.vllm_base_url,
        model_name=args.model,
        temperature=args.temperature,
        top_p=args.top_p
    )
    
    # Test client connection
    try:
        test_response = client.chat([{"role": "user", "content": "Hello"}])
        print(f"✅ Client connection successful!")
    except Exception as e:
        print(f"❌ Client connection failed: {e}")
        return 1

    # Load and sample dataset
    print("📚 Loading GSM8K dataset...")
    try:
        dataset = load_dataset("openai/gsm8k", name="main", split="test")
        df = pd.DataFrame({"question": dataset["question"], "answer": dataset["answer"]})
        sample_df = df.sample(min(args.num_samples, len(df)), random_state=args.seed)
        print(f"📊 Loaded {len(sample_df)} problems")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return 1
    
    # Process problems
    task_args = [(row, client, idx) for idx, (_, row) in enumerate(sample_df.iterrows())]
    results = []
    
    print(f"🔄 Processing {len(task_args)} problems with {args.num_threads} threads...")
    
    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        with tqdm(total=len(task_args), desc="Progress") as pbar:
            futures = [executor.submit(process_problem, arg) for arg in task_args]
            
            for future in as_completed(futures):
                try:
                    correct, status = future.result()
                    results.append(correct)
                    
                    # Show errors
                    if not correct and ("Error" in status or "Parse error" in status):
                        tqdm.write(f"❌ {status}")
                except Exception as e:
                    results.append(False)
                    tqdm.write(f"❌ Unexpected error: {e}")
                
                pbar.update(1)

    # Results
    accuracy = sum(results) / len(results) if results else 0
    total_time = time.time() - start_time
    correct_count = sum(results)
    
    print(f"\n{'='*60}")
    print(f"📊 RESULTS")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Samples: {len(results)}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy:.4f} ({correct_count}/{len(results)})")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Avg Time: {total_time/len(results):.2f}s per sample")
    print(f"{'='*60}")
    
    # Save results
    results_file = f"gsm8k_results_{args.model.replace('/', '_')}_{args.num_samples}samples.txt"
    with open(results_file, 'w') as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Samples: {len(results)}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Time: {total_time:.2f}s\n")
    
    print(f"💾 Results saved to {results_file}")
    
    return 0 if accuracy > 0 else 1

if __name__ == "__main__":
    exit(main())