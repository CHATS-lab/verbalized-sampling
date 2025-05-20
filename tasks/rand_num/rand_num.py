from llms.vllm_openai import VLLM_OpenAI_Agent
from llms.openrouter import OpenRouterAgent
from argparse import ArgumentParser
from verbalized_sampling.tasks.rand_num.prompt import QUESTION, FORMAT_PROMPT, SAMPLE_QUESTION
import concurrent.futures
import json
import math

parser = ArgumentParser()
parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-70B-Instruct")
parser.add_argument("--format", action="store_true")
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--top_p", type=float, default=0.9)
parser.add_argument("--use_vllm", action="store_true")

# Sampling related arguments
parser.add_argument("--is_sampling", action="store_true")
parser.add_argument("--num_responses", type=int, default=3)
parser.add_argument("--num_samples", type=int, default=1)

# Parallelism related arguments
parser.add_argument("--num_workers", type=int, default=128)

# Output related arguments
parser.add_argument("--output_file", type=str, default="responses.jsonl")

if __name__ == "__main__":
    args = parser.parse_args()

    if args.format:
        sim_type = "sampling"
    else:
        sim_type = "chat"

    if args.use_vllm:
        MODEL_CLASS = VLLM_OpenAI_Agent
    else:
        MODEL_CLASS = OpenRouterAgent

    config = {
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    model = MODEL_CLASS(
        model_name=args.model_name,
        sim_type=sim_type,
        config=config,
    )

    if args.is_sampling:
        prompt = SAMPLE_QUESTION.format(num_samples=args.num_samples)
    else:
        prompt = QUESTION
    if args.format:
        prompt += "\n\n" + FORMAT_PROMPT

    prompts = [[{"role": "user", "content": prompt}]] * args.num_responses

    from tqdm import tqdm

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(model.chat, prompt) for prompt in prompts]
        responses = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing prompts"):
            responses.append(future.result())

    if args.is_sampling and args.format:
        # Analyze the distribution of responses
        all_samples = []
        sample_counts = {}
        probabilities = {}
        
        for response in responses:
            if isinstance(response, list):
                for sample in response:
                    text = sample.get("text")
                    prob = sample.get("probability")
                    
                    if text:
                        all_samples.append(text)
                        sample_counts[text] = sample_counts.get(text, 0) + 1
                        
                        # Store the probability for each text
                        if text not in probabilities:
                            probabilities[text] = []
                        probabilities[text].append(prob)
            
        # Calculate statistics
        total_samples = len(all_samples)
        print(f"\nAnalysis of {total_samples} samples:")
        
        if total_samples > 0:
            # Sort by frequency
            sorted_samples = sorted(sample_counts.items(), key=lambda x: x[1], reverse=True)
            
            print("\nTop 10 most frequent responses:")
            for text, count in sorted_samples[:10]:
                frequency = count / total_samples
                avg_prob = sum(probabilities[text]) / len(probabilities[text])
                print(f"  {text}: {count} occurrences ({frequency:.2%}), avg probability: {avg_prob:.4f}")
            
            # Calculate entropy of the distribution
            entropy = -sum((count/total_samples) * math.log2(count/total_samples) for count in sample_counts.values())
            print(f"\nEntropy of distribution: {entropy:.4f} bits")
            
            # Check if probabilities match frequencies
            print("\nProbability vs Actual Frequency:")
            for text, probs in probabilities.items():
                avg_prob = sum(probs) / len(probs)
                actual_freq = sample_counts[text] / total_samples
                print(f"  {text}: stated prob={avg_prob:.4f}, actual freq={actual_freq:.4f}, diff={avg_prob-actual_freq:.4f}")

    with open(args.output_file, "w") as f:
        for response in responses:
            if isinstance(response, list) or isinstance(response, dict):
                f.write(json.dumps(response))
            else:
                f.write(response)
            f.write("\n")
