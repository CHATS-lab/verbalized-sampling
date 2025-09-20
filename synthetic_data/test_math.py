#!/usr/bin/env python3
"""
Test script for evaluating Qwen models on math tasks.

Tests Qwen/Qwen3-4B-Base and Qwen/Qwen3-4B-Thinking-2507 performance
across different math datasets (MATH, AIME, AMC, etc.).
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, List
import argparse
from tqdm import tqdm

# Add verbalized_sampling to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from verbalized_sampling.tasks import get_task
from verbalized_sampling.tasks.math_eval import evaluate_math_answer, extract_boxed_answer
from verbalized_sampling.llms import get_model
from verbalized_sampling.methods import Method


class MathTestRunner:
    """Test runner for math model evaluation."""

    def __init__(self, models: List[str], datasets: List[str], num_samples: int = 50, random_seed: int = 42, base_url: str = "http://localhost:8000"):
        self.models = models
        self.datasets = datasets
        self.num_samples = num_samples
        self.random_seed = random_seed
        self.base_url = base_url
        self.results = {}

    def create_llm(self, model_name: str):
        """Create LLM instance for the given model using vLLM."""
        try:
            # Configuration for vLLM
            config = {
                "base_url": f"{self.base_url}/v1",
                "temperature": 0.1,  # Low temperature for math problems
                "max_tokens": 2048,
                "top_p": 0.9,
            }

            # Create LLM using vLLM
            llm = get_model(
                model_name=model_name,
                method=Method.DIRECT,  # Using direct method for math problems
                config=config,
                use_vllm=True,
                num_workers=1,
                strict_json=False
            )
            return llm
        except Exception as e:
            print(f"Warning: Could not create LLM for {model_name}: {e}")
            print(f"Make sure vLLM server is running at {self.base_url} with model {model_name}")
            return None

    def test_single_problem(self, model_name: str, llm, problem: Dict, dataset: str) -> Dict:
        """Test a single math problem."""
        try:
            # Create prompt
            instruction = self._get_instruction_for_dataset(dataset)
            prompt_text = f"{instruction}\n\nProblem: {problem['problem']}"

            # Get model response
            messages = [{"role": "user", "content": prompt_text}]

            start_time = time.time()
            # Use the _chat method from BaseLLM
            response = llm._chat(messages)
            inference_time = time.time() - start_time

            # Extract answer
            extracted_answer = extract_boxed_answer(response)

            # Evaluate
            is_correct = evaluate_math_answer(response, problem['answer'], dataset)

            return {
                'problem_id': problem.get('id', -1),
                'problem': problem['problem'][:100] + "...",  # Truncated for storage
                'reference_answer': problem['answer'],
                'model_response': response[:200] + "..." if len(response) > 200 else response,
                'extracted_answer': extracted_answer,
                'is_correct': is_correct,
                'inference_time': inference_time,
                'difficulty': problem.get('difficulty', None)
            }

        except Exception as e:
            return {
                'problem_id': problem.get('id', -1),
                'error': str(e),
                'is_correct': False,
                'inference_time': 0
            }

    def _get_instruction_for_dataset(self, dataset: str) -> str:
        """Get dataset-specific instructions."""
        base_instruction = "Solve the following math problem step by step."

        if dataset in ["math", "aime"]:
            return (f"{base_instruction} Show your work clearly and provide your final answer "
                   "in LaTeX format enclosed in \\boxed{{}}. For example, if the answer is 42, "
                   "write \\boxed{{42}}.")
        elif dataset == "amc":
            return (f"{base_instruction} Show your work clearly and provide your final numerical "
                   "answer enclosed in \\boxed{{}}. For example, if the answer is 42, "
                   "write \\boxed{{42}}.")
        else:  # minerva, olympiad_bench
            return (f"{base_instruction} Show your work clearly and provide your final answer "
                   "enclosed in \\boxed{{}}. Express your answer as clearly and simply as possible.")

    def test_dataset(self, model_name: str, dataset: str) -> Dict:
        """Test a model on a specific dataset."""
        print(f"\n>� Testing {model_name} on {dataset.upper()} dataset...")

        # Create LLM
        llm = self.create_llm(model_name)
        if llm is None:
            print(f"L Skipping {model_name} - could not create LLM")
            return {
                'model': model_name,
                'dataset': dataset,
                'error': 'Could not create LLM',
                'accuracy': 0.0,
                'total_problems': 0
            }

        # Load task and get problems
        try:
            task = get_task(
                f"math_{dataset}",
                model=llm,
                method="direct",
                num_prompts=min(self.num_samples, 1000),  # Cap at reasonable number
                num_responses=1,
                random_seed=self.random_seed
            )

            # Get sample of problems
            all_problems = task.problems
            if len(all_problems) < self.num_samples:
                sample_problems = all_problems
                print(f"9  Using all {len(all_problems)} problems (requested {self.num_samples})")
            else:
                import random
                random.seed(self.random_seed)
                sample_problems = random.sample(all_problems, self.num_samples)
                print(f"9  Testing on {len(sample_problems)} randomly sampled problems")

        except Exception as e:
            print(f"L Failed to load dataset {dataset}: {e}")
            return {
                'model': model_name,
                'dataset': dataset,
                'error': f'Failed to load dataset: {e}',
                'accuracy': 0.0,
                'total_problems': 0
            }

        # Test problems
        results = []
        correct_count = 0
        total_time = 0

        print(f"= Testing {len(sample_problems)} problems...")

        for problem in tqdm(sample_problems, desc="Progress"):
            result = self.test_single_problem(model_name, llm, problem, dataset)
            results.append(result)

            if result.get('is_correct', False):
                correct_count += 1
            total_time += result.get('inference_time', 0)

        # Calculate metrics
        accuracy = correct_count / len(results) if results else 0
        avg_time = total_time / len(results) if results else 0

        # Difficulty breakdown if available
        difficulty_breakdown = {}
        if any(r.get('difficulty') is not None for r in results):
            difficulties = {}
            for result in results:
                diff = result.get('difficulty')
                if diff is not None:
                    if diff not in difficulties:
                        difficulties[diff] = {'total': 0, 'correct': 0}
                    difficulties[diff]['total'] += 1
                    if result.get('is_correct', False):
                        difficulties[diff]['correct'] += 1

            difficulty_breakdown = {
                diff: data['correct'] / data['total'] if data['total'] > 0 else 0
                for diff, data in difficulties.items()
            }

        return {
            'model': model_name,
            'dataset': dataset,
            'accuracy': accuracy,
            'correct_count': correct_count,
            'total_problems': len(results),
            'avg_inference_time': avg_time,
            'difficulty_breakdown': difficulty_breakdown,
            'detailed_results': results[:10]  # Store first 10 for inspection
        }

    def run_all_tests(self) -> Dict:
        """Run all tests for all model-dataset combinations."""
        print("=� Starting Math Model Evaluation")
        print(f"Models: {self.models}")
        print(f"Datasets: {self.datasets}")
        print(f"Samples per dataset: {self.num_samples}")
        print("=" * 60)

        all_results = {}

        for model in self.models:
            model_results = {}
            print(f"\n> Testing model: {model}")

            for dataset in self.datasets:
                result = self.test_dataset(model, dataset)
                model_results[dataset] = result

                # Print summary
                if 'error' not in result:
                    print(f" {dataset}: {result['accuracy']:.3f} "
                          f"({result['correct_count']}/{result['total_problems']}) "
                          f"avg_time: {result['avg_inference_time']:.2f}s")
                else:
                    print(f"L {dataset}: {result['error']}")

            all_results[model] = model_results

        return all_results

    def print_summary(self, results: Dict):
        """Print a summary of all results."""
        print("\n" + "=" * 80)
        print("=� FINAL RESULTS SUMMARY")
        print("=" * 80)

        # Create comparison table
        print(f"{'Model':<35} {'Dataset':<15} {'Accuracy':<10} {'Problems':<10} {'Avg Time':<10}")
        print("-" * 80)

        for model_name, model_results in results.items():
            for dataset, result in model_results.items():
                if 'error' not in result:
                    print(f"{model_name:<35} {dataset:<15} "
                          f"{result['accuracy']:<10.3f} "
                          f"{result['total_problems']:<10} "
                          f"{result['avg_inference_time']:<10.2f}s")
                else:
                    print(f"{model_name:<35} {dataset:<15} ERROR")

        # Overall comparison
        print("\n=� MODEL COMPARISON (Average Accuracy)")
        print("-" * 50)
        for model_name, model_results in results.items():
            valid_results = [r for r in model_results.values() if 'error' not in r]
            if valid_results:
                avg_accuracy = sum(r['accuracy'] for r in valid_results) / len(valid_results)
                total_problems = sum(r['total_problems'] for r in valid_results)
                print(f"{model_name:<35} {avg_accuracy:.3f} ({total_problems} total problems)")

    def save_results(self, results: Dict, output_file: str):
        """Save detailed results to JSON file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{output_file}_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'config': {
                    'models': self.models,
                    'datasets': self.datasets,
                    'num_samples': self.num_samples,
                    'random_seed': self.random_seed
                },
                'results': results
            }, f, indent=2)

        print(f"\n=� Detailed results saved to: {filename}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Test Qwen models on math datasets using local vLLM server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test both models on MATH dataset with 20 samples
  python test_math.py --datasets math --num_samples 20

  # Test only the thinking model on all datasets
  python test_math.py --models Qwen/Qwen3-4B-Thinking-2507

  # Use custom vLLM server port
  python test_math.py --base_url http://localhost:8001

Before running:
  1. Start vLLM server with one of the models:
     vllm serve Qwen/Qwen3-4B-Base --port 8000
  2. To test multiple models, restart server with different model or use different ports
        """)

    parser.add_argument("--models", nargs="+",
                       default=["Qwen/Qwen3-4B-Base", "Qwen/Qwen3-4B-Thinking-2507"],
                       help="Models to test (Note: vLLM serves one model at a time)")
    parser.add_argument("--datasets", nargs="+",
                       default=["math", "aime", "amc"],
                       choices=["math", "aime", "amc", "minerva", "olympiad_bench"],
                       help="Datasets to test")
    parser.add_argument("--num_samples", type=int, default=50,
                       help="Number of problems per dataset")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--output", default="math_test_results",
                       help="Output file prefix")
    parser.add_argument("--base_url", default="http://localhost:8000",
                       help="vLLM server base URL")

    args = parser.parse_args()

    # Warning about multiple models
    if len(args.models) > 1:
        print("⚠️  WARNING: Testing multiple models requires restarting vLLM server")
        print("   vLLM serves one model at a time. You'll need to:")
        print("   1. Test first model")
        print("   2. Stop vLLM server")
        print("   3. Restart with second model")
        print("   4. Run script again for second model")
        print()
        input("Press Enter to continue with first model or Ctrl+C to abort...")

    # Create test runner
    runner = MathTestRunner(
        models=args.models,
        datasets=args.datasets,
        num_samples=args.num_samples,
        random_seed=args.seed,
        base_url=args.base_url
    )

    # Run tests
    try:
        results = runner.run_all_tests()
        runner.print_summary(results)
        runner.save_results(results, args.output)

        print(f"\n Testing complete!")
        return 0

    except KeyboardInterrupt:
        print("\n�  Testing interrupted by user")
        return 1
    except Exception as e:
        print(f"\nL Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())