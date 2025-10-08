#!/usr/bin/env python3
# Copyright 2025 CHATS-Lab. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Research API - Experimental framework (from README.md)

This example shows the original research API used in the paper experiments.
For quick experiments, use the simple API (examples 01-04). This API provides
full control over methods, tasks, and metrics for reproducible research.

Note: These examples mirror the README.md "Quick Start" and "Example Usage" sections.
"""

from verbalized_sampling.methods import Method
from verbalized_sampling.pipeline import run_quick_comparison
from verbalized_sampling.tasks import Task, get_task


def example_quick_comparison():
    """Example 1: Quick comparison (from README.md)."""
    print("=" * 80)
    print("Example 1: Quick Comparison - Research API")
    print("=" * 80)
    print()

    # Run a quick comparison between methods
    results = run_quick_comparison(
        task=Task.JOKE,
        methods=[Method.DIRECT, Method.VS_STANDARD],
        model_name="gpt-4o-mini",
        metrics=["diversity", "length", "ngram"],
        num_responses=50,
    )

    print("Results:")
    print(f"  VS-Standard diversity: {results['VS_STANDARD']['diversity']:.2f}")
    print(f"  Direct diversity: {results['DIRECT']['diversity']:.2f}")
    print()

    # Show all metrics
    for method, method_results in results.items():
        print(f"{method}:")
        for metric, value in method_results.items():
            print(f"  {metric}: {value:.3f}")
        print()


def example_task_usage():
    """Example 2: Task-based generation (from README.md)."""
    print("=" * 80)
    print("Example 2: Task-Based Generation")
    print("=" * 80)
    print()

    # Create a task
    task = get_task(Task.STORY, num_prompts=10, random_seed=42)

    print(f"Task: {task.name}")
    print(f"Number of prompts: {len(task.prompts)}")
    print()

    # Get a VS-Standard prompt
    vs_prompt = task.get_prompt(Method.VS_STANDARD, num_samples=5, prompt_index=0)
    print("VS-Standard prompt (first 200 chars):")
    print(vs_prompt[:200])
    print("...")
    print()

    # Note: In practice, you would call your model here:
    # responses = model.generate(vs_prompt)
    # parsed = task.parse_response(Method.VS_STANDARD, responses)
    # Returns: [{"text": "...", "probability": 0.15}, ...]

    print("Expected output format after parsing:")
    print('[{"text": "Story text...", "probability": 0.25}, ...]')
    print()


def example_chain_of_thought():
    """Example 3: Chain-of-thought reasoning (from README.md)."""
    print("=" * 80)
    print("Example 3: Chain-of-Thought VS")
    print("=" * 80)
    print()

    task = get_task(Task.JOKE, num_prompts=5, random_seed=100)

    # Get VS-CoT prompt
    cot_prompt = task.get_prompt(Method.VS_COT, num_samples=3, prompt_index=0)

    print("VS-CoT prompt (first 200 chars):")
    print(cot_prompt[:200])
    print("...")
    print()

    # Note: In practice:
    # cot_responses = model.generate(cot_prompt)
    # parsed_cot = task.parse_response(Method.VS_COT, cot_responses)
    # Returns: [{"reasoning": "...", "text": "...", "probability": 0.22}, ...]

    print("Expected output format with reasoning:")
    print('[{"reasoning": "Analysis...", "text": "Joke...", "probability": 0.30}, ...]')
    print()


def example_methods_comparison():
    """Example 4: Comparing different VS methods."""
    print("=" * 80)
    print("Example 4: Method Comparison")
    print("=" * 80)
    print()

    # Available methods
    methods = [
        Method.DIRECT,  # Standard prompting
        Method.VS_STANDARD,  # Verbalized Sampling (standard)
        Method.VS_COT,  # VS with chain-of-thought
        Method.VS_MULTI,  # VS with multi-turn
        Method.SEQUENCE,  # Sequence sampling
    ]

    print("Available methods for comparison:")
    for method in methods:
        print(f"  - {method.value}")
    print()

    # Example: Get prompts for each method
    task = get_task(Task.POEM, num_prompts=3, random_seed=200)

    print("Prompt lengths for each method:")
    for method in methods[:3]:  # Show first 3
        prompt = task.get_prompt(method, num_samples=5, prompt_index=0)
        print(f"  {method.value}: {len(prompt)} chars")
    print()


def example_evaluation_metrics():
    """Example 5: Using evaluation metrics."""
    print("=" * 80)
    print("Example 5: Evaluation Metrics")
    print("=" * 80)
    print()

    # Available metrics from the paper
    metrics = [
        "diversity",  # Distinct n-gram ratios
        "length",  # Response length statistics
        "ngram",  # N-gram overlap
        "self_bleu",  # Self-BLEU (lower = more diverse)
        "entropy",  # Entropy of distribution
    ]

    print("Available metrics:")
    for metric in metrics:
        print(f"  - {metric}")
    print()

    # Example comparison with metrics
    print("Running comparison with multiple metrics...")
    results = run_quick_comparison(
        task=Task.STORY,
        methods=[Method.DIRECT, Method.VS_STANDARD],
        model_name="gpt-4o-mini",
        metrics=metrics,
        num_responses=20,  # Smaller for demo
    )

    print("\nResults by method:")
    for method, method_results in results.items():
        print(f"\n{method}:")
        for metric, value in method_results.items():
            print(f"  {metric}: {value:.3f}")
    print()


def example_task_types():
    """Example 6: Different task types from the paper."""
    print("=" * 80)
    print("Example 6: Task Types from Paper")
    print("=" * 80)
    print()

    # Tasks from the paper
    tasks = [
        (Task.POEM, "Creative Writing - Poetry"),
        (Task.STORY, "Creative Writing - Stories"),
        (Task.JOKE, "Creative Writing - Jokes"),
        (Task.SIMPLE_QA, "Open-Ended QA"),
        (Task.DIALOGUE, "Dialogue Simulation"),
    ]

    print("Tasks available:")
    for task_enum, description in tasks:
        print(f"  - {task_enum.value}: {description}")
    print()

    # Example: Load and inspect a task
    story_task = get_task(Task.STORY, num_prompts=5, random_seed=42)
    print(f"Story task loaded:")
    print(f"  Number of prompts: {len(story_task.prompts)}")
    print(f"  First prompt: {story_task.prompts[0][:100]}...")
    print()


if __name__ == "__main__":
    print("=" * 80)
    print("RESEARCH API EXAMPLES")
    print("=" * 80)
    print()
    print("These examples demonstrate the research API used in the paper.")
    print("For simpler use cases, see examples 01-04 (Simple API).")
    print()

    example_quick_comparison()
    example_task_usage()
    example_chain_of_thought()
    example_methods_comparison()
    example_evaluation_metrics()
    example_task_types()

    print("=" * 80)
    print("For more details, see:")
    print("  - scripts/EXPERIMENTS.md - Full experiment replication guide")
    print("  - scripts/tasks/ - Individual task scripts")
    print("=" * 80)
