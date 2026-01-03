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

import argparse
from pathlib import Path
from typing import Any, Dict, List

from verbalized_sampling.methods import Method
from verbalized_sampling.pipeline import (
    EvaluationConfig,
    ExperimentConfig,
    Pipeline,
    PipelineConfig,
)
from verbalized_sampling.tasks import Task


def create_method_experiments(
    task: Task,
    model_name: str,
    temperature: float,
    top_p: float,
    methods: List[Dict[str, Any]],
    num_responses: int = 500,
) -> List[ExperimentConfig]:
    """Create experiments for testing specific method variations."""

    # Base configuration
    base = {
        "task": task,
        "model_name": model_name,
        "num_responses": num_responses,
        "num_prompts": 1,
        "target_words": 0,
        "temperature": temperature,
        "top_p": top_p,
        "random_seed": 42,
    }

    experiments = []
    for method_config in methods:
        # Create name
        name = f"{method_config['method'].value}"
        if method_config.get("strict_json"):
            name += " [strict]"
        if method_config.get("num_samples"):
            name += f" (samples={method_config['num_samples']})"

        experiments.append(ExperimentConfig(name=name, **base, **method_config))

    return experiments


def run_method_tests(
    task: Task,
    model_name: str,
    methods: List[Dict[str, Any]],
    metrics: List[str],  # "ngram"
    temperature: float,
    top_p: float,
    output_dir: str,
    num_workers: int = 16,
    num_responses: int = 500,
) -> None:
    """Run tests for specific method variations."""
    print("🔬 Running Method Tests")

    experiments = create_method_experiments(
        task, model_name, temperature, top_p, methods, num_responses
    )
    print(f"📊 {len(experiments)} methods to test")

    for i, exp in enumerate(experiments, 1):
        print(f"  {i}. {exp.name}")

    model_basename = model_name.replace("/", "_")
    config = PipelineConfig(
        experiments=experiments,
        evaluation=EvaluationConfig(metrics=metrics),
        output_base_dir=Path(f"{output_dir}/{model_basename}_{task.value}"),
        skip_existing=True,
        num_workers=num_workers,
    )

    pipeline = Pipeline(config)
    pipeline.run_complete_pipeline()
    print(f"✅ Done! Check {output_dir}/{model_basename}_{task.value}/pipeline_report.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run state name experiments")
    parser.add_argument("--model", type=str, default="gpt-4.1", help="Model to use")
    parser.add_argument(
        "--methods",
        type=str,
        default="direct,vs_standard",
        help="Comma-separated list of methods (e.g., 'direct,vs_standard,vs_cot')",
    )
    parser.add_argument("--num-responses", type=int, default=500, help="Number of responses")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument(
        "--output-dir", type=str, default="method_results_state_name", help="Output directory"
    )
    parser.add_argument("--num-workers", type=int, default=None, help="Number of workers")
    args = parser.parse_args()

    # Parse methods from command line
    method_names = [m.strip().lower() for m in args.methods.split(",")]
    method_map = {
        "direct": Method.DIRECT,
        "multi_turn": Method.MULTI_TURN,
        "sequence": Method.SEQUENCE,
        "vs_standard": Method.VS_STANDARD,
        "vs_cot": Method.VS_COT,
        "vs_multi": Method.VS_MULTI,
        "direct_cot": Method.DIRECT_COT,
    }

    methods = []
    for method_name in method_names:
        if method_name not in method_map:
            print(f"Warning: Unknown method '{method_name}', skipping")
            continue

        method = method_map[method_name]
        if method == Method.DIRECT:
            methods.append({"method": method, "strict_json": False, "num_samples": 1})
        elif method == Method.DIRECT_COT:
            methods.append({"method": method, "strict_json": True, "num_samples": 1})
        elif method == Method.VS_MULTI:
            methods.append(
                {
                    "method": method,
                    "strict_json": True,
                    "num_samples": 20,
                    "num_samples_per_prompt": 5,
                }
            )
        else:
            methods.append({"method": method, "strict_json": True, "num_samples": 20})

    if not methods:
        print("Error: No valid methods specified")
        exit(1)

    # Determine number of workers
    model_basename = args.model.replace("/", "_")
    if args.num_workers is None:
        num_workers = 16 if any(x in model_basename for x in ["claude", "gemini"]) else 32
    else:
        num_workers = args.num_workers

    run_method_tests(
        task=Task.STATE_NAME,
        model_name=args.model,
        methods=methods,
        metrics=["response_count"],
        temperature=args.temperature,
        top_p=args.top_p,
        output_dir=args.output_dir,
        num_workers=num_workers,
        num_responses=args.num_responses,
    )

    # run_method_tests(
    #     task=Task.STATE_NAME,
    #     model_name="gpt-4.1-mini",
    #     methods=methods,
    #     metrics=["response_count"],
    #     temperature=0.7,
    #     top_p=1.0,
    #     output_dir="method_results_state_name",
    # )

    # run_method_tests(
    #     task=Task.STATE_NAME,
    #     model_name="gpt-4.1",
    #     methods=methods,
    #     metrics=["response_count"],
    #     temperature=0.7,
    #     top_p=1.0,
    #     output_dir="method_results_state_name",
    # )

    # run_method_tests(
    #     task=Task.STATE_NAME,
    #     model_name="google/gemini-2.5-flash",
    #     methods=methods,
    #     metrics=["response_count"],
    #     temperature=0.7,
    #     top_p=1.0,
    #     output_dir="method_results_state_name",
    # )

    # run_method_tests(
    #     task=Task.STATE_NAME,
    #     model_name="google/gemini-2.5-pro",
    #     methods=methods,
    #     metrics=["response_count"],
    #     temperature=0.7,
    #     top_p=1.0,
    #     output_dir="method_results_state_name",
    # )

    # run_method_tests(
    #     task=Task.STATE_NAME,
    #     model_name="anthropic/claude-4-sonnet",
    #     methods=methods,
    #     metrics=["response_count"],
    #     temperature=0.7,
    #     top_p=1.0,
    #     output_dir="method_results_state_name",
    # )

    # run_method_tests(
    #     task=Task.STATE_NAME,
    #     model_name="o3",
    #     methods=methods,
    #     metrics=["response_count"],
    #     temperature=0.7,
    #     top_p=1.0,
    #     output_dir="method_results_state_name",
    # )

    # run_method_tests(
    #     task=Task.STATE_NAME,
    #     model_name="llama-3.1-70b-instruct",
    #     methods=methods,
    #     metrics=["response_count"],
    #     temperature=0.7,
    #     top_p=1.0,
    #     output_dir="method_results_state_name",
    # )

    # run_method_tests(
    #     task=Task.STATE_NAME,
    #     model_name="deepseek-r1",
    #     methods=methods,
    #     metrics=["response_count"],
    #     temperature=0.7,
    #     top_p=1.0,
    #     output_dir="method_results_state_name",
    # )


# # Direct (Baseline)
# results = run_quick_comparison(
#     task=Task.STATE_NAME,
#     methods=[Method.DIRECT],
#     model_name="google/gemini-2.5-flash-preview", # google/gemini-2.5-flash-preview, openai/gpt-4.1
#     metrics=["response_count"], # diversity, ttct, creativity_index, length
#     output_dir=Path("comparison_results/direct"),
#     num_responses=NUM_RESPONSES,
#     num_samples=1, # how many times to sample from the model
#     num_prompts=NUM_PROMPTS, # how many samples from the prompt dataset to generate
#     strict_json=False,
#     rerun=True,
#     **MODEL_PARAMS
# )

# # Structure without probability
# results = run_quick_comparison(
#     task=Task.STATE_NAME,
#     methods=[Method.STRUCTURE], # Method.STRUCTURE, Method.VS_STANDARD
#     model_name="google/gemini-2.0-flash-001", # google/gemini-2.5-flash-preview, openai/gpt-4.1
#     metrics=["response_count"], # diversity, ttct, creativity_index, length
#     output_dir=Path("comparison_results/sequence"),
#     num_responses=NUM_RESPONSES,
#     num_samples=NUM_SAMPLES, # how many times to sample from the model
#     num_prompts=NUM_PROMPTS, # how many samples from the prompt dataset to generate
#     strict_json=True,
#     rerun=True,
#     **MODEL_PARAMS
# )


# # Structure with probabilitys
# results = run_quick_comparison(
#     task=Task.STATE_NAME,
#     methods=[Method.VS_STANDARD], # Method.STRUCTURE, Method.VS_STANDARD
#     model_name="google/gemini-2.0-flash-001", # google/gemini-2.5-flash-preview, openai/gpt-4.1
#     metrics=["response_count"], # diversity, ttct, creativity_index, length
#     output_dir=Path("comparison_results/sequence_with_prob"),
#     num_responses=NUM_RESPONSES,
#     num_samples=NUM_SAMPLES, # how many times to sample from the model
#     num_prompts=1, # how many samples from the prompt dataset to generate
#     strict_json=True,
#     rerun=True,
#     **MODEL_PARAMS
# )


# Results will include:
# - Generated responses for each method
# - Evaluation results for all metrics
# - Comparison plots
# - HTML summary report
