#!/usr/bin/env python3
"""
Safety evaluation script using HarmBench dataset and verbalized sampling methods.

This script evaluates LLM safety using the HarmBench safety dataset with various
verbalized sampling methods and the StrongReject evaluation methodology.
"""

from verbalized_sampling.pipeline import Pipeline, PipelineConfig, ExperimentConfig, EvaluationConfig
from verbalized_sampling.tasks import Task
from verbalized_sampling.methods import Method
from pathlib import Path
from typing import List, Dict, Any
import sys


def create_safety_experiments(
    model_name: str,
    temperature: float,
    top_p: float,
    methods: List[Dict[str, Any]],
    num_prompts: int = 500,
    num_responses: int = 5,
) -> List[ExperimentConfig]:
    """Create safety evaluation experiments for testing different method variations."""

    # Base configuration for safety evaluation
    base = {
        'task': Task.SAFETY,  # Need to add SAFETY to Task enum if not exists
        'model_name': model_name,
        'num_responses': num_responses,
        'num_prompts': num_prompts,
        'target_words': 0,
        'temperature': temperature,
        'top_p': top_p,
        'random_seed': 42,
        "use_vllm": False,
    }

    experiments = []
    for method_config in methods:
        # Create descriptive experiment name
        name = f"{method_config['method'].value}"
        if method_config.get('strict_json'):
            name += " [strict JSON]"
        if method_config.get('num_samples'):
            name += f" (samples={method_config['num_samples']})"
        if method_config.get('probability_definition'):
            name += f" [{method_config['probability_definition']}]"

        experiments.append(ExperimentConfig(
            name=name,
            **base,
            **method_config
        ))

    return experiments


def run_safety_evaluation(
    model_name: str,
    methods: List[Dict[str, Any]],
    temperature: float = 0.7,
    top_p: float = 1.0,
    output_dir: str = "safety_evaluation_results",
    num_prompts: int = 500,
    num_responses: int = 5,
) -> None:
    """Run safety evaluation tests using verbalized sampling methods."""
    print("🔒 Running Safety Evaluation")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Temperature: {temperature}, Top-p: {top_p}")
    print(f"Prompts: {num_prompts}, Responses per prompt: {num_responses}")
    print()

    experiments = create_safety_experiments(
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        methods=methods,
        num_prompts=num_prompts,
        num_responses=num_responses
    )

    print(f"📊 {len(experiments)} safety experiments to run:")
    for i, exp in enumerate(experiments, 1):
        print(f"  {i}. {exp.name}")
    print()

    # Configure pipeline with safety evaluation
    model_basename = model_name.replace("/", "_")
    config = PipelineConfig(
        experiments=experiments,
        evaluation=EvaluationConfig(
            metrics=["safety"],  # Use safety evaluator
            # You can add additional safety-focused metrics here
            # metrics=["safety", "refusal_quality", "harmful_content_detection"]
        ),
        output_base_dir=Path(f"{output_dir}/{model_basename}_safety"),
        skip_existing=True,
    )

    pipeline = Pipeline(config)
    pipeline.run_complete_pipeline()

    print("✅ Safety evaluation complete!")
    print(f"📋 Results available at: {output_dir}/{model_basename}_safety/pipeline_report.html")
    print("🔍 Key metrics to review:")
    print("  - Average safety score (lower is safer)")
    print("  - Refusal rate (higher indicates better safety)")
    print("  - Compliance rate (lower indicates better safety)")
    print("  - Harmful response rate (should be near 0%)")


if __name__ == "__main__":
    # Safety evaluation configuration

    # Test various sampling methods for safety evaluation
    safety_methods = [
        {
            'method': Method.DIRECT,
            'strict_json': False,
            'num_samples': 1,
        },
        {
            'method': Method.DIRECT_COT,
            'strict_json': True,
            'num_samples': 1,
        },
        {
            'method': Method.MULTI_TURN,
            'strict_json': True,
            'num_samples': 5,
        },
        {
            'method': Method.SEQUENCE,
            'strict_json': True,
            'num_samples': 5,
        },
        {
            'method': Method.VS_STANDARD,
            'strict_json': True,
            'num_samples': 5,
        },
        {
            'method': Method.VS_COT,
            'strict_json': True,
            'num_samples': 5,
        },
        {
            'method': Method.VS_MULTI,
            'strict_json': True,
            'num_samples': 5,
            'num_samples_per_prompt': 2,
        }
        # {
        #     'method': Method.VS_STANDARD,
        #     'strict_json': True,
        #     'num_samples': 3,
        #     'probability_definition': 'explicit',  # Use explicit probability definition for safety
        # },
        # {
        #     'method': Method.VS_COT,
        #     'strict_json': True,
        #     'num_samples': 3,
        # },
        # {
        #     'method': Method.MULTI_TURN,
        #     'strict_json': False,
        #     'num_samples': 3,
        # },
    ]

    # Models to test (add your preferred models)
    models = [
        "openai/gpt-4.1",
        # "openai/gpt-4.1-mini",
        # "google/gemini-2.5-flash",
        # # "meta-llama/llama-3.1-70b-instruct",
        # "meta-llama/Llama-3.1-70B-Instruct",
        # "meta-llama/Llama-3.1-70B",
        # "anthropic/claude-4-sonnet",
        # "google/gemini-2.5-pro",
        # "anthropic/claude-3.7-sonnet",
        # "anthropic/claude-4-sonnet",
        # "openai/o3",
        # "deepseek/deepseek-r1-0528",
        # "openai/o3",
        # "meta-llama/Llama-3.1-405B-Instruct-FP8",
        # "Qwen/Qwen3-235B-A22B-Instruct-2507",
    ]

    # Safety evaluation parameters
    evaluation_config = {
        'temperature': 0.3,  # Lower temperature for more consistent safety behavior
        'top_p': 0.9,        # Slightly restricted sampling
        'num_prompts': 500,   # Number of safety prompts to test (max from HarmBench standard config)
        'num_responses': 5,   # Responses per prompt
        'output_dir': "generated_data/safety_evaluation_results"
    }

    # Run safety evaluation for each model
    for model in models:
        print(f"\n🤖 Evaluating model: {model}")
        print("=" * 60)

        try:
            run_safety_evaluation(
                model_name=model,
                methods=safety_methods,
                **evaluation_config
            )

            print(f"✅ Completed safety evaluation for {model}")

        except Exception as e:
            print(f"❌ Error evaluating {model}: {str(e)}")
            continue