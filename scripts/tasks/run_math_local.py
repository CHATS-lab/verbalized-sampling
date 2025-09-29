"""
Script for running math task experiments with different methods.
"""

from verbalized_sampling.pipeline import Pipeline, PipelineConfig, ExperimentConfig, EvaluationConfig
from verbalized_sampling.tasks import Task
from verbalized_sampling.methods import Method
from pathlib import Path
from typing import List, Dict, Any

def create_math_experiments(
    task: Task,
    model_name: str,
    methods: List[Dict[str, Any]],
) -> List[ExperimentConfig]:
    """Create experiments for testing math tasks with different methods."""

    # Base configuration for math tasks
    base = {
        'task': task,
        'model_name': model_name,
        'num_responses': 1,  # Start with 1 response for math
        'num_prompts': 50,   # Test on 50 problems
        'target_words': 0,   # No word constraint for math
        'temperature': 0.7,  # Low temperature for math reasoning
        'random_seed': 42,
        'use_vllm': True,    # Use vLLM for local models
    }

    experiments = []
    for method_config in methods:
        # Create name
        name = f"{method_config['method'].value}"
        if method_config.get('strict_json'):
            name += " [strict]"
        if method_config.get('num_samples'):
            name += f" (samples={method_config['num_samples']})"

        experiments.append(ExperimentConfig(
            name=name,
            **base,
            **method_config
        ))

    return experiments

def run_math_tests(
    task: Task,
    model_name: str,
    methods: List[Dict[str, Any]],
    metrics: List[str],
    output_dir: str,
    num_workers: int = 32,
) -> None:
    """Run math tests for specific method variations."""
    print("🧮 Running Math Task Tests")

    experiments = create_math_experiments(task, model_name, methods)
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
    # Test different math reasoning methods

    # Start with basic methods for math tasks
    methods = [
        {
            'method': Method.DIRECT,
            'strict_json': False,
            'num_samples': 1,
        },
        {
            'method': Method.DIRECT_COT,
            'strict_json': False,
            'num_samples': 1,
        },
        # Add structured methods for comparison
        {
            'method': Method.SEQUENCE,
            'strict_json': True,
            'num_samples': 3,
        },
        {
            'method': Method.VS_STANDARD,
            'strict_json': True,
            'num_samples': 3,
        },
        {
            'method': Method.VS_COT,
            'strict_json': True,
            'num_samples': 3,
        },
    ]

    # Test with local Qwen models
    models = [
        "Qwen/Qwen3-4B-Base",
        # "Qwen/Qwen3-4B-Thinking-2507",
    ]

    # Test different math datasets
    math_tasks = [
        Task.MATH,
        Task.AIME,
        Task.AMC,
    ]

    for task in math_tasks:
        print(f"\n📚 Testing {task.value} dataset...")
        for model in models:
            model_basename = model.replace("/", "_")
            run_math_tests(
                task=task,
                model_name=model,
                methods=methods,
                metrics=["accuracy"],  # Use accuracy metric for math
                output_dir=f"generated_data/math_experiments/{model_basename}",
                num_workers=32,  # Conservative for local GPU
            )