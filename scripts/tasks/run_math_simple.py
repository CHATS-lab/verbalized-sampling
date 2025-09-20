"""
Simple script for running math tasks with standard generation.
Similar to run_jokes_local.py but focused on math reasoning.
"""

from verbalized_sampling.pipeline import Pipeline, PipelineConfig, ExperimentConfig, EvaluationConfig
from verbalized_sampling.tasks import Task
from verbalized_sampling.methods import Method
from pathlib import Path

def run_math_generation(model_name: str, task: Task, output_dir: str = "generated_data/math_simple"):
    """Run simple math generation experiment."""

    print(f"🧮 Running Math Generation: {model_name} on {task.value}")

    # Simple configuration for standard generation
    experiment = ExperimentConfig(
        name="direct_generation",
        task=task,
        model_name=model_name,
        method=Method.DIRECT,
        num_responses=1,
        num_prompts=20,  # Small test set
        target_words=0,
        temperature=0.1,  # Low temperature for math
        random_seed=42,
        use_vllm=True,
        strict_json=False,
    )

    model_basename = model_name.replace("/", "_")
    config = PipelineConfig(
        experiments=[experiment],
        evaluation=EvaluationConfig(metrics=["accuracy"]),
        output_base_dir=Path(f"{output_dir}/{model_basename}_{task.value}"),
        skip_existing=False,
        num_workers=32,
    )

    pipeline = Pipeline(config)
    pipeline.run_complete_pipeline()
    print(f"✅ Results: {output_dir}/{model_basename}_{task.value}/pipeline_report.html")

if __name__ == "__main__":
    # Simple test with one model and one math dataset

    # Model to test (change as needed)
    model = "Qwen/Qwen3-4B-Base"

    # Test on MATH dataset
    run_math_generation(
        model_name=model,
        task=Task.MATH,
        output_dir="generated_data/math_simple"
    )

    print("🎉 Math generation test complete!")
    print("To test other models/datasets, edit the script and change:")
    print("  - model = 'Qwen/Qwen3-4B-Thinking-2507'")
    print("  - task = Task.AIME  # or Task.AMC")