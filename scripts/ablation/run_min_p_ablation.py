"""
Script for running min_p ablation study for VLLM models.
Tests different min_p values: [0.0, 0.01, 0.02, 0.05, 0.1] with Qwen3-235B, Llama-3.1-70B, and Llama-3.1-405B.
Compares methods: DIRECT, SEQUENCE, STRUCTURE_WITH_PROB.
Note: min_p only works with use_vllm=True.
"""

from verbalized_sampling.pipeline import Pipeline, PipelineConfig, ExperimentConfig, EvaluationConfig
from verbalized_sampling.tasks import Task
from verbalized_sampling.methods import Method
from pathlib import Path
from typing import List, Dict, Any

def create_min_p_ablation_experiments(
    task: Task,
    base_config: Dict[str, Any] = None,
) -> List[ExperimentConfig]:
    """Create experiments for testing different min_p values."""

    # Default base configuration
    base = {
        'task': task,
        'num_responses': 30,
        'num_prompts': 100,
        'target_words': 200,
        'random_seed': 42,
        'temperature': 0.7,
        'top_p': 0.9,
        'num_samples': 5,
        'use_vllm': True,  # Required for min_p support
    }
    if base_config:
        base.update(base_config)

    experiments = []

    # Grid search parameters
    min_p_values = [0.0, 0.01, 0.02, 0.05, 0.1]
    models = [
        # "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "meta-llama/Llama-3.1-70B-Instruct",
        # "meta-llama/Llama-3.1-405B-Instruct-FP8",
    ]
    methods = [Method.DIRECT, Method.SEQUENCE, Method.VS_STANDARD]

    for model in models:
        for method in methods:
            for min_p in min_p_values:
                # DIRECT method only uses num_samples=1
                if method == Method.DIRECT:
                    num_samples = 1
                    strict_json = False
                else:
                    num_samples = base['num_samples']
                    strict_json = True

                experiments.append(ExperimentConfig(
                    name=f"{model}_{method.value}_min_p_{min_p}",
                    model_name=model,
                    method=method,
                    strict_json=strict_json,
                    num_samples=num_samples,
                    min_p=min_p,
                    **{k: v for k, v in base.items() if k not in ['num_samples']}
                ))

    return experiments

def run_min_p_ablation():
    """Run the min_p ablation study."""

    # Configuration
    task = Task.POEM  # You can change this to other tasks
    output_dir = Path("ablation_data/min_p_ablation")

    # Create experiments
    experiments = create_min_p_ablation_experiments(task)

    # Evaluation metrics focused on diversity and quality
    evaluation_config = EvaluationConfig(
        metrics=["diversity", "ngram"],
        num_workers=128
    )

    # Pipeline configuration
    pipeline_config = PipelineConfig(
        experiments=experiments,
        evaluation=evaluation_config,
        output_base_dir=output_dir,
        num_workers=128,
        skip_existing=True,
        rerun=False,
        title="Min-p Ablation Study for VLLM Models"
    )

    # Run pipeline
    pipeline = Pipeline(pipeline_config)
    results = pipeline.run_complete_pipeline()

    print(f"\n✅ Min-p ablation study completed!")
    print(f"📊 Results saved to: {output_dir}")
    print(f"📈 Report available at: {output_dir}/pipeline_report.html")

    return results

if __name__ == "__main__":
    import typer

    def main(
        task: str = typer.Option("poem", help="Task to run ablation on"),
        output_dir: str = typer.Option("ablation_data/min_p_ablation", help="Output directory"),
        rerun: bool = typer.Option(False, help="Rerun all experiments"),
        num_responses: int = typer.Option(30, help="Number of responses per experiment"),
        num_prompts: int = typer.Option(100, help="Number of prompts per experiment"),
    ):
        """Run min_p ablation study for VLLM models."""

        # Parse task
        task_obj = Task.POEM

        # Create experiments with custom config
        base_config = {
            'num_responses': num_responses,
            'num_prompts': num_prompts,
        }

        experiments = create_min_p_ablation_experiments(task_obj, base_config)

        # Evaluation metrics
        evaluation_config = EvaluationConfig(
            metrics=["diversity", "ngram"],
            num_workers=128
        )

        # Pipeline configuration
        pipeline_config = PipelineConfig(
            experiments=experiments,
            evaluation=evaluation_config,
            output_base_dir=Path(output_dir),
            num_workers=128,
            skip_existing=not rerun,
            rerun=rerun,
            title=f"Min-p Ablation Study - {task_obj.value}"
        )

        # Run pipeline
        pipeline = Pipeline(pipeline_config)
        results = pipeline.run_complete_pipeline()

        print(f"\n✅ Min-p ablation study completed!")
        print(f"📊 Results saved to: {output_dir}")

        return results

    typer.run(main)