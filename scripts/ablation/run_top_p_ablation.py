"""
Script for running top_p ablation study for diversity optimization.
Tests different top_p values: [0.7, 0.8, 0.9, 0.95, 1.0] with GPT-4.1 and Gemini-2.5-Flash.
Compares methods: DIRECT, SEQUENCE, STRUCTURE_WITH_PROB.
"""

from verbalized_sampling.pipeline import Pipeline, PipelineConfig, ExperimentConfig, EvaluationConfig
from verbalized_sampling.tasks import Task
from verbalized_sampling.methods import Method
from pathlib import Path
from typing import List, Dict, Any

def create_top_p_ablation_experiments(
    task: Task,
    base_config: Dict[str, Any] = None,
) -> List[ExperimentConfig]:
    """Create experiments for testing different top_p values."""

    # Default base configuration
    base = {
        'task': task,
        'num_responses': 30,
        'num_prompts': 100,
        'target_words': 200,
        'random_seed': 42,
        'temperature': 0.7,
        'num_samples': 5,
    }
    if base_config:
        base.update(base_config)

    experiments = []

    # Grid search parameters
    top_p_values = [0.7, 0.8, 0.9, 0.95, 1.0]
    models = ["gpt-4.1", "gemini-2.5-flash"]
    methods = [Method.DIRECT, Method.SEQUENCE, Method.VS_STANDARD]

    for model in models:
        for method in methods:
            for top_p in top_p_values:
                # DIRECT method only uses num_samples=1
                if method == Method.DIRECT:
                    num_samples = 1
                    strict_json = False
                else:
                    num_samples = base['num_samples']
                    strict_json = True

                experiments.append(ExperimentConfig(
                    name=f"{model}_{method.value}_top_p_{top_p}",
                    model_name=model,
                    method=method,
                    strict_json=strict_json,
                    num_samples=num_samples,
                    top_p=top_p,
                    **{k: v for k, v in base.items() if k not in ['num_samples']}
                ))

    return experiments

def run_top_p_ablation():
    """Run the top_p ablation study."""

    # Configuration
    task = Task.POEM  # You can change this to other tasks
    output_dir = Path("ablation_data/top_p_ablation")

    # Create experiments
    experiments = create_top_p_ablation_experiments(task)

    # Evaluation metrics focused on diversity
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
        title="Top-p Ablation Study for Diversity Optimization"
    )

    # Run pipeline
    pipeline = Pipeline(pipeline_config)
    results = pipeline.run_complete_pipeline()

    print(f"\n✅ Top-p ablation study completed!")
    print(f"📊 Results saved to: {output_dir}")
    print(f"📈 Report available at: {output_dir}/pipeline_report.html")

    return results

if __name__ == "__main__":
    import typer

    def main(
        task: str = typer.Option("poem", help="Task to run ablation on"),
        output_dir: str = typer.Option("ablation_data/top_p_ablation", help="Output directory"),
        rerun: bool = typer.Option(False, help="Rerun all experiments"),
        num_responses: int = typer.Option(30, help="Number of responses per experiment"),
        num_prompts: int = typer.Option(100, help="Number of prompts per experiment"),
    ):
        """Run top_p ablation study for diversity optimization."""

        # Parse task
        task_obj = Task.POEM

        # Create experiments with custom config
        base_config = {
            'num_responses': num_responses,
            'num_prompts': num_prompts,
        }

        experiments = create_top_p_ablation_experiments(task_obj, base_config)

        # Evaluation metrics
        evaluation_config = EvaluationConfig(
            metrics=["diversity", "ngram"],
            num_workers=64
        )

        # Pipeline configuration
        pipeline_config = PipelineConfig(
            experiments=experiments,
            evaluation=evaluation_config,
            output_base_dir=Path(output_dir),
            num_workers=128,
            skip_existing=not rerun,
            rerun=rerun,
            title=f"Top-p Ablation Study - {task_obj.value}"
        )

        # Run pipeline
        pipeline = Pipeline(pipeline_config)
        results = pipeline.run_complete_pipeline()

        print(f"\n✅ Top-p ablation study completed!")
        print(f"📊 Results saved to: {output_dir}")

        return results

    typer.run(main)