"""
Script for running sampling candidates (num_samples) ablation study.
Tests different num_samples values: [3, 5, 8, 10, 15, 20] with GPT-4.1 and Gemini-2.5-Flash.
Compares methods: DIRECT, SEQUENCE, STRUCTURE_WITH_PROB.
"""

from verbalized_sampling.pipeline import Pipeline, PipelineConfig, ExperimentConfig, EvaluationConfig
from verbalized_sampling.tasks import Task
from verbalized_sampling.methods import Method
from pathlib import Path
from typing import List, Dict, Any

def create_sampling_candidates_ablation_experiments(
    task: Task,
    base_config: Dict[str, Any] = None,
) -> List[ExperimentConfig]:
    """Create experiments for testing different num_samples values."""

    # Default base configuration
    base = {
        'task': task,
        'num_responses': 30,
        'num_prompts': 100,
        'target_words': 200,
        'random_seed': 42,
        'temperature': 0.7,
        'top_p': 0.9,
    }
    if base_config:
        base.update(base_config)

    experiments = []

    # Grid search parameters
    num_samples_values = [3, 5, 10, 15, 20]
    models = ["gpt-4.1", "gemini-2.5-flash"]
    methods = [Method.DIRECT, Method.SEQUENCE, Method.VS_STANDARD]

    for model in models:
        for method in methods:
            if method == Method.DIRECT:
                # DIRECT method only uses num_samples=1
                experiments.append(ExperimentConfig(
                    name=f"{model}_{method.value}_samples_1",
                    model_name=model,
                    method=method,
                    strict_json=False,
                    num_samples=1,
                    **base
                ))
            else:
                # Other methods test different num_samples values
                for num_samples in num_samples_values:
                    experiments.append(ExperimentConfig(
                        name=f"{model}_{method.value}_samples_{num_samples}",
                        model_name=model,
                        method=method,
                        strict_json=True,
                        num_samples=num_samples,
                        **base
                    ))

    return experiments

def run_sampling_candidates_ablation():
    """Run the sampling candidates ablation study."""

    # Configuration
    task = Task.POEM  # You can change this to other tasks
    output_dir = Path("ablation_data/sampling_candidates_ablation")

    # Create experiments
    experiments = create_sampling_candidates_ablation_experiments(task)

    # Evaluation metrics focused on diversity and quality
    evaluation_config = EvaluationConfig(
        metrics=["diversity", "ngram"],
        num_workers=64
    )

    # Pipeline configuration
    pipeline_config = PipelineConfig(
        experiments=experiments,
        evaluation=evaluation_config,
        output_base_dir=output_dir,
        num_workers=128,
        skip_existing=True,
        rerun=False,
        title="Sampling Candidates (num_samples) Ablation Study"
    )

    # Run pipeline
    pipeline = Pipeline(pipeline_config)
    results = pipeline.run_complete_pipeline()

    print(f"\n✅ Sampling candidates ablation study completed!")
    print(f"📊 Results saved to: {output_dir}")
    print(f"📈 Report available at: {output_dir}/pipeline_report.html")

    return results

if __name__ == "__main__":
    import typer

    def main(
        task: str = typer.Option("poem", help="Task to run ablation on"),
        output_dir: str = typer.Option("ablation_data/sampling_candidates_ablation", help="Output directory"),
        rerun: bool = typer.Option(False, help="Rerun all experiments"),
        num_responses: int = typer.Option(30, help="Number of responses per experiment"),
        num_prompts: int = typer.Option(100, help="Number of prompts per experiment"),
    ):
        """Run sampling candidates (num_samples) ablation study."""

        # Parse task
        task_obj = Task.POEM

        num_samples_values = [3, 5, 10, 15, 20]
        # Create experiments with custom config
        base_config = {
            'num_responses': num_responses,
            'num_prompts': num_prompts,
        }

        # Modified function to use custom num_samples_values
        def create_custom_experiments():
            base = {
                'task': task_obj,
                'num_responses': num_responses,
                'num_prompts': num_prompts,
                'target_words': 200,
                'random_seed': 42,
                'temperature': 0.7,
                'top_p': 0.9,
            }

            experiments = []
            models = ["gpt-4.1", "gemini-2.5-flash"]
            methods = [Method.DIRECT, Method.SEQUENCE, Method.VS_STANDARD]

            for model in models:
                for method in methods:
                    if method == Method.DIRECT:
                        experiments.append(ExperimentConfig(
                            name=f"{model}_{method.value}_samples_1",
                            model_name=model,
                            method=method,
                            strict_json=False,
                            num_samples=1,
                            **base
                        ))
                    else:
                        for num_samples in num_samples_values:
                            experiments.append(ExperimentConfig(
                                name=f"{model}_{method.value}_samples_{num_samples}",
                                model_name=model,
                                method=method,
                                strict_json=True,
                                num_samples=num_samples,
                                **base
                            ))

            return experiments

        experiments = create_custom_experiments()

        print(f"ℹ️  Testing num_samples values: {num_samples_values}")

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
            title=f"Sampling Candidates Ablation - {task_obj.value}"
        )

        # Run pipeline
        pipeline = Pipeline(pipeline_config)
        results = pipeline.run_complete_pipeline()

        print(f"\n✅ Sampling candidates ablation study completed!")
        print(f"📊 Results saved to: {output_dir}")

        return results

    typer.run(main)