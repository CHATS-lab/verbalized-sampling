"""
Script for running num_samples ablation study with grid search.
Tests different num_samples values: [3, 5, 10, 20] with fixed target_words=100 and num_responses=100.
Only uses GPT-4.1 model with methods: DIRECT, SEQUENCE, MULTI_TURN, STRUCTURE_WITH_PROB.
"""

from verbalized_sampling.pipeline import Pipeline, PipelineConfig, ExperimentConfig, EvaluationConfig
from verbalized_sampling.tasks import Task
from verbalized_sampling.methods import Method
from pathlib import Path
from typing import List, Dict, Any

def create_num_samples_ablation_experiments(
    task: Task,
    model_name: str,
    base_config: Dict[str, Any] = None,
) -> List[ExperimentConfig]:
    """Create experiments for testing different num_samples values."""
    
    # Default base configuration
    base = {
        'task': task,
        'model_name': model_name,
        'num_responses': 100,
        'num_prompts': 100,
        'target_words': 100,
        'random_seed': 42,
        'temperature': 0.7,
        'top_p': 0.9,
    }
    if base_config:
        base.update(base_config)

    experiments = []
    
    # Grid search parameters
    num_samples_values = [3, 5, 10, 20]
    methods = [Method.DIRECT, Method.SEQUENCE, Method.MULTI_TURN, Method.VS_STANDARD]
    
    for method in methods:
        for num_samples in num_samples_values:
            # DIRECT method only uses num_samples=1
            if method == Method.DIRECT:
                if num_samples == 3:  # Only run DIRECT once (use first num_samples value)
                    experiments.append(ExperimentConfig(
                        name=f"direct_num_samples_1",
                        method=method,
                        strict_json=False,
                        num_samples=1,
                        **base
                    ))
            else:
                # Other methods use the actual num_samples values
                strict_json = method in [Method.SEQUENCE, Method.MULTI_TURN, Method.VS_STANDARD]
                experiments.append(ExperimentConfig(
                    name=f"{method.value.lower()}_num_samples_{num_samples}",
                    method=method,
                    strict_json=strict_json,
                    num_samples=num_samples,
                    **base
                ))
    
    return experiments

def run_num_samples_ablation_study(
    task: Task,
    model_name: str,
    output_dir: str,
    metrics: List[str] = None,
    base_config: Dict[str, Any] = None,
    num_workers: int = 128,
) -> None:
    """Run a comprehensive num_samples ablation study."""
    print(f"🔬 Running Num Samples Ablation Study for {model_name}")
    print(f"📊 Task: {task.value}")
    print(f"📈 Grid search: num_samples=[3, 5, 10, 20], target_words=100, num_responses=100")
    
    if metrics is None:
        metrics = ["diversity", "ngram", "creative_writing_v3"]
    
    experiments = create_num_samples_ablation_experiments(task, model_name, base_config)
    print(f"📊 {len(experiments)} experiments to run")
    
    # Group experiments by method for better organization
    method_groups = {}
    for exp in experiments:
        method = exp.method.value
        if method not in method_groups:
            method_groups[method] = []
        method_groups[method].append(exp)
    
    for method, exps in method_groups.items():
        print(f"  📋 {method}: {len(exps)} experiments")
        for exp in exps:
            print(f"    - {exp.name}")
    
    model_basename = model_name.replace("/", "_")
    config = PipelineConfig(
        experiments=experiments,
        evaluation=EvaluationConfig(metrics=metrics),
        output_base_dir=Path(f"{output_dir}/{model_basename}_{task.value}_num_samples_ablation"),
        skip_existing=True,
        num_workers=num_workers,
    )
    
    pipeline = Pipeline(config)
    pipeline.run_complete_pipeline()
    print(f"✅ Done! Check {output_dir}/{model_basename}_{task.value}_num_samples_ablation/pipeline_report.html")

if __name__ == "__main__":
    # Models to test
    models = [
        "openai/gpt-4.1",
        "google/gemini-2.5-flash",
    ]
    
    # Task to use
    task = Task.POEM
    
    # Metrics to evaluate
    metrics = ["diversity", "ngram", "length", "creative_writing_v3"]
    
    # Output directory
    output_dir = "num_samples_ablation_results"
    
    # Run ablation studies for each model
    for model in models:
        print(f"\n{'='*60}")
        print(f"Running num_samples ablation for {model}")
        print(f"{'='*60}")
        
        # Run ablation study
        run_num_samples_ablation_study(
            task=task,
            model_name=model,
            output_dir=output_dir,
            metrics=metrics,
            num_workers=128,
        )
    
    print(f"\n🎉 All ablation studies completed!")
    print(f"📁 Results saved in: {output_dir}")
    print(f"📊 Check the pipeline reports for detailed analysis")