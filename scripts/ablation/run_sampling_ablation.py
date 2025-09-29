"""
Script for running sampling parameter ablation studies.
Tests different temperature and top-p values and compares with direct/sequence sampling.
"""

from verbalized_sampling.pipeline import Pipeline, PipelineConfig, ExperimentConfig, EvaluationConfig
from verbalized_sampling.tasks import Task
from verbalized_sampling.methods import Method
from pathlib import Path
from typing import List, Dict, Any
import itertools

def create_sampling_ablation_experiments(
    task: Task,
    model_name: str,
    base_config: Dict[str, Any] = None,
) -> List[ExperimentConfig]:
    """Create experiments for testing different sampling parameters."""
    
    # Default base configuration
    base = {
        'task': task,
        'model_name': model_name,
        'num_responses': 50,
        'num_prompts': 20,
        'target_words': 200,
        'random_seed': 42,
    }
    if base_config:
        base.update(base_config)

    experiments = []
    
    # 1. Direct sampling baseline with different temperatures
    temperatures = [0.3, 0.7, 1.0, 1.5]
    for temp in temperatures:
        experiments.append(ExperimentConfig(
            name=f"direct_temp_{temp}",
            method=Method.DIRECT,
            temperature=temp,
            top_p=0.9,
            strict_json=False,
            num_samples=1,
            **base
        ))
    
    # 2. Direct sampling with different top-p values
    top_p_values = [0.99, 0.95, 0.8, 0.7]
    for top_p in top_p_values:
        experiments.append(ExperimentConfig(
            name=f"direct_top_p_{top_p}",
            method=Method.DIRECT,
            temperature=0.7,
            top_p=top_p,
            strict_json=False,
            num_samples=1,
            **base
        ))
    
    # 3. Sequence sampling with different temperatures
    for temp in temperatures:
        experiments.append(ExperimentConfig(
            name=f"sequence_temp_{temp}",
            method=Method.SEQUENCE,
            temperature=temp,
            top_p=0.9,
            strict_json=True,
            num_samples=5,
            **base
        ))
    
    # 4. Sequence sampling with different top-p values
    for top_p in top_p_values:
        experiments.append(ExperimentConfig(
            name=f"sequence_top_p_{top_p}",
            method=Method.SEQUENCE,
            temperature=0.7,
            top_p=top_p,
            strict_json=True,
            num_samples=5,
            **base
        ))
    
    # 6. Structure with probability sampling with different temperatures
    for temp in temperatures:
        experiments.append(ExperimentConfig(
            name=f"structure_with_prob_temp_{temp}",
            method=Method.VS_STANDARD,
            temperature=temp,
            top_p=0.9,
            strict_json=True,
            num_samples=5,
            **base
        ))
    
    for top_p in top_p_values:
        experiments.append(ExperimentConfig(
            name=f"structure_with_prob_top_p_{top_p}",
            method=Method.VS_STANDARD,
            temperature=0.7,
            top_p=top_p,
            strict_json=True,
            num_samples=5,
            **base
        ))
    return experiments

def run_sampling_ablation_study(
    task: Task,
    model_name: str,
    output_dir: str,
    metrics: List[str] = None,
    base_config: Dict[str, Any] = None,
    num_workers: int = 128,
) -> None:
    """Run a comprehensive sampling parameter ablation study."""
    print(f"🔬 Running Sampling Ablation Study for {model_name}")
    print(f"📊 Task: {task.value}")
    
    if metrics is None:
        metrics = ["diversity", "ngram", "creative_writing_v3"]
    
    experiments = create_sampling_ablation_experiments(task, model_name, base_config)
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
        output_base_dir=Path(f"{output_dir}/{model_basename}_{task.value}_sampling_ablation"),
        skip_existing=True,
        num_workers=num_workers,
    )
    
    pipeline = Pipeline(config)
    pipeline.run_complete_pipeline()
    print(f"✅ Done! Check {output_dir}/{model_basename}_{task.value}_sampling_ablation/pipeline_report.html")

def run_focused_comparison(
    task: Task,
    model_name: str,
    output_dir: str,
    metrics: List[str] = None,
    num_workers: int = 128,
) -> None:
    """Run a focused comparison of best sampling parameters vs verbalized methods."""
    print(f"🎯 Running Focused Comparison for {model_name}")
    
    if metrics is None:
        metrics = ["diversity", "ngram", "creative_writing_v3"]
    
    # Base configuration
    base = {
        'task': task,
        'model_name': model_name,
        'num_responses': 50,
        'num_prompts': 20,
        'target_words': 200,
        'random_seed': 42,
    }
    
    # Best sampling parameters (to be determined from ablation study)
    best_temp = 0.9  # High temperature for creativity
    best_top_p = 0.9  # High top-p for diversity
    
    experiments = [
        # Direct sampling with best parameters
        ExperimentConfig(
            name="direct_best_params",
            method=Method.DIRECT,
            temperature=best_temp,
            top_p=best_top_p,
            strict_json=False,
            num_samples=1,
            **base
        ),
        # Sequence sampling with best parameters
        ExperimentConfig(
            name="sequence_best_params",
            method=Method.SEQUENCE,
            temperature=best_temp,
            top_p=best_top_p,
            strict_json=True,
            num_samples=5,
            **base
        ),
        # Multi-turn sampling with best parameters
        ExperimentConfig(
            name="multi_turn_best_params",
            method=Method.MULTI_TURN,
            temperature=best_temp,
            top_p=best_top_p,
            strict_json=True,
            num_samples=5,
            **base
        ),
        # Structure with probability sampling with best parameters
        ExperimentConfig(
            name="structure_with_prob_best_params",
            method=Method.VS_STANDARD,
            temperature=best_temp,
            top_p=best_top_p,
            strict_json=True,
            num_samples=5,
            **base
        ),
    ]
    
    model_basename = model_name.replace("/", "_")
    config = PipelineConfig(
        experiments=experiments,
        evaluation=EvaluationConfig(metrics=metrics),
        output_base_dir=Path(f"{output_dir}/{model_basename}_{task.value}_focused_comparison"),
        skip_existing=True,
        num_workers=num_workers,
    )
    
    pipeline = Pipeline(config)
    pipeline.run_complete_pipeline()
    print(f"✅ Done! Check {output_dir}/{model_basename}_{task.value}_focused_comparison/pipeline_report.html")

if __name__ == "__main__":
    # Models to test
    models = [
        "openai/gpt-4.1",
        "google/gemini-2.5-flash",
    ]
    
    # Task to use
    task = Task.POEM
    
    # Metrics to evaluate
    metrics = ["diversity", "ngram", "creative_writing_v3"]
    
    # Output directory
    output_dir = "sampling_ablation_results"
    
    # Run ablation studies for each model
    for model in models:
        print(f"\n{'='*60}")
        print(f"Running experiments for {model}")
        print(f"{'='*60}")
        
        # Adjust workers based on model
        num_workers = 32 if "claude" in model else 128
        
        # Run comprehensive ablation study
        run_sampling_ablation_study(
            task=task,
            model_name=model,
            output_dir=output_dir,
            metrics=metrics,
            num_workers=num_workers,
        )
        
        # Run focused comparison
        run_focused_comparison(
            task=task,
            model_name=model,
            output_dir=output_dir,
            metrics=metrics,
            num_workers=num_workers,
        )
    
    print(f"\n🎉 All experiments completed!")
    print(f"📁 Results saved in: {output_dir}")
    print(f"📊 Check the pipeline reports for detailed analysis") 