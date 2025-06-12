"""
Unified experiment runner that can handle ablation studies, grid searches, and task-specific experiments.
"""

from verbalized_sampling.pipeline import Pipeline, PipelineConfig, ExperimentConfig, EvaluationConfig
from verbalized_sampling.tasks import Task
from verbalized_sampling.prompts import Method
from pathlib import Path
import itertools
from typing import List, Dict, Any, Optional

def create_experiments(
    task: Task,
    model_name: str,
    experiment_type: str = "ablation",
    base_params: Optional[Dict[str, Any]] = None,
    compare_params: Optional[Dict[str, List[Any]]] = None,
) -> List[ExperimentConfig]:
    """Create experiments based on the specified type and parameters."""
    
    # Default base configuration
    base = {
        'task': task,
        'model_name': model_name,
        'num_responses': 50,
        'num_samples': 5,
        'num_prompts': 10,
        'temperature': 0.7,
        'random_seed': 42,
    }
    if base_params:
        base.update(base_params)

    # Default comparison parameters
    if compare_params is None:
        compare_params = {
            'method': [Method.DIRECT, Method.STRUCTURE_WITH_PROB],
            'strict_json': [False, True],
        }

    experiments = []
    
    if experiment_type == "ablation":
        # Generate experiments for ablation study
        for values in itertools.product(*compare_params.values()):
            params = dict(zip(compare_params.keys(), values))
            
            # Skip invalid combinations
            if params['method'] == Method.DIRECT and params['strict_json']:
                continue
            
            # Create name
            name = f"{params['method'].value}"
            if params['method'] == Method.STRUCTURE_WITH_PROB:
                name += f" (samples={params.get('num_samples', 5)})"
            if params['strict_json']:
                name += " [strict]"
            
            experiments.append(ExperimentConfig(
                name=name,
                **base,
                **params
            ))
    
    elif experiment_type == "grid":
        # Generate experiments for grid search
        for values in itertools.product(*compare_params.values()):
            params = dict(zip(compare_params.keys(), values))
            name = "_".join(f"{k}={v}" for k, v in params.items())
            experiments.append(ExperimentConfig(
                name=name,
                **base,
                **params
            ))
    
    return experiments

def run_experiment(
    task: Task,
    model_name: str,
    experiment_type: str = "ablation",
    base_params: Optional[Dict[str, Any]] = None,
    compare_params: Optional[Dict[str, List[Any]]] = None,
    metrics: List[str] = ["diversity", "length", "ngram"],
    output_dir: str = "experiment_results",
    skip_existing: bool = True,
) -> None:
    """Run the specified experiment type."""
    print(f"🔬 Running {experiment_type.capitalize()} Experiment")
    
    experiments = create_experiments(
        task=task,
        model_name=model_name,
        experiment_type=experiment_type,
        base_params=base_params,
        compare_params=compare_params,
    )
    
    print(f"📊 {len(experiments)} experiments to run")
    for i, exp in enumerate(experiments, 1):
        print(f"  {i}. {exp.name}")
    
    model_basename = model_name.replace("/", "_")
    config = PipelineConfig(
        experiments=experiments,
        evaluation=EvaluationConfig(metrics=metrics),
        output_base_dir=Path(f"{output_dir}/{model_basename}_{task.value}"),
        skip_existing=skip_existing,
    )
    
    pipeline = Pipeline(config)
    pipeline.run_complete_pipeline()
    print(f"✅ Done! Check {output_dir}/{model_basename}_{task.value}/pipeline_report.html")

if __name__ == "__main__":
    # Example usage for different experiment types
    
    # 1. Simple ablation study
    run_experiment(
        task=Task.JOKE,
        model_name="anthropic/claude-sonnet-4",
        experiment_type="ablation",
        compare_params={
            'method': [Method.DIRECT, Method.STRUCTURE_WITH_PROB],
            'strict_json': [False, True],
            'num_samples': [3, 5, 10],
        }
    )
    
    # 2. Grid search example
    # run_experiment(
    #     task=Task.POEM,
    #     model_name="openai/gpt-4.1",
    #     experiment_type="grid",
    #     compare_params={
    #         'temperature': [0.5, 0.7, 0.9],
    #         'num_samples': [3, 5, 7],
    #     }
    # ) 