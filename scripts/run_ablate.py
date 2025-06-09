"""
Simple ablation study runner for focused parameter comparisons (3-5 parameters max).
Easy to configure, easy to understand.
"""

from verbalized_sampling.pipeline import Pipeline, PipelineConfig, ExperimentConfig, EvaluationConfig
from verbalized_sampling.tasks import Task
from verbalized_sampling.prompts import Method
from pathlib import Path
import itertools

def create_ablation_experiments():
    """Define your ablation study here - keep it simple and focused."""
    
    # Base configuration (what stays the same)
    base = {
        'task': Task.POEM,
        'model_name': "openai/gpt-4.1",
        'num_responses': 100,
        'num_samples': 5,
        'sample_size': 5,
        'temperature': 0.7,
    }
    
    # What you want to compare (keep this small - 3-5 dimensions max)
    compare = {
        'method': [Method.DIRECT, Method.STRUCTURE, Method.STRUCTURE_WITH_PROB, Method.SEQUENCE],
        'strict_json': [False, True],
        # 'method': [Method.STRUCTURE_WITH_PROB],
        # 'strict_json': [False],
    }
    
    # Generate experiments
    experiments = []
    for values in itertools.product(*compare.values()):
        params = dict(zip(compare.keys(), values))
        
        # Skip invalid combinations
        if params['method'] == Method.DIRECT and params['strict_json']:
            continue
        
        # Create name
        name = f"{params['method'].value}_{'strict' if params['strict_json'] else 'normal'}"
        
        experiments.append(ExperimentConfig(
            name=name,
            **base,
            **params
        ))
    
    return experiments

def run_ablation():
    """Run the ablation study."""
    print("🔬 Running Ablation Study")
    
    experiments = create_ablation_experiments()
    print(f"📊 {len(experiments)} experiments to run")
    
    # Print experiment names for quick overview
    for i, exp in enumerate(experiments, 1):
        print(f"  {i}. {exp.name}")
    
    # Setup and run
    config = PipelineConfig(
        experiments=experiments,
        evaluation=EvaluationConfig(metrics=["diversity", "length"]),
        output_base_dir=Path("ablation_results"),
        # rerun=True
    )
    
    pipeline = Pipeline(config)
    return pipeline.run_complete_pipeline()

if __name__ == "__main__":
    results = run_ablation()
    print("✅ Done! Check ablation_results/pipeline_report.html") 