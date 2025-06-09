"""
Comprehensive parameter sweep for extensive exploration.
Configure your dimensions and let it run.
"""

from verbalized_sampling.pipeline import Pipeline, PipelineConfig, ExperimentConfig, EvaluationConfig
from verbalized_sampling.tasks import Task
from verbalized_sampling.prompts import Method
from pathlib import Path
import itertools

def create_sweep_experiments():
    """Define your comprehensive sweep here."""
    
    # Base configuration
    base = {
        'task': Task.POEM,
        'top_p': 0.9,
        'random_seed': 42,
    }
    
    # All dimensions to sweep over
    dimensions = {
        'model_name': [
            "openai/gpt-4.1",
            "openai/gpt-4o-mini",
        ],
        'method': [
            Method.DIRECT,
            Method.STRUCTURE, 
            Method.STRUCTURE_WITH_PROB,
            Method.SEQUENCE
        ],
        'strict_json': [False, True],
        'temperature': [0.7, 1.0],
        'num_responses': [10, 20],
        'num_samples': [5, 10],
        'sample_size': [5, 10],
    }
    
    # Generate all combinations
    experiments = []
    total_combinations = 1
    for values in dimensions.values():
        total_combinations *= len(values)
    
    print(f"🌊 Generating sweep: {total_combinations:,} possible combinations")
    
    for values in itertools.product(*dimensions.values()):
        params = dict(zip(dimensions.keys(), values))
        
        # Filter out invalid combinations
        if params['method'] == Method.DIRECT and params['strict_json']:
            continue
        
        # Create descriptive name
        model_short = params['model_name'].split('/')[-1]
        name = (f"{model_short}_{params['method'].value}_"
                f"{'strict' if params['strict_json'] else 'normal'}_"
                f"temp{params['temperature']}_"
                f"resp{params['num_responses']}_"
                f"samp{params['num_samples']}x{params['sample_size']}")
        
        experiments.append(ExperimentConfig(
            name=name,
            **base,
            **params
        ))
    
    return experiments

def run_sweep():
    """Run the comprehensive sweep."""
    print("🌊 Running Comprehensive Parameter Sweep")
    
    experiments = create_sweep_experiments()
    print(f"📊 {len(experiments):,} experiments to run")
    
    # Show quick stats
    models = set(exp.model_name.split('/')[-1] for exp in experiments)
    methods = set(exp.method.value for exp in experiments)
    print(f"📈 {len(models)} models × {len(methods)} methods × various parameters")
    
    # Confirm before running
    response = input(f"\nThis will run {len(experiments):,} experiments. Continue? (y/N): ")
    if not response.lower().startswith('y'):
        print("Cancelled.")
        return
    
    # Setup and run
    config = PipelineConfig(
        experiments=experiments,
        evaluation=EvaluationConfig(metrics=["diversity", "creativity_index", "length"]),
        output_base_dir=Path("sweep_results"),
        rerun=True,
        num_workers=128
    )
    
    pipeline = Pipeline(config)
    return pipeline.run_complete_pipeline()

if __name__ == "__main__":
    results = run_sweep()
    print("✅ Done! Check sweep_results/pipeline_report.html") 