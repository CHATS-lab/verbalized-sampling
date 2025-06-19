"""
Script for testing specific method variations and configurations.
"""

from verbalized_sampling.pipeline import Pipeline, PipelineConfig, ExperimentConfig, EvaluationConfig
from verbalized_sampling.tasks import Task
from verbalized_sampling.methods import Method
from pathlib import Path
from typing import List, Dict, Any

def create_method_experiments(
    task: Task,
    model_name: str,
    methods: List[Dict[str, Any]],
) -> List[ExperimentConfig]:
    """Create experiments for testing specific method variations."""
    
    # Base configuration
    base = {
        'task': task,
        'model_name': model_name,
        'num_responses': 100,
        'num_prompts': 5,
        'target_words': 200,
        'temperature': 1,
        'random_seed': 42,
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

def run_method_tests(
    task: Task,
    model_name: str,
    methods: List[Dict[str, Any]],
    metrics: List[str], # "ngram"
    output_dir: str,
) -> None:
    """Run tests for specific method variations."""
    print("🔬 Running Method Tests")
    
    experiments = create_method_experiments(task, model_name, methods)
    print(f"📊 {len(experiments)} methods to test")
    
    for i, exp in enumerate(experiments, 1):
        print(f"  {i}. {exp.name}")
    
    model_basename = model_name.replace("/", "_")
    config = PipelineConfig(
        experiments=experiments,
        evaluation=EvaluationConfig(metrics=metrics),
        output_base_dir=Path(f"{output_dir}/{model_basename}_{task.value}"),
        skip_existing=True,
    )
    
    pipeline = Pipeline(config)
    pipeline.run_complete_pipeline()
    print(f"✅ Done! Check {output_dir}/{model_basename}_{task.value}/pipeline_report.html")

if __name__ == "__main__":
    # Example usage for testing different method variations
    
    # Test multi-turn and JSON mode variations
    methods = [
        {
            'method': Method.DIRECT,
            'strict_json': False,
            'num_samples': 1,
        },
        {
            'method': Method.MULTI_TURN,
            'strict_json': True,
            'num_samples': 5,
        },
        {
            'method': Method.SEQUENCE,
            'strict_json': True,
            'num_samples': 5,
        },
        {
            'method': Method.STRUCTURE_WITH_PROB,
            'strict_json': True,
            'num_samples': 5,
        },
    ]
    
    # run_method_tests(
    #     task=Task.POEM,
    #     model_name="openai/gpt-4.1",
    #     methods=methods,
    #     metrics=["diversity", "ngram", "ttct"], # "ngram", "ttct"
    #     output_dir="method_results_poem",
    # ) 
    # run_method_tests(
    #     task=Task.SPEECH,
    #     model_name="openai/gpt-4.1",
    #     methods=methods,
    #     metrics=["diversity", "ttct"], # "ngram", "ttct"
    #     output_dir="method_results_speech",
    # ) 
    run_method_tests(
        task=Task.BOOK,
        model_name="openai/gpt-4.1",
        methods=methods,
        metrics=["diversity", "ttct"], # "ngram", "ttct"
        output_dir="method_results_book",
    ) 
    # run_method_tests(
    #     task=Task.JOKE,
    #     model_name="google/gemini-2.0-flash-001",
    #     methods=methods,
    #     metrics=["diversity", "ngram", "ttct"],
    #     output_dir="method_results_jokes",
    # )