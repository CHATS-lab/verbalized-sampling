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
        'num_responses': 30, # 30
        'num_prompts': 100, # 100
        'target_words': 0,
        'temperature': 0.7,
        'random_seed': 42,
        # 'use_vllm': True,/
    }

    # story, target_words: 500, num_responses: 
    # ablation: vary num_responses
    # ablation: base model
    
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
    num_workers: int = 128,
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
        num_workers=num_workers,
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
            "method": Method.DIRECT_COT,
            "strict_json": True,
            "num_samples": 1,
        },
        {
            'method': Method.SEQUENCE,
            'strict_json': True,
            'num_samples': 5,
        },
        {
            'method': Method.MULTI_TURN,
            'strict_json': True,
            'num_samples': 5,
        },
        {
            'method': Method.VS_STANDARD,
            'strict_json': True,
            'num_samples': 5,
        },
        {
            'method': Method.VS_COT,
            'strict_json': True,
            'num_samples': 5,
        },
        {
            'method': Method.VS_MULTI,
            'strict_json': True,
            'num_samples': 5,
            'num_samples_per_prompt': 2,
        }
    ]
     
    models = [
        # "openai/gpt-4.1",
        # "openai/gpt-4.1-mini",
        # "google/gemini-2.5-flash",
        # "anthropic/claude-4-sonnet",
        # "anthropic/claude-3.7-sonnet",
        "google/gemini-2.5-pro",
        # "openai/o3",
        # "deepseek/deepseek-r1-0528",

        # "Qwen/Qwen3-235B-A22B-Instruct-2507",
        # "meta-llama/Llama-3.1-70B-Instruct"
        # "meta-llama/llama-3.1-70b-instruct"
        # "openai/o3",
    ]
    for model in models:
        model_basename = model.replace("/", "_")
        run_method_tests(
            task=Task.JOKE,
            model_name=model,
            methods=methods,
            metrics=["ngram"],
            output_dir=f"generated_data/joke_experiments_final/{model_basename}",
            num_workers=16 if "claude" in model_basename else 128,
        )