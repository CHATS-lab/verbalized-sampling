from verbalized_sampling.pipeline import run_quick_comparison, Pipeline, PipelineConfig, ExperimentConfig, EvaluationConfig
from verbalized_sampling.tasks import Task
from verbalized_sampling.methods import Method
from pathlib import Path
from typing import List, Dict, Any
import sys

def create_method_experiments(
    task: Task,
    model_name: str,
    temperature: float,
    top_p: float,
    methods: List[Dict[str, Any]],
    probability_definition: str,
) -> List[ExperimentConfig]:
    """Create experiments for testing specific method variations."""
    
    experiments = []
    for i, method_config in enumerate(methods):
        # Create descriptive name
        name = f"{method_config['method'].value}"
        if method_config.get('strict_json'):
            name += " [strict]"
        if method_config.get('num_samples'):
            name += f" (samples={method_config['num_samples']})"
        
        # Use unique random seed for each experiment to ensure different prompt selections
        unique_random_seed = 42
        
        # Merge configurations with method_config taking precedence
        config = {
            'name': name,
            'task': task,
            'model_name': model_name,
            'num_responses': 40, # 500
            'num_prompts': 40, # 5
            'target_words': 0, 
            'temperature': temperature,
            'top_p': top_p,
            'random_seed': unique_random_seed,  # Unique seed for each experiment
            'use_vllm': False,
            'probability_definition': probability_definition,
            **method_config  # method_config overrides base values
        }
        
        # Validate required fields
        if 'method' not in method_config:
            raise ValueError(f"Missing 'method' in method_config: {method_config}")
        
        experiments.append(ExperimentConfig(**config))
    
    return experiments


def run_method_tests(
    task: Task,
    model_name: str,
    methods: List[Dict[str, Any]],
    probability_definition: str,
    metrics: List[str], # "ngram"
    temperature: float,
    top_p: float,
    output_dir: str,
    num_workers: int = 16,
    rerun: bool = False,  # Add rerun option
) -> None:
    """Run tests for specific method variations."""
    print("🔬 Running Method Tests")
    
    experiments = create_method_experiments(task, model_name, temperature, top_p, methods, probability_definition)
    print(f"📊 {len(experiments)} methods to test")
    
    for i, exp in enumerate(experiments, 1):
        print(f"  {i}. {exp.name} (seed: {exp.random_seed})")
    
    # Run each experiment separately to avoid overwriting
    for exp in experiments:
        print(f"\n🔄 Running experiment: {exp.name}")
        
        model_basename = model_name.replace("/", "_")
        config = PipelineConfig(
            experiments=[exp],  # Single experiment
            evaluation=EvaluationConfig(metrics=metrics),
            output_base_dir=Path(f"{output_dir}/{exp.method.value}/{model_basename}/{exp.probability_definition}"),
            skip_existing=True,  # Always run experiments to ensure fresh results
            rerun=rerun,  # Use rerun flag to force clean runs
        )
        print(f"📂 Output directory: {output_dir}/{exp.method.value}/{model_basename}/{exp.probability_definition}")
        print(f"🎲 Random seed: {exp.random_seed}")
        
        pipeline = Pipeline(config)
        pipeline.run_complete_pipeline()
        print(f"✅ Done! Check {output_dir}/{exp.method.value}/{model_basename}/{exp.probability_definition}/pipeline_report.html")


if __name__ == "__main__":
    # Example usage for testing different method variations

    num_samples = 20
    probability_definitions = ["implicit", "explicit", "relative", "percentage", "confidence", "nll", "perplexity"]

    # Build methods list by iterating all probability definitions
    methods = [
        # {
        #     'method': Method.VS_STANDARD,
        #     'strict_json': True,
        #     'num_samples': num_samples,
        # }
        {
            'method': Method.VS_MULTI,
            'strict_json': True,
            'num_samples': num_samples,
            'num_samples_per_prompt': 10,
        }
    ]
    # If you want to add other methods, you can append to the list here

    models = [
        # "gpt-4.1-mini",
        "gpt-4.1",
        "gemini-2.5-flash",
        # "gemini-2.5-pro",
        # "llama-3.1-70b-instruct",
        # "meta-llama/Llama-3.1-70B-Instruct",
        # "meta-llama/Llama-3.1-70B",
        # "qwen3-235b",
        # "claude-4-sonnet",
        # "deepseek-r1",
        # "o3",
    ]
    
    # Set rerun=True to force clean runs (this will delete existing outputs and start fresh)
    rerun = False  # Change to False if you want to preserve existing results
    
    for model in models:
        model_basename = model.replace("/", "_")
        for prob_def in probability_definitions:
            run_method_tests(
                task=Task.STATE_NAME,
                model_name=model,
                methods=methods,
                probability_definition=prob_def,
                metrics=["response_count"],
                temperature=0.9,
                top_p=1.0,
                output_dir="ablation_bias_task",
                num_workers=16 if any(x in model_basename for x in ["claude", "gemini"]) else 32,
                rerun=rerun,  # Pass rerun flag
            )
    
    