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
) -> List[ExperimentConfig]:
    """Create experiments for testing specific method variations."""
    
    experiments = []
    for method_config in methods:
        # Create descriptive name
        name = f"{method_config['method'].value}"
        if method_config.get('strict_json'):
            name += " [strict]"
        if method_config.get('num_samples'):
            name += f" (samples={method_config['num_samples']})"
        
        # Merge configurations with method_config taking precedence
        config = {
            'name': name,
            'task': task,
            'model_name': model_name,
            'num_responses': 20,
            'num_prompts': 1,
            'target_words': 0, 
            'temperature': temperature,
            'top_p': top_p,
            'random_seed': 42,
            'use_vllm': False,
            'probability_definition': "default",
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
    metrics: List[str], # "ngram"
    temperature: float,
    top_p: float,
    output_dir: str,
    num_workers: int = 16,
) -> None:
    """Run tests for specific method variations."""
    print("🔬 Running Method Tests")
    
    experiments = create_method_experiments(task, model_name, temperature, top_p, methods)
    print(f"📊 {len(experiments)} methods to test")
    
    for i, exp in enumerate(experiments, 1):
        print(f"  {i}. {exp.name}")
    
    model_basename = model_name.replace("/", "_")
    config = PipelineConfig(
        experiments=experiments,
        evaluation=EvaluationConfig(metrics=metrics),
        output_base_dir=Path(f"{output_dir}/{model_basename}_{exp.probability_definition}"),
        skip_existing=True,
    )
    
    pipeline = Pipeline(config)
    pipeline.run_complete_pipeline()
    print(f"✅ Done! Check {output_dir}/{model_basename}_{exp.probability_definition}/pipeline_report.html")

if __name__ == "__main__":
    # Example usage for testing different method variations
    
    # Test multi-turn and JSON mode variations
    num_samples = 20
    probability_definitions = "nll" # ["implicit", "explicit", "relative", "confidence", "perplexity", "nll"]
    methods = [
        # {
        #     'method': Method.DIRECT,
        #     'strict_json': False,
        #     'num_samples': 1,
        # },
        # {
        #     'method': Method.DIRECT_COT,
        #     'strict_json': True,
        #     'num_samples': 1,
        # },
        # {
        #     'method': Method.MULTI_TURN,
        #     'strict_json': False,
        #     'num_samples': num_samples,
        # },
        # {
        #     'method': Method.SEQUENCE,
        #     'strict_json': True,
        #     'num_samples': num_samples,
        # },
        {
            'method': Method.STRUCTURE_WITH_PROB,
            'strict_json': True,
            'num_samples': num_samples,
            'probability_definition': probability_definitions,
        },
        # {
        #     'method': Method.CHAIN_OF_THOUGHT,
        #     'strict_json': True,
        #     'num_samples': num_samples,
        #     'probability_definition': "default",
        # },
        # {
        #     'method': Method.COMBINED,
        #     'strict_json': True,
        #     'num_samples': num_samples,
        #     'num_samples_per_prompt': 10,
        #     'probability_definition': "default",
        # }
    ]



    models = [
        "gpt-4.1-mini",
        # "gpt-4.1",
        # "gemini-2.5-flash",
        # "gemini-2.5-pro",
        # "llama-3.1-70b-instruct",
        # "meta-llama/Llama-3.1-70B-Instruct",
        # "meta-llama/Llama-3.1-70B",
        # "qwen3-235b",
        # "claude-4-sonnet",
        # "deepseek-r1",
        # "o3",
    ]
    for model in models:
        model_basename = model.replace("/", "_")
        run_method_tests(
            task=Task.STATE_NAME,
            model_name=model,
            methods=methods,
            metrics=["response_count"],
            temperature=0.7,
            top_p=1.0,
            output_dir="ablation_bias_task",
            num_workers=16 if any(x in model_basename for x in ["claude", "gemini"]) else 32,
        )
    
    # run_method_tests(
    #     task=Task.STATE_NAME,
    #     model_name="gpt-4.1-mini",
    #     methods=methods,
    #     metrics=["response_count"],
    #     temperature=0.7,
    #     top_p=1.0,    
    #     output_dir="method_results_bias",
    # )


    # run_method_tests(
    #     task=Task.STATE_NAME,
    #     model_name="gpt-4.1",
    #     methods=methods,
    #     metrics=["response_count"],
    #     temperature=0.7,
    #     top_p=1.0,
    #     output_dir="method_results_bias",
    # )


    # run_method_tests(
    #     task=Task.STATE_NAME,
    #     model_name="google/gemini-2.5-flash", 
    #     methods=methods,
    #     metrics=["response_count"],
    #     temperature=0.7,
    #     top_p=1.0,
    #     output_dir="method_results_bias",
    # )


    # run_method_tests(
    #     task=Task.STATE_NAME,
    #     model_name="google/gemini-2.5-pro", 
    #     methods=methods,
    #     metrics=["response_count"],
    #     temperature=0.7,
    #     top_p=1.0,
    #     output_dir="method_results_bias",
    # )

    
    # run_method_tests(
    #     task=Task.STATE_NAME,
    #     model_name="anthropic/claude-4-sonnet", 
    #     methods=methods,
    #     metrics=["response_count"],
    #     temperature=0.7,
    #     top_p=1.0,
    #     output_dir="method_results_bias",
    # )

    # run_method_tests(
    #     task=Task.STATE_NAME,
    #     model_name="o3", 
    #     methods=methods,
    #     metrics=["response_count"],
    #     temperature=0.7,
    #     top_p=1.0,
    #     output_dir="method_results_bias",
    # )

    # run_method_tests(
    #     task=Task.STATE_NAME,
    #     model_name="llama-3.1-70b-instruct", 
    #     methods=methods,
    #     metrics=["response_count"],
    #     temperature=0.7,
    #     top_p=1.0,
    #     output_dir="method_results_bias",
    # )

    # run_method_tests(
    #     task=Task.STATE_NAME,
    #     model_name="deepseek-r1", 
    #     methods=methods,
    #     metrics=["response_count"],
    #     temperature=0.7,
    #     top_p=1.0,
    #     output_dir="method_results_bias",
    # )


