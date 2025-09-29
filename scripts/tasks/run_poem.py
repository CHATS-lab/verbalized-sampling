from verbalized_sampling.pipeline import run_quick_comparison, Pipeline, PipelineConfig, ExperimentConfig, EvaluationConfig
from verbalized_sampling.tasks import Task
from verbalized_sampling.methods import Method
from pathlib import Path
from typing import List, Dict, Any
from argparse import ArgumentParser
def create_method_experiments(
    task: Task,
    model_name: str,
    temperature: float,
    top_p: float,
    methods: List[Dict[str, Any]],
) -> List[ExperimentConfig]:
    """Create experiments for testing specific method variations."""
    
    # Base configuration
    base = {
        'task': task,
        'model_name': model_name,
        'num_responses': 30,
        'num_prompts': 100, # current total: 300; total: 4326
        'target_words': 200, 
        'temperature': temperature,
        'top_p': top_p,
        'random_seed': 42,
        'use_vllm': True, # Use litellm for all models
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
        output_base_dir=Path(f"{output_dir}/{model_basename}_{task.value}"),
        skip_existing=True,
        num_workers=num_workers,
    )
    
    pipeline = Pipeline(config)
    pipeline.run_complete_pipeline()
    print(f"✅ Done! Check {output_dir}/{model_basename}_{task.value}/pipeline_report.html")


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--model", type=str, default="meta-llama/Llama-3.1-70B")
    args = args.parse_args()

    methods = [
        {
            'method': Method.DIRECT,
            'strict_json': False,
            'num_samples': 1,
        },
        {
            'method': Method.DIRECT_COT,
            'strict_json': True,
            'num_samples': 1,
        },
        {
            'method': Method.MULTI_TURN,
            'strict_json': True,
            'num_samples': 5,
        },
        # {
        #     'method': Method.MULTI_TURN,
        #     'strict_json': False,
        #     'num_samples': 5,
        # },
        {
            'method': Method.SEQUENCE,
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


    # models = [args.model]
    models = [
        # "openai/gpt-4.1",
        # "openai/gpt-4.1-mini",
        # "google/gemini-2.5-flash",
        # # "meta-llama/llama-3.1-70b-instruct",
        # "meta-llama/Llama-3.1-70B-Instruct",
        # "meta-llama/Llama-3.1-70B",
        # "anthropic/claude-4-sonnet",
        # "google/gemini-2.5-pro",
        "anthropic/claude-3.7-sonnet",
        # "openai/o3",
        # "deepseek/deepseek-r1-0528",
        # "openai/o3",
        # "meta-llama/Llama-3.1-405B-Instruct-FP8",
        # "Qwen/Qwen3-235B-A22B-Instruct-2507",
    ]
    for model in models:
        model_basename = model.replace("/", "_")
        run_method_tests(
            task=Task.POEM,
            model_name=model,
            methods=methods,
            metrics=[
                "diversity", 
                "ngram", 
                "creative_writing_v3", 
                "length"],
            temperature=0.7,
            top_p=1.0,
            output_dir=f"poem_experiments_final/{model_basename}",
            num_workers=32 if "claude" in model_basename else 128,
        )
