
from verbalized_sampling.pipeline import run_quick_comparison, Pipeline, PipelineConfig, ExperimentConfig, EvaluationConfig
from verbalized_sampling.tasks import Task
from verbalized_sampling.methods import Method
from pathlib import Path
from typing import List, Dict, Any

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
        'num_responses': 600,
        'num_prompts': 1, # current total: 300; total: 4326
        'target_words': 0, 
        'temperature': temperature,
        'top_p': top_p,
        'random_seed': 42,
        "use_vllm": False,
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
    )
    
    pipeline = Pipeline(config)
    pipeline.run_complete_pipeline()
    print(f"✅ Done! Check {output_dir}/{model_basename}_{task.value}/pipeline_report.html")


if __name__ == "__main__":
    num_samples = 5
    methods = [
        {
            'method': Method.DIRECT,
            'strict_json': False,
            'num_samples': 1,
        },
        {
            'method': Method.DIRECT_COT,
            'strict_json': False,
            'num_samples': 1,
        },
        {
            'method': Method.MULTI_TURN,
            'strict_json': False,
            'num_samples': num_samples,
        },
        {
            'method': Method.SEQUENCE,
            'strict_json': True,
            'num_samples': num_samples,
        },
        {
            'method': Method.VS_STANDARD,
            'strict_json': True,
            'num_samples': num_samples,
        },
        {
            'method': Method.VS_COT,
            'strict_json': True,
            'num_samples': num_samples,
        },
        {
            'method': Method.VS_MULTI,
            'strict_json': True,
            'num_samples': num_samples,
            'num_samples_per_prompt': 3,
        }
    ]


    models = [
        # "gpt-4.1-mini",
        # "gpt-4.1",
        # "gemini-2.5-flash",
        # "gemini-2.5-pro",
        # # "meta-llama/Llama-3.1-70B-Instruct"
        "claude-4-sonnet",
        # "deepseek-r1",
        # "o3",
    ]
    for model in models:
        model_basename = model.replace("/", "_")
        run_method_tests(
            task=Task.RAND_NUM,
            model_name=model,
            methods=methods,
            metrics=[],
            temperature=0.7,
            top_p=1.0,
            output_dir="method_results_rng",
            num_workers=1 if any(x in model_basename for x in ["claude", "gemini"]) else 16,
        )


 