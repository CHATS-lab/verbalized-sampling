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
    use_vllm: bool = True,
) -> List[ExperimentConfig]:
    """Create experiments for testing specific method variations."""
    
    # Base configuration
    base = {
        'task': task,
        'model_name': model_name,
        'num_responses': 40,
        'num_prompts': 40, # current total: 40; 
        'target_words': 0, 
        'temperature': temperature,
        'top_p': top_p,
        'random_seed': 42,
        'use_vllm': use_vllm, # Use litellm for all models
    }
    
    experiments = []
    for method_config in methods:
        # Create name
        name = f"{method_config['method'].value}"
        if method_config.get('strict_json'):
            name += " [strict]"
        if method_config.get('num_samples'):
            name += f" (samples={method_config['num_samples']})"
        if "probability_definition" in method_config:
            name += f" (prob_def={method_config['probability_definition']})"
        if "probability_tuning" in method_config:
            name += f" (prob_tuning={method_config['probability_tuning']})"

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
    rerun: bool = False,
    use_vllm: bool = True,
) -> None:
    """Run tests for specific method variations."""
    print("🔬 Running Method Tests")
    
    experiments = create_method_experiments(task, model_name, temperature, top_p, methods, use_vllm=use_vllm)
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

    probability_tunings = [
        -1,
        0.9,
        0.5,
        0.1,
        0.05,
        0.01,
        # 0.005
    ]


    num_samples = 20
    num_samples_per_prompt = 10
    methods = [
        {
            'method': Method.DIRECT,
            'strict_json': False,
            'num_samples': num_samples,
        },
        # {
        #     'method': Method.DIRECT_COT,
        #     'strict_json': True,
        #     'num_samples': num_samples,
        # },
        # {
        #     'method': Method.MULTI_TURN,
        #     'strict_json': True,
        #     'num_samples': num_samples,
        # },
        {
            'method': Method.SEQUENCE,
            'strict_json': True,
            'num_samples': num_samples,
        },
    ]
    for prob_def in probability_tunings:
        methods.append({
            'method': Method.VS_STANDARD,
            'strict_json': True,
            'num_samples': num_samples,
            'probability_definition': "explicit",
            'probability_tuning': prob_def,
        })
        methods.append({
            'method': Method.VS_MULTI,
            'strict_json': True,
            'num_samples': num_samples,
            'num_samples_per_prompt': num_samples_per_prompt,
            'probability_definition': "confidence",
            'probability_tuning': prob_def,
        })


    # models = [args.model]
    models = [
        # "gpt-4.1",
        "google/gemini-2.5-flash",
    ]
    for model in models:
        model_basename = model.replace("/", "_")
        run_method_tests(
            task=Task.STATE_NAME,
            # prompt_path="data/state_name.txt",
            model_name=model,
            methods=methods,
            metrics=["response_count"],
            temperature=0.7,
            top_p=1.0,
            output_dir="bias_experiments_prob_tuning",
            # output_dir="openended_qa_general",
            num_workers=16 if any(x in model_basename for x in ["claude", "gemini", "llama", "deepseek"]) else 32,
            rerun=False,
            use_vllm=False,
        )