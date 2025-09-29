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
        'num_responses': 30,
        'num_prompts': 100, # current total: 300; total: 4326
        'target_words': 200, 
        'temperature': temperature,
        'top_p': top_p,
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
        evaluation=EvaluationConfig(metrics=metrics, num_responses_per_prompt=30),
        output_base_dir=Path(f"{output_dir}/{model_basename}_{task.value}"),
        skip_existing=True,
        num_workers=num_workers,
    )
    
    pipeline = Pipeline(config)
    pipeline.run_complete_pipeline()
    print(f"✅ Done! Check {output_dir}/{model_basename}_{task.value}/pipeline_report.html")


if __name__ == "__main__":
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
        #     'strict_json': True,
        #     'num_samples': 5,
        # },
        {
            'method': Method.SEQUENCE,
            'strict_json': True,
            'num_samples': 5,
        },
        # {
        #     'method': Method.VS_STANDARD,
        #     'strict_json': True,
        #     'num_samples': 5,
        #     'probability_definition': "explicit"
        # },
        # {
        #     'method': Method.VS_COT,
        #     'strict_json': True,
        #     'num_samples': 5,
        #     'probability_definition': "explicit"
        # },
        # {
        #     'method': Method.VS_MULTI,
        #     'strict_json': True,
        #     'num_samples': 5,
        #     'probability_definition': "confidence"
        # },
    ]


    models = [
        # "openai/gpt-4.1",
        # "openai/gpt-4.1-mini",
        # "google/gemini-2.5-flash",
        # "meta-llama/Llama-3.1-70B-Instruct",
        # "meta-llama/Llama-3.1-70B-Instruct",
        # "anthropic/claude-4-sonnet",
        # "google/gemini-2.5-pro",
        # "anthropic/claude-3.7-sonnet",
        # "openai/o3",
        "deepseek/deepseek-r1-0528",
    ]
    for model in models:
        model_basename = model.replace("/", "_")
        run_method_tests(
            task=Task.BOOK,
            model_name=model,
            methods=methods,
            metrics=["diversity", "ngram", "creative_writing_v3", "length"],
            temperature=0.5,
            top_p=1.0,
            # output_dir=f"story_experiments_max_diversity/{model_basename}",
            output_dir=f"story_experiments_test/{model_basename}",
            num_workers=32 if "claude" in model_basename else 128,
        )


    # run_method_tests(
    #     task=Task.POEM,
    #     model_name="gpt-4.1-mini", # google/gemini-2.5-pro, gpt-4.1, anthropic/claude-4-sonnet
    #     methods=methods,
    #     metrics=["diversity"],
    #     temperature=0.7,
    #     top_p=1.0,    
    #     output_dir="method_results_poem",
    # )


    # run_method_tests(
    #     task=Task.POEM,
    #     model_name="gpt-4.1", 
    #     methods=methods,
    #     metrics=["diversity"],
    #     temperature=0.7,
    #     top_p=1.0,
    #     output_dir="method_results_poem",
    # )


    # run_method_tests(
    #     task=Task.POEM,
    #     model_name="google/gemini-2.5-flash",
    #     methods=methods,
    #     metrics=["diversity"],
    #     temperature=0.7,
    #     top_p=1.0,
    #     output_dir="method_results_poem",
    # )


    # run_method_tests(
    #     task=Task.POEM,
    #     model_name="google/gemini-2.5-pro",
    #     methods=methods,
    #     metrics=["diversity"],
    #     temperature=0.7,
    #     top_p=1.0,
    #     output_dir="method_results_poem",
    # )

    
    # run_method_tests(
    #     task=Task.POEM,
    #     model_name="anthropic/claude-4-sonnet",
    #     methods=methods,
    #     metrics=["diversity"],
    #     temperature=0.7,
    #     top_p=1.0,
    #     output_dir="method_results_poem",
    # )

    # run_method_tests(
    #     task=Task.POEM,
    #     model_name="o3",
    #     methods=methods,
    #     metrics=["diversity"],
    #     temperature=0.7,
    #     top_p=1.0,
    #     output_dir="method_results_poem",
    # )

    # run_method_tests(
    #     task=Task.POEM,
    #     model_name="llama-3.1-70b-instruct",
    #     methods=methods,
    #     metrics=["diversity"],
    #     temperature=0.7,
    #     top_p=1.0,
    #     output_dir="method_results_poem",
    # )

    # run_method_tests(
    #     task=Task.POEM,
    #     model_name="deepseek-r1",
    #     methods=methods,
    #     metrics=["diversity"],
    #     temperature=0.7,
    #     top_p=1.0,
    #     output_dir="method_results_poem",
    # )




# NUM_RESPONSES = 100 # how many responses to generate for each prompt
# NUM_SAMPLES = 10 # how many times to sample from the model
# NUM_PROMPTS = 1 # 4326 samples, how many prompts to sample from the prompt dataset
# TARGET_WORDS = 0

# MODEL_PARAMS = {
#     "temperature": 0.7,
#     "top_p": 0.9,
# }

# # Direct (Baseline)
# results = run_quick_comparison(
#     task=Task.SIMPLE_QA,
#     methods=[Method.DIRECT],
#     model_name="openai/gpt-4.1", # google/gemini-2.5-flash-preview, openai/gpt-4.1
#     metrics=["factuality"], # factuality, diversity, ttct, creativity_index, length
#     output_dir=Path("comparison_results/direct"),
#     num_responses=NUM_RESPONSES,
#     num_samples=1,
#     num_prompts=NUM_PROMPTS, # how many samples from the prompt dataset to generate
#     target_words=TARGET_WORDS,
#     rerun=True,
#     create_backup=False,
#     strict_json=False,
#     **MODEL_PARAMS
# )

# # Structure without probability
# results = run_quick_comparison(
#     task=Task.SIMPLE_QA,
#     methods=[Method.STRUCTURE], # Method.STRUCTURE, Method.VS_STANDARD
#     model_name="openai/gpt-4.1", # google/gemini-2.5-flash-preview, openai/gpt-4.1
#     metrics=["factuality"], # diversity, ttct, creativity_index, length
#     output_dir=Path("comparison_results/structure"),
#     num_responses=NUM_RESPONSES,
#     num_samples=NUM_SAMPLES, # how many times to sample from the model
#     num_prompts=NUM_PROMPTS, # how many samples from the prompt dataset to generate
#     target_words=TARGET_WORDS,
#     strict_json=True,
#     rerun=True,
#     **MODEL_PARAMS
# )


# # Structure with probabilitys
# results = run_quick_comparison(
#     task=Task.SIMPLE_QA,
#     methods=[Method.VS_STANDARD], # Method.STRUCTURE, Method.VS_STANDARD
#     model_name="openai/gpt-4.1", # google/gemini-2.5-flash-preview, openai/gpt-4.1
#     metrics=["factuality"], # diversity, ttct, creativity_index, length
#     output_dir=Path("comparison_results/sequence_with_prob"),
#     num_responses=NUM_RESPONSES,
#     num_samples=NUM_SAMPLES, # how many times to sample from the model
#     num_prompts=NUM_PROMPTS, # how many samples from the prompt dataset to generate
#     target_words=TARGET_WORDS,
#     strict_json=True,
#     rerun=True,
#     **MODEL_PARAMS
# )




# Results will include:
# - Generated responses for each method
# - Evaluation results for all metrics
# - Comparison plots
# - HTML summary report
