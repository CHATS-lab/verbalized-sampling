"""
Focused comparison script to demonstrate that verbalized sampling methods
outperform direct sampling even with optimized temperature and top-p parameters.
"""

from verbalized_sampling.pipeline import Pipeline, PipelineConfig, ExperimentConfig, EvaluationConfig
from verbalized_sampling.tasks import Task
from verbalized_sampling.methods import Method
from pathlib import Path
from typing import List, Dict, Any

def create_optimized_comparison_experiments(
    task: Task,
    model_name: str,
    base_config: Dict[str, Any] = None,
) -> List[ExperimentConfig]:
    """Create experiments comparing optimized sampling vs verbalized methods."""
    
    # Default base configuration
    base = {
        'task': task,
        'model_name': model_name,
        'num_responses': 50,
        'num_prompts': 20,
        'target_words': 200,
        'random_seed': 42,
    }
    if base_config:
        base.update(base_config)

    experiments = []
    
    # Test multiple optimized sampling configurations
    optimized_configs = [
        # Conservative sampling
        {"temp": 0.3, "top_p": 0.9, "name": "conservative"},
        # Balanced sampling  
        {"temp": 0.7, "top_p": 0.95, "name": "balanced"},
        # Creative sampling
        {"temp": 1.0, "top_p": 0.98, "name": "creative"},
        # Very creative sampling
        {"temp": 1.5, "top_p": 0.99, "name": "very_creative"},
    ]
    
    # 1. Direct sampling with optimized parameters
    for config in optimized_configs:
        experiments.append(ExperimentConfig(
            name=f"direct_{config['name']}",
            method=Method.DIRECT,
            temperature=config['temp'],
            top_p=config['top_p'],
            strict_json=False,
            num_samples=1,
            **base
        ))
    
    # 2. Verbalized sampling methods with standard parameters
    verbalized_methods = [
        (Method.SEQUENCE, "sequence"),
        (Method.MULTI_TURN, "multi_turn"), 
        (Method.VS_STANDARD, "vs_standard"),
    ]
    
    for method, name in verbalized_methods:
        experiments.append(ExperimentConfig(
            name=f"{name}_standard",
            method=method,
            temperature=0.7,  # Standard temperature
            top_p=0.9,        # Standard top-p
            strict_json=True,
            num_samples=5,
            **base
        ))
    
    # 3. Verbalized methods with optimized parameters (to show they can be further improved)
    for method, name in verbalized_methods:
        # Use the most creative settings for verbalized methods
        experiments.append(ExperimentConfig(
            name=f"{name}_optimized",
            method=method,
            temperature=1.0,  # Higher temperature for creativity
            top_p=0.98,       # Higher top-p for diversity
            strict_json=True,
            num_samples=5,
            **base
        ))
    
    return experiments

def run_optimized_comparison(
    task: Task,
    model_name: str,
    output_dir: str,
    metrics: List[str] = None,
    base_config: Dict[str, Any] = None,
    num_workers: int = 128,
) -> None:
    """Run optimized sampling comparison study."""
    print(f"🎯 Running Optimized Sampling Comparison for {model_name}")
    print(f"📊 Task: {task.value}")
    
    if metrics is None:
        metrics = ["diversity", "ngram", "creative_writing_v3"]
    
    experiments = create_optimized_comparison_experiments(task, model_name, base_config)
    print(f"📊 {len(experiments)} experiments to run")
    
    # Group experiments by type
    direct_exps = [exp for exp in experiments if exp.method == Method.DIRECT]
    verbalized_exps = [exp for exp in experiments if exp.method != Method.DIRECT]
    
    print(f"  📋 Direct sampling (optimized): {len(direct_exps)} experiments")
    for exp in direct_exps:
        print(f"    - {exp.name} (temp={exp.temperature}, top_p={exp.top_p})")
    
    print(f"  📋 Verbalized methods: {len(verbalized_exps)} experiments")
    for exp in verbalized_exps:
        print(f"    - {exp.name} (temp={exp.temperature}, top_p={exp.top_p})")
    
    model_basename = model_name.replace("/", "_")
    config = PipelineConfig(
        experiments=experiments,
        evaluation=EvaluationConfig(metrics=metrics),
        output_base_dir=Path(f"{output_dir}/{model_basename}_{task.value}_optimized_comparison"),
        skip_existing=True,
        num_workers=num_workers,
    )
    
    pipeline = Pipeline(config)
    pipeline.run_complete_pipeline()
    print(f"✅ Done! Check {output_dir}/{model_basename}_{task.value}_optimized_comparison/pipeline_report.html")

def run_quick_demo(
    task: Task,
    model_name: str,
    output_dir: str,
    metrics: List[str] = None,
    num_workers: int = 128,
) -> None:
    """Run a quick demo with fewer experiments for faster results."""
    print(f"⚡ Running Quick Demo for {model_name}")
    
    if metrics is None:
        metrics = ["diversity", "ngram", "creative_writing_v3"]
    
    # Base configuration with fewer samples for quick testing
    base = {
        'task': task,
        'model_name': model_name,
        'num_responses': 30,
        'num_prompts': 10,
        'target_words': 200,
        'random_seed': 42,
    }
    
    experiments = [
        # Best direct sampling configuration
        ExperimentConfig(
            name="direct_best",
            method=Method.DIRECT,
            temperature=1.0,
            top_p=0.98,
            strict_json=False,
            num_samples=1,
            **base
        ),
        # Standard verbalized methods
        ExperimentConfig(
            name="sequence_standard",
            method=Method.SEQUENCE,
            temperature=0.7,
            top_p=0.9,
            strict_json=True,
            num_samples=5,
            **base
        ),
        ExperimentConfig(
            name="structure_with_prob_standard",
            method=Method.VS_STANDARD,
            temperature=0.7,
            top_p=0.9,
            strict_json=True,
            num_samples=5,
            **base
        ),
    ]
    
    model_basename = model_name.replace("/", "_")
    config = PipelineConfig(
        experiments=experiments,
        evaluation=EvaluationConfig(metrics=metrics),
        output_base_dir=Path(f"{output_dir}/{model_basename}_{task.value}_quick_demo"),
        skip_existing=True,
        num_workers=num_workers,
    )
    
    pipeline = Pipeline(config)
    pipeline.run_complete_pipeline()
    print(f"✅ Done! Check {output_dir}/{model_basename}_{task.value}_quick_demo/pipeline_report.html")

if __name__ == "__main__":
    # Models to test
    models = [
        "openai/gpt-4.1",
        "google/gemini-2.5-flash",
    ]
    
    # Task to use
    task = Task.POEM
    
    # Metrics to evaluate
    metrics = ["diversity", "ngram", "creative_writing_v3"]
    
    # Output directory
    output_dir = "sampling_comparison_results"
    
    # Run experiments for each model
    for model in models:
        print(f"\n{'='*60}")
        print(f"Running experiments for {model}")
        print(f"{'='*60}")
        
        # Adjust workers based on model
        num_workers = 32 if "claude" in model else 128
        
        # Run quick demo first (faster)
        run_quick_demo(
            task=task,
            model_name=model,
            output_dir=output_dir,
            metrics=metrics,
            num_workers=num_workers,
        )
        
        # Run full optimized comparison
        run_optimized_comparison(
            task=task,
            model_name=model,
            output_dir=output_dir,
            metrics=metrics,
            num_workers=num_workers,
        )
    
    print(f"\n🎉 All experiments completed!")
    print(f"📁 Results saved in: {output_dir}")
    print(f"📊 Check the pipeline reports for detailed analysis")
    print(f"\n💡 Expected Results:")
    print(f"   - Verbalized methods should outperform direct sampling even with optimized parameters")
    print(f"   - Higher temperature/top-p should improve direct sampling but not match verbalized methods")
    print(f"   - Verbalized methods can be further improved with optimized parameters") 