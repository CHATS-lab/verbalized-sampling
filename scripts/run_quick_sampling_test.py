"""
Quick sampling parameter test to demonstrate verbalized methods vs optimized direct sampling.
"""

from verbalized_sampling.pipeline import Pipeline, PipelineConfig, ExperimentConfig, EvaluationConfig
from verbalized_sampling.tasks import Task
from verbalized_sampling.methods import Method
from pathlib import Path

def run_quick_sampling_test():
    """Run a quick test comparing optimized direct sampling vs verbalized methods."""
    
    # Models to test
    models = [
        "openai/gpt-4.1",
        "google/gemini-2.5-flash",
    ]
    
    task = Task.POEM
    metrics = ["diversity", "ngram", "creative_writing_v3"]
    output_dir = "quick_sampling_test"
    
    for model in models:
        print(f"\n{'='*50}")
        print(f"Testing {model}")
        print(f"{'='*50}")
        
        # Base configuration
        base = {
            'task': task,
            'model_name': model,
            'num_responses': 30,
            'num_prompts': 10,
            'target_words': 200,
            'random_seed': 42,
        }
        
        # Create experiments
        experiments = [
            # Direct sampling with conservative parameters
            ExperimentConfig(
                name="direct_conservative",
                method=Method.DIRECT,
                temperature=0.3,
                top_p=0.9,
                strict_json=False,
                num_samples=1,
                **base
            ),
            # Direct sampling with creative parameters
            ExperimentConfig(
                name="direct_creative",
                method=Method.DIRECT,
                temperature=1.0,
                top_p=0.98,
                strict_json=False,
                num_samples=1,
                **base
            ),
            # Direct sampling with very creative parameters
            ExperimentConfig(
                name="direct_very_creative",
                method=Method.DIRECT,
                temperature=1.5,
                top_p=0.99,
                strict_json=False,
                num_samples=1,
                **base
            ),
            # Sequence sampling (verbalized method)
            ExperimentConfig(
                name="sequence_standard",
                method=Method.SEQUENCE,
                temperature=0.7,
                top_p=0.9,
                strict_json=True,
                num_samples=5,
                **base
            ),
            # Structure with probability sampling (verbalized method)
            ExperimentConfig(
                name="structure_with_prob_standard",
                method=Method.STRUCTURE_WITH_PROB,
                temperature=0.7,
                top_p=0.9,
                strict_json=True,
                num_samples=5,
                **base
            ),
        ]
        
        print(f"📊 Running {len(experiments)} experiments:")
        for exp in experiments:
            print(f"  - {exp.name} (temp={exp.temperature}, top_p={exp.top_p})")
        
        # Configure pipeline
        model_basename = model.replace("/", "_")
        num_workers = 32 if "claude" in model else 128
        
        config = PipelineConfig(
            experiments=experiments,
            evaluation=EvaluationConfig(metrics=metrics),
            output_base_dir=Path(f"{output_dir}/{model_basename}_{task.value}"),
            skip_existing=True,
            num_workers=num_workers,
        )
        
        # Run pipeline
        pipeline = Pipeline(config)
        pipeline.run_complete_pipeline()
        
        print(f"✅ Completed {model}")
        print(f"📁 Results: {output_dir}/{model_basename}_{task.value}/pipeline_report.html")
    
    print(f"\n🎉 All quick tests completed!")
    print(f"📊 Expected outcome: Verbalized methods should outperform direct sampling")
    print(f"   even with the most creative temperature/top-p settings")

if __name__ == "__main__":
    run_quick_sampling_test() 