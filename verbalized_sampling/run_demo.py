from verbalized_sampling.pipeline import run_quick_comparison
from verbalized_sampling.tasks import Task
from verbalized_sampling.prompts import Method
from pathlib import Path

# Quick comparison
results = run_quick_comparison(
    task=Task.POEM,
    methods=[Method.DIRECT, Method.STRUCTURE, Method.STRUCTURE_WITH_PROB, Method.SEQUENCE],
    # methods=[Method.SEQUENCE],
    model_name="openai/gpt-4.1",
    metrics=["diversity", "length", "quality"],
    output_dir=Path("comparison_results"),
    num_responses=50,
    num_samples=5,
    rerun=True,
)

# Results will include:
# - Generated responses for each method
# - Evaluation results for all metrics
# - Comparison plots
# - HTML summary report
