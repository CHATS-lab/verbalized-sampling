from verbalized_sampling.pipeline import run_quick_comparison
from verbalized_sampling.tasks import Task
from verbalized_sampling.prompts import Method
from pathlib import Path
import sys

# Quick comparison
results = run_quick_comparison(
    task=Task.STATE_NAME,
    # methods=[Method.DIRECT, Method.STRUCTURE, Method.STRUCTURE_WITH_PROB],
    methods=[Method.STRUCTURE_WITH_PROB],
    model_name="openai/gpt-4.1",
    metrics=["diversity"], # diversity, ttct, creativity_index, length
    output_dir=Path("comparison_results"),
    num_responses=200,
    rerun=True,
)

# Results will include:
# - Generated responses for each method
# - Evaluation results for all metrics
# - Comparison plots
# - HTML summary report
