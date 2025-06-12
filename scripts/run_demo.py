from verbalized_sampling.pipeline import run_quick_comparison
from verbalized_sampling.tasks import Task
from verbalized_sampling.prompts import Method
from pathlib import Path

NUM_RESPONSES = 5
NUM_SAMPLES = 5

# Quick comparison
# results = run_quick_comparison(
#     task=Task.POEM,
#     methods=[Method.DIRECT], # Method.STRUCTURE, Method.STRUCTURE_WITH_PROB, Method.SEQUENCE
#     model_name="openai/gpt-4.1",
#     metrics=["diversity"],
#     output_dir=Path("comparison_results/poem"),
#     num_responses=NUM_RESPONSES,
#     sample_size=1,
#     rerun=True
# )


results = run_quick_comparison(
    task=Task.POEM,
    methods=[Method.STRUCTURE],
    model_name="openai/gpt-4.1",
    metrics=["diversity"],
    output_dir=Path("comparison_results/poem"),
    num_responses=NUM_RESPONSES,
    num_samples=NUM_SAMPLES,
    sample_size=1,
    strict_json=True,
    rerun=True,
)

# Results will include:
# - Generated responses for each method
# - Evaluation results for all metrics
# - Comparison plots
# - HTML summary report
