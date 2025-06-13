from verbalized_sampling.pipeline import run_quick_comparison
from verbalized_sampling.tasks import Task
from verbalized_sampling.methods import Method
from pathlib import Path

NUM_RESPONSES = 5
NUM_SAMPLES = 5

# Quick comparison
results = run_quick_comparison(
    task=Task.POEM,
    methods=[Method.SEQUENCE],
    # methods=[Method.MULTI_TURN],
    model_name="openai/gpt-4.1",
    # model_name="anthropic/claude-sonnet-4",
    metrics=["diversity"],
    output_dir=Path("comparison_results/poem"),
    num_responses=NUM_RESPONSES,
    num_samples=NUM_SAMPLES,
    num_prompts=1,
    rerun=True,
    # strict_json=True,
    # skip_existing=True
)


# results = run_quick_comparison(
#     task=Task.POEM,
#     methods=[Method.STRUCTURE],
#     model_name="openai/gpt-4.1",
#     metrics=["quality"],
#     output_dir=Path("comparison_results/poem"),
#     num_responses=NUM_RESPONSES,
#     num_samples=NUM_SAMPLES,
#     num_prompts=1,
#     strict_json=True,
#     rerun=True,
# )

# Results will include:
# - Generated responses for each method
# - Evaluation results for all metrics
# - Comparison plots
# - HTML summary report
