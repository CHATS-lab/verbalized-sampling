from verbalized_sampling.pipeline import run_quick_comparison
from verbalized_sampling.tasks import Task
from verbalized_sampling.methods import Method
from pathlib import Path

NUM_RESPONSES = 5
NUM_SAMPLES = 5

# Quick comparison
results = run_quick_comparison(
    task=Task.POEM,
    methods=[Method.DIRECT, Method.SEQUENCE, Method.MULTI_TURN,Method.STRUCTURE_WITH_PROB],
    # methods=[Method.MULTI_TURN],
    # model_name="openai/gpt-4.1",
    model_name="anthropic/claude-4-sonnet",
    metrics=["diversity", "quality", "creative_writing_v3"],
    output_dir=Path("comparison_results/claude_4_sonnet"),
    num_responses=NUM_RESPONSES,
    num_samples=NUM_SAMPLES,
    num_prompts=10,
    rerun=True,
    strict_json=True,
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
