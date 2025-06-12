from verbalized_sampling.pipeline import run_quick_comparison
from verbalized_sampling.tasks import Task
from verbalized_sampling.prompts import Method
from pathlib import Path
import sys

NUM_RESPONSES = 5
NUM_SAMPLES = 5

MODEL_PARAMS = {
    "temperature": 1,
    "top_p": 1.0,
}


# # Direct (Baseline)
# results = run_quick_comparison(
#     task=Task.STATE_NAME,
#     methods=[Method.DIRECT],
#     model_name="google/gemini-2.5-flash-preview", # google/gemini-2.5-flash-preview, openai/gpt-4.1
#     metrics=["response_count"], # diversity, ttct, creativity_index, length
#     output_dir=Path("comparison_results/direct"),
#     num_responses=NUM_RESPONSES,
#     sample_size=1, # how many samples from the prompt dataset to generate
#     strict_json=False,
#     rerun=True,
#     **MODEL_PARAMS
# )

# Structure without probability
results = run_quick_comparison(
    task=Task.STATE_NAME,
    methods=[Method.STRUCTURE], # Method.STRUCTURE, Method.STRUCTURE_WITH_PROB
    model_name="google/gemini-2.5-flash-preview", # google/gemini-2.5-flash-preview, openai/gpt-4.1
    metrics=["response_count"], # diversity, ttct, creativity_index, length
    output_dir=Path("comparison_results/sequence"),
    num_responses=NUM_RESPONSES,
    num_samples=NUM_SAMPLES, # how many times to sample from the model
    sample_size=1, # how many samples from the prompt dataset to generate
    strict_json=True,
    rerun=True,
    **MODEL_PARAMS
)


# # Structure with probabilitys
# results = run_quick_comparison(
#     task=Task.STATE_NAME,
#     methods=[Method.STRUCTURE_WITH_PROB], # Method.STRUCTURE, Method.STRUCTURE_WITH_PROB
#     model_name="google/gemini-2.0-flash-001", # google/gemini-2.5-flash-preview, openai/gpt-4.1
#     metrics=["response_count"], # diversity, ttct, creativity_index, length
#     output_dir=Path("comparison_results/sequence_with_prob"),
#     num_responses=NUM_RESPONSES,
#     num_samples=NUM_SAMPLES, # how many times to sample from the model
#     sample_size=1, # how many samples from the prompt dataset to generate
#     strict_json=True,
#     rerun=True,
#     **MODEL_PARAMS
# )




# Results will include:
# - Generated responses for each method
# - Evaluation results for all metrics
# - Comparison plots
# - HTML summary report
