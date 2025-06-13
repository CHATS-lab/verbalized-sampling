from verbalized_sampling.pipeline import run_quick_comparison
from verbalized_sampling.tasks import Task
from verbalized_sampling.prompts import Method
from pathlib import Path
import sys

NUM_RESPONSES = 5
NUM_SAMPLES = 5
SAMPLE_SIZE = 500 # 4326 samples

MODEL_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.9,
}

# # Direct (Baseline)
# results = run_quick_comparison(
#     task=Task.SIMPLE_QA,
#     methods=[Method.DIRECT],
#     model_name="openai/gpt-4.1", # google/gemini-2.5-flash-preview, openai/gpt-4.1
#     metrics=["factuality"], # factuality, diversity, ttct, creativity_index, length
#     output_dir=Path("comparison_results/direct"),
#     num_responses=NUM_RESPONSES,
#     num_samples=1,
#     sample_size=SAMPLE_SIZE, # how many samples from the prompt dataset to generate
#     rerun=True,
#     create_backup=False,
#     strict_json=False,
#     **MODEL_PARAMS
# )

# # Structure without probability
# results = run_quick_comparison(
#     task=Task.SIMPLE_QA,
#     methods=[Method.STRUCTURE], # Method.STRUCTURE, Method.STRUCTURE_WITH_PROB
#     model_name="openai/gpt-4.1", # google/gemini-2.5-flash-preview, openai/gpt-4.1
#     metrics=["factuality"], # diversity, ttct, creativity_index, length
#     output_dir=Path("comparison_results/structure"),
#     num_responses=NUM_RESPONSES, 
#     num_samples=NUM_SAMPLES, # how many times to sample from the model
#     sample_size=SAMPLE_SIZE, # how many samples from the prompt dataset to generate
#     strict_json=True,
#     rerun=True,
#     **MODEL_PARAMS
# )


# Structure with probabilitys
results = run_quick_comparison(
    task=Task.SIMPLE_QA,
    methods=[Method.STRUCTURE_WITH_PROB], # Method.STRUCTURE, Method.STRUCTURE_WITH_PROB
    model_name="openai/gpt-4.1", # google/gemini-2.5-flash-preview, openai/gpt-4.1
    metrics=["factuality"], # diversity, ttct, creativity_index, length
    output_dir=Path("comparison_results/structure_with_prob"),
    num_responses=NUM_RESPONSES,
    num_samples=NUM_SAMPLES, # how many times to sample from the model
    sample_size=SAMPLE_SIZE, # how many samples from the prompt dataset to generate
    strict_json=True,
    rerun=True,
    **MODEL_PARAMS
)




# Results will include:
# - Generated responses for each method
# - Evaluation results for all metrics
# - Comparison plots
# - HTML summary report
