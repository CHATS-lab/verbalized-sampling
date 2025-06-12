"""
Example script for running state name experiments.
"""

from verbalized_sampling.pipeline import run_quick_comparison
from verbalized_sampling.tasks import Task
from verbalized_sampling.methods import Method
from pathlib import Path

def run_state_name_example():
    """Run a state name experiment example."""
    NUM_RESPONSES = 5

    MODEL_PARAMS = {
        "temperature": 1.0,
        "top_p": 1.0,
    }

    # Structure without probability
    results = run_quick_comparison(
        task=Task.STATE_NAME,
        methods=[Method.STRUCTURE],
        model_name="google/gemini-2.5-flash-preview-05-20",
        metrics=["response_count"],
        output_dir=Path("comparison_results/sequence"),
        num_responses=NUM_RESPONSES,
        num_samples=5,
        sample_size=1,
        strict_json=True,
        rerun=True,
        **MODEL_PARAMS
    )

if __name__ == "__main__":
    run_state_name_example() 