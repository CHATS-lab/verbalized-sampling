"""Unified math task implementation for all math datasets."""

from ..base import BaseTask
from typing import List, Dict, Union
import random
import datasets


class MathTask(BaseTask):
    """
    Unified task for math problem solving across multiple datasets.

    Supports: MATH, AIME, AMC, MINERVA, OLYMPIAD_BENCH datasets.
    """

    SUPPORTED_DATASETS = {
        "math": "MATH dataset - LaTeX math problems with string answers",
        "aime": "AIME competition problems with string answers",
        "amc": "AMC competition problems with numeric answers",
        "minerva": "Minerva physics/advanced problems with list answers",
        "olympiad_bench": "Olympiad competition problems with list answers"
    }

    def __init__(self, dataset: str = "math", **kwargs):
        """
        Initialize the MathTask.

        Args:
            dataset: Which math dataset to use ("math", "aime", "amc", "minerva", "olympiad_bench")
            num_prompts: Number of prompts to randomly sample from the dataset
            random_seed: Random seed for reproducible sampling
        """
        if dataset not in self.SUPPORTED_DATASETS:
            raise ValueError(f"Dataset '{dataset}' not supported. Available: {list(self.SUPPORTED_DATASETS.keys())}")

        super().__init__(**kwargs)
        self.dataset = dataset

        # Find the data path dynamically
        import pathlib
        current_file = pathlib.Path(__file__)
        project_root = current_file.parent.parent.parent.parent  # Go up to verbalize-sampling root
        self.data_path = project_root / "data" / "math" / dataset

        # Load the dataset
        self._load_dataset()

        self.metadata = {
            "task_type": f"math_{dataset}",
            "dataset": dataset,
            "total_prompts": len(self.problems),
            "num_prompts": self.num_prompts,
            "random_seed": self.random_seed,
            "description": f"Math problem solving using {self.SUPPORTED_DATASETS[dataset]}"
        }

    def _load_dataset(self):
        """Load the specified math dataset."""
        try:
            ds = datasets.load_from_disk(self.data_path)
            self.problems = []

            for i, item in enumerate(ds):
                problem_data = {
                    "id": i,
                    "problem": item["problem"],
                    "answer": item["answer"],
                    "dataset_type": self.dataset
                }

                # Add difficulty if available
                if "difficulty" in item:
                    problem_data["difficulty"] = item["difficulty"]

                self.problems.append(problem_data)

        except Exception as e:
            raise RuntimeError(f"Failed to load dataset '{self.dataset}': {e}")

    def get_prompt(self) -> List[Union[List[Dict[str, str]], str]]:
        """Get prompts for the math task."""
        # Sample problems
        random.seed(self.random_seed)
        if self.num_prompts <= len(self.problems):
            sampled_problems = random.sample(self.problems, self.num_prompts)
        else:
            # If we need more prompts than problems, sample with replacement
            sampled_problems = random.choices(self.problems, k=self.num_prompts)

        prompts = []
        for problem in sampled_problems:
            # Format the problem - prompt template will be applied during inference
            question = problem['problem']
            prompt_text = f"Question: {question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."

            # Store problem metadata for evaluation
            if not hasattr(self, '_problem_metadata'):
                self._problem_metadata = {}

            prompt_id = len(prompts)
            self._problem_metadata[prompt_id] = {
                "answer": problem["answer"],
                "dataset_type": self.dataset,
                "problem_id": problem["id"],
                "difficulty": problem.get("difficulty")
            }

            # Return as message format for proper chat handling
            prompts.append(prompt_text)

        return prompts


    @property
    def task_type(self) -> str:
        return f"math_{self.dataset}"

    def get_problem_metadata(self, prompt_id: int) -> Dict:
        """Get metadata for a specific problem by prompt ID."""
        if not hasattr(self, '_problem_metadata'):
            return {}
        return self._problem_metadata.get(prompt_id, {})