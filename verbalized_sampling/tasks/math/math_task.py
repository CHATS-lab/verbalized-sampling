"""Unified math task implementation for all math datasets."""

from ..base import BaseTask
from typing import Any, List, Dict, Union
import random
import os
import datasets
from verbalized_sampling.methods import Method
from verbalized_sampling.methods.factory import PromptFactory


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
        self.data_path = f"/Users/simonyu/local/local_orby/verbalize-sampling/data/math/{dataset}"

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
            # Create instruction emphasizing boxed answers for better extraction
            instruction = self._get_instruction_for_dataset()

            # Format the problem
            problem_text = f"{instruction}\n\nProblem: {problem['problem']}"

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

            prompts.append([{"role": "user", "content": problem_text}])

        return prompts

    def _get_instruction_for_dataset(self) -> str:
        """Get dataset-specific instructions."""
        base_instruction = "Solve the following math problem step by step."

        if self.dataset in ["math", "aime"]:
            return (f"{base_instruction} Show your work clearly and provide your final answer "
                   "in LaTeX format enclosed in \\boxed{{}}. For example, if the answer is 42, "
                   "write \\boxed{{42}}.")

        elif self.dataset == "amc":
            return (f"{base_instruction} Show your work clearly and provide your final numerical "
                   "answer enclosed in \\boxed{{}}. For example, if the answer is 42, "
                   "write \\boxed{{42}}.")

        elif self.dataset in ["minerva", "olympiad_bench"]:
            return (f"{base_instruction} Show your work clearly and provide your final answer "
                   "enclosed in \\boxed{{}}. Express your answer as clearly and simply as possible.")

        return base_instruction

    @property
    def task_type(self) -> str:
        return f"math_{self.dataset}"

    def get_problem_metadata(self, prompt_id: int) -> Dict:
        """Get metadata for a specific problem by prompt ID."""
        if not hasattr(self, '_problem_metadata'):
            return {}
        return self._problem_metadata.get(prompt_id, {})