"""
Safety task using HarmBench dataset for evaluating AI safety.

This task loads potentially harmful prompts from the HarmBench dataset
and evaluates how well models handle them safely.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from ..base import BaseTask


class SafetyTask(BaseTask):
    """Safety evaluation task using HarmBench safety prompts."""

    def __init__(
        self,
        data_dir: str = "data",
        max_prompts: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize the safety task.

        Args:
            data_dir: Directory containing safety data files
            max_prompts: Maximum number of prompts to load (None for all)
        """
        super().__init__(**kwargs)
        self.data_dir = Path(data_dir)
        self.max_prompts = max_prompts
        self._prompts = None
        self.metadata = {
            "task_type": "safety",
            "total_prompts": 0,
            "num_prompts": self.num_prompts,
            "random_seed": self.random_seed,
            "description": "Safety evaluation using potentially harmful prompts from HarmBench dataset",
            "data_source": "HarmBench"
        }

    @property
    def task_type(self) -> str:
        return "safety"

    def load_prompts(self) -> List[str]:
        """Load safety prompts from HarmBench data."""
        if self._prompts is not None:
            return self._prompts

        # Try different possible file locations
        possible_files = [
            self.data_dir / "safety.jsonl",
            self.data_dir / "safety.txt",
            Path("data/safety.jsonl"),
            Path("data/safety.txt"),
        ]

        prompts = []
        data_file = None

        for file_path in possible_files:
            if file_path.exists():
                data_file = file_path
                break

        if data_file is None:
            raise FileNotFoundError(
                f"Safety data not found in any of: {[str(f) for f in possible_files]}. "
                "Please run 'python scripts/load_harmbench_data.py' to generate the data."
            )

        # Load prompts based on file format
        if data_file.suffix == ".jsonl":
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    prompts.append(data['prompt'])
        elif data_file.suffix == ".txt":
            with open(data_file, 'r', encoding='utf-8') as f:
                prompts = [line.strip() for line in f if line.strip()]

        # Shuffle and limit prompts if specified
        if self.random_seed is not None:
            random.seed(self.random_seed)
            random.shuffle(prompts)

        if self.max_prompts is not None:
            prompts = prompts[:self.max_prompts]

        self._prompts = prompts
        self.metadata["total_prompts"] = len(prompts)
        print(f"Loaded {len(prompts)} safety prompts from {data_file}")

        return prompts

    def get_prompts(self, num_prompts: Optional[int] = None) -> List[str]:
        """
        Get safety prompts for evaluation.

        Args:
            num_prompts: Number of prompts to return (None for all loaded)

        Returns:
            List of safety prompt strings
        """
        all_prompts = self.load_prompts()

        if num_prompts is None:
            return all_prompts

        return all_prompts[:num_prompts]

    def get_prompt_metadata(self) -> Dict[str, Any]:
        """Get metadata about the safety prompts."""
        prompts = self.load_prompts()

        return {
            "total_prompts": len(prompts),
            "data_source": "HarmBench",
            "task_type": "safety_evaluation",
            "evaluation_framework": "StrongReject",
            "description": "Safety evaluation using potentially harmful prompts from HarmBench dataset"
        }

    def validate_response(self, response: str) -> bool:
        """
        Validate that a response is properly formatted.

        For safety tasks, we accept any string response as the safety
        evaluation will be done by the SafetyEvaluator.
        """
        return isinstance(response, str) and len(response.strip()) > 0

    def get_task_description(self) -> str:
        """Get a human-readable description of the task."""
        return (
            "Safety evaluation task using HarmBench dataset. "
            "Models are presented with potentially harmful prompts and "
            "evaluated on their ability to refuse unsafe requests while "
            "remaining helpful for legitimate use cases."
        )