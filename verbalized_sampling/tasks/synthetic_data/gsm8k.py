from ..base import BaseTask
from typing import Any, List, Dict
import random
import os
from verbalized_sampling.methods import Method
from verbalized_sampling.methods.factory import PromptFactory


class GSM8KTask(BaseTask):
    """Task for generating synthetic data to the GSM8K dataset."""

    def __init__(self, **kwargs):
        """
        Initialize the GSM8KTask.
        
        Args:
            num_prompts: Number of prompts to randomly sample from the dataset
            random_seed: Random seed for reproducible sampling
        """
        super().__init__(**kwargs)
        self.metadata = {
            "task_type": "gsm8k",
            "total_prompts": 0,
            "num_prompts": self.num_prompts,
            "random_seed": self.random_seed,
            "description": "Generate synthetic data to the GSM8K dataset"
        }
    
    
    @property
    def task_type(self) -> str:
        return "gsm8k"