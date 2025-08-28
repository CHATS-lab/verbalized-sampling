from ..base import BaseTask
from typing import Any, List, Dict
import random
import os
from verbalized_sampling.methods import Method
from verbalized_sampling.methods.factory import PromptFactory


class SyntheticNegativeTask(BaseTask):
    """Task for generating negative synthetic data."""

    def __init__(self, **kwargs):
        """
        Initialize the SyntheticNegativeTask.
        
        Args:
            num_prompts: Number of prompts to randomly sample from the dataset
            random_seed: Random seed for reproducible sampling
        """
        super().__init__(**kwargs)
        self.metadata = {
            "task_type": "synthetic_negative",
            "total_prompts": 0,
            "num_prompts": self.num_prompts,
            "random_seed": self.random_seed,
            "description": "Generate negative synthetic data"
        }
    
    
    @property
    def task_type(self) -> str:
        return "synthetic_negative"