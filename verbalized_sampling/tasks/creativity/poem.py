from ..base import BaseTask
from typing import Any, List, Dict
import random
import os
from verbalized_sampling.prompts import Method
from verbalized_sampling.prompts.factory import PromptFactory

class PoemTask(BaseTask):
    """Task for generating poems from starting line prompts."""
    
    def __init__(self, **kwargs):
        """
        Initialize the PoemTask.
        
        Args:
            num_prompts: Number of prompts to randomly sample from the dataset
            random_seed: Random seed for reproducible sampling
        """
        super().__init__(**kwargs)
        self.metadata = {
            "task_type": "poem",
            "total_prompts": 0,
            "num_prompts": self.num_prompts,
            "random_seed": self.random_seed,
            "description": "Poetry generation task with starting line prompts"
        }
    
    @property
    def task_type(self) -> str:
        return "poem" 