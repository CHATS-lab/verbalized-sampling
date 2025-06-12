from ..base import BaseTask
from typing import Any, List, Dict
import random
import os
from verbalized_sampling.methods import Method
from verbalized_sampling.methods.factory import PromptFactory

class JokeTask(BaseTask):
    """Task for generating jokes from prompts."""
    
    def __init__(self, **kwargs):
        """
        Initialize the JokeTask.
        
        Args:
            num_prompts: Number of prompts to randomly sample from the dataset
            random_seed: Random seed for reproducible sampling
        """
        super().__init__(**kwargs)
        self.metadata = {
            "task_type": "joke",
            "total_prompts": 0,
            "num_prompts": self.num_prompts,
            "random_seed": self.random_seed,
            "description": "Joke generation task with various prompts"
        }
    
    @property
    def task_type(self) -> str:
        return "joke"
    
    def get_metadata(self) -> dict:
        """Get task metadata."""
        return self.metadata
