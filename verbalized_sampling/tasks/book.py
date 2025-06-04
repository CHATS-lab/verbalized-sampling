from .base import BaseTask
from typing import Any, List, Dict
import random
import os
from verbalized_sampling.prompts import Method
from verbalized_sampling.prompts.factory import PromptFactory

class BookTask(BaseTask):
    """Task for generating book/novel continuations from prompts."""
    
    def __init__(self, **kwargs):
        """
        Initialize the BookTask.
        
        Args:
            sample_size: Number of prompts to randomly sample from the dataset
            random_seed: Random seed for reproducible sampling
        """
        super().__init__(**kwargs)
    
    def get_metadata(self) -> dict:
        """Get task metadata."""
        return {
            "task_type": "book",
            "total_prompts": len(self._prompts) if self._prompts else 0,
            "sample_size": self.sample_size,
            "random_seed": self.random_seed,
            "description": "Novel/book continuation task with prompts from literary works"
        }