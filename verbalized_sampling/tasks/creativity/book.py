from ..base import BaseTask
from typing import Any, List, Dict
import random
import os
from verbalized_sampling.methods import Method
from verbalized_sampling.methods.factory import PromptFactory

class BookTask(BaseTask):
    """Task for generating book/novel continuations from prompts."""
    
    def __init__(self, **kwargs):
        """
        Initialize the BookTask.
        
        Args:
            num_prompts: Number of prompts to randomly sample from the dataset
            random_seed: Random seed for reproducible sampling
        """
        super().__init__(**kwargs)
        self.metadata = {
            "task_type": "book",
            "total_prompts": 0,
            "num_prompts": self.num_prompts,
            "random_seed": self.random_seed,
            "description": "Novel/book continuation task with prompts from literary works"
        }

    @property
    def task_type(self) -> str:
        return "book" 
    
    # def get_metadata(self) -> dict:
    #     """Get task metadata."""
    #     return {
    #         "task_type": "book",
    #         "total_prompts": len(self._prompts) if self._prompts else 0,
    #         "num_prompts": self.num_prompts,
    #         "random_seed": self.random_seed,
    #         "description": "Novel/book continuation task with prompts from literary works"
    #     }