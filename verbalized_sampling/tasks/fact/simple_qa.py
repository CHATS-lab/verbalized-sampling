from ..base import BaseTask
from typing import Any, List, Dict
import random
import os
from verbalized_sampling.methods import Method
from verbalized_sampling.methods.factory import PromptFactory


class SimpleQATask(BaseTask):
    """Task for generating random state names."""

    def __init__(self, **kwargs):
        """
        Initialize the SimpleQATask.
        
        Args:
            num_prompts: Number of prompts to randomly sample from the dataset
            random_seed: Random seed for reproducible sampling
        """
        super().__init__(**kwargs)
        self.metadata = {
            "task_type": "simple_qa",
            "total_prompts": 0,
            "num_prompts": self.num_prompts,
            "random_seed": self.random_seed,
            "description": "Generate answer to the SimpleQA dataset from OpenAI."
        }
    
    
    @property
    def task_type(self) -> str:
        return "simple_qa"