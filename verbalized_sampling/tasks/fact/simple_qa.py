import os
import json
import random
from ..base import BaseTask
from textwrap import dedent
from typing import Any, List, Dict
from rich.progress import Progress
from verbalized_sampling.prompts import Method
from verbalized_sampling.prompts.factory import PromptFactory

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List
import json
from rich.progress import Progress
from verbalized_sampling.prompts import (
    PromptFactory, 
    Method
)
from verbalized_sampling.llms import BaseLLM
from verbalized_sampling.prompts.schema import get_schema


class SimpleQATask(BaseTask):
    """Task for generating random state names."""

    def __init__(self, **kwargs):
        """
        Initialize the SimpleQATask.
        
        Args:
            sample_size: Number of prompts to randomly sample from the dataset
            random_seed: Random seed for reproducible sampling
        """
        super().__init__(**kwargs)
        self.metadata = {
            "task_type": "simple_qa",
            "total_prompts": 0,
            "sample_size": self.sample_size,
            "random_seed": self.random_seed,
            "description": "Generate answer to the SimpleQA dataset from OpenAI."
        }
    
    
    @property
    def task_type(self) -> str:
        return "simple_qa"