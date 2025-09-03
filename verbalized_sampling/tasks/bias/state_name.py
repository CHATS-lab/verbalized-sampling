from ..base import BaseTask
from typing import Any, List, Dict, Union
import random
import os
from verbalized_sampling.methods import Method
from verbalized_sampling.methods.factory import PromptFactory


class StateNameTask(BaseTask):
    """Task for generating random state names."""

    def __init__(self, **kwargs):
        """
        Initialize the StateNameTask.
        
        Args:
            num_prompts: Number of prompts to randomly sample from the dataset
            random_seed: Random seed for reproducible sampling
        """
        super().__init__(**kwargs)
        self.metadata = {
            "task_type": "state_name",
            "total_prompts": 0,
            "num_prompts": self.num_prompts,
            "random_seed": self.random_seed,
            "description": "Generate a state name randomly."
        }
    
    
    @property
    def task_type(self) -> str:
        return "state_name"

    


    # def get_prompt(self, num_samples: int = 1) -> str:
    #     """Get the prompt for the task."""
    #     FORMAT_SYSTEM_PROMPT_NON_SAMPLING = dedent("""
    #     You are simulating answers to a given question.
    #     """).strip()

    #     FORMAT_SYSTEM_PROMPT_SAMPLING = dedent("""
    #     You are simulating answers to a given question.
    #     Randomly generate {num_samples} plausible and diverse responses to the user's question, also providing the empirical probability of each response.
    #     """).strip()

    #     FORMAT_USER_PROMPT = dedent("""
    #     Question: {question}

    #     Return the output in JSON format with the key "responses", which should be a list of dictionaries. Each dictionary must include:
    #     - "text": the {name_type} named in the response
    #     - "probability": the empirical probability of that response (value between 0 and 1)

    #     Only output the JSON object—no additional explanation or text.
    #     """).strip()

    #     if self.format == "direct":
    #         return FORMAT_SYSTEM_PROMPT_NON_SAMPLING + FORMAT_USER_PROMPT
    #     else:
    #         return FORMAT_SYSTEM_PROMPT_SAMPLING + FORMAT_USER_PROMPT
    