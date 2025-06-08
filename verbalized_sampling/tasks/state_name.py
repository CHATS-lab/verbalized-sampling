import os
import json
import random
from .base import BaseTask
from textwrap import dedent
from typing import Any, List, Dict
from verbalized_sampling.prompts import Method
from verbalized_sampling.prompts.factory import PromptFactory


class StateNameTask(BaseTask):
    """Task for generating random state names."""

    def __init__(self, **kwargs):
        """
        Initialize the StateNameTask.
        
        Args:
            sample_size: Number of prompts to randomly sample from the dataset
            random_seed: Random seed for reproducible sampling
        """
        super().__init__(**kwargs)
        self._metadata = {
            "task_type": "state_name",
            "total_prompts": 0,
            "sample_size": self.sample_size,
            "random_seed": self.random_seed,
            "description": "Generate a state name randomly."
        }
    
    def parse_response(self, method: Method, response: str) -> Any:
        """Parse the model's response based on the method."""
        if method in [Method.STRUCTURE, Method.STRUCTURE_WITH_PROB]:
            # Try to parse as JSON for structured methods
            try:
                # Clean up response if it contains markdown code blocks
                if "```json" in response:
                    start = response.find("```json") + 7
                    end = response.find("```", start)
                    if end != -1:
                        response = response[start:end].strip()
                elif "```" in response:
                    start = response.find("```") + 3
                    end = response.rfind("```")
                    if end != -1 and end > start:
                        response = response[start:end].strip()
                
                # Try to parse as JSON
                parsed = json.loads(response)
                if isinstance(parsed, dict) and "responses" in parsed:
                    return parsed["responses"]
                return parsed
            except json.JSONDecodeError:
                # If JSON parsing fails, return the raw response
                return response
        
        # For direct and sequence methods, return as-is
        return response
    
    
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
    