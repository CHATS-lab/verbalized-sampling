from .base import BaseTask
from typing import Any
from verbalized_sampling.prompts import Method
from verbalized_sampling.prompts.factory import PromptFactory

class CreativeStoryTask(BaseTask):
    """Task for generating creative stories."""
    
    def get_prompt(self, method: Method, num_samples: int = 1) -> str:
        """Get the prompt for the task."""
        return PromptFactory.get_prompt(
            "creative_story", 
            method, 
            num_samplings=num_samples)
    
    def parse_response(self, method: Method, response: str) -> Any:
        """Parse the model's response."""
        return response