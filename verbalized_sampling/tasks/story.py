from .base import BaseTask
from typing import Any, List, Dict
from verbalized_sampling.prompts import Method
from verbalized_sampling.prompts.factory import PromptFactory

class CreativeStoryTask(BaseTask):
    """Task for generating creative stories."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metadata = {
            "task_type": "creative_story",
            "total_prompts": 0,
            "sample_size": self.sample_size,
            "random_seed": self.random_seed,
            "description": "Creative story generation task with prompts from literary works"
        }
    
    def parse_response(self, method: Method, response: str) -> Any:
        """Parse the model's response."""
        return response
    
    @property
    def metadata(self) -> dict:
        """Get task metadata."""
        return self._metadata
    
    @property
    def task_type(self) -> str:
        return "creative_story"