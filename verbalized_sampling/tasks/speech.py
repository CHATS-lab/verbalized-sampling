from .base import BaseTask
from typing import Any, List, Dict
import random
import os
from verbalized_sampling.prompts import Method
from verbalized_sampling.prompts.factory import PromptFactory

class SpeechTask(BaseTask):
    """Task for generating speeches from starting sentence prompts."""
    
    def __init__(self, **kwargs):
        """
        Initialize the SpeechTask.
        
        Args:
            sample_size: Number of prompts to randomly sample from the dataset
            random_seed: Random seed for reproducible sampling
        """
        super().__init__(**kwargs)
    
    def parse_response(self, method: Method, response: str) -> Any:
        """Parse the model's response based on the method."""
        if method in [Method.STRUCTURE, Method.STRUCTURE_WITH_PROB]:
            # Try to parse as JSON for structured methods
            import json
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
    
    def get_metadata(self) -> dict:
        """Get task metadata."""
        return {
            "task_type": "speech",
            "total_prompts": len(self._prompts) if self._prompts else 0,
            "sample_size": self.sample_size,
            "random_seed": self.random_seed,
            "description": "Speech generation task with starting sentence prompts"
        } 