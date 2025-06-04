from .base import BaseTask
from typing import Any, List
import random
import os
from verbalized_sampling.prompts import Method
from verbalized_sampling.prompts.factory import PromptFactory

class SpeechTask(BaseTask):
    """Task for generating speeches from starting sentence prompts."""
    
    def __init__(self, sample_size: int = 1, random_seed: int = 42):
        """
        Initialize the SpeechTask.
        
        Args:
            sample_size: Number of prompts to randomly sample from the dataset
            random_seed: Random seed for reproducible sampling
        """
        self.sample_size = sample_size
        self.random_seed = random_seed
        self._prompts = None
        self._load_prompts()
    
    def _load_prompts(self) -> None:
        """Load prompts from the speech.txt data file."""
        data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'speech.txt')
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                all_prompts = [line.strip() for line in f if line.strip()]
            
            # Set random seed for reproducibility
            random.seed(self.random_seed)
            
            # Sample prompts if sample_size is specified and less than total
            if self.sample_size > 0 and self.sample_size < len(all_prompts):
                self._prompts = random.sample(all_prompts, self.sample_size)
            else:
                self._prompts = all_prompts
            
            # Reset random seed to avoid affecting other operations
            random.seed()
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Speech prompts file not found at {data_path}")
        except Exception as e:
            raise Exception(f"Error loading speech prompts: {e}")
    
    def get_prompts(self) -> List[str]:
        """Get all sampled prompts."""
        return self._prompts.copy()
    
    def get_prompt(self, method: Method, num_samples: int = 1, prompt_index: int = 0) -> str:
        """
        Get a specific prompt or use it with PromptFactory for structured methods.
        
        Args:
            method: The sampling method
            num_samples: Number of samples for structured methods
            prompt_index: Index of the prompt to use (default: 0)
        """
        if not self._prompts:
            raise ValueError("No prompts loaded")
        
        if prompt_index >= len(self._prompts):
            raise ValueError(f"Prompt index {prompt_index} out of range. Available prompts: {len(self._prompts)}")
        
        selected_prompt = self._prompts[prompt_index]
        
        # For direct method, return the prompt as-is
        if method == Method.DIRECT:
            return selected_prompt
        
        # For structured methods, extract the actual prompt content
        # The prompts in speech.txt start with "Please write a speech starting with the following sentence: "
        prompt_prefix = "Please write a speech starting with the following sentence: "
        if selected_prompt.startswith(prompt_prefix):
            actual_prompt = selected_prompt[len(prompt_prefix):]
            
            # Create a custom message for structured methods
            if method == Method.SEQUENCE:
                from verbalized_sampling.prompts.prompt import SEQUENCE_PROMPT
                system_prompt = SEQUENCE_PROMPT.format(num_samplings=num_samples)
                return f"System: {system_prompt}\nUser: Write a speech starting with the following sentence: {actual_prompt}"
            elif method == Method.STRUCTURE:
                from verbalized_sampling.prompts.prompt import FORMAT_WITHOUT_PROBABILITY_PROMPT
                system_prompt = FORMAT_WITHOUT_PROBABILITY_PROMPT.format(num_samplings=num_samples)
                return f"System: {system_prompt}\nUser: Write a speech starting with the following sentence: {actual_prompt}"
            elif method == Method.STRUCTURE_WITH_PROB:
                from verbalized_sampling.prompts.prompt import FORMAT_WITH_PROBABILITY_PROMPT
                system_prompt = FORMAT_WITH_PROBABILITY_PROMPT.format(num_samplings=num_samples)
                return f"System: {system_prompt}\nUser: Write a speech starting with the following sentence: {actual_prompt}"
        
        # Fallback: return the original prompt
        return selected_prompt
    
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