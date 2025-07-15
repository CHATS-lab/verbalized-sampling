from ..base import BaseTask
from typing import Any, List
import random
import os

class PoemTask(BaseTask):
    """Task for generating poems from starting line prompts."""
    
    def __init__(self, **kwargs):
        """
        Initialize the PoemTask.
        
        Args:
            num_prompts: Number of prompts to randomly sample from the dataset
            random_seed: Random seed for reproducible sampling
        """
        super().__init__(**kwargs)
        self.metadata = {
            "task_type": "poem",
            "total_prompts": 0,
            "num_prompts": self.num_prompts,
            "random_seed": self.random_seed,
            "description": "Poetry generation task with starting line prompts"
        }
        
    def parse_response(self, response: str) -> Any:
        """Parse the model's response."""
        return response
    
    @property
    def metadata(self) -> dict:
        """Get task metadata."""
        return self._metadata
    
    @metadata.setter
    def metadata(self, value: dict):
        """Set task metadata."""
        self._metadata = value
    
    @property
    def task_type(self) -> str:
        return "poem"
    
    def get_prompts(self) -> List[str]:
        """Load and return poem prompts from the data file."""
        # Get the path to the data file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, "..", "..", "..", "data", "poem_titles.txt")
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Clean up the lines and create prompts
            prompts = []
            for line in lines:
                line = line.strip()
                if line:  # Skip empty lines
                    prompts.append(line)
            
            # Update metadata with total prompts
            self.metadata["total_prompts"] = len(prompts)
            
            # Sample prompts if needed
            if self.all_possible:
                return prompts
            else:
                random.seed(self.random_seed)
                return random.sample(prompts, min(self.num_prompts, len(prompts)))
                
        except FileNotFoundError:
            print(f"Warning: Could not find poem data file at {data_path}")
            # Fallback to a few default prompts
            default_prompts = [
                "I stand alone in darkness,",
                "Yellow moon peeps at me",
                "We met for supper in your flat-bottomed boat.",
                "Love and Sleep",
                "From childhood's hour I have not been"
            ]
            return random.sample(default_prompts, min(self.num_prompts, len(default_prompts)))