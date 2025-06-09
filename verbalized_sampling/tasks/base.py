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

class BaseTask(ABC):
    """Base class for all tasks."""
    
    def __init__(self,
                 model: BaseLLM,
                 method: Method,
                 num_responses: int = 3,
                 num_samples: int = 5,
                 sample_size: int = 1,
                 random_seed: int = 42,
                 all_possible: bool = False,
                 strict_json: bool = False,
                 ):
        self.model = model
        self.method = method
        self.num_responses = num_responses
        self.num_samples = num_samples
        self.sample_size = sample_size
        self.random_seed = random_seed
        self.all_possible = all_possible
        self.strict_json = strict_json
        
    def get_prompt(self) -> List[List[Dict[str, str]]]:
        """Get the prompt for the task."""
        return PromptFactory.get_prompt(
            self.task_type, 
            self.method, 
            num_samplings=self.num_samples,
            sample_size=self.sample_size,
            random_seed=self.random_seed,
            all_possible=self.all_possible,
            strict_json=self.strict_json
        )
    
    def parse_response(self, response: str) -> Any:
        """Parse the model's response."""
        try:
            parsed = json.loads(response)
            if isinstance(parsed, dict):
                parsed = parsed["responses"]
            return parsed
        except Exception as e:
            print(f"Error parsing response: {e}")
            return response
    
    def run(
        self,
        progress: Progress = None,
        task_id: int = None,
    ) -> List[Any]:
        """Run the task with the given model."""
        
        prompts = [prompt for prompt in self.get_prompt() for _ in range(self.num_responses)]
        results = self.model.chat(prompts, schema=get_schema(self.method))
        parsed_results = []
        for result in results:
            parsed = self.parse_response(result)
            if parsed is not None:
                if isinstance(parsed, list):
                    parsed_results.extend(parsed)
                else:
                    parsed_results.append(parsed)
            else:
                parsed_results.append(result)
            if progress and task_id is not None:
                progress.update(task_id, advance=1)
        
        return parsed_results
    
    def save_results(self, results: List[Any], output_file: Path):
        """Save the results to a file."""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            for result in results:
                if isinstance(result, (dict, list)):
                    f.write(json.dumps(result))
                else:
                    f.write(str(result))
                f.write("\n")