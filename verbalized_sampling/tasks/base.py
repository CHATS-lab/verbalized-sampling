from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List
import json
from rich.progress import Progress
from verbalized_sampling.prompts import PromptFactory, PromptTemplate, SamplingPromptTemplate, Method
from verbalized_sampling.llms import BaseLLM

class BaseTask(ABC):
    """Base class for all tasks."""
    
    @abstractmethod
    def get_prompt(self, method: Method, num_samples: int = 1) -> str:
        """Get the prompt for the task."""
        pass
    
    @abstractmethod
    def parse_response(self, method: Method, response: str) -> Any:
        """Parse the model's response."""
        pass
    
    def run(
        self,
        model: BaseLLM,
        method: Method,
        num_responses: int = 3,
        num_samples: int = 1,
        num_workers: int = 128,
        progress: Progress = None,
        task_id: int = None,
    ) -> List[Any]:
        """Run the task with the given model."""
        
        prompts = [self.get_prompt(method, num_samples)] * num_responses
        results = model.chat(prompts)
        parsed_results = []
        for result in results:
            parsed = self.parse_response(method, result)
            if parsed is not None:
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