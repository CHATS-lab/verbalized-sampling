from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List
import json
from rich.progress import Progress

class BaseTask(ABC):
    """Base class for all tasks."""
    
    @abstractmethod
    def get_prompt(self, num_samples: int = 1) -> str:
        """Get the prompt for the task."""
        pass
    
    @abstractmethod
    def parse_response(self, response: str) -> Any:
        """Parse the model's response."""
        pass
    
    def run(
        self,
        model: Any,
        num_responses: int = 3,
        num_samples: int = 1,
        num_workers: int = 128,
        progress: Progress = None,
        task_id: int = None,
    ) -> List[Any]:
        """Run the task with the given model."""
        import concurrent.futures
        
        prompt = self.get_prompt(num_samples)
        prompts = [[{"role": "user", "content": prompt}]] * num_responses
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(model.chat, prompt) for prompt in prompts]
            for future in concurrent.futures.as_completed(futures):
                response = future.result()
                parsed = self.parse_response(response)
                if parsed is not None:
                    results.append(parsed)
                if progress and task_id is not None:
                    progress.update(task_id, advance=1)
        
        return results
    
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