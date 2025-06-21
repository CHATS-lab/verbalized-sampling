from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List
import json
from rich.progress import Progress
from verbalized_sampling.methods import (
    PromptFactory, 
    Method,
    is_method_multi_turn,
    ResponseParser
)
import concurrent.futures
from verbalized_sampling.llms import BaseLLM
from verbalized_sampling.methods.schema import get_schema

class BaseTask(ABC):
    """Base class for all tasks."""
    
    def __init__(self,
                 model: BaseLLM,
                 method: Method,
                 num_responses: int = 3,
                 num_samples: int = 5,
                 num_prompts: int = 5,
                 target_words: int = 200,
                 random_seed: int = 42,
                 all_possible: bool = False,
                 strict_json: bool = False,
                 ):
        self.model = model
        self.method = method
        self.num_responses = num_responses
        self.num_samples = num_samples
        self.num_prompts = num_prompts
        self.target_words = target_words
        self.random_seed = random_seed
        self.all_possible = all_possible
        self.strict_json = strict_json
        self.max_turns = num_samples
        
    def get_prompt(self) -> List[List[Dict[str, str]]]:
        """Get the prompt for the task."""
        return PromptFactory.get_prompt(
            self.task_type, 
            self.method, 
            num_samplings=self.num_samples,
            num_prompts=self.num_prompts,
            target_words=self.target_words,
            random_seed=self.random_seed,
            all_possible=self.all_possible,
            strict_json=self.strict_json
        )
    
    def _run_multi_turn(
        self,
        progress: Progress = None,
        task_id: int = None,
    ) -> List[Any]:
        """Run multi-turn conversations."""
        initial_prompts = [prompt for prompt in self.get_prompt() for _ in range(self.num_responses)]
        all_results = []
        
        def _run_whole_conversation(initial_prompt: List[Dict[str, str]]):
            chat_history = initial_prompt.copy()
            turn_responses = []
            for turn in range(self.max_turns):
                if turn == 0:
                    current_prompts = initial_prompt
                else:
                    continuation_prompt = PromptFactory.get_multi_turn_continuation(chat_history)
                    current_prompts = continuation_prompt
                # print(f"Current prompts: {current_prompts}")
                result = self.model._chat(current_prompts)

                initial_prompt_content = initial_prompt[-1]["content"]
                response_data = {
                    "prompt": initial_prompt_content,
                    "responses": [{
                        "text": result,
                        "turn": turn + 1,
                    }]
                }
                turn_responses.append(response_data)
                chat_history.append({"role": "assistant", "content": str(result)})

            return turn_responses
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=128) as executor:
            futures = [executor.submit(_run_whole_conversation, initial_prompt) for initial_prompt in initial_prompts]
            for future in concurrent.futures.as_completed(futures):
                turn_responses = future.result()
                # print(f"Turn responses: {turn_responses}")
                all_results.extend(turn_responses)
                if progress and task_id is not None:
                    progress.update(task_id, advance=len(turn_responses))
        
        return all_results

    def run(
        self,
        progress: Progress = None,
        task_id: int = None,
    ) -> List[Any]:
        """Run the task with the given model."""
        
        print("Task parameters:")
        print(f"  task_type: {self.task_type}")
        print(f"  method: {self.method}")
        print(f"  model: {self.model}")
        print(f"  num_responses: {self.num_responses}")
        print(f"  num_samples: {self.num_samples}")
        print(f"  num_prompts: {self.num_prompts}")
        print(f"  target_words: {self.target_words}")
        print(f"  random_seed: {self.random_seed}")
        print(f"  max_turns: {self.max_turns}")
        
        # Check if this is a multi-turn method
        if is_method_multi_turn(self.method):
            return self._run_multi_turn(progress, task_id)
        
        # Original single-turn logic
        prompts = [prompt for prompt in self.get_prompt() for _ in range(self.num_responses)]
        results = self.model.chat(prompts, schema=get_schema(self.method))
        parsed_results = []

        print("Prompts: ", prompts)
        # print("Results: ", results)
        
        for prompt, result in zip(prompts, results):
            prompt_text = prompt[-1]["content"]
            # Use the ResponseParser to get unified format
            parsed_responses = ResponseParser.parse_response(self.method, result)
            
            parsed_results.append({
                "prompt": prompt_text,
                "responses": parsed_responses
            })
            
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