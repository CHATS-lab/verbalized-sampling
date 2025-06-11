from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List
import json
from rich.progress import Progress
from verbalized_sampling.prompts import (
    PromptFactory, 
    Method,
    is_method_multi_turn
)
import concurrent.futures
from verbalized_sampling.llms import BaseLLM
from verbalized_sampling.prompts.schema import get_schema
from tqdm import trange

class BaseTask(ABC):
    """Base class for all tasks."""
    
    def __init__(self,
                 model: BaseLLM,
                 method: Method,
                 num_responses: int = 3,
                 num_samples: int = 5,
                 sample_size: int = 5,
                 random_seed: int = 42,
                 all_possible: bool = False,
                 strict_json: bool = False,
                 max_turns: int = 3,  # Add parameter for multi-turn
                 ):
        self.model = model
        self.method = method
        self.num_responses = num_responses
        self.num_samples = num_samples
        self.sample_size = sample_size
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
            sample_size=self.sample_size,
            random_seed=self.random_seed,
            all_possible=self.all_possible,
            strict_json=self.strict_json
        )
    
    def parse_response(self, response: Any) -> Any:
        """Parse the model's response.
        
        Args:
            response: Can be either a string (JSON) or a list of responses
            
        Returns:
            Parsed response in the expected format
        """

        # Old code by Simon
        # If response is already a list, return it directly
        # if isinstance(response, list):
        #     if len(response) == 1:
        #         response = response[0]
        #         if "response" in response:
        #             response = response["response"]
        #     else:
        #         return response

        if isinstance(response, (list, dict)):
            return response
            
        # If response is a string, try to parse it as JSON
        try:
            # Check if response contains code block markers
            if "```" in response:
                # Remove code block markers and any language specifiers
                response = response.replace("```json", "").replace("```", "").strip()
            # elif response.startswith("[") and response.endswith("]"):
            #     response = response[1:-1]
            parsed = json.loads(response)
            if isinstance(parsed, dict):
                parsed = parsed["responses"]
            return parsed
        except Exception as e:
            print(f"Error parsing response: {e}")
            return response
    
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
                
                result = self.model._chat(current_prompts)

                initial_prompt_content = initial_prompt[-1]["content"]
                response_data = {
                    "prompt": initial_prompt_content,
                    "response": result,
                    "turn": turn + 1,
                }
                turn_responses.append(response_data)
                chat_history.append({"role": "assistant", "content": str(result)})

            return turn_responses
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=128) as executor:
            futures = [executor.submit(_run_whole_conversation, initial_prompt) for initial_prompt in initial_prompts]
            for future in concurrent.futures.as_completed(futures):
                turn_responses = future.result()
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
        print(f"  num_responses: {self.num_responses}")
        print(f"  sample_size: {self.sample_size}")
        print(f"  random_seed: {self.random_seed}")
        print(f"  max_turns: {self.max_turns}")
        print(f"  model: {self.model}")
        # Check if this is a multi-turn method
        if is_method_multi_turn(self.method):
            return self._run_multi_turn(progress, task_id)
        
        # Original single-turn logic
        prompts = [prompt for prompt in self.get_prompt() for _ in range(self.num_responses)]
        results = self.model.chat(prompts, schema=get_schema(self.method))
        print("Results: ", results)
        parsed_results = []
        current_batch = []
        
        for prompt, result in zip(prompts, results):
            prompt = prompt[-1]["content"]
            parsed = self.parse_response(result)
            # Old code by Simon
            # if parsed is not None:
            #     if isinstance(parsed, list):
            #         for item in parsed:
            #             if isinstance(item, dict):
            #                 item["prompt"] = prompt
            #                 parsed_results.append(item)
            #             else:
            #                 parsed_results.append({"prompt": prompt, "response": item})
            #     elif isinstance(parsed, dict):
            #         if ("response" in parsed) and (isinstance(parsed["response"], list)):
            #             for item in parsed["response"]:
            #                 item["prompt"] = prompt
            #                 parsed_results.append(item)
            #         else:
            #             parsed["prompt"] = prompt
            #             parsed_results.append(parsed)
            #         # parsed_results.append(parsed)
            #     else:
            #         parsed_results.append({"prompt": prompt, "response": parsed})

            if self.method == Method.DIRECT:
                current_batch.append({"prompt": prompt, "response": result})
            else:
                parsed = self.parse_response(result)
                
                if parsed is not None:
                    if isinstance(parsed, list):
                        for item in parsed:
                            if isinstance(item, dict):
                                item["prompt"] = prompt
                                current_batch.append(item)
                            else:
                                current_batch.append({"prompt": prompt, "response": item})
                    elif isinstance(parsed, dict):
                        parsed["prompt"] = prompt
                        current_batch.append(parsed)
                    else:
                        current_batch.append({"prompt": prompt, "response": parsed})
                else:
                    current_batch.append({"prompt": prompt, "response": result})
                
            # When we have collected num_samples items, add the batch to results
            if len(current_batch) == self.num_samples:
                parsed_results.append(current_batch)
                current_batch = []
                
            if progress and task_id is not None:
                progress.update(task_id, advance=1)
        
        # print(parsed_results)

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