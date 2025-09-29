from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Union
import json
import ast
from rich.progress import Progress
from verbalized_sampling.methods import (
    PromptFactory, 
    Method,
    is_method_multi_turn,
    is_method_combined,
    is_method_base_model,
    ResponseParser
)
import concurrent.futures
from verbalized_sampling.llms import BaseLLM
from verbalized_sampling.methods.schema import get_schema
import math
from tqdm import tqdm

class BaseTask(ABC):
    """Base class for all tasks."""
    
    def __init__(self,
                 model: BaseLLM,
                 method: Method,
                 num_responses: int = 3,
                 num_samples: int = 5,
                 num_prompts: int = 5,
                 num_samples_per_prompt: int = 2,
                 target_words: int = 200,
                 random_seed: int = 42,
                 all_possible: bool = False,
                 strict_json: bool = False,
                 probability_definition: str = "default",
                 probability_tuning: float = -1,
                 custom_prompts: list = None,
                 ):
        self.model = model
        self.method = method
        self.num_responses = num_responses
        self.num_samples = num_samples
        self.num_prompts = num_prompts
        self.num_samples_per_prompt = num_samples_per_prompt
        self.target_words = target_words
        self.random_seed = random_seed
        self.all_possible = all_possible
        self.strict_json = strict_json
        self.probability_definition = probability_definition
        self.probability_tuning = probability_tuning
        self.max_turns = num_samples
        self.custom_prompts = custom_prompts
        
    def get_prompt(self) -> List[Union[List[Dict[str, str]], str]]:
        """Get the prompt for the task."""
        return PromptFactory.get_prompt(
            self.task_type, 
            self.method, 
            num_samplings=self.num_samples,
            num_prompts=self.num_prompts,
            num_samples_per_prompt=self.num_samples_per_prompt,
            target_words=self.target_words,
            random_seed=self.random_seed,
            all_possible=self.all_possible,
            strict_json=self.strict_json,
            probability_definition=self.probability_definition,
            probability_tuning=self.probability_tuning,
            custom_prompts=self.custom_prompts,
        )


    def _run_combined(self, progress: Progress = None, task_id: int = None) -> List[Any]:
        """Run VS-Multi (vs_multi) multi-turn conversations with structured responses."""
        initial_prompts = [prompt for prompt in self.get_prompt() for _ in range(self.num_responses)]
        all_results = []
        
        num_turns = int(self.num_samples // self.num_samples_per_prompt)
        remainder = self.num_samples % self.num_samples_per_prompt
        
        def _run_whole_conversation(initial_prompt: List[Dict[str, str]]):
            chat_history = initial_prompt.copy()
            turn_responses = []
            global_index = 0
            
            def process_turn(current_prompts, num_samples):
                nonlocal global_index
                result = self.model._chat_with_format(current_prompts, schema=get_schema(self.method, probability_definition=self.probability_definition))
                # print("Result: ", result)
                
                parsed_responses = ResponseParser.parse_response(self.method, result)
                for i, response in enumerate(parsed_responses):
                    if isinstance(response, str):
                        # If the response is a string, convert it to a dict
                        response = {"text": response}
                    response["index"] = global_index + i
                
                # print("parsed_responses: ", parsed_responses)
                response_data = {
                    "prompt": initial_prompt[-1]["content"],
                    "responses": parsed_responses
                }
                turn_responses.append(response_data)
                chat_history.append({"role": "assistant", "content": str(result)})
                global_index += len(parsed_responses)
            
            # Process regular turns
            for turn in range(num_turns):
                current_prompts = (initial_prompt if turn == 0 else 
                                 PromptFactory.get_combined_continuation(chat_history, self.num_samples_per_prompt, self.task_type, self.target_words))
                process_turn(current_prompts, self.num_samples_per_prompt)
            
            # Process remainder turn
            if remainder > 0:
                continuation_prompt = PromptFactory.get_combined_continuation(chat_history, remainder, self.task_type, self.target_words)
                process_turn(continuation_prompt, remainder)
            
            return turn_responses
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.model.num_workers) as executor:
            futures = [executor.submit(_run_whole_conversation, initial_prompt) for initial_prompt in initial_prompts]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Running combined method"):
                turn_responses = future.result()
                all_results.extend(turn_responses)
                if progress and task_id is not None:
                    progress.update(task_id, advance=len(turn_responses))
        
        return all_results


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
                    continuation_prompt = PromptFactory.get_multi_turn_continuation(chat_history, self.task_type, self.target_words)
                    current_prompts = continuation_prompt
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
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.model.num_workers) as executor:
            futures = [executor.submit(_run_whole_conversation, initial_prompt) for initial_prompt in initial_prompts]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Running multi-turn method"):
                turn_responses = future.result()
                all_results.extend(turn_responses)
                if progress and task_id is not None:
                    progress.update(task_id, advance=len(turn_responses))
        
        return all_results
    
    def _run_base_model(
        self,
        progress: Progress = None,
        task_id: int = None,
    ) -> List[Any]:
        """Run base model completions with oversampling and filtering."""
        import math

        oversample_factor = 1.5
        num_prompts = len(self.get_prompt())
        prompts = [prompt for prompt in self.get_prompt() for _ in range(math.ceil(self.num_responses * oversample_factor))]

        results = self.model.chat(prompts)
        parsed_results = []

        # Group results by original prompt (since we oversampled)
        for i in range(num_prompts):
            # Get all completions for this prompt
            prompt = self.get_prompt()[i]
            start = i * math.ceil(self.num_responses * oversample_factor)
            end = (i + 1) * math.ceil(self.num_responses * oversample_factor)
            completions = results[start:end]

            # Filter completions: non-empty and at least 20 words
            valid_completions = []
            for result in completions:
                parsed = ResponseParser.parse_response(self.method, result)
                # parsed can be a list or a string, handle both
                if isinstance(parsed, list):
                    for resp in parsed:
                        text = resp if isinstance(resp, str) else resp.get("text", "")
                        if text and len(text.split()) >= 20:
                            valid_completions.append(resp)
                else:
                    text = parsed if isinstance(parsed, str) else parsed.get("text", "")
                    if text and len(text.split()) >= 20:
                        valid_completions.append(parsed)

                if len(valid_completions) >= self.num_responses:
                    break

            # For base models, the prompt is a string, not a message list
            prompt_text = prompt if isinstance(prompt, str) else prompt[-1]["content"]
            for i in range(len(valid_completions)):
                parsed_results.append({
                    "prompt": prompt_text,
                    "responses": [valid_completions[i]]
                })
                if progress and task_id is not None:
                    progress.update(task_id, advance=1)

        return parsed_results

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
        print(f"  num_samples_per_prompt: {self.num_samples_per_prompt}")
        print(f"  target_words: {self.target_words}")
        print(f"  random_seed: {self.random_seed}")
        print(f"  max_turns: {self.max_turns}")
        
        # Check if this is a multi-turn method
        if is_method_multi_turn(self.method):
            return self._run_multi_turn(progress, task_id)
        
        if is_method_combined(self.method):
            return self._run_combined(progress, task_id)
        
        # Check if this is a base model method
        if is_method_base_model(self.method):
            return self._run_base_model(progress, task_id)
        
        # Original single-turn logic
        prompts = [prompt for prompt in self.get_prompt() for _ in range(self.num_responses)]
        results = self.model.chat(prompts, schema=get_schema(self.method, probability_definition=self.probability_definition))
        print("Schema: ", get_schema(self.method, probability_definition=self.probability_definition))
        parsed_results = []

        # print("Prompts: ", prompts)
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
