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
        self.max_turns = num_responses
        
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
        # If response is already a list, return it directly
        if isinstance(response, list):
            return response
            
        # If response is a string, try to parse it as JSON
        try:
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
        initial_prompts = self.get_prompt()
        all_results = []
        
        for conv_id, initial_prompt in enumerate(initial_prompts):
            # Initialize conversation with the first prompt
            chat_history = initial_prompt.copy()
            
            for turn in range(self.max_turns):
                if turn == 0:
                    # First turn: use the initial prompt as-is
                    current_prompts = [initial_prompt for _ in range(self.num_responses)]
                else:
                    # Subsequent turns: get continuation prompts
                    continuation_prompt = PromptFactory.get_multi_turn_continuation(chat_history)
                    current_prompts = [continuation_prompt for _ in range(self.num_responses)]
                
                # Get responses for this turn
                results = self.model.chat(current_prompts, schema=get_schema(self.method))
                
                # Process results for this turn
                turn_responses = []
                for i, result in enumerate(results):
                    initial_prompt_content = initial_prompt[-1]["content"]
                    # prompt_content = current_prompts[i][-1]["content"]
                    parsed = self.parse_response(result)
                    
                    if parsed is not None:
                        if isinstance(parsed, list):
                            for item in parsed:
                                if isinstance(item, dict):
                                    item.update({
                                        "prompt": initial_prompt_content,
                                        "turn": turn + 1,
                                        "conversation_id": conv_id,
                                        "response_id": i
                                    })
                                    turn_responses.append(item)
                                    all_results.append(item)
                                else:
                                    response_data = {
                                        "prompt": initial_prompt_content, 
                                        "response": item,
                                        "turn": turn + 1,
                                        "conversation_id": conv_id,
                                        "response_id": i
                                    }
                                    turn_responses.append(response_data)
                                    all_results.append(response_data)
                        elif isinstance(parsed, dict):
                            parsed.update({
                                "prompt": initial_prompt_content,
                                "turn": turn + 1,
                                "conversation_id": conv_id,
                                "response_id": i
                            })
                            turn_responses.append(parsed)
                            all_results.append(parsed)
                        else:
                            response_data = {
                                "prompt": initial_prompt_content, 
                                "response": parsed,
                                "turn": turn + 1,
                                "conversation_id": conv_id,
                                "response_id": i
                            }
                            turn_responses.append(response_data)
                            all_results.append(response_data)
                    else:
                        response_data = {
                            "prompt": initial_prompt_content, 
                            "response": result,
                            "turn": turn + 1,
                            "conversation_id": conv_id,
                            "response_id": i
                        }
                        turn_responses.append(response_data)
                        all_results.append(response_data)
                    
                    if progress and task_id is not None:
                        progress.update(task_id, advance=1)
                
                # Use the first response to continue the conversation
                if turn_responses:
                    first_response = turn_responses[0]
                    response_text = first_response.get("response", str(first_response.get("text", "")))
                    chat_history.append({"role": "assistant", "content": str(response_text)})
        
        return all_results

    def run(
        self,
        progress: Progress = None,
        task_id: int = None,
    ) -> List[Any]:
        """Run the task with the given model."""
        
        # Check if this is a multi-turn method
        if is_method_multi_turn(self.method):
            return self._run_multi_turn(progress, task_id)
        
        # Original single-turn logic
        prompts = [prompt for prompt in self.get_prompt() for _ in range(self.num_responses)]
        results = self.model.chat(prompts, schema=get_schema(self.method))
        parsed_results = []
        for prompt, result in zip(prompts, results):
            prompt = prompt[-1]["content"]
            parsed = self.parse_response(result)
            if parsed is not None:
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict):
                            item["prompt"] = prompt
                            parsed_results.append(item)
                        else:
                            parsed_results.append({"prompt": prompt, "response": item})
                elif isinstance(parsed, dict):
                    parsed["prompt"] = prompt
                    parsed_results.append(parsed)
                else:
                    parsed_results.append({"prompt": prompt, "response": parsed})
            else:
                parsed_results.append({"prompt": prompt, "response": result})
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