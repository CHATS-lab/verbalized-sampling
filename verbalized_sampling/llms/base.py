from abc import ABC, abstractmethod
from typing import Any, Dict, List, Callable, T, Union
import concurrent.futures
from pydantic import BaseModel
from tqdm import tqdm
from functools import partial

class BaseLLM(ABC):
    """Base class for all LLM interfaces."""
    
    def __init__(self, 
                 model_name: str, 
                 config: Dict[str, Any], 
                 num_workers: int = 1, 
                 strict_json: bool = False):
        self.model_name = model_name
        self.config = config
        self.num_workers = num_workers
        self.strict_json = strict_json
    
    @abstractmethod
    def _chat(self, message: List[Dict[str, str]]) -> str:
        """Send a single message to the model and get the response."""
        pass

    @abstractmethod
    def _chat_with_format(self, message: List[Dict[str, str]], schema: BaseModel) -> str:
        """Send a single message to the model and get the response in JSON format."""
        pass
    
    def _complete(self, prompt: str) -> str:
        """Send a completion prompt to the model and get the response."""
        # Default implementation - subclasses can override
        pass
    
    def chat(self, messages: List[Union[List[Dict[str, str]], str]], schema: BaseModel = None) -> List[str]:
        # Handle mixed list of chat messages and completion prompts
        results = []
        
        def _func(message: List[Dict[str, str]]) -> str:
            if isinstance(message, str):
                return self._complete(message)
            else:
                if not self.strict_json:
                    return self._chat(message)
                else:
                    if schema is None:
                        raise ValueError("Schema is required for strict JSON mode.")
                    return self._chat_with_format(message, schema)
        
        return self._parallel_execute(_func, messages)


    def _parallel_execute(self, func: Callable[[List[Dict[str, str]]], T], messages_list: List[List[Dict[str, str]]]) -> List[T]:
        """Execute function in parallel while maintaining order of responses."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks and keep track of their order
            future_to_index = {
                executor.submit(func, messages): i 
                for i, messages in enumerate(messages_list)
            }
            
            # Initialize results list with None
            results = [None] * len(messages_list)
            
            # As futures complete, put them in the correct position with tqdm progress
            with tqdm(
                total=len(messages_list), 
                desc="Processing messages",
                unit="msg",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            ) as pbar:
                for future in concurrent.futures.as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        results[index] = future.result()
                        pbar.set_postfix({"completed": f"#{index}"})
                    except Exception as e:
                        pbar.write(f"Error processing message {index}: {e}")
                        results[index] = None
                    pbar.update(1)
            
            return results