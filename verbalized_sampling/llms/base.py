from abc import ABC, abstractmethod
from typing import Any, Dict, List, Callable, T
import concurrent.futures
from pydantic import BaseModel
from tqdm import tqdm

class VerbalizedSamplingResponse(BaseModel):
    response: str
    probability: float

class VerbalizedSamplingResponseList(BaseModel):
    responses: List[VerbalizedSamplingResponse]

def get_json_schema_from_pydantic(model: BaseModel) -> Dict[str, Any]:
    return model.model_json_schema()

class BaseLLM(ABC):
    """Base class for all LLM interfaces."""
    
    def __init__(self, 
                 model_name: str, 
                 config: Dict[str, Any], 
                 num_workers: int = 1, 
                 is_structured: bool = False):
        self.model_name = model_name
        self.config = config
        self.num_workers = num_workers
        self.is_structured = is_structured
    
    @abstractmethod
    def _chat(self, message: List[Dict[str, str]]) -> str:
        """Send a single message to the model and get the response."""
        pass

    @abstractmethod
    def _chat_with_format(self, message: List[Dict[str, str]]) -> str:
        """Send a single message to the model and get the response in JSON format."""
        pass
    
    def chat(self, messages: List) -> List[str]:
        CHAT_FUNC = self._chat_with_format if self.is_structured else self._chat
        if self.num_workers > 1:
            return self._parallel_execute(CHAT_FUNC, messages)
        else:
            return [CHAT_FUNC(message) for message in messages] 


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