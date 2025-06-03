from abc import ABC, abstractmethod
from typing import Any, Dict, List

class BaseLLM(ABC):
    """Base class for all LLM interfaces."""
    
    def __init__(self, model_name: str, sim_type: str, config: Dict[str, Any]):
        self.model_name = model_name
        self.sim_type = sim_type
        self.config = config
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Send a chat message to the model and get the response."""
        pass 