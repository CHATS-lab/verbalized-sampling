from typing import Any, Dict, List
import openrouter
from .base import BaseLLM

class OpenRouterLLM(BaseLLM):
    """OpenRouter implementation for various models."""
    
    def __init__(self, model_name: str, sim_type: str, config: Dict[str, Any]):
        super().__init__(model_name, sim_type, config)
        self.client = openrouter.Client()
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Send a chat message to the model and get the response."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.config.get("temperature", 0.7),
            top_p=self.config.get("top_p", 0.9),
        )
        return response.choices[0].message.content 