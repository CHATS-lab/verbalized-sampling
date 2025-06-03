from typing import Any, Dict, List
from .base import BaseLLM
from vllm import LLM, SamplingParams

class VLLMOpenAI(BaseLLM):
    """vLLM implementation for OpenAI models."""
    
    def __init__(self, model_name: str, sim_type: str, config: Dict[str, Any]):
        super().__init__(model_name, sim_type, config)
        self.llm = LLM(model=model_name)
        self.sampling_params = SamplingParams(
            temperature=config.get("temperature", 0.7),
            top_p=config.get("top_p", 0.9),
        )
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Send a chat message to the model and get the response."""
        # Convert messages to prompt format
        prompt = ""
        for msg in messages:
            if msg["role"] == "user":
                prompt += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                prompt += f"Assistant: {msg['content']}\n"
        
        # Get response
        outputs = self.llm.generate(prompt, self.sampling_params)
        return outputs[0].outputs[0].text 