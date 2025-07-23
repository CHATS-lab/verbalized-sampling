from typing import Any, Dict, List
from pydantic import BaseModel
import json
import time
import litellm
from .base import BaseLLM

litellm.drop_params = True
LITELLM_MODELS_MAPPING = {
    "anthropic/claude-4-sonnet" : "anthropic/claude-sonnet-4-20250514",
    "anthropic/claude-4-opus" : "anthropic/claude-opus-4-20250514",
    "claude-4-sonnet" : "anthropic/claude-sonnet-4-20250514",
    "claude-4-opus" : "anthropic/claude-opus-4-20250514",
    "claude-3.7-sonnet" : "anthropic/claude-3-7-sonnet-20250219",
    "anthropic/claude-3.7-sonnet" : "anthropic/claude-3-7-sonnet-20250219",
    "google/gemini-2.5-pro" : "gemini/gemini-2.5-pro",
    "google/gemini-2.5-flash" : "gemini/gemini-2.5-flash",
    "google/gemini-2.0-flash" : "gemini/gemini-2.0-flash",
    # "meta-llama/llama-3.3-70b-instruct" : "fireworks_ai/accounts/fireworks/models/llama-v3p3-70b-instruct",
}


class LiteLLM(BaseLLM):
    """LiteLLM implementation for Anthropic Claude Sonnet 4."""
    
    def __init__(self, 
                 model_name: str = "claude-3-5-sonnet-20241022", 
                 config: Dict[str, Any] = None, 
                 num_workers: int = 1, 
                 strict_json: bool = False):
        if model_name in LITELLM_MODELS_MAPPING:
            model_name = LITELLM_MODELS_MAPPING[model_name]

        super().__init__(model_name, config, num_workers, strict_json)
    
    def _chat(self, messages: List[Dict[str, str]]) -> str:
        """Send a single message to Claude Sonnet 4 and get the response."""
        return self._completion(messages, json_schema=None)
    
    def _chat_with_format(self, messages: List[Dict[str, str]], schema: BaseModel) -> str:
        """Send a single message to Claude Sonnet 4 and get the response in JSON format."""
        # Convert Pydantic schema to JSON schema format
        result = self._completion(messages, json_schema=schema)
        return json.dumps(result) if isinstance(result, dict) else result
    
    def _completion(
        self,
        messages: List[Dict[str, str]],
        json_schema: Dict = None
    ):
        """Completion method based on the provided reference."""
        tries = 5
        backoff = 1
        
        for i in range(tries):
            try:
                # Prepare request parameters for litellm
                request_params = {
                    "model": self.model_name,
                    "messages": messages,
                    **self.config  # Include temperature, max_tokens, etc.
                }
                
                if json_schema is not None:
                    request_params["response_format"] = json_schema
                
                # Make the API call using litellm
                response = litellm.completion(**request_params)
                
                # Extract content from response
                content = response.choices[0].message.content
                
                if json_schema is not None:
                    parsed_content = json.loads(content)
                    # print("Parsed content: ", parsed_content)
                    return parsed_content
                else:
                    return content
                    
            except Exception as e:
                if "JSON" in str(type(e)).upper():
                    print("JSON error:")
                    print("Content:", content if 'content' in locals() else "No content")
                    print(e)

                if i < tries - 1:
                    print(f"Error: {e}. Retrying...")
                    time.sleep(backoff)
                    backoff = min(30, backoff * 2)
                else:
                    print("Reached maximum number of tries. Raising.")
                    return None
