from typing import Any, Dict, List, Callable, TypeVar
from .base import BaseLLM, VerbalizedSamplingResponseList, get_json_schema_from_pydantic
import json
from openai import OpenAI
import os

T = TypeVar('T')

OPENROUTER_MODELS_MAPPING = {
    # Claude models
    "claude-3-opus": "anthropic/claude-3-opus",
    "claude-3.5-haiku": "anthropic/claude-3.5-haiku",
    "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",
    "claude-3.7-sonnet": "anthropic/claude-3.7-sonnet",
    # Gemini models
    "gemini-2.0-flash": "google/gemini-2.0-flash-001",
    "gemini-2.5-flash": "google/gemini-2.5-flash-preview",
    "gemini-2.5-pro": "google/gemini-2.5-pro-preview",
}

class OpenRouterLLM(BaseLLM):
    """OpenRouter implementation for various models."""
    
    def __init__(self, model_name: str, config: Dict[str, Any], num_workers: int = 1, is_structured: bool = False):
        super().__init__(model_name, config, num_workers, is_structured)
        
        if model_name in OPENROUTER_MODELS_MAPPING:
            self.model_name = OPENROUTER_MODELS_MAPPING[model_name]
            
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )
        # Set up response format for structured output
        self.response_format = None if not self.is_structured else get_json_schema_from_pydantic(VerbalizedSamplingResponseList)

    def _chat(self, messages: List[Dict[str, str]]) -> str:
        """Basic chat functionality without structured response format."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.config.get("temperature", 0.7),
            top_p=self.config.get("top_p", 0.9),
        )
        response = response.choices[0].message.content
        if response:
            response = response.replace("\n", "")
        return response

    def _chat_with_format(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Chat with structured response format."""
        try:
            if "claude" in self.model_name:
                # Claude doesn't support structured output format in OpenRouter
                response = self._chat(messages)
                return self._parse_response(response)
            else:
                # For Gemini and other models that support structured output
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.config.get("temperature", 0.7),
                    top_p=self.config.get("top_p", 0.9),
                    response_format=self.response_format,
                )
                response = completion.choices[0].message.content
                return self._parse_response(response)
        except Exception as e:
            print(f"Error in _chat_with_format: {e}")
            # Fallback to regular chat if parsing fails
            response = self._chat(messages)
            return self._parse_response(response)

    def _parse_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse the response into a list of text and probability pairs."""
        try:
            if isinstance(response, str):
                parsed = json.loads(response)
                return [
                    {
                        "response": resp["response"],
                        "probability": resp["probability"]
                    }
                    for resp in parsed.get("responses", [])
                ]
        except Exception as e:
            print(f"Error parsing response: {e}")
            # If parsing fails, return a single response with probability 1.0
            return [{"response": response, "probability": 1.0}]