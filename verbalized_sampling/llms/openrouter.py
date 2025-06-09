from typing import Any, Dict, List, Callable, TypeVar
from .base import BaseLLM
import json
from openai import OpenAI
import os
from pydantic import BaseModel

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
    
    def __init__(self, model_name: str, config: Dict[str, Any], num_workers: int = 1, strict_json: bool = False):
        super().__init__(model_name, config, num_workers, strict_json)
        
        if model_name in OPENROUTER_MODELS_MAPPING:
            self.model_name = OPENROUTER_MODELS_MAPPING[model_name]
            
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )

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

    def _chat_with_format(self, messages: List[Dict[str, str]], schema: BaseModel) -> List[Dict[str, Any]]:
        """Chat with structured response format."""
        schema_json = schema.model_json_schema()
        
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.config.get("temperature", 0.7),
            top_p=self.config.get("top_p", 0.9),
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": schema_json
                }
            }
        )
        
        response = completion.choices[0].message.content
        if response:
            parsed_response = self._parse_response_with_schema(response, schema)
            return parsed_response
        return []

    def _parse_response_with_schema(self, response: str, schema: BaseModel) -> List[Dict[str, Any]]:
        """Parse the response based on the provided schema."""
        try:
            if isinstance(response, str):
                parsed = json.loads(response)
                
                # Validate the parsed response against the schema
                validated_data = schema(**parsed)
                
                # Handle different schema types
                if hasattr(validated_data, 'responses'):
                    # For schemas with a 'responses' field (SequenceResponse, StructuredResponseList, etc.)
                    responses = validated_data.responses
                    
                    if isinstance(responses, list):
                        result = []
                        for resp in responses:
                            if hasattr(resp, 'text') and hasattr(resp, 'probability'):
                                # ResponseWithProbability
                                result.append({
                                    "response": resp.text,
                                    "probability": resp.probability
                                })
                            elif hasattr(resp, 'text'):
                                # Response
                                result.append({
                                    "response": resp.text,
                                    "probability": 1.0
                                })
                            elif isinstance(resp, str):
                                # SequenceResponse (list of strings)
                                result.append({
                                    "response": resp,
                                    "probability": 1.0
                                })
                        return result
                else:
                    # For direct response schemas (Response)
                    if hasattr(validated_data, 'text'):
                        return [{
                            "response": validated_data.text,
                            "probability": getattr(validated_data, 'probability', 1.0)
                        }]
                    
                # Fallback: return the raw validated data
                return [{"response": str(validated_data), "probability": 1.0}]
                
        except Exception as e:
            print(f"Error parsing response with schema: {e}")
            # If parsing fails, return a single response with probability 1.0
            return [{"response": response, "probability": 1.0}]

    def _parse_response(self, response: str) -> List[Dict[str, Any]]:
        """Legacy parse method - kept for backward compatibility."""
        try:
            if isinstance(response, str):
                parsed = json.loads(response)
                return [
                    {
                        "response": resp["text"],
                        "probability": resp["probability"] if "probability" in resp else 1.0
                    }
                    for resp in parsed.get("responses", [])
                ]
        except Exception as e:
            print(f"Error parsing response: {e}")
            # If parsing fails, return a single response with probability 1.0
            return [{"response": response, "probability": 1.0}]