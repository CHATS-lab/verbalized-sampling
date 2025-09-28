from typing import Any, Dict, List, Callable, TypeVar
from .base import BaseLLM
import json
from openai import OpenAI
import os
from pydantic import BaseModel
import time

T = TypeVar('T')

OPENROUTER_MODELS_MAPPING = {
    # Claude models
    "claude-3-opus": "anthropic/claude-3-opus",
    "claude-3.5-haiku": "anthropic/claude-3.5-haiku",
    "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",
    "claude-3.7-sonnet": "anthropic/claude-3.7-sonnet",
    # Gemini models
    "gemini-2.0-flash": "google/gemini-2.0-flash-001",
    "gemini-2.5-flash": "google/gemini-2.5-flash",
    "gemini-2.5-pro": "google/gemini-2.5-pro",
    # OpenAI models
    "gpt-4.1-mini": "openai/gpt-4.1-mini",
    "gpt-4.1": "openai/gpt-4.1",
    # meta-llama models
    "llama-3.1-70b-instruct": "meta-llama/llama-3.1-70b-instruct",
    # DeepSeek models
    "deepseek-r1": "deepseek/deepseek-r1-0528",
    # Qwen models
    "qwen3-235b": "qwen/qwen3-235b-a22b-2507",

}

class OpenRouterLLM(BaseLLM):
    """OpenRouter implementation for various models."""
    
    def __init__(self, model_name: str, config: Dict[str, Any], num_workers: int = 1, strict_json: bool = False):
        super().__init__(model_name, config, num_workers, strict_json)
        
        if model_name in OPENROUTER_MODELS_MAPPING:
            self.model_name = OPENROUTER_MODELS_MAPPING[model_name]
        
        if "gemini-2.5-flash" in model_name:
            config["reasoning"] = {"exclude": "true"}
            
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            # api_key=os.environ.get("OPENAI_API_KEY"),
        )

    def _chat(self, messages: List[Dict[str, str]]) -> str:
        """Basic chat functionality without structured response format."""
        if "deepseek" in self.model_name:
            provider_args = {"provider": {
                    "require_parameters": True,
                    "only": ["fireworks"]
                }}
        else:
            provider_args = None

        # Build parameters dynamically
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.config.get("temperature", 0.7),
            "top_p": self.config.get("top_p", 0.9),
            "provider": provider_args,
        }

        # Only add min_p if it's provided in config
        if "min_p" in self.config:
            params["min_p"] = self.config["min_p"]

        try:
            response = self.client.chat.completions.create(**params)
            response = response.choices[0].message.content
            if response:
                response = response.replace("\n", "")
            return response
        except Exception as e:
            print(f"Error in OpenRouter chat: {e}")
            return ""

    def _chat_with_format(self, messages: List[Dict[str, str]], schema: BaseModel) -> List[Dict[str, Any]]:
        """Chat with structured response format."""
        tries = 10
        backoff = 1
        for i in range(tries):
            try:
                if "deepseek" in self.model_name:
                    provider_args = {
                        "provider": {
                            "require_parameters": True,
                            "only": ["fireworks"]
                        }
                    }
                else:
                    provider_args = None
                    
                if isinstance(schema, BaseModel):
                    schema = schema.model_json_schema()
                
                # print("Schema: ", schema)
                
                # Build parameters dynamically
                params = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": self.config.get("temperature", 0.7),
                    "top_p": self.config.get("top_p", 0.9),
                    "response_format": schema,
                    "provider": provider_args,
                }

                # Only add min_p if it's provided in config
                if "min_p" in self.config:
                    params["extra_body"] = {
                        "min_p": self.config["min_p"]
                    }

                completion = self.client.chat.completions.create(**params)
                
                if completion is None or not completion.choices:
                    print(f"Error: No response from OpenRouter API for model {self.model_name}")
                    raise Exception(f"Error: No response from OpenRouter API for model {self.model_name}")
                
                response = completion.choices[0].message.content
                if response:
                    parsed_response = self._parse_response_with_schema(response, schema)
                    if not parsed_response:
                        print(f"Error: Empty response from OpenRouter API for model {self.model_name}")
                        raise Exception(f"Error: Empty response from OpenRouter API for model {self.model_name}")
                    return parsed_response
                else:
                    print(f"Error: Empty response from OpenRouter API for model {self.model_name}")
                    raise Exception(f"Error: Empty response from OpenRouter API for model {self.model_name}")
            except Exception as e:
                print(f"Error in OpenRouter chat_with_format: {e}")
                
                if i < tries - 1:
                    print(f"Retrying in {backoff} seconds...")
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    print("Max retries reached. Returning empty response.")
                    return [{"response": "", "probability": 1.0}]

    def _parse_response_with_schema(self, response: str, schema: BaseModel) -> List[Dict[str, Any]]:
        """Parse the response based on the provided schema."""
        try:
            if isinstance(response, str):
                parsed = json.loads(response)
                
                # Handle double-escaped JSON strings (i.e., string inside a string)
                if isinstance(parsed, str):
                    parsed = json.loads(parsed)
                
                # Handle different schema types
                if "responses" in parsed:
                    # For schemas with a 'responses' field (SequenceResponse, StructuredResponseList, etc.)
                    responses = parsed["responses"]
                    # print('RESPONSES: ', responses)
                    
                    if isinstance(responses, list):
                        result = []
                        for resp in responses:
                            if isinstance(resp, dict) and "text" in resp and any(key in resp for key in ["probability", "confidence", "perplexity", "nll"]):
                                # Combine probability/confidence/perplexity fields
                                if "probability" in resp:
                                    prob = resp["probability"]
                                if "confidence" in resp:
                                    prob = resp["confidence"]
                                if "perplexity" in resp:
                                    prob = resp["perplexity"]
                                if "nll" in resp:
                                    prob = resp["nll"]
                                result.append({
                                    "response": resp["text"],
                                    "probability": prob
                                })
                            elif isinstance(resp, dict) and "text" in resp:
                                # Response
                                result.append({
                                    "response": resp["text"],
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
                    if "text" in parsed:
                        return [{
                            "response": parsed["text"],
                            "probability": parsed.get("probability", 1.0)
                        }]
                    elif 'response' in parsed:
                        return [{
                            "response": parsed["response"],
                            "probability": parsed.get("probability", 1.0)
                        }]
                    
                # Fallback: return the raw validated data
                return [{"response": str(parsed), "probability": 1.0}]
                
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