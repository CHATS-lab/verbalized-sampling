from typing import Any, Dict, List
from .base import BaseLLM
from openai import OpenAI
import json
from pydantic import BaseModel

class VLLMOpenAI(BaseLLM):
    """vLLM implementation for OpenAI compatible requests."""
    
    def __init__(self, model_name: str, config: Dict[str, Any], num_workers: int = 1, strict_json: bool = False):
        super().__init__(model_name, config, num_workers, strict_json)
        self.client = OpenAI(
            base_url=config.get("base_url", "http://localhost:8000/v1"),
        )

    def _chat(self, messages: List[Dict[str, str]]) -> str:
        """Basic chat functionality without structured response format."""
        # if 'max_tokens' not in self.config:
        #     self.config['max_tokens'] = 4096
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **self.config
        )
        response = response.choices[0].message.content
        # if response:
        #     response = response.replace("\n", "")
        return response

    def _chat_with_format(self, messages: List[Dict[str, str]], schema: BaseModel) -> List[Dict[str, Any]]:
        """Chat with structured response format using guided decoding."""
        try:
            if isinstance(schema, BaseModel):
                schema_json = schema.model_json_schema()
            else:
                schema_json = schema
            
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                response_format=schema_json,
                **self.config
            )
            response = completion.choices[0].message.content
            
            # Parse the JSON response
            parsed_json = json.loads(response)
            return parsed_json
            
        except Exception as e:
            print(f"Error in guided decoding: {e}")
            # Fallback to regular chat if parsing fails
            response = self._chat(messages)
            try:
                # Try to parse the response as JSON
                parsed_json = json.loads(response)
                parsed_responses = []
                
                for resp in parsed_json.get("responses", []):
                    parsed_responses.append({
                        "text": resp.get("text", ""),
                        "probability": resp.get("probability", 0.0)
                    })
                    
                return parsed_responses
            except:
                # If all else fails, return a single response with probability 1.0
                return [{"text": response, "probability": 1.0}]

    def _complete(self, prompt: str) -> str:
        """Send a completion prompt to the model and get the response."""
        # Use appropriate stop tokens for base models
        config = self.config.copy()
        if 'stop' not in config:
            config['stop'] = [
                "<|endoftext|>", 
                "</s>",
                "<|end_of_text|>",
                "### User:",
                "### Assistant:"
            ]
        
        if "max_tokens" not in config:
            config["max_tokens"] = 400
        
        response = self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            **config
        )
        response_text = response.choices[0].text
        if response_text:
            response_text = response_text.strip()
        return response_text