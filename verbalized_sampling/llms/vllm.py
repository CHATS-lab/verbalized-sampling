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
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **self.config
        )
        response = response.choices[0].message.content
        if response:
            response = response.replace("\n", "")
        return response

    def _chat_with_format(self, messages: List[Dict[str, str]], schema: BaseModel) -> List[Dict[str, Any]]:
        """Chat with structured response format using guided decoding."""
        try:
            if isinstance(schema, BaseModel):
                schema_json = schema.model_json_schema()
            else:
                schema_json = schema
                
            # completion = self.client.beta.chat.completions.parse(
            #     model=self.model_name,
            #     messages=messages,
            #     response_format=schema_json,
            #     extra_body=dict(guided_decoding_backend="auto"),
            #     **self.config
            # )
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
            # parsed_responses = []
            
            # message = completion.choices[0].message
            # assert message.parsed
            
            # # Extract responses from the parsed message
            # if hasattr(message.parsed, 'responses'):
            #     for resp in message.parsed.responses:
            #         parsed_responses.append({
            #             "text": resp.text,
            #             "probability": resp.probability
            #         })
            #     return parsed_responses
            # else:
            #     raise ValueError("No responses found in the parsed message")
            
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