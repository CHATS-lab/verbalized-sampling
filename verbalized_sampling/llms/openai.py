import os
from typing import Any, Dict, Generator, List
import json

from openai import OpenAI
from pydantic import BaseModel

from verbalized_sampling.llms.base import BaseLLM

GPT_MODELS_MAPPING = {
    "gpt-4-turbo": "gpt-4-turbo",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4o": "gpt-4o", # gpt-4o-2024-03-27
    "gpt-4.1-mini": "gpt-4.1-mini", # gpt-4.1-mini-2025-04-14
    "gpt-4.1": "gpt-4.1", # gpt-4.1-2025-04-14
    "openai/gpt-4.1": "gpt-4.1",
    "openai/gpt-4.1-mini": "gpt-4.1-mini",
    "openai/gpt-4o": "gpt-4o",
    "openai/gpt-4o-mini": "gpt-4o-mini",
    # "gpt-4.5": "gpt-4.5-preview",
    # OpenAI reasoning models
    "o1-mini": "o1-mini",
    "o1": "o1",
    "o3-mini": "o3-mini",
    "o3": "o3",
    "o4-mini": "o4-mini-2025-04-16",
    "openai/o3": "o3",
    "openai/o3-mini": "o3-mini",
}


class OpenAILLM(BaseLLM):
    def __init__(self, model_name: str, config: Dict[str, Any], num_workers: int = 1, strict_json: bool = False):
        super().__init__(model_name, config, num_workers, strict_json)
        
        if model_name in GPT_MODELS_MAPPING:
            self.model_name = GPT_MODELS_MAPPING[model_name] 

        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        if "ftjob-" in self.model_name:
            status = self.client.fine_tuning.jobs.retrieve(self.model_name)
            self.model_name = status.fine_tuned_model
        
        # Handle O3/O4 models that don't support temperature and top_p
        if "o3" in self.model_name or "o4" in self.model_name:
            # Remove temperature and top_p for O3/O4 models
            self.config.pop('temperature', None)
            self.config.pop('top_p', None)
            if 'max_tokens' in self.config:
                self.config.update({'max_completion_tokens': self.config.pop('max_tokens')})


    def _chat(self, messages: List[Dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **self.config,
        )
        response = response.choices[0].message.content
        if response:
            response = response.replace("\n", "")
            if response.startswith('"') and response.endswith('"'):
                response = response[1:-1]
        return response

    
    def _parse_response_with_schema(self, response: str) -> List[Dict[str, Any]]:
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
                    
                    if isinstance(responses, list):
                        result = []
                        for resp in responses:
                            if isinstance(resp, dict) and "text" in resp and any(key in resp for key in ["probability", "confidence", "perplexity", "nll"]):
                                # Combine probability/confidence/perplexity fields
                                if "probability" in resp:
                                    prob = resp["probability"]
                                elif "confidence" in resp:
                                    prob = resp["confidence"]
                                elif "perplexity" in resp:
                                    prob = resp["perplexity"]
                                elif "nll" in resp:
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
                    elif "response" in parsed:
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


    def _chat_with_format(self, messages: List[Dict[str, str]], schema: BaseModel) -> List[Dict[str, Any]]:
        # print(f"Schema: {schema}")
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **self.config,
                response_format=schema,
            )
            response = completion.choices[0].message.content

            # print(f"Response: {response}")
            parsed_response = self._parse_response_with_schema(response)
            # print(f"Structured Output Response:\n" + "\n".join(str(resp) for resp in parsed_response))
            return parsed_response
        except Exception as e:
            print(f"Error: {e}")
            return []