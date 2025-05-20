import os
from typing import Any, Dict, Generator, List
import json
import requests

from pydantic import BaseModel
from openai import OpenAI
from llms.lm_model import LM_Agent


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


class OpenRouterAgent(LM_Agent):
    def __init__(
        self,
        model_name,
        sim_type,
        config={},
    ):
        super().__init__(model_name, sim_type, config)

        if model_name in OPENROUTER_MODELS_MAPPING:
            self.model_name = OPENROUTER_MODELS_MAPPING[model_name]

        self.client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )
            
        # Fix for the response_format implementation
        self.response_format = None
        if self.sim_type == "sampling":
            self.response_format = {
                "type": "json_schema",  # This was missing and is required for OpenRouter
                "json_schema": {        # This wrapper was missing
                    "name": "responses_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "responses": {
                                "type": "array",
                                "description": "A list of different possible responses to the interlocutor's message, each with a response text and a probability.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "text": {
                                            "type": "string",
                                            "description": "The text of the response."
                                        },
                                        "probability": {
                                            "type": "number",
                                            "description": "The confidence level of the response."
                                        }
                                    },
                                    "required": [
                                        "text",
                                        "probability"
                                    ],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": [
                            "responses"
                        ],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }


    def _chat(self, messages) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **self.config,
        )
        response = response.choices[0].message.content
        if response:
            response = response.replace("\n", "")
        return response


    def parse_response(self, response) -> List[Dict[str, Any]]:
        parsed_responses = []
        if isinstance(response, str):
            response = json.loads(response)
            for resp in response["responses"]:
                parsed_responses.append({
                    "text": resp["text"],
                    "probability": resp["probability"]
                })
        return parsed_responses


    def _chat_with_format(self, messages) -> List[Dict[str, Any]]:
        try:
            if "claude" in self.model_name:
                # Claude doesn't support the structured output format in OpenRouter
                multiple_responses = self._chat(messages)
                # print(f"Multiple Responses: {multiple_responses}")
                return self.parse_response(multiple_responses)
            else:
                # For Gemini and other models that support structured output
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    **self.config,
                    response_format=self.response_format,
                )
                response = completion.choices[0].message.content
                print(f"Structured Output Response: {response}")
                return self.parse_response(response)
        except Exception as e:
            print(f"Error in _chat_with_format: {e}")
            # Fallback to regular chat if parsing fails
            response = self._chat(messages)
            return self.parse_response(response)