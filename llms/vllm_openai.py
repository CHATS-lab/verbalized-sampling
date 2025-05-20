from openai import OpenAI
from llms.lm_model import LM_Agent
import os
from typing import List, Dict, Any
import json
from pydantic import BaseModel

class VerbalizedSamplingResponse(BaseModel):
    text: str
    probability: float

class VerbalizedSamplingResponseList(BaseModel):
    responses: List[VerbalizedSamplingResponse]

class VLLM_OpenAI_Agent(LM_Agent):
    def __init__(
        self,
        model_name,
        sim_type,
        config={},
    ):
        super().__init__(model_name, sim_type, config)

        self.client = OpenAI(
            base_url="http://localhost:8000/v1"
        )
            
        # Fix for the response_format implementation
        self.response_format = VerbalizedSamplingResponseList

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


    # def parse_response(self, response) -> List[Dict[str, Any]]:
    #     parsed_responses = []
    #     if isinstance(response, str):
    #         response = json.loads(response)
    #         for resp in response["responses"]:
    #             parsed_responses.append({
    #                 "text": resp["text"],
    #                 "probability": resp["probability"]
    #             })
    #     return parsed_responses


    def _chat_with_format(self, messages) -> List[Dict[str, Any]]:
        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=messages,
                response_format=self.response_format,
                extra_body=dict(guided_decoding_backend="auto"),
            )
            response = completion.choices[0].message.content
            
            # Parse the JSON response
            parsed_json = json.loads(response)
            parsed_responses = []
            
            message = completion.choices[0].message
            # print(message)
            assert message.parsed
            
            # Extract responses from the parsed message
            parsed_responses = []
            if hasattr(message.parsed, 'responses'):
                for resp in message.parsed.responses:
                    parsed_responses.append({
                        "text": resp.text,
                        "probability": resp.probability
                    })
            else:
                raise ValueError("No responses found in the parsed message")
            return parsed_responses
            
        except Exception as e:
            print(f"Error in _chat_with_format: {e}")
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