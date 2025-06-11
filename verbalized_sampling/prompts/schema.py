from pydantic import BaseModel, Field, ConfigDict
from typing import List
from .factory import Method

class Response(BaseModel):
    model_config = ConfigDict(extra='forbid')
    text: str = Field(..., description="The response text")

class ResponseWithProbability(BaseModel):
    model_config = ConfigDict(extra='forbid')
    text: str = Field(..., description="The response text")
    probability: float = Field(..., description="The probability of the response")

class SequenceResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')
    responses: List[str] = Field(..., description="List of responses")

StructuredResponseList = {
    "type": "json_schema",  # Required for OpenRouter
    "json_schema": {
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
                        },
                        "required": ["text"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["responses"],
            "additionalProperties": False
        },
        "strict": True
    }
}

StructuredResponseListWithProbability = {
    "type": "json_schema",  # Required for OpenRouter
    "json_schema": {
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
                                "description": "How likely each response would be (value between 0 and 1)"
                            }
                        },
                        "required": ["text", "probability"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["responses"],
            "additionalProperties": False
        },
        "strict": True
    }
}

def get_schema(method: Method) -> BaseModel:
    if method == Method.SEQUENCE:
        return SequenceResponse
    elif method == Method.STRUCTURE:
        return StructuredResponseList
    elif method == Method.STRUCTURE_WITH_PROB:
        return StructuredResponseListWithProbability
    else:
        return None