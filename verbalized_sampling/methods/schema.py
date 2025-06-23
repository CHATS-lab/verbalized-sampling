from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any
from .factory import Method

# class Response(BaseModel):
#     model_config = ConfigDict(extra='forbid')
#     text: str = Field(..., description="The response text")

# class ResponseWithProbability(BaseModel):
#     model_config = ConfigDict(extra='forbid')
#     text: str = Field(..., description="The response text")
#     probability: float = Field(..., description="The probability of the response", ge=0.0, le=1.0)

# class SequenceResponse(BaseModel):
#     model_config = ConfigDict(extra='forbid')
#     responses: List[str] = Field(..., description="List of responses")

# class StructuredResponseList(BaseModel):
#     model_config = ConfigDict(extra='forbid')
#     responses: List[Response] = Field(..., description="List of responses")

# class StructuredResponseListWithProbability(BaseModel):
#     model_config = ConfigDict(extra='forbid')
#     responses: List[ResponseWithProbability] = Field(..., description="List of responses with probabilities")

# Tool calling schemas for Claude/Anthropic
def get_tool_schema(method: Method) -> List[Dict[str, Any]]:
    """Get tool calling schema for the specified method."""
    
    if method == Method.SEQUENCE:
        return [{
            "name": "generate_responses",
            "description": "Generate multiple responses in sequence format",
            "input_schema": {
                "type": "object",
                "properties": {
                    "responses": {
                        "type": "array",
                        "description": "List of response strings",
                        "items": {
                            "type": "string",
                            "description": "A response text"
                        }
                    }
                },
                "required": ["responses"]
            }
        }]
    
    elif method == Method.STRUCTURE:
        return [{
            "name": "generate_structured_responses",
            "description": "Generate structured responses with text only",
            "input_schema": {
                "type": "object",
                "properties": {
                    "responses": {
                        "type": "array",
                        "description": "List of response strings",
                        "items": {
                            "type": "string",
                            "description": "The response text"
                        }
                    }
                },
                "required": ["responses"]
            }
        }]
    
    elif method == Method.STRUCTURE_WITH_PROB:
        return [{
            "name": "generate_responses_with_probability",
            "description": "Generate structured responses with probability scores",
            "input_schema": {
                "type": "object",
                "properties": {
                    "responses": {
                        "type": "array",
                        "description": "List of response strings",
                        "items": {
                            "type": "string",
                            "description": "The response text"
                        }
                    },
                    "probabilities": {
                        "type": "array",
                        "description": "List of probability values corresponding to each response",
                        "items": {
                            "type": "number",
                            "description": "Probability score between 0 and 1 indicating how likely this response would be",
                        }
                    }
                },
                "required": ["responses", "probabilities"]
            }
        }]
    
    elif method == Method.CHAIN_OF_THOUGHT:
        return [{
            "name": "generate_with_reasoning",
            "description": "Generate responses with step-by-step reasoning",
            "input_schema": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Step-by-step reasoning process"
                    },
                    "responses": {
                        "type": "array",
                        "description": "List of responses based on the reasoning",
                        "items": {
                            "type": "string",
                            "description": "The response text"
                        }
                    }
                },
                "required": ["reasoning", "responses"]
            }
        }]
    
    elif method == Method.SELF_REFLECTION:
        return [{
            "name": "generate_with_reflection",
            "description": "Generate responses with self-reflection and confidence scores",
            "input_schema": {
                "type": "object",
                "properties": {
                    "responses": {
                        "type": "array",
                        "description": "List of response strings",
                        "items": {
                            "type": "string",
                            "description": "The response text"
                        }
                    },
                    "reflections": {
                        "type": "array",
                        "description": "List of self-reflections corresponding to each response",
                        "items": {
                            "type": "string",
                            "description": "Self-reflection on the quality and appropriateness of this response"
                        }
                    },
                    "confidences": {
                        "type": "array",
                        "description": "List of confidence scores corresponding to each response",
                        "items": {
                            "type": "number",
                            "description": "Confidence score between 0 and 1",
                        }
                    }
                },
                "required": ["responses", "reflections", "confidences"]
            }
        }]
    
    else:
        return None

# Legacy JSON schema support for OpenAI/OpenRouter

SequenceResponse ={
    "type": "json_schema",
    "json_schema": {
        "name": "sequence_responses_schema",
        "schema": {
            "type": "object",
            "properties": {
                "responses": {
                    "type": "array",
                    "description": "A list of string responses.",
                    "items": {
                        "type": "string",
                        "description": "Individual response as a string.",
                        "minLength": 1
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

StructuredResponseList = {
    "type": "json_schema",  # Required for OpenRouter
    "json_schema": {
        "name": "structured_schema",
        "schema": {
            "type": "object",
            "properties": {
                "responses": {
                    "type": "array",
                    "description": "A list of dicts, each with a 'text' field, representing possible responses to the input prompt.",
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
        "name": "structured_with_prob_schema",
        "schema": {
            "type": "object",
            "properties": {
                "responses": {
                    "type": "array",
                    "description": "A list of dicts, each with a 'text' and 'probability' field, representing possible responses to the input prompt and corresponding probabilities of each response.",
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


def get_schema(method: Method, use_tools: bool = False) -> Any:
    """Get schema for the specified method.
    
    Args:
        method: The sampling method
        use_tools: Whether to use tool calling (True for Claude) or JSON schema (False for OpenAI/OpenRouter)
    """
    if method == Method.SEQUENCE:
        return SequenceResponse
    elif method == Method.STRUCTURE:
        return StructuredResponseList
    elif method == Method.STRUCTURE_WITH_PROB:
        return StructuredResponseListWithProbability
    else:
        return None

def is_claude_model(model_name: str) -> bool:
    """Check if the model is a Claude model that should use tool calling."""
    return "claude" in model_name.lower() or "anthropic" in model_name.lower()


# Legacy JSON schema support for OpenAI/OpenRouter
# SequenceResponse = {
#     "type": "json_schema",
#     "json_schema": {
#         "name": "sequence_responses_schema",
#         "schema": {
#             "type": "object",
#             "properties": {
#                 "responses": {
#                     "type": "array",
#                     "description": "List of response strings",
#                     "items": {
#                         "type": "string",
#                         "description": "A single response candidate"
#                     }
#                 }
#             },
#             "required": ["responses"],
#             "additionalProperties": False
#         },
#         "strict": True
#     }
# }