from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any
from .factory import Method

def _get_probability_field_info(probability_definition: str) -> tuple[str, str]:
    """Get field name and description based on probability definition type.
    
    Returns:
        tuple: (field_name, field_description)
    """
    field_info = {
        "implicit": ("probability", "how likely this response would be (from 0.0 to 1.0)."),
        "explicit": ("probability", "the estimated probability from 0.0 to 1.0 of this response given the input prompt (relative to the full distribution)."),
        "relative": ("probability", "a probability value between 0.0 and 1.0, reflecting the relative likelihood of this response given the input."),
        "percentage": ("probability", "the probability of this response relative to the full distribution, expressed as a percentage from 0% to 100%."),
        "confidence": ("confidence", "the normalized likelihood score between 0.0 and 1.0 that indicates how representative or typical this response is compared to the full distribution."),
        "perplexity": ("perplexity", "the exponentiated average negative log likelihood of the response tokens, where lower values indicate higher model certainty in predicting each token."),
        "nll": ("nll", "the sum of the negative log probabilities of each token in the response given the input prompt, with smaller values reflecting higher model confidence."),
    }
    
    return field_info.get(probability_definition)

def _create_structured_response_with_field_schema(probability_definition: str) -> Dict[str, Any]:
    """Create a structured response schema with the appropriate field type."""
    field_name, field_description = _get_probability_field_info(probability_definition)
    
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"structured_with_{field_name}_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "responses": {
                        "type": "array",
                        "description": f"A list of dicts, each with a 'text' and '{field_name}' field, representing possible responses to the input prompt and corresponding {field_name} scores of each response.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string",
                                    "description": "The text of the response."
                                },
                                field_name: {
                                    "type": "number",
                                    "description": field_description,
                                }
                            },
                            "required": ["text", field_name],
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

def _create_chain_of_thought_schema(probability_definition: str) -> Dict[str, Any]:
    """Create a chain of thought schema with the appropriate field type."""
    field_name, field_description = _get_probability_field_info(probability_definition)
    
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"chain_of_thought_{field_name}_response",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Detailed reasoning behind the responses."
                    },
                    "responses": {
                        "type": "array",
                        "description": f"List of potential responses with their {field_name} scores.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string",
                                    "description": "The text of the response."
                                },
                                field_name: {
                                    "type": "number",
                                    "description": field_description,
                                }
                            },
                            "required": ["text", field_name],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["reasoning", "responses"],
                "additionalProperties": False
            }
        }
    }

def _create_combined_schema(probability_definition: str) -> Dict[str, Any]:
    """Create a VS-Multi (vs_multi) schema with the appropriate field type."""
    field_name, field_description = _get_probability_field_info(probability_definition)
    
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"combined_{field_name}_response",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "responses": {
                        "type": "array",
                        "description": f"A list of response objects containing text and {field_name}.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string",
                                    "description": "The text response."
                                },
                                field_name: {
                                    "type": "number",
                                    "description": field_description,
                                }
                            },
                            "required": ["text", field_name],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["responses"],
                "additionalProperties": False
            }
        }
    }

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
def get_tool_schema(method: Method, probability_definition: str = "default") -> List[Dict[str, Any]]:
    """Get tool calling schema for the specified method.
    
    Args:
        method: The sampling method
        probability_definition: Type of probability definition to use
    """

    if method == Method.DIRECT_COT:
        return [{
            "name": "generate_response_with_reasoning",
            "description": "Generate a response with step-by-step reasoning",
            "input_schema": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Step-by-step reasoning process"
                    },
                    "response": {
                        "type": "string",
                        "description": "The response text"
                    }
                },
                "required": ["reasoning", "response"]
            }
        }]
        
    elif method == Method.SEQUENCE:
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
    
    elif method == Method.VS_STANDARD:
        field_name, field_description = _get_probability_field_info(probability_definition)
        return [{
            "name": "generate_responses_with_probability",
            "description": f"Generate structured responses with {field_name} scores",
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
                    field_name: {
                        "type": "array",
                        "description": f"List of {field_name} values corresponding to each response",
                        "items": {
                            "type": "number",
                            "description": field_description,
                        }
                    }
                },
                "required": ["responses", field_name]
            }
        }]
    
    elif method == Method.VS_COT:
        field_name, field_description = _get_probability_field_info(probability_definition)
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
                        "description": f"List of potential responses with their {field_name} scores.",
                        "items": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "The text of the response."
                            },
                            field_name: {
                                "type": "number",
                                "description": field_description,
                            }
                        },
                        "required": [
                            "text",
                            field_name
                        ],
                        "additionalProperties": False
                        }
                    }
                },
                "required": ["reasoning", "responses"]
            }
        }]
    
    elif method == Method.VS_MULTI:
        field_name, field_description = _get_probability_field_info(probability_definition)
        return [{
            "name": "generate_with_reasoning",
            "description": "Generate responses with step-by-step reasoning",
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
                    field_name: {
                        "type": "array",
                        "description": f"List of {field_name} values corresponding to each response",
                        "items": {
                            "type": "number",
                            "description": field_description,
                        }
                    }
                },
                "required": ["responses", field_name]
            }
        }]
    
    else:
        return None

# Legacy JSON schema support for OpenAI/OpenRouter
DirectCoTResponse = {
    "type": "json_schema",
    "json_schema": {
        "name": "direct_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Step-by-step reasoning process"
                },
                "response": {
                    "type": "string",
                    "description": "The response text"
                }
            },
            "required": ["reasoning", "response"],
            "additionalProperties": False
        },
        "strict": True
    }
}


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




def get_schema(method: Method, use_tools: bool = False, probability_definition: str = "default") -> Any:
    """Get schema for the specified method.
    
    Args:
        method: The sampling method
        use_tools: Whether to use tool calling (True for Claude) or JSON schema (False for OpenAI/OpenRouter)
        probability_definition: Type of probability definition to use
    """
    if method == Method.DIRECT_COT:
        return DirectCoTResponse
    elif method == Method.SEQUENCE:
        return SequenceResponse
    elif method == Method.STRUCTURE:
        return StructuredResponseList
    elif method == Method.VS_STANDARD:
        return _create_structured_response_with_field_schema(probability_definition)
    elif method == Method.VS_COT:
        return _create_chain_of_thought_schema(probability_definition)
    elif method == Method.VS_MULTI:
        return _create_combined_schema(probability_definition)
    else:
        return None

def is_claude_model(model_name: str) -> bool:
    """Check if the model is a Claude model that should use tool calling."""
    return "claude" in model_name.lower() or "anthropic" in model_name.lower()

def get_appropriate_schema(method: Method, model_name: str, probability_definition: str = "default") -> Any:
    """Get the appropriate schema for a method and model, considering probability definitions.
    
    Args:
        method: The sampling method
        model_name: The model name to determine schema type
        probability_definition: Type of probability definition to use
    
    Returns:
        The appropriate schema (tool calling for Claude, JSON schema for others)
    """
    if is_claude_model(model_name):
        return get_tool_schema(method, probability_definition)
    else:
        return get_schema(method, use_tools=False, probability_definition=probability_definition)


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

# StructuredResponseListWithProbability = {
#     "type": "json_schema",  # Required for OpenRouter
#     "json_schema": {
#         "name": "structured_with_prob_schema",
#         "schema": {
#             "type": "object",
#             "properties": {
#                 "responses": {
#                     "type": "array",
#                     "description": "A list of dicts, each with a 'text' and 'probability' field, representing possible responses to the input prompt and corresponding probabilities of each response.",
#                     "items": {
#                         "type": "object",
#                         "properties": {
#                             "text": {
#                                 "type": "string",
#                                 "description": "The text of the response."
#                             },
#                             "probability": {
#                                 "type": "number",
#                                 # "description": "How likely each response would be (value between 0 and 1)"
#                                 # "description": "The estimated probability from 0.0 to 1.0 of this response given the input prompt (relative to the full answer space).",
#                                 "description": "a probability value between 0.0 and 1.0, reflecting the relative likelihood of this response given the input.",
#                             }
#                         },
#                         "required": ["text", "probability"],
#                         "additionalProperties": False
#                     }
#                 }
#             },
#             "required": ["responses"],
#             "additionalProperties": False
#         },
#         "strict": True
#     }
# }

# StructuredResponseListWithPerplexity = {
#     "type": "json_schema",  # Required for OpenRouter
#     "json_schema": {
#         "name": "structured_with_perplexity_schema",
#         "schema": {
#             "type": "object",
#             "properties": {
#                 "responses": {
#                     "type": "array",
#                     "description": "A list of dicts, each with a 'text' and 'perplexity' field, representing possible responses to the input prompt and corresponding perplexities of each response.",
#                     "items": {
#                         "type": "object",
#                         "properties": {
#                             "text": {
#                                 "type": "string",
#                                 "description": "The text of the response."
#                             },
#                             "perplexity": {
#                                 "type": "number",
#                                 "description": "The perplexity of the response."
#                             }
#                         },
#                         "required": ["text", "perplexity"],
#                         "additionalProperties": False
#                     }
#                 }
#             },
#             "required": ["responses"],
#             "additionalProperties": False
#         },
#         "strict": True
#     }
# }

# ChainOfThoughtResponse = {
#     "type": "json_schema",
#     "json_schema": {
#         "name": "VS-CoT (vs_cot)_response",
#         "strict": True,
#         "schema": {
#             "type": "object",
#             "properties": {
#                 "reasoning": {
#                     "type": "string",
#                     "description": "Detailed reasoning behind the responses."
#                 },
#                 "responses": {
#                     "type": "array",
#                     "description": "List of potential responses with their probabilities.",
#                     "items": {
#                     "type": "object",
#                     "properties": {
#                         "text": {
#                             "type": "string",
#                             "description": "The text of the response."
#                         },
#                         "probability": {
#                             "type": "number",
#                             # "description": "The likelihood of this response being correct.",
#                             # "description": "The estimated probability from 0.0 to 1.0 of this response given the input prompt (relative to the full answer space).",
#                             "description": "a probability value between 0.0 and 1.0, reflecting the relative likelihood of this response given the input.",
#                         }
#                     },
#                     "required": ["text", "probability"],
#                     "additionalProperties": False
#                     }
#                 }
#             },
#             "required": ["reasoning", "responses"],
#             "additionalProperties": False
#         }
#     }
# }

# CombinedResponse = {
#     "type": "json_schema",
#     "json_schema": {
#         "name": "VS-Multi (vs_multi)_response",
#         "strict": True,
#         "schema": {
#             "type": "object",
#             "properties": {
#                 "responses": {
#                     "type": "array",
#                     "description": "A list of response objects containing text and probability.",
#                     "items": {
#                         "type": "object",
#                         "properties": {
#                             "text": {
#                                 "type": "string",
#                                 "description": "The text response."
#                             },
#                             "probability": {
#                                 "type": "number",
#                                 # "description": "a score from 0.0 to 1.0 representing how likely or typical the response is."
#                                 # "description": "The estimated probability from 0.0 to 1.0 of this response given the input prompt (relative to the full answer space).",
#                                 "description": "a probability value between 0.0 and 1.0, reflecting the relative likelihood of this response given the input.",
#                             }
#                             # "confidence": {
#                             #     "type": "number",
#                             #     "description": "a score from 0.0 to 1.0 representing how likely or typical the response is."
#                             # }
#                         },
#                         "required": ["text", "probability"],
#                         # "required": ["text", "confidence"],
#                         "additionalProperties": False
#                     }
#                 }
#             },
#             "required": ["responses"],
#             "additionalProperties": False
#         }
#     }
# }