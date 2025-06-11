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

class StructuredResponseList(BaseModel):
    model_config = ConfigDict(extra='forbid')
    responses: List[Response] = Field(..., description="List of responses")

class StructuredResponseListWithProbability(BaseModel):
    model_config = ConfigDict(extra='forbid')
    responses: List[ResponseWithProbability] = Field(..., description="List of responses with probabilities")

def get_schema(method: Method) -> BaseModel:
    if method == Method.SEQUENCE:
        return SequenceResponse
    elif method == Method.STRUCTURE:
        return StructuredResponseList
    elif method == Method.STRUCTURE_WITH_PROB:
        return StructuredResponseListWithProbability
    else:
        return None