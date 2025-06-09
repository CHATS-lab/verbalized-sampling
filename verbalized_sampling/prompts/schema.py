from pydantic import BaseModel, Field
from typing import List
from .factory import Method

class Response(BaseModel):
    text: str = Field(..., description="The response text")

class ResponseWithProbability(BaseModel):
    text: str = Field(..., description="The response text")
    probability: float = Field(..., description="The probability of the response")

class SequenceResponse(BaseModel):
    responses: List[str] = Field(..., description="List of responses")

class StructuredResponseList(BaseModel):
    responses: List[Response] = Field(..., description="List of responses")

class StructuredResponseListWithProbability(BaseModel):
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