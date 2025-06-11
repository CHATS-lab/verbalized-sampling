"""
Sampling method definitions for verbalized sampling experiments.
"""

from enum import Enum
from .factory import PromptFactory, PromptTemplate, SamplingPromptTemplate, Method, is_method_structured, is_method_multi_turn
from .parser import ResponseParser, parse_response_by_method

__all__ = [
    'PromptFactory', 
    'PromptTemplate', 
    'SamplingPromptTemplate', 
    'Method',
    'is_method_structured',
    'is_method_multi_turn',
    'ResponseParser',
    'parse_response_by_method'
] 

class Method(str, Enum):
    """Available sampling methods for verbalized sampling experiments.
    
    Each method represents a different approach to sampling from LLMs,
    with varying levels of structure and control.
    """
    
    DIRECT = "direct"
    """Direct sampling method.
    
    Uses the prompt as-is without any additional structure. This is the
    baseline method that represents standard LLM generation.
    """
    
    SEQUENCE = "sequence"
    """Sequential sampling method.
    
    Generates multiple responses in a Python list format. Useful for
    generating diverse responses in a structured way.
    """
    
    STRUCTURE = "structure"
    """Structured sampling method.
    
    Uses JSON format with a response field. Enforces structured output
    while maintaining flexibility in the response content.
    """
    
    STRUCTURE_WITH_PROB = "structure_with_prob"
    """Structured sampling with probability.
    
    Uses JSON format with response and probability fields. Allows the model
    to express confidence in its responses.
    """
    
    MULTI_TURN = "multi_turn"
    """Multi-turn conversation sampling.
    
    Uses a conversational format with multiple turns. Useful for
    complex reasoning and step-by-step generation.
    """