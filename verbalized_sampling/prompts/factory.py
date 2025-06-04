from typing import Dict, Any, List, Optional
from enum import Enum
import os
import random
from pydantic import BaseModel
from .prompt import (
    SEQUENCE_PROMPT,
    FORMAT_WITHOUT_PROBABILITY_PROMPT,
    FORMAT_WITH_PROBABILITY_PROMPT,
)

class Method(str, Enum):
    DIRECT = "direct"
    SEQUENCE = "sequence"
    STRUCTURE = "structure"
    STRUCTURE_WITH_PROB = "structure_with_prob"
    MULTI_TURN = "multi_turn"

def is_method_structured(method: Method) -> bool:
    return method in [Method.STRUCTURE, Method.STRUCTURE_WITH_PROB]

class PromptTemplate(BaseModel):
    """Base class for prompt templates."""
    system_prompt: str = "Generate 5 different responses to your interlocutor that are coherent with the chat history and aligned with your persona. Output in JSON format with keys: 'responses' (list of dicts with 'text' and 'probability'). The probability field represents the empirical probability of each response, ranging from 0 to 1. Only output the JSON object, no other text."
    user_prompt: str
    response_format: Optional[Dict[str, Any]] = None

class SamplingPromptTemplate(PromptTemplate):
    """Template for sampling tasks."""
    num_samples: int = 1
    temperature: float = 1.0
    top_p: float = 1.0

class PromptFactory:
    """Factory for creating prompts for different models and tasks."""
    @staticmethod
    def get_prompt(task: str, method: Method, **kwargs) -> List[Dict[str, str]]:
        """Get a prompt for a specific task and format."""
        prompt_path = f"data/{task}.txt"

        if not os.path.exists(prompt_path):
            raise ValueError(f"Prompt file {prompt_path} not found.")
        prompts = []
        with open(prompt_path, "r") as f:
            for line in f:
                prompts.append(line)
        
        # TODO add selection of prompts
        target_prompt = random.choice(prompts)

        if method == Method.DIRECT:
            return [
                {"role": "user", "content": target_prompt}
            ]
        elif method == Method.SEQUENCE:
            formatted_prompt = SEQUENCE_PROMPT.format(num_samplings=kwargs.get("num_samplings", 5))
            return [
                {"role": "system", "content": formatted_prompt},
                {"role": "user", "content": target_prompt}
            ]
        elif method == Method.STRUCTURE:
            formatted_prompt = FORMAT_WITHOUT_PROBABILITY_PROMPT.format(num_samplings=kwargs.get("num_samplings", 5))
            return [
                {"role": "system", "content": formatted_prompt},
                {"role": "user", "content": target_prompt}
            ]
        elif method == Method.STRUCTURE_WITH_PROB:
            formatted_prompt = FORMAT_WITH_PROBABILITY_PROMPT.format(num_samplings=kwargs.get("num_samplings", 5))
            return [
                {"role": "system", "content": formatted_prompt},
                {"role": "user", "content": target_prompt}
            ]
        elif method == Method.MULTI_TURN:
            raise NotImplementedError("Multi-turn prompts are not supported yet.")