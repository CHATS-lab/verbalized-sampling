from typing import Dict, Any, List, Optional
from enum import Enum
import os
import random
from pydantic import BaseModel
from .prompt import (
    SEQUENCE_PROMPT,
    STRUCTURE_RESPONSE_ONLY_PROMPT,
    STRUCTURE_WITH_PROBABILITY_PROMPT,
    MULTI_TURN_INITIAL_PROMPT,
    MULTI_TURN_CONTINUE_PROMPT,
    MULTI_TURN_FINAL_PROMPT,
    CHAIN_OF_THOUGHT_PROMPT,
    SELF_REFLECTION_PROMPT,
    TEMPERATURE_SAMPLING_PROMPT,
    # Legacy imports for backward compatibility
    FORMAT_WITHOUT_PROBABILITY_PROMPT,
    FORMAT_WITH_PROBABILITY_PROMPT,
)

class Method(str, Enum):
    DIRECT = "direct"
    SEQUENCE = "sequence" 
    STRUCTURE = "structure"
    STRUCTURE_WITH_PROB = "structure_with_prob"
    MULTI_TURN = "multi_turn"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    SELF_REFLECTION = "self_reflection"
    TEMPERATURE_SAMPLING = "temperature_sampling"

def is_method_structured(method: Method) -> bool:
    """Check if a method requires structured JSON output."""
    return method in [
        Method.STRUCTURE, 
        Method.STRUCTURE_WITH_PROB,
        Method.CHAIN_OF_THOUGHT,
        Method.SELF_REFLECTION,
        Method.TEMPERATURE_SAMPLING
    ]

def is_method_multi_turn(method: Method) -> bool:
    """Check if a method requires multi-turn interaction."""
    return method == Method.MULTI_TURN

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
            formatted_prompt = STRUCTURE_RESPONSE_ONLY_PROMPT.format(num_samplings=kwargs.get("num_samplings", 5))
            return [
                {"role": "system", "content": formatted_prompt},
                {"role": "user", "content": target_prompt}
            ]
        elif method == Method.STRUCTURE_WITH_PROB:
            formatted_prompt = STRUCTURE_WITH_PROBABILITY_PROMPT.format(num_samplings=kwargs.get("num_samplings", 5))
            return [
                {"role": "system", "content": formatted_prompt},
                {"role": "user", "content": target_prompt}
            ]
        elif method == Method.CHAIN_OF_THOUGHT:
            formatted_prompt = CHAIN_OF_THOUGHT_PROMPT.format(num_samplings=kwargs.get("num_samplings", 5))
            return [
                {"role": "system", "content": formatted_prompt},
                {"role": "user", "content": target_prompt}
            ]
        elif method == Method.SELF_REFLECTION:
            formatted_prompt = SELF_REFLECTION_PROMPT.format(num_samplings=kwargs.get("num_samplings", 5))
            return [
                {"role": "system", "content": formatted_prompt},
                {"role": "user", "content": target_prompt}
            ]
        elif method == Method.TEMPERATURE_SAMPLING:
            formatted_prompt = TEMPERATURE_SAMPLING_PROMPT.format(num_samplings=kwargs.get("num_samplings", 5))
            return [
                {"role": "system", "content": formatted_prompt},
                {"role": "user", "content": target_prompt}
            ]
        elif method == Method.MULTI_TURN:
            formatted_prompt = MULTI_TURN_INITIAL_PROMPT.format(num_samplings=kwargs.get("num_samplings", 5))
            return [
                {"role": "system", "content": formatted_prompt},
                {"role": "user", "content": target_prompt}
            ]
        else:
            raise ValueError(f"Unsupported method: {method}")

    @staticmethod
    def get_multi_turn_continuation(turn_number: int, total_turns: int, original_prompt: str) -> List[Dict[str, str]]:
        """Get continuation prompt for multi-turn sampling."""
        if turn_number == total_turns:
            continuation_prompt = MULTI_TURN_FINAL_PROMPT.format(
                current_turn=turn_number, 
                total_turns=total_turns
            )
        else:
            continuation_prompt = MULTI_TURN_CONTINUE_PROMPT.format(
                current_turn=turn_number, 
                total_turns=total_turns
            )
        
        return [
            {"role": "user", "content": f"{continuation_prompt}\n\nOriginal prompt: {original_prompt}"}
        ]