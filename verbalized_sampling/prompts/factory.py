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
    
    PROMPT_MAP = {
        Method.SEQUENCE: SEQUENCE_PROMPT,
        Method.STRUCTURE: STRUCTURE_RESPONSE_ONLY_PROMPT,
        Method.STRUCTURE_WITH_PROB: STRUCTURE_WITH_PROBABILITY_PROMPT,
        Method.CHAIN_OF_THOUGHT: CHAIN_OF_THOUGHT_PROMPT,
        Method.SELF_REFLECTION: SELF_REFLECTION_PROMPT,
        Method.TEMPERATURE_SAMPLING: TEMPERATURE_SAMPLING_PROMPT,
        Method.MULTI_TURN: MULTI_TURN_INITIAL_PROMPT,
    }
    @staticmethod
    def pack_prompt(
        method: Method,
        prompt: str,
        chat_history: List[Dict[str, str]] = None,
        num_samplings: int = 5,
    ) -> List[Dict[str, str]]:
        
        """Pack a prompt for a specific method."""
        if method == Method.DIRECT:
            return [{"role": "user", "content": prompt}]
        else:
            system_prompt = PromptFactory.PROMPT_MAP[method].format(num_samplings=num_samplings)
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
    
    @staticmethod
    def get_prompt(
        task: str, 
        method: Method, 
        num_samplings: int = 5,
        sample_size: int = None,
        random_seed: int = None,
    ) -> List[List[Dict[str, str]]]:
        """Get a prompt for a specific task and format.
        
        Returns:
            List[List[Dict[str, str]]]: A list of prompts, each containing a system and user message.
        """
        prompt_path = f"data/{task}.txt"

        if not os.path.exists(prompt_path):
            raise ValueError(f"Prompt file {prompt_path} not found.")
        prompts = []
        with open(prompt_path, "r") as f:
            for line in f:
                prompts.append(line)
        
        # TODO add selection of prompts
        if (sample_size is not None) and (random_seed is not None):
            random.seed(random_seed)
            prompts = random.sample(prompts, sample_size)
        else:
            prompts = random.sample(prompts, 1)

        return [PromptFactory.pack_prompt(method, prompt) for prompt in prompts]

    @staticmethod
    def get_multi_turn_continuation(turn_number: int, total_turns: int, chat_history: List[Dict[str, str]]) -> List[Dict[str, str]]:
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
        
        return chat_history + [{"role": "user", "content": continuation_prompt}]