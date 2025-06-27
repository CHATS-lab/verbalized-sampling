from typing import Dict, Any, List, Optional
from enum import Enum
import os
import random
from pydantic import BaseModel
from .prompt import (
    BASE_PROMPT,
    STANDARD_PROMPT,
    STANDARD_ALL_POSSIBLE_PROMPT,
    SEQUENCE_FORMAT_PROMPT,
    STRUCTURE_FORMAT_PROMPT,
    STRUCTURE_WITH_PROBABILITY_FORMAT_PROMPT,
    CHAIN_OF_THOUGHT_PROMPT,
    MULTI_TURN_CONTINUE_PROMPT,
    BASE_PROMPT_TARGET_WORDS,
    STANDARD_PROMPT_TARGET_WORDS,
    STANDARD_ALL_POSSIBLE_PROMPT_TARGET_WORDS,
    STANDARD_COMBINED_PROMPT,
    STANDARD_COMBINED_PROMPT_TARGET_WORDS,
    COMBINED_CONTINUE_PROMPT,
)

class Method(str, Enum):
    """Available sampling methods for verbalized sampling experiments."""
    DIRECT = "direct"
    SEQUENCE = "sequence" 
    STRUCTURE = "structure"
    STRUCTURE_WITH_PROB = "structure_with_prob"
    MULTI_TURN = "multi_turn"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    COMBINED = "combined"
    # SELF_REFLECTION = "self_reflection"
    # TEMPERATURE_SAMPLING = "temperature_sampling"

def is_method_structured(method: Method) -> bool:
    """Check if a method requires structured JSON output."""
    return method in [
        Method.STRUCTURE, 
        Method.STRUCTURE_WITH_PROB,
        Method.CHAIN_OF_THOUGHT,
        Method.COMBINED,
        # Method.SELF_REFLECTION,
        # Method.TEMPERATURE_SAMPLING,
    ]

def is_method_multi_turn(method: Method) -> bool:
    """Check if a method requires multi-turn interaction."""
    return method == Method.MULTI_TURN

def is_method_combined(method: Method) -> bool:
    """Check if a method requires combined sampling."""
    return method == Method.COMBINED

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
        Method.SEQUENCE: SEQUENCE_FORMAT_PROMPT,
        Method.STRUCTURE: STRUCTURE_FORMAT_PROMPT,
        Method.STRUCTURE_WITH_PROB: STRUCTURE_WITH_PROBABILITY_FORMAT_PROMPT,
        Method.CHAIN_OF_THOUGHT: CHAIN_OF_THOUGHT_PROMPT,
        Method.COMBINED: STANDARD_COMBINED_PROMPT,
    }

    @staticmethod
    def pack_prompt(
        method: Method,
        prompt: str,
        chat_history: List[Dict[str, str]] = None,
        num_samplings: int = 5,
        num_samples_per_prompt: int = 2,
        target_words: int = 0,
        all_possible: bool = False,
        strict_json: bool = False,
    ) -> List[Dict[str, str]]:
        
        if (method == Method.DIRECT) or (method == Method.MULTI_TURN):
            if target_words > 0:
                system_prompt = BASE_PROMPT_TARGET_WORDS.format(num_samplings=num_samplings, target_words=target_words)
            else:
                system_prompt = BASE_PROMPT.format(num_samplings=num_samplings)
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]

        if method == Method.COMBINED:
            if target_words > 0:
                system_prompt = STANDARD_COMBINED_PROMPT_TARGET_WORDS.format(num_samplings=num_samples_per_prompt, target_words=target_words)
            else:
                system_prompt = STANDARD_COMBINED_PROMPT.format(num_samplings=num_samples_per_prompt)
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        
        # print(f"Method: {method}")
        if all_possible:
            if target_words > 0:
                system_prompt = STANDARD_ALL_POSSIBLE_PROMPT_TARGET_WORDS.format(num_samplings=num_samplings, target_words=target_words)
            else:
                system_prompt = STANDARD_ALL_POSSIBLE_PROMPT.format(num_samplings=num_samplings)
        elif method == Method.CHAIN_OF_THOUGHT:
            system_prompt = CHAIN_OF_THOUGHT_PROMPT.format(num_samplings=num_samplings)
        else:
            if target_words > 0:
                system_prompt = STANDARD_PROMPT_TARGET_WORDS.format(num_samplings=num_samplings, target_words=target_words)
            else:
                system_prompt = STANDARD_PROMPT.format(num_samplings=num_samplings)
        
        if strict_json:
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        elif method in PromptFactory.PROMPT_MAP:
            system_prompt = f"{system_prompt}\n\n{PromptFactory.PROMPT_MAP[method]}"
        else:
            raise ValueError(f"Method {method} not supported.")

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

    @staticmethod
    def get_multi_turn_continuation(chat_history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Get continuation prompt for multi-turn sampling."""
        continuation_prompt = MULTI_TURN_CONTINUE_PROMPT
        return chat_history + [{"role": "user", "content": continuation_prompt}]

    @staticmethod
    def get_combined_continuation(chat_history: List[Dict[str, str]], num_samplings_per_prompt: int = 3) -> List[Dict[str, str]]:
        """Get continuation prompt for combined sampling."""
        continuation_prompt = COMBINED_CONTINUE_PROMPT.format(num_samplings=num_samplings_per_prompt)
        return chat_history + [{"role": "user", "content": continuation_prompt}]
    
    @staticmethod
    def get_prompt(
        task: str, 
        method: Method, 
        num_samplings: int = 5,
        num_prompts: int = None,
        num_samples_per_prompt: int = 2,
        random_seed: int = None,
        target_words: int = 200,
        **kwargs,
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
        if (num_prompts is not None) and (random_seed is not None):
            random.seed(random_seed)
            prompts = random.sample(prompts, min(num_prompts, len(prompts)))

        print(f"Num samplings: {num_samplings}, Method: {method}, Sample size: {num_prompts}, Random seed: {random_seed}")
        return [PromptFactory.pack_prompt(method, prompt, num_samplings=num_samplings, num_samples_per_prompt=num_samples_per_prompt, target_words=target_words, **kwargs) for prompt in prompts]