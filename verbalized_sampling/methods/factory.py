from typing import Dict, Any, List, Optional
from enum import Enum
import os
import random
from pydantic import BaseModel
from .prompt import (
    TaskType,
    PromptTemplateFactory,
    BasePromptTemplate,
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

    # DIRECT = "Baseline"
    # SEQUENCE = "Sequence" 
    # STRUCTURE = "Structure"
    # STRUCTURE_WITH_PROB = "Verbalized Sampling"
    # MULTI_TURN = "Multi-turn"
    # CHAIN_OF_THOUGHT = "Verbalized Sampling (CoT)"
    # COMBINED = "Verbalized Sampling (Combined)"
    # SELF_REFLECTION = "self_reflection"
    # TEMPERATURE_SAMPLING = "temperature_sampling"

def is_method_structured(method: Method) -> bool:
    """Check if a method requires structured JSON output."""
    return method in [
        Method.STRUCTURE, 
        Method.STRUCTURE_WITH_PROB,
        Method.CHAIN_OF_THOUGHT,
        Method.COMBINED,
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
    
    # Map methods to format types for the new prompt system
    METHOD_TO_FORMAT = {
        Method.SEQUENCE: "sequence",
        Method.STRUCTURE: "structure", 
        Method.STRUCTURE_WITH_PROB: "structure_with_prob",
        Method.CHAIN_OF_THOUGHT: "chain_of_thought",
        Method.COMBINED: "combined",
    }

    @staticmethod
    def _get_task_type_from_task_name(task: str) -> TaskType:
        """Map task names to TaskType enum."""
        task_mapping = {
            # Creativity tasks
            "book": TaskType.CREATIVITY,
            "joke": TaskType.CREATIVITY,
            "poem": TaskType.CREATIVITY,
            "speech": TaskType.CREATIVITY,
            "story": TaskType.CREATIVITY,
            
            # Commonsense tasks
            "simple_qa": TaskType.COMMONSENSE,
            
            # Bias tasks
            "rand_num": TaskType.BIAS,
            "state_name": TaskType.BIAS,
            
            # Default to creativity for unknown tasks
        }
        return task_mapping.get(task, TaskType.CREATIVITY)

    @staticmethod
    def _get_prompt_type_from_method(method: Method, all_possible: bool = False) -> str:
        """Map method to prompt type."""
        if method == Method.DIRECT or method == Method.MULTI_TURN:
            return "base"
        elif method == Method.COMBINED:
            return "combined"
        elif all_possible:
            return "standard_all_possible"
        elif method == Method.CHAIN_OF_THOUGHT:
            return "chain_of_thought"
        else:
            return "standard"

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
        task_type: TaskType = None,
    ) -> List[Dict[str, str]]:
        """Pack a prompt using the new class-based prompt system."""
        
        # Get prompt type based on method
        prompt_type = PromptFactory._get_prompt_type_from_method(method, all_possible)
        
        # Initialize system_prompt to None
        system_prompt = None
        
        # Get the prompt template
        try:
            if method == Method.DIRECT or method == Method.MULTI_TURN:
                system_prompt = PromptTemplateFactory.get_prompt(
                    task_type=task_type,
                    prompt_type=prompt_type,
                    target_words=target_words
                )
            else:
                system_prompt = PromptTemplateFactory.get_prompt(
                    task_type=task_type,
                    prompt_type=prompt_type,
                    num_samplings=num_samplings,
                    num_samples_per_prompt=num_samples_per_prompt if method == Method.COMBINED else None,
                    target_words=target_words
                )
        except Exception as e:
            print(f"Warning: Could not get prompt from new system: {e}")
        
        # Add format prompt if needed
        if not strict_json and method in PromptFactory.METHOD_TO_FORMAT:
            format_type = PromptFactory.METHOD_TO_FORMAT[method]
            template = PromptTemplateFactory.get_template(task_type)
            format_prompt = template.get_format_prompt(format_type, num_samplings)

            system_prompt = f"{system_prompt}{format_prompt}"
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

    @staticmethod
    def get_multi_turn_continuation(chat_history: List[Dict[str, str]], task: str, target_words: int) -> List[Dict[str, str]]:
        """Get continuation prompt for multi-turn sampling."""
        task_type = PromptFactory._get_task_type_from_task_name(task)
        template = PromptTemplateFactory.get_template(task_type)
        continuation_prompt = template.get_continue_prompt(num_samplings=1, target_words=target_words)
        # print("Continuation prompt: ", continuation_prompt)
        
        return chat_history + [{"role": "user", "content": continuation_prompt}]

    @staticmethod
    def get_combined_continuation(chat_history: List[Dict[str, str]], num_samplings_per_prompt: int, task: str, target_words: int) -> List[Dict[str, str]]:
        """Get continuation prompt for combined sampling."""
        task_type = PromptFactory._get_task_type_from_task_name(task)
        template = PromptTemplateFactory.get_template(task_type)
        continuation_prompt = template.get_continue_prompt(num_samplings=num_samplings_per_prompt, target_words=target_words)
        # print("Continuation prompt: ", continuation_prompt)
        
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
        
        # Determine task type for new prompt system
        task_type = PromptFactory._get_task_type_from_task_name(task)
        
        return [
            PromptFactory.pack_prompt(
                method, 
                prompt, 
                num_samplings=num_samplings, 
                num_samples_per_prompt=num_samples_per_prompt, 
                target_words=target_words, 
                task_type=task_type,
                **kwargs
            ) 
            for prompt in prompts
        ]