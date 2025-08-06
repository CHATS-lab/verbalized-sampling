from typing import Dict, Any, List, Optional, Union
from enum import Enum
import os
import random
import numpy as np
from datasets import load_dataset
from pydantic import BaseModel
from .prompt import (
    TaskType,
    PromptTemplateFactory,
    BasePromptTemplate,
)

class Method(str, Enum):
    """Available sampling methods for verbalized sampling experiments."""
    DIRECT = "direct"
    DIRECT_BASE = "direct_base"
    DIRECT_COT = "direct_cot"
    SEQUENCE = "sequence" 
    STRUCTURE = "structure"
    STRUCTURE_WITH_PROB = "structure_with_prob"
    MULTI_TURN = "multi_turn"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    COMBINED = "combined"


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

def is_method_base_model(method: Method) -> bool:
    """Check if a method is for base models (no chat template)."""
    return method == Method.DIRECT_BASE

class PromptTemplate(BaseModel):
    """Base class for prompt templates."""
    system_prompt: str
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
            
            # Synthetic data tasks
            "gsm8k": TaskType.SYNTHETIC_DATA,
            "livecodebench": TaskType.SYNTHETIC_DATA,
            
            # Default to creativity for unknown tasks
        }
        return task_mapping.get(task, TaskType.CREATIVITY)

    @staticmethod
    def _get_prompt_type_from_method(method: Method, all_possible: bool = False) -> str:
        """Map method to prompt type."""
        if method == Method.DIRECT or method == Method.MULTI_TURN:
            return "base"
        elif method == Method.DIRECT_BASE:
            return "base_model"
        elif method == Method.DIRECT_COT:
            return "base_cot"
        elif method == Method.STRUCTURE_WITH_PROB:
            return "vs_standard"
        elif method == Method.CHAIN_OF_THOUGHT:
            return "vs_cot"
        elif method == Method.COMBINED:
            return "vs_multi_turn"
        elif all_possible:
            return "standard_all_possible"
        else: # Method.SEQUENCE, Method.STRUCTURE
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
        task_name: str = None,
    ) -> Union[List[Dict[str, str]], str]:
        """Pack a prompt using the new class-based prompt system."""
        
        # Get prompt type based on method
        prompt_type = PromptFactory._get_prompt_type_from_method(method, all_possible)
        
        # Initialize system_prompt to None
        system_prompt = None
        
        # Get the prompt template
        try:
            if method == Method.DIRECT or method == Method.MULTI_TURN or method == Method.DIRECT_COT or method == Method.DIRECT_BASE:
                system_prompt = PromptTemplateFactory.get_prompt(
                    task_type=task_type,
                    prompt_type=prompt_type,
                    target_words=target_words,
                    task_name=task_name
                )
            else:
                system_prompt = PromptTemplateFactory.get_prompt(
                    task_type=task_type,
                    prompt_type=prompt_type,
                    num_samplings=num_samplings,
                    num_samples_per_prompt=num_samples_per_prompt if method == Method.COMBINED else None,
                    target_words=target_words,
                    task_name=task_name
                )
        except Exception as e:
            print(f"Warning: Could not get prompt from new system: {e}")
        
        # Add format prompt if needed
        if not strict_json and method in PromptFactory.METHOD_TO_FORMAT:
            format_type = PromptFactory.METHOD_TO_FORMAT[method]
            template = PromptTemplateFactory.get_template(task_type)
            format_prompt = template.get_format_prompt(format_type, num_samplings)

            system_prompt = f"{system_prompt}{format_prompt}"
        
        # print("System prompt: ", system_prompt)
        # print("User prompt: ", prompt)
        
        # Handle base model format (no chat template, just completion)
        if method == Method.DIRECT_BASE:
            # Format for base model completion using the same pattern as test_base_model.py
            combined_prompt = f"### User: {system_prompt}\n{prompt}\n### Assistant: "
            return combined_prompt
        
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
        print("Continuation prompt: ", continuation_prompt)
        
        return chat_history + [{"role": "user", "content": continuation_prompt}]

    @staticmethod
    def get_combined_continuation(chat_history: List[Dict[str, str]], num_samplings_per_prompt: int, task: str, target_words: int) -> List[Dict[str, str]]:
        """Get continuation prompt for combined sampling."""
        task_type = PromptFactory._get_task_type_from_task_name(task)
        template = PromptTemplateFactory.get_template(task_type)
        continuation_prompt = template.get_continue_prompt(num_samplings=num_samplings_per_prompt, target_words=target_words)
        print("Continuation prompt: ", continuation_prompt)
        
        return chat_history + [{"role": "user", "content": continuation_prompt}]
    
    @staticmethod
    def get_gsm8k_task_prompts(num_icl_example: int, random_seed: int) -> List[str]:
        """Get prompts for the GSM8K task."""
        ds = load_dataset("gsm8k", "main", split="train")
        np.random.seed(random_seed)
        idxs = np.random.choice(range(len(ds)), num_icl_example, replace=False)
        icl_examples = [ds[int(i)] for i in idxs]
        
        user_prompts = f"""Generate grade school math word problems that involve a sequence of basic arithmetic calculations (addition, subtraction, multiplication, division).
        A bright middle school student should be able to solve each problem. Problems require no concepts beyond the level of early Algebra.
        
        For each problem:
        - Specify the question.
        - Then provide a brief reasoning and the numerical answer.
        - The answer should be given after four hash marks (####) at the end of the reasoning.

        Format your generated problems as follows:
        Question: [your question]
        Answer: [your brief reasoning and answer, ending with #### [numerical answer]]

        Here are some examples you can use as inspiration:
        Example 1: Question: {icl_examples[0]['question']}\nAnswer: {icl_examples[0]['answer']}
        Example 2: Question: {icl_examples[1]['question']}\nAnswer: {icl_examples[1]['answer']}
        Example 3: Question: {icl_examples[2]['question']}\nAnswer: {icl_examples[2]['answer']}

        Ensure that each problem you generate is unique in both topic and content compared to the given examples.
        Only include the question and answer in your response, and always begin your response with the question.
        """
        return [user_prompts]
    
    @staticmethod
    def get_livecodebench_task_prompts(num_icl_example: int, random_seed: int) -> List[str]:
        """Get prompts for generating synthetic LiveCodeBench-style coding problems."""
        ds = load_dataset("livecodebench/code_generation_lite", version_tag="release_v6", split="test", trust_remote_code='True')
        np.random.seed(random_seed)
        idxs = np.random.choice(range(len(ds)), num_icl_example, replace=False)
        icl_examples = [ds[int(i)] for i in idxs]
        
        user_prompt = f"""Generate examples of natural language programming-esque tasks with a specified test input and the resulting output answer. 
        Provide your examples in the following format:
        "Question:
        [question]
        Test Input:
        [test_input]
        Reasoning:
        [reasoning]
        Answer:
        [answer]"

        Here are some examples:
        Example 1: {icl_examples[0]['question_content']}
        Example 2: {icl_examples[1]['question_content']}
        Example 3: {icl_examples[2]['question_content']}

        Generate different problems following this format. Your question should be different in content from the examples. 
        Make sure to only provide only the question, test input, reasoning, and answer. Start each example with the question. """

        return [user_prompt]
    
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
    ) -> List[Union[List[Dict[str, str]], str]]:
        """Get a prompt for a specific task and format.
        
        Returns:
            List[Union[List[Dict[str, str]], str]]: A list of prompts, each containing either:
                - A list of system and user messages (for chat models)
                - A string prompt (for base models)
        """
        prompts = []
        if task == "gsm8k":
            prompts = PromptFactory.get_gsm8k_task_prompts(num_icl_example=3, random_seed=random_seed)
        elif task == "livecodebench":
            prompts = PromptFactory.get_livecodebench_task_prompts(num_icl_example=3, random_seed=random_seed)
        elif (task == "poem") and (method == Method.DIRECT_BASE): # Handle poem task with clean data
            prompt_path = "data/poem_titles.txt"
        else:
            prompt_path = f"data/{task}.txt"

        # Only try to read from file if we don't have prompts from the special task methods
        if not prompts:
            if not os.path.exists(prompt_path):
                raise ValueError(f"Prompt file {prompt_path} not found.")
            
            prompts = []
            with open(prompt_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        prompts.append(line)
        
        # TODO add selection of prompts
        if (num_prompts is not None) and (random_seed is not None):
            random.seed(random_seed)
            prompts = random.sample(prompts, min(num_prompts, len(prompts)))

        print(f"Num samplings: {num_samplings}, Method: {method}, Sample size: {num_prompts}, Random seed: {random_seed}")
        
        # Determine task type for new prompt system
        task_type = PromptFactory._get_task_type_from_task_name(task)
        
        packed_prompts = []
        for prompt in prompts:
            packed_prompt = PromptFactory.pack_prompt(
                method, 
                prompt, 
                num_samplings=num_samplings, 
                num_samples_per_prompt=num_samples_per_prompt, 
                target_words=target_words, 
                task_type=task_type,
                task_name=task,
                **kwargs
            )
            packed_prompts.append(packed_prompt)
        
        return packed_prompts