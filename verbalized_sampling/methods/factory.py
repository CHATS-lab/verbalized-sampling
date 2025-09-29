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
    # Standard baseline methods
    DIRECT = "direct"
    DIRECT_BASE = "direct_base"
    DIRECT_COT = "direct_cot"
    SEQUENCE = "sequence"
    STRUCTURE = "structure"
    MULTI_TURN = "multi_turn"

    # New VS method names (paper-aligned)
    VS_STANDARD = "vs_standard"           # Primary VS method with probabilities
    VS_COT = "vs_cot"                    # VS with chain-of-thought
    VS_MULTI = "vs_multi"                # VS with multi-turn/vs_multi sampling

    # Legacy aliases for backwards compatibility (deprecated)
    STRUCTURE_WITH_PROB = "vs_standard"  # Use VS_STANDARD instead
    CHAIN_OF_THOUGHT = "vs_cot"        # Use VS_COT instead
    COMBINED = "vs_multi"                        # Use VS_MULTI instead

    # Additional methods
    STANDARD_ALL_POSSIBLE = "standard_all_possible"

    @property
    def paper_name(self) -> str:
        """Get the paper-published name for this method."""
        mapping = {
            "vs_standard": "VS-Standard",
            "vs_standard": "VS-Standard",  # Legacy
            "vs_cot": "VS-CoT",
            "vs_cot": "VS-CoT",  # Legacy
            "vs_multi": "VS-Multi",
            "vs_multi": "VS-Multi",  # Legacy
        }
        return mapping.get(self.value, self.value.replace("_", "-").title())


# Migration mapping for backwards compatibility
METHOD_MIGRATION_MAP = {
    "vs_standard": "vs_standard",
    "vs_cot": "vs_cot",
    "vs_multi": "vs_multi"
}

REVERSE_MIGRATION_MAP = {v: k for k, v in METHOD_MIGRATION_MAP.items()}


def migrate_method_name(old_name: str) -> str:
    """Convert old method name to new name."""
    return METHOD_MIGRATION_MAP.get(old_name, old_name)


def legacy_method_name(new_name: str) -> str:
    """Convert new method name to legacy name for data access."""
    return REVERSE_MIGRATION_MAP.get(new_name, new_name)


def normalize_method(method: Union[str, Method]) -> Method:
    """Normalize method to use new naming, with migration support."""
    if isinstance(method, str):
        # Try migration first
        migrated = migrate_method_name(method)
        try:
            return Method(migrated)
        except ValueError:
            # Fall back to original if migration fails
            return Method(method)
    return method


def is_method_structured(method: Method) -> bool:
    """Check if a method requires structured JSON output."""
    return method in [
        Method.STRUCTURE,
        Method.VS_STANDARD,  # Legacy
        Method.VS_STANDARD,          # New
        Method.VS_COT,     # Legacy
        Method.VS_COT,               # New
        Method.VS_MULTI,             # Legacy
        Method.VS_MULTI,             # New
        Method.STANDARD_ALL_POSSIBLE,
    ]

def is_method_multi_turn(method: Method) -> bool:
    """Check if a method requires multi-turn interaction."""
    return method == Method.MULTI_TURN

def is_method_combined(method: Method) -> bool:
    """Check if a method requires VS-Multi (vs_multi) sampling."""
    return method in [Method.VS_MULTI, Method.VS_MULTI]  # Support both old and new

def is_vs_method(method: Method) -> bool:
    """Check if a method is a Verbalized Sampling method."""
    return method in [
        Method.VS_STANDARD, Method.VS_COT, Method.VS_MULTI,
        Method.VS_STANDARD, Method.VS_COT, Method.VS_MULTI  # Legacy support
    ]

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
        Method.DIRECT_COT: "direct_cot",

        # New VS method names (primary)
        Method.VS_STANDARD: "vs_standard",
        Method.VS_COT: "vs_cot",
        Method.VS_MULTI: "vs_multi",

        # Legacy method names (deprecated but supported for backwards compatibility)
        Method.VS_STANDARD: "vs_standard",
        Method.VS_COT: "vs_cot",
        Method.VS_MULTI: "vs_multi",
    }
    
    # Available probability definition types
    PROBABILITY_DEFINITIONS = {
        "default": "Standard probability definition",
        "implicit": "Simple likelihood definition",
        "explicit": "Explicit probability definition", 
        "relative": "Relative likelihood definition",
        "percentage": "Percentage likelihood definition",
        "confidence": "Confidence score definition",
        "perplexity": "Perplexity-based definition",
        "nll": "Negative log likelihood definition",
    }

    @staticmethod
    def get_available_probability_definitions() -> Dict[str, str]:
        """Get available probability definition types and their descriptions."""
        return PromptFactory.PROBABILITY_DEFINITIONS.copy()
    
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
            "amc_aime_math": TaskType.SYNTHETIC_DATA,
            
            # Synthetic negative tasks
            "synthetic_negative": TaskType.SYNTHETIC_NEGATIVE,

            # Safety tasks
            "safety": TaskType.SAFETY,

            # Math tasks
            "math_math": TaskType.MATH,
            "math_aime": TaskType.MATH,
            "math_amc": TaskType.MATH,
            "math_minerva": TaskType.MATH,
            "math_olympiad_bench": TaskType.MATH,

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
        elif method == Method.VS_COT:
            return "vs_cot"
        elif method == Method.VS_MULTI:
            return "vs_multi"
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
        probability_definition: str = None,
        probability_tuning: float = -1,
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
                    num_samples_per_prompt=num_samples_per_prompt if method == Method.VS_MULTI else None,
                    target_words=target_words,
                    task_name=task_name
                )
        except Exception as e:
            print(f"Warning: Could not get prompt from new system: {e}")
        
        # Add format prompt if needed
        if not strict_json and method in PromptFactory.METHOD_TO_FORMAT:
            format_type = PromptFactory.METHOD_TO_FORMAT[method]
            template = PromptTemplateFactory.get_template(task_type)
            format_prompt = template.get_format_prompt(
                format_type, 
                num_samplings, 
                probability_definition,
                probability_tuning
            )
            system_prompt = f"{system_prompt}{format_prompt}"
        
        
        print("System prompt: ", system_prompt)
        print("User prompt: ", prompt)
        
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
        print("Multi-turn continuation prompt: ", continuation_prompt)
        
        return chat_history + [{"role": "user", "content": continuation_prompt}]

    @staticmethod
    def get_combined_continuation(chat_history: List[Dict[str, str]], num_samplings_per_prompt: int, task: str, target_words: int) -> List[Dict[str, str]]:
        """Get continuation prompt for VS-Multi (vs_multi) sampling."""
        task_type = PromptFactory._get_task_type_from_task_name(task)
        template = PromptTemplateFactory.get_template(task_type)
        continuation_prompt = template.get_continue_prompt(num_samplings=num_samplings_per_prompt, target_words=target_words)
        print("VS-Multi continuation prompt: ", continuation_prompt)
        
        return chat_history + [{"role": "user", "content": continuation_prompt}]
    
    @staticmethod
    def get_gsm8k_task_prompts(num_icl_example: int, random_seed: int) -> List[str]:
        """Get prompts for the GSM8K task."""
        user_prompts = f"""Generate a grade school math word problem that involves a sequence of basic arithmetic calculations (addition, subtraction, multiplication, division).
        A bright middle school student should be able to solve the problem. The difficulty of the problem should be similar to typical middle school math problems.

        Format the generated problem as follows:
        Question: [question]
        """
        return [user_prompts]

    @staticmethod
    def get_amc_and_aime_math_task_prompts(num_icl_example: int, random_seed: int) -> List[str]:
        """Get prompts for the AMC and AIME math task."""
        user_prompt = f"""Generate a math competition problem in the style of AMC 10, AMC 12, or AIME.

Knowledge Coverage:
Use secondary or high school mathematics — arithmetic, algebra, counting & probability, number theory, combinatorics, geometry, trigonometry, pre-calculus, and common contest techniques (inequalities such as AM–GM or Cauchy–Schwarz, symmetry, invariants, clever manipulations).

Format Requirements:
- Clearly state a single math problem under a line starting with “Question:”.
- Provide the difficulty level under a line starting with “Difficulty:”, using exactly one of: AMC or AIME.
- The answer must be a specific number or simplified expression (no multiple-choice).

Constraints:
- The problem must be self-contained and well-posed.
- Do not require advanced undergraduate mathematics (e.g., advanced calculus, abstract algebra).
- Avoid obscure tricks; rely only on creative applications of standard high-school math.
- Keep the difficulty level and the style consistent with official AMC/AIME problems.

Format exactly as follows:
Question:
[problem statement in natural language]
Difficulty:
[difficulty level, exactly one of: AMC or AIME]
        """
# Output Style Example (do not copy):
# Question: What is the degree measure of the acute angle formed by lines with slopes $2$ and $\frac{1}{3}$? 
# Difficulty: AMC

# Question: Let $p$ be the least prime number for which there exists a positive integer $n$ such that $n^{4}+1$ is divisible by $p^{2}$. Find the least positive integer $m$ such that $m^{4}+1$ is divisible by $p^{2}$.
# Difficulty: AIME
        return [user_prompt]
    
    @staticmethod
    def get_livecodebench_task_prompts(num_icl_example: int, random_seed: int) -> List[str]:
        """Get prompts for generating synthetic LiveCodeBench-style coding problems."""
        user_prompt = f"""Generate a programming challenge in the style of competitive programming platforms (e.g., LeetCode, AtCoder, Codeforces).
        The problem must be:
        - Self-contained and clearly stated.
        - Include only the task description, input/output format, and constraints.
        - At a specified difficulty level (easy, medium, or hard), appropriate for coding interviews or algorithmic contests like LeetCode, AtCoder, Codeforces.

        For the problem, output only in the following format:
        Question:
        [problem statement in natural language]
        Difficulty:
        [difficulty level]
        """
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
        custom_prompts: Optional[List[str]] = None,
        **kwargs,
    ) -> List[Union[List[Dict[str, str]], str]]:
        """Get a prompt for a specific task and format.
        
        Returns:
            List[Union[List[Dict[str, str]], str]]: A list of prompts, each containing either:
                - A list of system and user messages (for chat models)
                - A string prompt (for base models)
        """
        # If custom prompts are provided, use them directly
        if custom_prompts is not None and len(custom_prompts) > 0:
            prompts = list(custom_prompts)
        else:
            prompts = []
        if not custom_prompts and task == "gsm8k":
            prompts = PromptFactory.get_gsm8k_task_prompts(num_icl_example=3, random_seed=random_seed)
        elif not custom_prompts and task == "amc_aime_math":
            prompts = PromptFactory.get_amc_and_aime_math_task_prompts(num_icl_example=3, random_seed=random_seed)
        elif not custom_prompts and task == "livecodebench":
            prompts = PromptFactory.get_livecodebench_task_prompts(num_icl_example=3, random_seed=random_seed)
        elif not custom_prompts and (task == "poem") and (method == Method.DIRECT_BASE): # Handle poem task with clean data
            prompt_path = "data/poem_titles.txt"
        # elif task == "safety":
        #     prompt_path = "data/safety"
        elif not custom_prompts:
            prompt_path = f"data/{task}.txt"

        # Only try to read from file if we don't have prompts from the special task methods
        if not prompts and not custom_prompts:
            if not os.path.exists(prompt_path):
                raise ValueError(f"Prompt file {prompt_path} not found.")
            
            prompts = []
            with open(prompt_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        prompts.append(line)
        
        # Selection of prompts (if custom prompts given, sample from them if requested)
        if num_prompts is not None:
            if random_seed is not None:
                random.seed(random_seed)
            if len(prompts) > num_prompts:
                prompts = random.sample(prompts, num_prompts)

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
