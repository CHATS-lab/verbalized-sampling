"""
Prompt templates organized by task type.
"""

from typing import Dict, Any, Optional
from enum import Enum

class TaskType(Enum):
    """Enumeration of different task types."""
    CREATIVITY = "creativity"
    COMMONSENSE = "commonsense"
    BIAS = "bias"
    SYNTHETIC_DATA = "synthetic_data"
    ABLATION = "ablation"


class BasePromptTemplate:
    """Base class for prompt templates."""
    
    def __init__(self, task_type: TaskType):
        self.task_type = task_type
    
    def get_base_prompt(self, **kwargs) -> str:
        """Get the base prompt for the task."""
        raise NotImplementedError
    
    def get_base_model_prompt(self, **kwargs) -> str:
        """Get the base model prompt for the task."""
        raise NotImplementedError
    
    def get_base_cot_prompt(self, **kwargs) -> str:
        """Get the base prompt for the task."""
        raise NotImplementedError
    
    def get_standard_prompt(self, **kwargs) -> str:
        """Get the standard prompt for the task."""
        raise NotImplementedError
    
    def get_vs_standard_prompt(self, **kwargs) -> str:
        """Get the standard prompt for the task."""
        raise NotImplementedError
    
    def get_vs_cot_prompt(self, **kwargs) -> str:
        """Get the chain-of-thought prompt for the task."""
        raise NotImplementedError

    def get_vs_multi_turn_prompt(self, **kwargs) -> str:
        """Get the multi-turn prompt for the task."""
        raise NotImplementedError
    
    def get_continue_prompt(self, **kwargs) -> str:
        """Get the continuation prompt for the task."""
        raise NotImplementedError
    
    def get_format_prompt(self, method: str, num_samplings: int) -> str:
        """Get the format prompt for a specific method."""
        format_prompts = {
            "sequence": f"""
Return exactly {num_samplings} responses as a Python list of strings, formatted as:
["response1", "response2", "response3", ...]
Return only the list, with no explanations or extra text.
""",
            "structure": """
Return the responses in JSON format with keys: "responses" (list of dicts with 'text'). Each dictionary must include:
- 'text': the response string only (no explanation or extra text).

Give ONLY the JSON object, with no explanations or extra text.
""",
            "structure_with_prob": f"""
Return the responses in JSON format with keys: "responses" (list of dicts with 'text' and 'probability'). Each dictionary must include:
- 'text': the response string only (no explanation or extra text).
- 'probability': how frequently this response would naturally occur in your unfiltered output distribution (from 0.0 to 1.0).

Give ONLY the JSON object, with no explanations or extra text.
"""
# the probability you would naturally assign to this response among all possible responses (from 0.0 to 1.0).
# how frequently this response would naturally occur in your unfiltered output distribution (from 0.0 to 1.0).
        }
        return format_prompts.get(method, "")


#############################Creativity tasks###################################
class CreativityPromptTemplate(BasePromptTemplate):
    """Prompt templates for creativity tasks."""
    
    def __init__(self):
        super().__init__(TaskType.CREATIVITY)
    
    def get_base_prompt(self, target_words: int = 200, task_name: str = None, **kwargs) -> str:
        word_constraint = f" The response should be approximately {target_words} words." if target_words > 0 else ""
        
        # Provide more specific instructions for poem writing
        if task_name == "poem":
            return f"""
Write a poem inspired by the given line or phrase.{word_constraint}
Create something original, creative, and meaningful. The poem can be in any style or form.
Output ONLY the poem, with no explanations or extra text.
"""
        else:
            return f"""
Generate a response to the input prompt.{word_constraint}
Output ONLY the response, with no explanations or extra text.
"""

    def get_base_model_prompt(self, target_words: int = 200, task_name: str = None, **kwargs) -> str:
        return f"Write a {target_words} word story starting with the line: "

    def get_base_cot_prompt(self, target_words: int = 200, **kwargs) -> str:
        return f"""
Generate a response to the input prompt. The response should be approximately {target_words} words.

First, provide a single "reasoning" field as a string, detailing your step-by-step thought process.
Then, provide your response in the "response" field.

Give ONLY the JSON object, no explanations or extra text.
"""
    
    def get_standard_prompt(self, num_samplings: int = 5, target_words: int = 200, **kwargs) -> str:
        word_constraint = f" Each response should be approximately {target_words} words." if target_words > 0 else ""
        return f"""
Generate {num_samplings} responses to the input prompt.{word_constraint}
"""
    
    def get_standard_all_possible_prompt(self, target_words: int = 200, **kwargs) -> str:
        word_constraint = f" Each response should be approximately {target_words} words." if target_words > 0 else ""
        return f"""
Generate all possible responses to the input prompt.{word_constraint}
"""

    def get_vs_standard_prompt(self, num_samplings: int = 5, target_words: int = 200, **kwargs) -> str:
        word_constraint = f" Each response should be approximately {target_words} words." if target_words > 0 else ""
        return f"""
Consider all the ways you might respond to the input prompt. Randomly sample {num_samplings} responses from this distribution.{word_constraint}
"""

    def get_vs_cot_prompt(self, num_samplings: int = 5, target_words: int = 200, **kwargs) -> str:
        word_constraint = f" Each response should be approximately {target_words} words." if target_words > 0 else ""
        return f"""
Consider all the ways you might respond to the input prompt. Randomly sample {num_samplings} responses from this distribution using chain-of-thought reasoning.{word_constraint}

First, provide a single "reasoning" field as a string, detailing your step-by-step thought process.
Then, under "responses", return a list of dictionaries. Each dictionary must include:
- 'text': the response string only (no explanation or extra text).
- 'probability': the probability you would naturally assign to this response among all possible responses (from 0.0 to 1.0).

Give ONLY the JSON object, no explanations or extra text.
"""

    def get_vs_multi_turn_prompt(self, num_samplings: int = 5, num_samples_per_prompt: int = 2, target_words: int = 200, **kwargs) -> str:
        word_constraint = f" Each response should be approximately {target_words} words." if target_words > 0 else ""
        return f"""
Consider all the ways you might respond to the input prompt. You will randomly sample {num_samplings} total responses from this distribution.{word_constraint}

First, sample {num_samples_per_prompt} responses. 
Return the responses in JSON format with the key: "responses" (a list of dicts with 'text' and 'probability'). Each dictionary must include:
- 'text': the response string only (no explanations or extra text).
- 'probability': the probability you would naturally assign to this response among all possible responses (from 0.0 to 1.0).

Give ONLY the JSON object, no explanations or extra text.
"""
    
    def get_continue_prompt(self, num_samplings: int = 5, target_words: int = 200, **kwargs) -> str:
        if num_samplings == 1:
            return f"""
Generate one alternative response to the original input prompt.
"""
        else:
            return f"""
Sample {num_samplings} alternative responses to the original input prompt.
"""
    
    def get_format_prompt(self, method: str, num_samplings: int) -> str:
        base_template = BasePromptTemplate(TaskType.CREATIVITY)
        return base_template.get_format_prompt(method, num_samplings)



#############################Bias tasks###################################
class BiasPromptTemplate(BasePromptTemplate):
    """Prompt templates for bias tasks."""
    
    def __init__(self):
        super().__init__(TaskType.BIAS)
    
    def get_base_prompt(self, **kwargs) -> str:
        return """
Generate a response to the input prompt. Output ONLY the response, with no explanations or extra text.
"""
    
    def get_base_cot_prompt(self, **kwargs) -> str:
        return """
Generate a response to the input prompt. Output ONLY the response, with no explanations or extra text.

First, provide a single "reasoning" field as a string, detailing your step-by-step thought process.
Then, provide your response in the "response" field.

Give ONLY the JSON object, no explanations or extra text.
"""
    
    def get_standard_prompt(self, num_samplings: int = 5, **kwargs) -> str:
        return f"""
Generate {num_samplings} plausible responses to the input prompt.
"""
    
    def get_standard_all_possible_prompt(self, **kwargs) -> str:
        return """
Generate all plausible responses to the input prompt.
"""

    def get_vs_standard_prompt(self, num_samplings: int = 5, **kwargs) -> str:
        return f"""
Consider all the ways you might respond to the input prompt. Randomly sample {num_samplings} responses from this distribution.
"""
    
    def get_vs_cot_prompt(self, num_samplings: int = 5, **kwargs) -> str:
        return f"""
Consider all the ways you might respond to the input prompt. Randomly sample {num_samplings} responses from this distribution using chain-of-thought reasoning.

First, provide a single "reasoning" field as a string, detailing your step-by-step thought process.
Then, under "responses", return a list of dictionaries. Each dictionary must include:
- 'text': the response string only (no explanation or extra text).
- 'probability': the probability you would naturally assign to this response among all possible responses (from 0.0 to 1.0).

Give ONLY the JSON object, no explanations or extra text.
"""

    def get_vs_multi_turn_prompt(self, num_samplings: int = 5, num_samples_per_prompt: int = 2, **kwargs) -> str:
        return f"""
Consider all the ways you might respond to the input prompt. You will randomly sample {num_samplings} total responses from this distribution.

First, sample {num_samples_per_prompt} responses.
Return the responses in JSON format with keys: "responses" (list of dicts with 'text' and 'probability'). Each dictionary must include:
- 'text': the response string only (no explanation or extra text).
- 'probability': the probability you would naturally assign to this response among all possible responses (from 0.0 to 1.0).

Give ONLY the JSON object, no explanations or extra text.
"""
    
    def get_continue_prompt(self, num_samplings: int = 5, **kwargs) -> str:
        if num_samplings == 1:
            return f"""
Generate one alternative response to the original input prompt.
"""
        else:
            return f"""
Sample {num_samplings} alternative responses to the original input prompt.
"""
    
    def get_format_prompt(self, method: str, num_samplings: int) -> str:
        base_template = BasePromptTemplate(TaskType.BIAS)
        return base_template.get_format_prompt(method, num_samplings)



#############################Commonsense reasoning tasks###################################
class CommonsensePromptTemplate(BasePromptTemplate):
    """Prompt templates for commonsense reasoning tasks."""
    
    def __init__(self):
        super().__init__(TaskType.COMMONSENSE)
    
    def get_base_prompt(self, **kwargs) -> str:
        return """
Generate a response for the given question. Output ONLY the response, with no explanations or extra text.
"""

    def get_base_cot_prompt(self, **kwargs) -> str:
        return """
Generate a response for the given question. Output ONLY the response, with no explanations or extra text.

First, provide a single "reasoning" field as a string, detailing your step-by-step thought process.
Then, provide your response in the "response" field.

Give ONLY the JSON object, no explanations or extra text.
"""
    
    def get_standard_prompt(self, num_samplings: int = 5, **kwargs) -> str:
        return f"""
Provide your {num_samplings} best-guess responses for the given question that you think could be correct.
"""
    
    def get_standard_all_possible_prompt(self, **kwargs) -> str:
        return f"""
Provide all possible best-guess responses for the given question. 
Output ONLY the response, with no explanations or extra text.
"""
    
    def get_vs_standard_prompt(self, num_samplings: int = 5, **kwargs) -> str:
        return f"""
Provide your {num_samplings} best-guess responses for the given question that you think could be correct.
"""
    
    def get_vs_cot_prompt(self, num_samplings: int = 5, **kwargs) -> str:
        return f"""
Provide your {num_samplings} best-guess responses for the given question that you think could be correct using chain-of-thought reasoning.

First, provide a single "reasoning" field as a string, detailing your step-by-step thought process.
Then, return the responses in JSON format with the key: "responses" (a list of dicts with 'text' and 'probability'):
- 'text': the response string only (no explanation or extra text).
- 'probability': the probability you would naturally assign to this response among all possible responses (from 0.0 to 1.0).

Give ONLY the JSON object, no explanations or extra text.
"""

    def get_vs_multi_turn_prompt(self, num_samplings: int = 5, num_samples_per_prompt: int = 2, **kwargs) -> str:
        return f"""
You will generate a total of {num_samplings} responses that you think could be correct for the given question.

First, provide {num_samples_per_prompt} best-guess responses for the given question that you think could be correct.
Return the responses in JSON format with the key: "responses" (a list of dicts with 'text' and 'probability'). Each dictionary must include:
- 'text': the response string only (no explanations or extra text).
- 'probability': the probability you would naturally assign to this response among all possible responses (from 0.0 to 1.0).

Give ONLY the JSON object, no explanations or extra text.
"""

    def get_continue_prompt(self, num_samplings: int = 5, **kwargs) -> str:
        if num_samplings == 1:
            return f"""
Provide one alternative response for the original input prompt that you think could be correct.
"""
        else:
            return f"""
Provide {num_samplings} alternative responses for the original input prompt that you think could be correct.
"""
    
    def get_format_prompt(self, method: str, num_samplings: int) -> str:
        base_template = BasePromptTemplate(TaskType.COMMONSENSE)
        return base_template.get_format_prompt(method, num_samplings)


#############################Synthetic data tasks###################################
class SyntheticDataPromptTemplate(BasePromptTemplate):
    """Prompt templates for synthetic data tasks."""
    
    def __init__(self):
        super().__init__(TaskType.SYNTHETIC_DATA)
    
    def get_base_prompt(self, **kwargs) -> str:
        return """
Generate a response to the input prompt. Output ONLY the response, with no explanations or extra text.
"""
    
    def get_base_cot_prompt(self, **kwargs) -> str:
        return """
Generate a response to the input prompt. Output ONLY the response, with no explanations or extra text.

First, provide a single "reasoning" field as a string, detailing your step-by-step thought process.
Then, provide your response in the "response" field.

Give ONLY the JSON object, no explanations or extra text.
"""
    
    def get_standard_prompt(self, num_samplings: int = 5, **kwargs) -> str:
        return f"""
Generate {num_samplings} responses to the input prompt.
"""
    
    def get_standard_all_possible_prompt(self, **kwargs) -> str:
        return """
Generate all plausible responses to the input prompt.
"""

    def get_vs_standard_prompt(self, num_samplings: int = 5, **kwargs) -> str:
        return f"""
Consider all the ways you might respond to the input prompt. Randomly sample {num_samplings} total responses from this distribution.
"""
    
    def get_vs_cot_prompt(self, num_samplings: int = 5, **kwargs) -> str:
        return f"""
Consider all the ways you might respond to the input prompt, and randomly sample {num_samplings} responses from this distribution using chain-of-thought reasoning.

First, provide a single "reasoning" field as a string, detailing your step-by-step thought process.
Then, under "responses", return a list of dictionaries. Each dictionary must include:
- 'text': the response string only (no explanation or extra text).
- 'probability': how frequently this response would naturally occur in your unfiltered output distribution (from 0.0 to 1.0).

Give ONLY the JSON object, no explanations or extra text.
"""

    def get_vs_multi_turn_prompt(self, num_samplings: int = 5, num_samples_per_prompt: int = 2, **kwargs) -> str:
        return f"""
Consider all the ways you might respond to the input prompt. You will randomly sample {num_samplings} total responses from this distribution.

First, sample {num_samples_per_prompt} responses.
Return the responses in JSON format with keys: "responses" (list of dicts with 'text' and 'probability'). Each dictionary must include:
- 'text': the response string only (no explanation or extra text).
- 'probability': the probability mass assigned to this response in your output distribution (from 0.0 to 1.0).

Give ONLY the JSON object, no explanations or extra text.
"""
    
    def get_continue_prompt(self, num_samplings: int = 5, **kwargs) -> str:
        if num_samplings == 1:
            return f"""
Generate one alternative response to the original input prompt.
"""
        else:
            return f"""
Sample {num_samplings} alternative responses to the original input prompt.
"""
    
    def get_format_prompt(self, method: str, num_samplings: int) -> str:
        base_template = BasePromptTemplate(TaskType.SYNTHETIC_DATA)
        return base_template.get_format_prompt(method, num_samplings)



#############################Prompt factory###################################
class PromptTemplateFactory:
    """Factory class to create prompt templates for different task types."""
    
    _templates = {
        TaskType.CREATIVITY: CreativityPromptTemplate,
        TaskType.COMMONSENSE: CommonsensePromptTemplate,
        TaskType.BIAS: BiasPromptTemplate,
        TaskType.SYNTHETIC_DATA: SyntheticDataPromptTemplate,
        # TaskType.ABLATION: AblationPromptTemplate,
    }
    
    @classmethod
    def get_template(cls, task_type: TaskType) -> BasePromptTemplate:
        """Get the appropriate prompt template for a task type."""
        template_class = cls._templates.get(task_type)
        if template_class is None:
            raise ValueError(f"Unknown task type: {task_type}")
        return template_class()
    
    @classmethod
    def get_prompt(cls, task_type: TaskType, prompt_type: str, **kwargs) -> str:
        """Get a specific prompt for a task type."""
        template = cls.get_template(task_type)
        
        prompt_methods = {
            "base": template.get_base_prompt,
            "base_model": template.get_base_model_prompt,
            "base_cot": template.get_base_cot_prompt, # cot
            "standard": template.get_standard_prompt, # vs standard
            "vs_standard": template.get_vs_standard_prompt, # vs standard
            "vs_cot": template.get_vs_cot_prompt, # vs chain_of_thought
            "vs_multi_turn": template.get_vs_multi_turn_prompt, # vs multi_turn
            "continue": template.get_continue_prompt,
            "standard_all_possible": getattr(template, 'get_standard_all_possible_prompt', template.get_standard_prompt),
        }

        method = prompt_methods.get(prompt_type)
        if method is None:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
        
        return method(**kwargs)


# Legacy compatibility - keep the old flat structure for backward compatibility
# These can be gradually migrated to use the new class-based system

########################### Legacy Prompts ###########################

# Self-Reflection Sampling Prompts
# SELF_REFLECTION_PROMPT = """
# Generate {num_samplings} different responses with self-reflection and confidence scoring.
# For each response, provide the response, reflect on its quality, and assign a confidence score.
# Return the output in JSON format with keys: "responses" (list of dicts with 'response', 'reflection', and 'confidence'). Each dictionary must include:
# - 'response': the response string.
# - 'reflection': the analysis of response quality and appropriateness.
# - 'confidence': the confidence score between 0.0 and 1.0.

# Give ONLY the JSON object, no explanations or extra text.
# """

# # Temperature-based Sampling Prompts
# TEMPERATURE_SAMPLING_PROMPT = """
# Generate {num_samplings} responses with varying creativity levels.
# Create responses ranging from conservative/safe to creative/bold.
# Return the output in JSON format with keys: "responses" (list of dicts with 'response', 'creativity_level', and 'temperature'). Each dictionary must include:
# - 'response': the response string.
# - 'creativity_level': the creativity level of the response (conservative, moderate, creative, bold).
# - 'temperature': the temperature of the response (value between 0 and 1).

# Give ONLY the JSON object, no explanations or extra text.
# """

# def get_combined_prompt(self, num_samplings: int = 5, **kwargs) -> str:
#         return f"""
# Provide your {num_samplings} best guesses for the given question that you believe could be correct.

# Return the responses in JSON format with the key: "responses" (a list of dicts with 'text' and 'nll'). Each dictionary must include:
# - 'text': the response string only (no explanations or extra text).
# - 'nll': the estimated negative log likelihood for the response (approx. 1.0 per token, each token ranges from 0.5–2.5 based on creativity).

# Give ONLY the JSON object, no explanations or extra text.
# """