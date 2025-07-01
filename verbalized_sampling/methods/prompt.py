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
    ABLATION = "ablation"


class BasePromptTemplate:
    """Base class for prompt templates."""
    
    def __init__(self, task_type: TaskType):
        self.task_type = task_type
    
    def get_base_prompt(self, **kwargs) -> str:
        """Get the base prompt for the task."""
        raise NotImplementedError
    
    def get_standard_prompt(self, **kwargs) -> str:
        """Get the standard prompt for the task."""
        raise NotImplementedError
    
    def get_combined_prompt(self, **kwargs) -> str:
        """Get the combined prompt for the task."""
        raise NotImplementedError
    
    def get_chain_of_thought_prompt(self, **kwargs) -> str:
        """Get the chain-of-thought prompt for the task."""
        raise NotImplementedError
    
    def get_continue_prompt(self, **kwargs) -> str:
        """Get the continuation prompt for the task."""
        raise NotImplementedError
    
    def get_format_prompt(self, method: str, num_samplings: int) -> str:
        """Get the format prompt for a specific method."""
        format_prompts = {
            "sequence": """
Return ALL responses as a Python list of strings, in the following format:
["response1", "response2", "response3", ...]

The list must contain exactly {num_samplings} strings, each representing a unique response.
Output ONLY the list, with no explanations or extra text.
""",
            "structure": """
Return the responses in JSON format with keys: "responses" (list of dicts with 'text'). Each dictionary must include:
- 'text': the response string only (no explanation or extra text).

Give ONLY the JSON object, with no explanations or extra text.
""",
            "structure_with_prob": """
Return the responses in JSON format with keys: "responses" (list of dicts with 'text' and 'probability'). Each dictionary must include:
- 'text': the response string only (no explanation or extra text).
- 'probability': the empirical probability representing how likely each response would be (0.0 to 1.0).

Give ONLY the JSON object, with no explanations or extra text.
"""
        }
        return format_prompts.get(method, "")


#############################Creativity tasks###################################
class CreativityPromptTemplate(BasePromptTemplate):
    """Prompt templates for creativity tasks."""
    
    def __init__(self):
        super().__init__(TaskType.CREATIVITY)
    
    def get_base_prompt(self, target_words: int = 200, **kwargs) -> str:
        return f"""
Generate a response to the input prompt. The response should be approximately {target_words} words.
Output ONLY the response, with no explanations or extra text.
"""
    
    def get_standard_prompt(self, num_samplings: int = 5, target_words: int = 200, **kwargs) -> str:
        return f"""
Generate {num_samplings} creative and diverse responses to the input prompt. Each response should be approximately {target_words} words.
Maximizing both creativity and diversity, while ensuring that each response remains high-quality and relevant to the input prompt.
"""
    
    def get_standard_all_possible_prompt(self, target_words: int = 200, **kwargs) -> str:
        return f"""
Generate all possible responses to the input prompt. Each response should be approximately {target_words} words.
Maximizing both creativity and diversity, while ensuring that each response remains high-quality and relevant to the input prompt.
"""

    def get_chain_of_thought_prompt(self, num_samplings: int = 5, target_words: int = 200, **kwargs) -> str:
        return f"""
Generate {num_samplings} creative and diverse responses to the input prompt using chain-of-thought reasoning. Each response should be approximately {target_words} words.
Maximizing both creativity and diversity, while ensuring that each response remains high-quality and relevant to the input prompt.

First, provide a single "reasoning" field as a string, detailing your step-by-step thought process.
Then, under "responses", return a list of dictionaries. Each dictionary must include:
- 'text': the response string only (no explanation or extra text).
- 'probability': the empirical probability representing how likely each response would be (0.0 to 1.0).

Give ONLY the JSON object, no explanations or extra text.
"""

    def get_combined_prompt(self, num_samplings: int = 5, num_samples_per_prompt: int = 2, target_words: int = 200, **kwargs) -> str:
        return f"""
You will generate a total of {num_samplings} creative and diverse responses to the input prompt. Each response should be approximately {target_words} words.
Maximizing both creativity and diversity of the responses, while ensuring that each response remains high-quality and relevant to the input prompt.

First, generate {num_samples_per_prompt} creative and diverse responses. 
Return the responses in JSON format with the key: "responses" (a list of dicts with 'text' and 'confidence'). Each dictionary must include:
- 'text': the response string only (no explanations or extra text).
- 'confidence': a score from 0.0 to 1.0 representing how likely or typical the response is (1.0 = very typical/common, 0.0 = highly original/creative).

Give ONLY the JSON object, no explanations or extra text.
"""
    
    def get_continue_prompt(self, num_samplings: int = 5, target_words: int = 200, **kwargs) -> str:
        if num_samplings == 1:
            return f"""
Generate an alternative response to the original input prompt.
"""
        else:
            return f"""
Generate {num_samplings} alternative responses to the original input prompt. Try to be as creative and diverse as possible.
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
    
    def get_standard_prompt(self, num_samplings: int = 5, **kwargs) -> str:
        return f"""
Generate {num_samplings} plausible and diverse responses to the input prompt.
Maximizing the diversity of the responses, while ensuring that each response remains high-quality and relevant to the input prompt.
"""
    
    def get_standard_all_possible_prompt(self, **kwargs) -> str:
        return """
Generate all plausible responses to the input prompt.
Maximizing the diversity of the responses, while ensuring that each response remains high-quality and relevant to the input prompt.
"""
    
    def get_chain_of_thought_prompt(self, num_samplings: int = 5, **kwargs) -> str:
        return f"""
Generate {num_samplings} plausible and diverse responses to the input prompt using chain-of-thought reasoning.
Maximizing the diversity of the responses, while ensuring that each response remains high-quality and relevant to the input prompt.

First, provide a single "reasoning" field as a string, detailing your step-by-step thought process.
Then, under "responses", return a list of dictionaries. Each dictionary must include:
- 'text': the response string only (no explanation or extra text).
- 'probability': the empirical probability representing how likely each response would be (0.0 to 1.0).

Give ONLY the JSON object, no explanations or extra text.
"""

    def get_combined_prompt(self, num_samplings: int = 5, num_samples_per_prompt: int = 2, **kwargs) -> str:
        return f"""
You will generate a total of {num_samplings} plausible and diverse responses to the input prompt.
Maximizing the diversity of the responses, while ensuring that each response remains high-quality and relevant to the input prompt.

First, generate {num_samples_per_prompt} responses. Try to be as diverse as possible.

Return the responses in JSON format with keys: "responses" (list of dicts with 'text' and 'probability'). Each dictionary must include:
- 'text': the response string only (no explanation or extra text).
- 'confidence': a score from 0.0 to 1.0 representing how likely or typical the response is (1.0 = very typical/common, 0.0 = highly original/rare).

Give ONLY the JSON object, no explanations or extra text.
"""
    
    def get_continue_prompt(self, num_samplings: int = 5, **kwargs) -> str:
        if num_samplings == 1:
            return f"""
Generate an alternative response to the original input prompt.
"""
        else:
            return f"""
Generate {num_samplings} alternative plausible responses to the original input prompt. 
Try to be as diverse as possible while maintaining high-quality and relevance to the input prompt.
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
Provide your best-guess response for the given question. Output ONLY the response, with no explanations or extra text.
"""
    
    def get_standard_prompt(self, num_samplings: int = 5, **kwargs) -> str:
        return f"""
Provide your {num_samplings} best-guess responses for the given question that you believe could be correct.
"""
    
    def get_standard_all_possible_prompt(self, **kwargs) -> str:
        return f"""
Provide all possible best-guess responses for the given question that you believe could be correct. 
Output ONLY the response, with no explanations or extra text.
"""
    
    def get_chain_of_thought_prompt(self, num_samplings: int = 5, **kwargs) -> str:
        return f"""
Provide your {num_samplings} best-guess responses for the given question that you believe could be correct using chain-of-thought reasoning.

First, provide a single "reasoning" field as a string, detailing your step-by-step thought process.
Then, under "responses", return a list of dictionaries. Each dictionary must include:
- 'text': the response string only (no explanation or extra text).
- 'probability': the empirical probability representing how likely each response is to be correct (0.0 to 1.0).

Give ONLY the JSON object, no explanations or extra text.
"""

    def get_combined_prompt(self, num_samplings: int = 5, num_samples_per_prompt: int = 2, **kwargs) -> str:
        return f"""
You will generate a total of {num_samplings} responses for the given question that you believe could be correct.

First, provide {num_samples_per_prompt} best-guess responses that you believe could be correct.
Return the responses in JSON format with the key: "responses" (a list of dicts with 'text' and 'confidence'). Each dictionary must include:
- 'text': the response string only (no explanations or extra text).
- 'confidence': a score from 0.0 to 1.0 representing how likely or typical the response is (1.0 = very typical/commonsense, 0.0 = very unlikely/unconventional).

Give ONLY the JSON object, no explanations or extra text.
"""


    def get_continue_prompt(self, num_samplings: int = 5, **kwargs) -> str:
        if num_samplings == 1:
            return f"""
Provide an alternative response that you think could be correct for the original question.
"""
        else:
            return f"""
Provide {num_samplings} alternative responses that you believe could be correct for the original question.
"""
    
    def get_format_prompt(self, method: str, num_samplings: int) -> str:
        if method == "structure_with_prob":
            return """
Return the responses in JSON format with keys: "responses" (list of dicts with 'text' and 'probability'). Each dictionary must include:
- 'text': the response string only (no explanation or extra text).
- 'probability': the empirical probability representing how likely each response is to be correct (0.0 to 1.0).

Give ONLY the JSON object, with no explanations or extra text.
"""
        else:
            # Use the same format prompts as creativity tasks
            base_template = BasePromptTemplate(TaskType.COMMONSENSE)
            return base_template.get_format_prompt(method, num_samplings)


#############################Prompt factory###################################
class PromptTemplateFactory:
    """Factory class to create prompt templates for different task types."""
    
    _templates = {
        TaskType.CREATIVITY: CreativityPromptTemplate,
        TaskType.COMMONSENSE: CommonsensePromptTemplate,
        TaskType.BIAS: BiasPromptTemplate,
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
            "standard": template.get_standard_prompt,
            "combined": template.get_combined_prompt,
            "chain_of_thought": template.get_chain_of_thought_prompt,
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