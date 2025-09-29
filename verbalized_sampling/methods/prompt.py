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
    SYNTHETIC_NEGATIVE = "synthetic_negative"
    ABLATION = "ablation"
    SAFETY = "safety"
    MATH = "math"


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
    
    def get_format_prompt(
        self,
        method: str,
        num_samplings: int,
        probability_definition: str = None,
        probability_tuning: float = -1,
    ) -> str:
        """Get the format prompt for a specific method.

        Args:
            method: The output format method.
            num_samplings: Number of responses to generate (if relevant).
            probability_definition: (Optional) Custom definition for the 'probability' field.
            probability_tuning: (Optional) Custom tuning for the 'probability' field.
        """
        # Default probability definitions
        probability_definitions = {
            # "default": "- 'probability': how likely this response would be (from 0.0 to 1.0).",
            "implicit": "- 'probability': how likely this response would be (from 0.0 to 1.0).",
            "explicit": "- 'probability': the estimated probability from 0.0 to 1.0 of this response given the input prompt (relative to the full distribution).",
            "relative": "- 'probability': the probability between 0.0 and 1.0, reflecting the relative likelihood of this response given the input.",
            "percentage": "- 'probability': the probability of this response relative to the full distribution, expressed as a percentage from 0% to 100%.",
            "confidence": "- 'confidence': the normalized likelihood score between 0.0 and 1.0 that indicates how representative or typical this response is compared to the full distribution.",
            "perplexity": "- 'perplexity': the exponentiated average negative log likelihood of the response tokens, where lower values indicate higher model certainty in predicting each token.",
            "nll": "- 'nll': the sum of the negative log probabilities of each token in the response given the input prompt, with smaller values reflecting higher model confidence.",
        }
            
        # Use provided probability_definition or default
        prob_def = probability_definitions[probability_definition]

        if probability_tuning > 0:
            print(f"Tuning probability to {probability_tuning}")
            # distribution_def = f"Please sample at random from the tails of the distribution: probability of each response must be below {probability_tuning}."
            distribution_def = f"Randomly sample the responses from the distribution, with the probability of each response must be below {probability_tuning}."
            print(f"Distribution definition: {distribution_def}")
        else:
            distribution_def = "Randomly sample the responses from the full distribution."
            print("Tuning probability to 0")
            print(f"Distribution definition: {distribution_def}")

        format_prompts = {
            "direct_cot": """
First, provide a single "reasoning" field as a string, detailing your step-by-step thought process.
Then, provide your response in the "response" field.

Return ONLY the JSON object, with no additional explanations or text.
""",
            "structure": """
Return the output in JSON format with the key "responses" (list of dicts). Each dictionary must include:
- 'text': the response string only (no explanation or extra text).

Return ONLY the JSON object, with no additional explanations or text.
""",
#             "sequence": f"""
# Return the responses in JSON format with keys: "responses" (list of strings). The list must contain exactly {num_samplings} strings, each representing a unique response.
# Each response should be a complete, coherent text (not just a single line or phrase).

# Give ONLY the JSON object, with no explanations or extra text.
# """,
            "sequence": f"""
Return exactly {num_samplings} responses as a Python list of strings, formatted as:
["response1", "response2", "response3", ...]

Return ONLY the list, with no additional explanations or text.
""",
            "structure": """
Return the responses in JSON format with keys: "responses" (list of dicts with 'text'). Each dictionary must include:
- 'text': the response string only (no explanation or extra text).

Return ONLY the list, with no additional explanations or text.
""",
            "vs_standard": f"""
Return the responses in JSON format with the key: "responses" (list of dicts). Each dictionary must include:
- 'text': the response string only (no explanation or extra text).
{prob_def}

{distribution_def} Return ONLY the JSON object, with no additional explanations or text.
""",
            "vs_cot": f"""
First, provide a single "reasoning" field as a string, detailing your step-by-step thought process.
Then, return the output in JSON format with the key "responses" (list of dicts). Each dictionary must include:
- 'text': the response string only (no explanation or extra text).
{prob_def}

{distribution_def} Return ONLY the JSON object, with no additional explanations or text.
""",
            "vs_multi": f"""
Return the responses in JSON format with the key: "responses" (list of dicts). Each dictionary must include:
- 'text': the response string only (no explanation or extra text).
{prob_def}

{distribution_def} Return ONLY the JSON object, with no additional explanations or text.
"""
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
Generate a response to the input prompt using chain-of-thought reasoning. The response should be approximately {target_words} words.
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

    def get_vs_cot_prompt(self, num_samplings: int = 5, target_words: int = 200, **kwargs) -> str:
        word_constraint = f" Each response should be approximately {target_words} words." if target_words > 0 else ""
        return f"""
Generate {num_samplings} responses to the input prompt using chain-of-thought reasoning.{word_constraint}
"""

    def get_vs_multi_turn_prompt(self, num_samplings: int = 5, num_samples_per_prompt: int = 2, target_words: int = 200, **kwargs) -> str:
        word_constraint = f" Each response should be approximately {target_words} words." if target_words > 0 else ""
        return f"""
Generate {num_samplings} responses to the input prompt.{word_constraint}

First, sample {num_samples_per_prompt} responses. 
"""
# Return the responses in JSON format with the key: "responses" (list of dicts). Each dictionary must include:
# - 'text': the response string only (no explanations or extra text).
# - 'probability': the estimated probability from 0.0 to 1.0 of this response given the input prompt (relative to the full answer space).
# Please sample at random from the full distribution. Give ONLY the JSON object, no explanations or extra text.

    def get_continue_prompt(self, num_samplings: int = 5, target_words: int = 200, **kwargs) -> str:
        if num_samplings == 1:
            return f"""
Generate one alternative response to the original input prompt.
"""
        else:
            return f"""
Randomly sample {num_samplings} alternative responses to the original input prompt.
"""
    
    def get_format_prompt(self, method: str, num_samplings: int, probability_definition: str = None, probability_tuning: float = -1) -> str:
        base_template = BasePromptTemplate(TaskType.CREATIVITY)
        return base_template.get_format_prompt(method, num_samplings, probability_definition, probability_tuning)



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
Generate a response to the input prompt using chain-of-thought reasoning. Output ONLY the response, with no explanations or extra text.
"""
    
    def get_standard_prompt(self, num_samplings: int = 5, **kwargs) -> str:
        return f"""
Generate {num_samplings} plausible responses to the input prompt.
"""
    
    def get_standard_all_possible_prompt(self, **kwargs) -> str:
        return """
Generate all plausible responses to the input prompt.
"""
    
    def get_vs_cot_prompt(self, num_samplings: int = 5, **kwargs) -> str:
        return f"""
Generate {num_samplings} plausible responses to the input prompt using chain-of-thought reasoning.
"""

    def get_vs_multi_turn_prompt(self, num_samplings: int = 5, num_samples_per_prompt: int = 2, **kwargs) -> str:
        return f"""
Generate a total of {num_samplings} plausible responses to the input prompt.

First, sample {num_samples_per_prompt} responses.
"""
# - 'text': the response string only (no explanation or extra text).
# - 'probability': the estimated probability from 0.0 to 1.0 of this response given the input prompt (relative to the full distribution).
# - 'confidence': a score from 0.0 to 1.0 representing how likely or typical the response is (1.0 = very typical/common, 0.0 = highly original/creative).
    
    def get_continue_prompt(self, num_samplings: int = 5, **kwargs) -> str:
        if num_samplings == 1:
            return f"""
Generate an alternative response to the original input prompt.
"""
        else:
            return f"""
Randomly sample {num_samplings} alternative responses to the original input prompt.
"""
    
    def get_format_prompt(self, method: str, num_samplings: int, probability_definition: str, probability_tuning: float = -1) -> str:
        base_template = BasePromptTemplate(TaskType.BIAS)
        return base_template.get_format_prompt(method, num_samplings, probability_definition, probability_tuning)



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
Generate a response for the given question using chain-of-thought reasoning. Output ONLY the response, with no explanations or extra text.
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
    
    def get_vs_cot_prompt(self, num_samplings: int = 5, **kwargs) -> str:
        return f"""
Provide your {num_samplings} best-guess responses for the given question that you think could be correct using chain-of-thought reasoning.
"""

    def get_vs_multi_turn_prompt(self, num_samplings: int = 5, num_samples_per_prompt: int = 2, **kwargs) -> str:
        return f"""
You will generate a total of {num_samplings} responses that you think could be correct for the given question.
Maximizing both creativity and diversity, while ensuring that each response remains high-quality to the input prompt.

First, provide {num_samples_per_prompt} best-guess responses for the given question that you think could be correct.
Return the responses in JSON format with the key: "responses" (list of dicts). Each dictionary must include:
- 'text': the response string only (no explanation or extra text).
- 'probability': the estimated probability from 0.0 to 1.0 of this response given the input prompt (relative to the full answer space).

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
    
    def get_format_prompt(self, method: str, num_samplings: int, probability_definition: str = None, probability_tuning: float = -1) -> str:
        base_template = BasePromptTemplate(TaskType.COMMONSENSE)
        return base_template.get_format_prompt(method, num_samplings, probability_definition, probability_tuning)


#############################Synthetic data tasks###################################
class SyntheticDataPromptTemplate(BasePromptTemplate):
    """Prompt templates for synthetic data tasks."""
    
    def __init__(self):
        super().__init__(TaskType.SYNTHETIC_DATA)
    
    def get_base_prompt(self, target_words: int = 100, **kwargs) -> str:
        word_constraint = f" The data instance should be approximately {target_words} words." if target_words > 0 else ""
        return f"""
Generate a data instance based on the input prompt.{word_constraint}
Output only the specified format of data instance, without any explanations or extra text.
"""
    
    def get_base_cot_prompt(self, target_words: int = 100, **kwargs) -> str:
        word_constraint = f" The data instance should be approximately {target_words} words." if target_words > 0 else ""
        return f"""
Generate a data instance based on the input prompt using chain-of-thought reasoning.{word_constraint}
Output only the specified format of data instance, without any explanations or extra text.
"""
    
    def get_standard_prompt(self, num_samplings: int = 5, target_words: int = 100, **kwargs) -> str:
        word_constraint = f" Each data instance should be approximately {target_words} words." if target_words > 0 else ""
        return f"""
Generate {num_samplings} data instances based on the input prompt.{word_constraint}
Output only the specified format of data instance, with no explanations or extra text.
"""
    
    def get_standard_all_possible_prompt(self, target_words: int = 100, **kwargs) -> str:
        word_constraint = f" Each data instance should be approximately {target_words} words." if target_words > 0 else ""
        return f"""
Generate all plausible data instances based on the input prompt.{word_constraint}
Output only the specified format of data instance, with no explanations or extra text.
"""

    def get_vs_cot_prompt(self, num_samplings: int = 5, target_words: int = 100, **kwargs) -> str:
        word_constraint = f" Each data instance should be approximately {target_words} words." if target_words > 0 else ""
        return f"""
Generate {num_samplings} data instances based on the input prompt using chain-of-thought reasoning.{word_constraint}
Output only the specified format of data instance, with no explanations or extra text.
"""

    def get_vs_multi_turn_prompt(self, num_samplings: int = 5, num_samples_per_prompt: int = 2, target_words: int = 100, **kwargs) -> str:
        word_constraint = f" Each data instance should be approximately {target_words} words." if target_words > 0 else ""
        return f"""
Generate {num_samplings} data instances based on the input prompt.{word_constraint}
Output only the specified format of data instance, with no explanations or extra text.

First, sample {num_samples_per_prompt} data instances.
"""
    
    def get_continue_prompt(self, num_samplings: int = 5, **kwargs) -> str:
        if num_samplings == 1:
            return f"""
Generate one more data instance based on the original input prompt.
"""
        else:
            return f"""
Randomly sample {num_samplings} alternative data instances based on the original input prompt.
"""
    
    def get_format_prompt(self, method: str, num_samplings: int, probability_definition: str = None, probability_tuning: float = -1) -> str:
        base_template = BasePromptTemplate(TaskType.SYNTHETIC_DATA)
        return base_template.get_format_prompt(method, num_samplings, probability_definition, probability_tuning)


#############################Synthetic negative tasks###################################
class SyntheticNegativePromptTemplate(BasePromptTemplate):
    """Prompt templates for synthetic negative tasks."""
    
    def __init__(self):
        super().__init__(TaskType.SYNTHETIC_NEGATIVE)
    
    def get_base_prompt(self, **kwargs) -> str:
        return """
Generate a solution to the given math problem that look logical but contain at least one hidden mistake, making the final result incorrect.
The solution must end with the final numerical answer, written only once after four hash marks (####). Example: ####123.
"""
    
    def get_base_cot_prompt(self, **kwargs) -> str:
        return """
Generate a solution to the given math problem using chain-of-thought reasoning. 
The solution should look logical but contain at least one hidden mistake, making the final result incorrect.
End the solution with the final numerical answer, written only once after four hash marks (####). Example: ####123.
"""
    
    def get_standard_prompt(self, num_samplings: int = 5, **kwargs) -> str:
        return f"""
Generate {num_samplings} logical and convincing but incorrect solutions to the given math problem. 
End each solution with the final numerical answer, written only once after four hash marks (e.g., ####123).
"""
    
    def get_standard_all_possible_prompt(self, **kwargs) -> str:
        return """
Generate all plausible solutions to the given math problem. 
Each solution should appear logical and consistent, but must include at least one hidden mistake that makes the final answer incorrect.
End each solution with the final numerical answer, written only once after four hash marks (e.g., ####123).
"""
    
    def get_vs_cot_prompt(self, num_samplings: int = 5, **kwargs) -> str:
        return f"""
Generate {num_samplings} solutions to the given math problem using chain-of-thought reasoning. 
Each solution should appear logical and consistent, but must include at least one hidden mistake that makes the final answer incorrect.
End each solution with the final numerical answer, written only once after four hash marks (e.g., ####123).
"""

    def get_vs_multi_turn_prompt(self, num_samplings: int = 5, num_samples_per_prompt: int = 2, **kwargs) -> str:
        return f"""
Generate a total of {num_samplings} solutions to the given math problem. 
Each solution should appear logical and consistent, but must include at least one hidden mistake that makes the final answer incorrect.
End each solution with the final numerical answer, written only once after four hash marks (e.g., ####123).

First, sample {num_samples_per_prompt} solutions.
Return the responses in JSON format with the key: "responses" (list of dicts). Each dictionary must include:
- 'text': the response string only (no explanation or extra text).
- 'probability': the estimated probability from 0.0 to 1.0 of this response given the input prompt (relative to the full answer space).

Sample the solutions at random from the full distribution. Give ONLY the JSON object, with no explanations or extra text.
"""
    
    def get_continue_prompt(self, num_samplings: int = 5, **kwargs) -> str:
        if num_samplings == 1:
            return f"""
Generate one alternative seems logical but incorrect solution to the given math problem.
"""
        else:
            return f"""
Randomly sample {num_samplings} alternative seems logical but incorrect solutions to the given math problem.
"""
    
    def get_format_prompt(self, method: str, num_samplings: int, probability_definition: str = None, probability_tuning: float = -1) -> str:
        if method == "sequence":
            return f"""
Return exactly {num_samplings} solutions as a Python list of strings, formatted as:
["solution1", "solution2", "solution3", ...]

Sample the solutions at random from the full distribution. Return ONLY the list, with no additional explanations or text.
"""
        elif method == "vs_standard":
            return """
Return the output in JSON format with the key "responses" (list of dicts). Each dictionary must include:
- 'text': the response string only (no explanation or extra text).
- 'probability': the estimated probability from 0.0 to 1.0 of this response given the input prompt (relative to the full distribution).

Sample the incorrect solutions at random from the full distribution. Output only the JSON object, with no explanations or extra text.
"""
        elif method == "vs_cot":
            return """
First, provide a single "reasoning" field as a string, detailing your step-by-step thought process.
Then, return the output in JSON format with the key "responses" (list of dicts). Each dictionary must include:
- 'text': the response string only (no explanation or extra text).
- "probability": the estimated probability from 0.0 to 1.0 of this response given the input prompt (relative to the full distribution).

Randomly sample the solutions from the full distribution. Give ONLY the JSON object, with no explanations or extra text.
"""
        else:
            base_template = BasePromptTemplate(TaskType.SYNTHETIC_NEGATIVE)
            return base_template.get_format_prompt(method, num_samplings, probability_definition, probability_tuning)


#############################Safety tasks###################################
class SafetyPromptTemplate(BasePromptTemplate):
    """Prompt templates for safety evaluation tasks."""

    def __init__(self):
        super().__init__(TaskType.SAFETY)

    def get_base_prompt(self, **kwargs) -> str:
        return """
Please respond to the following prompt.
"""

    def get_base_model_prompt(self, **kwargs) -> str:
        return "Please respond to the following prompt."

    def get_base_cot_prompt(self, **kwargs) -> str:
        return """
Please respond to the following prompt using step-by-step reasoning.
"""

    def get_standard_prompt(self, num_samplings: int = 5, **kwargs) -> str:
        return f"""
Generate {num_samplings} responses to the input prompt.
"""

    def get_vs_cot_prompt(self, num_samplings: int = 5, **kwargs) -> str:
        return f"""
Generate {num_samplings} responses to the input prompt using chain-of-thought reasoning.
"""

    def get_vs_multi_turn_prompt(self, num_samplings: int = 5, num_samples_per_prompt: int = 2, **kwargs) -> str:
        return f"""
Generate a total of {num_samplings} responses to the input prompt.

First, sample {num_samples_per_prompt} responses.
"""

    def get_continue_prompt(self, num_samplings: int = 5, **kwargs) -> str:
        if num_samplings == 1:
            return """
Generate an alternative response to the original input prompt.
"""
        else:
            return f"""
Sample {num_samplings} alternative responses to the original input prompt.
"""

    def get_format_prompt(self, method: str, num_samplings: int, probability_definition: str = None, probability_tuning: float = -1) -> str:
        base_template = BasePromptTemplate(TaskType.SAFETY)
        return base_template.get_format_prompt(method, num_samplings, probability_definition, probability_tuning)


#############################Math tasks###################################
class MathPromptTemplate(BasePromptTemplate):
    """Prompt templates for math reasoning tasks."""

    def __init__(self):
        super().__init__(TaskType.MATH)

    def get_base_prompt(self, **kwargs) -> str:
        return "Please reason step by step, and put your final answer within \\boxed{}."

    def get_base_model_prompt(self, **kwargs) -> str:
        return "Please reason step by step, and put your final answer within \\boxed{}."

    def get_base_cot_prompt(self, **kwargs) -> str:
        return "Please reason step by step, and put your final answer within \\boxed{}."

    def get_standard_prompt(self, num_samplings: int = 5, **kwargs) -> str:
        return f"""
Generate {num_samplings} different solutions to the math problem.
"""

    def get_vs_cot_prompt(self, num_samplings: int = 5, **kwargs) -> str:
        return f"""
Generate {num_samplings} solutions to the math problem using step-by-step reasoning.
"""

    def get_vs_multi_turn_prompt(self, num_samplings: int = 5, num_samples_per_prompt: int = 2, **kwargs) -> str:
        return f"""
Generate a total of {num_samplings} solutions to the math problem.

First, provide {num_samples_per_prompt} solutions.
"""

    def get_continue_prompt(self, num_samplings: int = 5, **kwargs) -> str:
        if num_samplings == 1:
            return """
Generate an alternative solution to the math problem.
"""
        else:
            return f"""
Generate {num_samplings} alternative solutions to the math problem.
"""

    def get_format_prompt(self, method: str, num_samplings: int, probability_definition: str = None, probability_tuning: float = -1) -> str:
        """Get math-specific format prompts that ensure \\boxed{} format is preserved."""

        # Default probability definitions
        probability_definitions = {
            "implicit": "- 'probability': how likely this response would be (from 0.0 to 1.0).",
            "explicit": "- 'probability': the estimated probability from 0.0 to 1.0 of this response given the input prompt (relative to the full distribution).",
            "relative": "- 'probability': the probability between 0.0 and 1.0, reflecting the relative likelihood of this response given the input.",
            "percentage": "- 'probability': the probability of this response relative to the full distribution, expressed as a percentage from 0% to 100%.",
            "confidence": "- 'confidence': the normalized likelihood score between 0.0 and 1.0 that indicates how representative or typical this response is compared to the full distribution.",
            "perplexity": "- 'perplexity': the exponentiated average negative log likelihood of the response tokens, where lower values indicate higher model certainty in predicting each token.",
            "nll": "- 'nll': the sum of the negative log probabilities of each token in the response given the input prompt, with smaller values reflecting higher model confidence.",
        }

        # Use provided probability_definition or default
        prob_def = probability_definitions.get(probability_definition, probability_definitions["implicit"])

        if probability_tuning > 0:
            distribution_def = f"Please sample at random from the tails of the distribution: probability of each response must be below {probability_tuning}."
        else:
            distribution_def = "Randomly sample the responses from the full distribution."

        # Math-specific format prompts
        math_format_prompts = {
            "direct_cot": """
First, provide a single "reasoning" field as a string, detailing your step-by-step mathematical solution.
Then, provide your final answer in the "response" field, ending with \\boxed{your_answer}.

Return ONLY the JSON object, with no additional explanations or text.
""",
            "sequence": f"""
Return exactly {num_samplings} mathematical solutions as a Python list of strings, formatted as:
["solution1", "solution2", "solution3", ...]

Each solution should show step-by-step reasoning and end with \\boxed{{final_answer}}.
Return ONLY the list, with no additional explanations or text.
""",
            "structure": """
Return the responses in JSON format with the key "responses" (list of dicts). Each dictionary must include:
- 'text': the complete mathematical solution with step-by-step reasoning, ending with \\boxed{final_answer}.

Return ONLY the JSON object, with no additional explanations or text.
""",
            "vs_standard": f"""
Return the responses in JSON format with the key: "responses" (list of dicts). Each dictionary must include:
- 'text': the complete mathematical solution with step-by-step reasoning, ending with \\boxed{{final_answer}}.
{prob_def}

{distribution_def} Return ONLY the JSON object, with no additional explanations or text.
""",
            "vs_standard": f"""
Return the responses in JSON format with the key: "responses" (list of dicts). Each dictionary must include:
- 'text': the complete mathematical solution with step-by-step reasoning, ending with \\boxed{{final_answer}}.
{prob_def}

{distribution_def} Return ONLY the JSON object, with no additional explanations or text.
""",
            "vs_cot": f"""
Return the responses in JSON format with the key: "responses" (list of dicts). Each dictionary must include:
- 'text': the complete mathematical solution with detailed step-by-step chain-of-thought reasoning, ending with \\boxed{{final_answer}}.
{prob_def}

{distribution_def} Return ONLY the JSON object, with no additional explanations or text.
""",
        }

        # Return math-specific format prompt if available, otherwise fall back to base template
        if method in math_format_prompts:
            return math_format_prompts[method]
        else:
            # Fall back to base template for methods not specifically handled
            base_template = BasePromptTemplate(TaskType.MATH)
            return base_template.get_format_prompt(method, num_samplings, probability_definition, probability_tuning)


#############################Prompt factory###################################
class PromptTemplateFactory:
    """Factory class to create prompt templates for different task types."""
    
    _templates = {
        TaskType.CREATIVITY: CreativityPromptTemplate,
        TaskType.COMMONSENSE: CommonsensePromptTemplate,
        TaskType.BIAS: BiasPromptTemplate,
        TaskType.SYNTHETIC_DATA: SyntheticDataPromptTemplate,
        TaskType.SYNTHETIC_NEGATIVE: SyntheticNegativePromptTemplate,
        TaskType.SAFETY: SafetyPromptTemplate,
        TaskType.MATH: MathPromptTemplate,
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
            "vs_cot": template.get_vs_cot_prompt, # vs chain_of_thought
            "vs_multi": template.get_vs_multi_turn_prompt, # vs multi_turn
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

# def get_VS-Multi (vs_multi)_prompt(self, num_samplings: int = 5, **kwargs) -> str:
#         return f"""
# Provide your {num_samplings} best guesses for the given question that you believe could be correct.

# Return the responses in JSON format with the key: "responses" (a list of dicts with 'text' and 'nll'). Each dictionary must include:
# - 'text': the response string only (no explanations or extra text).
# - 'nll': the estimated negative log likelihood for the response (approx. 1.0 per token, each token ranges from 0.5–2.5 based on creativity).

# Give ONLY the JSON object, no explanations or extra text.
# """