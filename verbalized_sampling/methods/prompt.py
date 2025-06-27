BASE_PROMPT = """
Generate a response to the input prompt. Output ONLY the response, no explanations or extra text.
"""

STANDARD_PROMPT = """
Generate {num_samplings} different responses to the input prompt. Try to be as creative and diverse as possible.
"""

# # Special version for the simple QA task
# STANDARD_PROMPT = """
# Provide your {num_samplings} best guesses for the given question. 
# """

# # Special version for the state name task
# STANDARD_PROMPT = """
# Generate {num_samplings} different responses to the input prompt. Try to be as diverse as possible.
# """

STANDARD_ALL_POSSIBLE_PROMPT = """
Generate all possible responses to the input prompt. Try to be as creative and diverse as possible.
"""

# Combined multi-turn and verbalized sampling together
STANDARD_COMBINED_PROMPT = """
Generate {num_samplings} plausible and diverse responses to the input prompt. Try to be as creative and diverse as possible.

Return the responses in JSON format with keys: "responses" (list of dicts with 'text' and 'probability'). Each dictionary must include:
- 'text': the response string (no explanation or extra text).
- 'probability': the empirical probability representing how likely each response would be (0.0 to 1.0).

Give ONLY the JSON object, no explanations or extra text.
"""


BASE_PROMPT_TARGET_WORDS = """
Generate a response to the input prompt with {target_words} target words.
Output ONLY the response, no explanations or extra text.
"""
STANDARD_PROMPT_TARGET_WORDS = """
Generate {num_samplings} different responses to the input prompt with {target_words} target words.
Try to be as creative and diverse as possible.
"""
STANDARD_ALL_POSSIBLE_PROMPT_TARGET_WORDS = """
Generate all possible responses to the input prompt with {target_words} target words.
Try to be as creative and diverse as possible.
"""
STANDARD_COMBINED_PROMPT_TARGET_WORDS = """
Generate {num_samplings} different responses to the input prompt with {target_words} target words.
Try to be as creative and diverse as possible.

Return the responses in JSON format with keys: "responses" (list of dicts with 'text' and 'probability'). Each dictionary must include:
- 'text': the response string (no explanation or extra text).
- 'probability': the empirical probability representing how likely each response would be (0.0 to 1.0).

Give ONLY the JSON object, no explanations or extra text.
"""

# SEQUENCE_FORMAT_PROMPT = """
# Give your responses in a Python list of strings format like: ["response1", "response2", "response3", ...]
# Output ONLY the list, no explanations or extra text.
# """
SEQUENCE_FORMAT_PROMPT = """
Return ALL responses as a Python list of strings, in the following format:
["response1", "response2", "response3", ...]
Output ONLY the list, no explanations or extra text.
"""

STRUCTURE_FORMAT_PROMPT = """
Return the responses in JSON format with keys: "responses" (list of dicts with 'text'). Each dictionary must include:
- 'text': the response string (no explanation or extra text).

Give ONLY the JSON object, no explanations or extra text.
"""

STRUCTURE_WITH_PROBABILITY_FORMAT_PROMPT = """
Return the responses in JSON format with keys: "responses" (list of dicts with 'text' and 'probability'). Each dictionary must include:
- 'text': the response string (no explanation or extra text).
- 'probability': the empirical probability representing how likely each response would be (0.0 to 1.0).

Give ONLY the JSON object, no explanations or extra text.
"""

MULTI_TURN_CONTINUE_PROMPT = """
Generate an alternative response to the original input prompt.
"""

COMBINED_CONTINUE_PROMPT = """
Generate {num_samplings} alternative responses to the original input prompt.
"""

# Chain-of-Thought Sampling Prompts
CHAIN_OF_THOUGHT_PROMPT = """
Provide {num_samplings} plausible responses to the input prompt using chain-of-thought reasoning.
First, provide a single "reasoning" field that details your step-by-step thought process.
Then, under "responses", return a list of dictionaries. Each dictionary must include:
- 'text': the response string (no explanation or extra text).
- 'probability': the empirical probability representing how likely each response would be (0.0 to 1.0).

Give ONLY the JSON object, no explanations or extra text.
"""

# CHAIN_OF_THOUGHT_PROMPT = """
# Generate {num_samplings} plausible and diverse responses using chain-of-thought reasoning.
# First, provide a single "reasoning" field that details your step-by-step thought process.
# Then, under "responses", return a list of dictionaries. Each dictionary must include:
# - 'text': the response string (no explanation or extra text).
# - 'probability': the empirical probability representing how likely each response would be (0.0 to 1.0).

# Give ONLY the JSON object, no explanations or extra text.
# """

# Self-Reflection Sampling Prompts
SELF_REFLECTION_PROMPT = """
Generate {num_samplings} different responses with self-reflection and confidence scoring.
For each response, provide the response, reflect on its quality, and assign a confidence score.
Return the output in JSON format with keys: "responses" (list of dicts with 'response', 'reflection', and 'confidence'). Each dictionary must include:
- 'response': the response string.
- 'reflection': the analysis of response quality and appropriateness.
- 'confidence': the confidence score between 0.0 and 1.0.

Give ONLY the JSON object, no explanations or extra text.
"""

# Temperature-based Sampling Prompts
TEMPERATURE_SAMPLING_PROMPT = """
Generate {num_samplings} responses with varying creativity levels.
Create responses ranging from conservative/safe to creative/bold.
Return the output in JSON format with keys: "responses" (list of dicts with 'response', 'creativity_level', and 'temperature'). Each dictionary must include:
- 'response': the response string.
- 'creativity_level': the creativity level of the response (conservative, moderate, creative, bold).
- 'temperature': the temperature of the response (value between 0 and 1).

Give ONLY the JSON object, no explanations or extra text.
"""