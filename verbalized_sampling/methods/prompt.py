BASE_PROMPT = """
Generate a response to the input prompt. Output ONLY the response, no explanations or extra text.
"""

STANDARD_PROMPT = """
Generate {num_samplings} different responses to the input prompt. Each response must be at least {min_words} words long.
Try to be as creative and diverse as possible.
"""

STANDARD_ALL_POSSIBLE_PROMPT = """
Generate all possible responses to the input prompt. Each response must be at least {min_words} words long.
Try to be as creative and diverse as possible.
"""

SEQUENCE_FORMAT_PROMPT = """
Give your responses in a Python list of strings format like: ["response1", "response2", "response3", ...]
Output ONLY the list, no explanations or extra text.
"""

STRUCTURE_FORMAT_PROMPT = """
Return the output in JSON format with keys: "responses" (list of dicts with 'text'). Each dictionary must include:
- 'text': the response string (no explanation or extra text).

Give ONLY the JSON object, no explanations or extra text.
"""

STRUCTURE_WITH_PROBABILITY_FORMAT_PROMPT = """
Return the output in JSON format with keys: "responses" (list of dicts with 'text' and 'probability'). Each dictionary must include:
- 'text': the response string (no explanation or extra text).
- 'probability': Assign probabilities representing how likely each response would be (value between 0 and 1).

Give ONLY the JSON object, no explanations or extra text.
"""

MULTI_TURN_CONTINUE_PROMPT = """
Generate an alternative response to the original input prompt.
"""

# Chain-of-Thought Sampling Prompts
CHAIN_OF_THOUGHT_PROMPT = """
Output {num_samplings} plausible and diverse responses using chain-of-thought reasoning.
For each response, first think through your reasoning, then provide the response.
Return the output in JSON format with keys: "responses" (list of dicts with "reasoning" and "response"). Each dictionary must include:
- 'reasoning': the step-by-step thinking process.
- 'response': the final response based on reasoning.

Give ONLY the JSON object, no explanations or extra text.
"""

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