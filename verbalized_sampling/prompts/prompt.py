# Sequential Sampling Prompts
SEQUENCE_PROMPT = """
Output {num_samplings} plausible and diverse responses, try to be as diverse as possible.
Give your responses in a Python list of strings format like: ["response1", "response2", "response3"]

Give ONLY the list, no explanations or extra text.
"""

# Structured Sampling Prompts (JSON format)
STRUCTURE_RESPONSE_ONLY_PROMPT = """
Output {num_samplings} plausible and diverse responses to the input prompt. Try to be as creative as possible.
Return the output in JSON format with keys: "responses" (list of dicts with "text"). Each dictionary must include:
- 'text': the response string.

Give ONLY the JSON object, no explanations or extra text.
"""

STRUCTURE_WITH_PROBABILITY_PROMPT = """
Output {num_samplings} plausible and diverse responses to the input prompt. Try to be as creative as possible.
Return the output in JSON format with keys: "responses" (list of dicts with "text" and "probability"). Each dictionary must include:
- 'text': the response string.
- 'probability': Assign probabilities representing how likely each response would be (value between 0 and 1).

Give ONLY the JSON object, no explanations or extra text.
"""

# Multi-turn Sampling Prompts
MULTI_TURN_INITIAL_PROMPT = """
You are engaged in a multi-turn text-based conversation.
You are required to output {num_samplings} plausible and diverse responses to the input prompt.

First, provide your initial response to the input prompt.
Then I will ask you to provide alternative responses.
Keep each response diverse and contextually appropriate.
"""

MULTI_TURN_CONTINUE_PROMPT = """
Now provide an alternative response to the original input prompt:
{input_prompt}
Make it different from your previous response(s) while maintaining quality and relevance.
Response {current_turn} of {total_turns}:
"""

MULTI_TURN_FINAL_PROMPT = """
Provide your final alternative response to the original input prompt:
{input_prompt}
Make it unique compared to all previous responses.
Final response ({current_turn} of {total_turns}):
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
Return the output in JSON format with keys: "responses" (list of dicts with "response", "reflection", and "confidence"). Each dictionary must include:
- 'response': the response string.
- 'reflection': the analysis of response quality and appropriateness.
- 'confidence': the confidence score between 0.0 and 1.0.

Give ONLY the JSON object, no explanations or extra text.
"""

# Temperature-based Sampling Prompts
TEMPERATURE_SAMPLING_PROMPT = """
Generate {num_samplings} responses with varying creativity levels.
Create responses ranging from conservative/safe to creative/bold.
Return the output in JSON format with keys: "responses" (list of dicts with "response", "creativity_level", and "temperature"). Each dictionary must include:
- 'response': the response string.
- 'creativity_level': the creativity level of the response (conservative, moderate, creative, bold).
- 'temperature': the temperature of the response (value between 0 and 1).

Give ONLY the JSON object, no explanations or extra text.
"""

# Backward and Forward Prompts (Legacy - kept for compatibility)
FORMAT_WITHOUT_PROBABILITY_PROMPT = STRUCTURE_RESPONSE_ONLY_PROMPT
FORMAT_WITH_PROBABILITY_PROMPT = STRUCTURE_WITH_PROBABILITY_PROMPT