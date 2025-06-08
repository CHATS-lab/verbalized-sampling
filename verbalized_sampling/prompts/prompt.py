# Sequential Sampling Prompts
SEQUENCE_PROMPT = """
Generate {num_samplings} different responses, try to be as diverse as possible.
Give your responses in a Python list of strings format like: ["response1", "response2", "response3"]
Only output the list, no other text or explanation.
"""

# Structured Sampling Prompts (JSON format)
STRUCTURE_RESPONSE_ONLY_PROMPT = """
Generate {num_samplings} different responses to the given prompt.
Output in JSON format with the following structure:
{{
    "responses": [
        {{"response": "your first response here"}},
        {{"response": "your second response here"}},
        ...
    ]
}}
Only output the JSON object, no other text or markdown formatting.
"""

# Bias might come from the examples
STRUCTURE_WITH_PROBABILITY_PROMPT = """
Generate {num_samplings} different responses to the given prompt.
Output in JSON format with the following structure:
{{
    "responses": [
        {{"response": "your first response here", "probability": 0.3}},
        {{"response": "your second response here", "probability": 0.25}},
        ...
    ]
}}
The probability field represents the empirical probability of each response (should sum to 1.0).
Only output the JSON object, no other text or markdown formatting.
"""

# Multi-turn Sampling Prompts
MULTI_TURN_INITIAL_PROMPT = """
You will engage in a multi-turn conversation to generate {num_samplings} different responses.
First, provide your initial response to the given prompt.
Then I will ask you to provide alternative responses.
Keep each response diverse and contextually appropriate.
"""

MULTI_TURN_CONTINUE_PROMPT = """
Now provide an alternative response to the original prompt. 
Make it different from your previous response(s) while maintaining quality and relevance.
Response {current_turn} of {total_turns}:
"""

MULTI_TURN_FINAL_PROMPT = """
Provide your final alternative response to the original prompt.
Make it unique compared to all previous responses.
Final response ({current_turn} of {total_turns}):
"""

# Chain-of-Thought Sampling Prompts
CHAIN_OF_THOUGHT_PROMPT = """
Generate {num_samplings} different responses using chain-of-thought reasoning.
For each response, first think through your reasoning, then provide the response.
Format as follows:
{{
    "responses": [
        {{
            "reasoning": "step-by-step thinking process",
            "response": "final response based on reasoning"
        }},
        ...
    ]
}}
Only output the JSON object, no other text.
"""

# Self-Reflection Sampling Prompts
SELF_REFLECTION_PROMPT = """
Generate {num_samplings} different responses with self-reflection and confidence scoring.
For each response, provide the response, reflect on its quality, and assign a confidence score.
Format as follows:
{{
    "responses": [
        {{
            "response": "your response here",
            "reflection": "analysis of response quality and appropriateness",
            "confidence": 0.85
        }},
        ...
    ]
}}
Confidence should be between 0.0 and 1.0. Only output the JSON object.
"""

# Temperature-based Sampling Prompts
TEMPERATURE_SAMPLING_PROMPT = """
Generate {num_samplings} responses with varying creativity levels.
Create responses ranging from conservative/safe to creative/bold.
Format as follows:
{{
    "responses": [
        {{
            "response": "your response here",
            "creativity_level": "conservative|moderate|creative|bold",
            "temperature": 0.3
        }},
        ...
    ]
}}
Vary the creativity levels across responses. Only output the JSON object.
"""

# Backward and Forward Prompts (Legacy - kept for compatibility)
FORMAT_WITHOUT_PROBABILITY_PROMPT = STRUCTURE_RESPONSE_ONLY_PROMPT
FORMAT_WITH_PROBABILITY_PROMPT = STRUCTURE_WITH_PROBABILITY_PROMPT