SEQUENCE_PROMPT = """
    Generate {num_samplings} different responses, try to be as diverse as possible.
    Give your responses in a Python list of strings.
"""

FORMAT_WITHOUT_PROBABILITY_PROMPT = """
    Generate {num_samplings} different responses to your interlocutor that are coherent with the chat history and aligned with your persona.
    Output in JSON format with keys: "responses" (list of dicts with "response").
    Only output the JSON object, no other text.
"""

FORMAT_WITH_PROBABILITY_PROMPT = """
    Generate {num_samplings} different responses to your interlocutor that are coherent with the chat history and aligned with your persona.
    Output in JSON format with keys: "responses" (list of dicts with "response" and "probability"). The probability field represents the empirical probability of each response, ranging from 0 to 1.
    Only output the JSON object, no other text.
"""