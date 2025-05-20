QUESTION = "Generate random numbers between 1 and 100"
SAMPLE_QUESTION = "Generate {num_samples} random numbers between 1 and 100"
FORMAT_PROMPT = """
For each choice, output in JSON format with keys: "responses" (list of dicts with "text" and "probability"). Each probability should be between 0 and 1.
Output in the following format: {{
    "responses": [
        {{
            "text": [Answer]
            "probability": [Probability]
        }}
    ]
}}
"""
