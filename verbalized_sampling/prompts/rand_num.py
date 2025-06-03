QUESTION = "Generate a random integer between 0 and 9."

SAMPLE_QUESTION = "Generate {num_samples} random integers between 0 and 9 as a list."

SEQUENCE_PROMPT = "Please provide the numbers as a Python list, e.g., [3, 7, 1]."

FORMAT_WITHOUT_PROBABILITY_PROMPT = (
    "Return your answer as a JSON object with a 'numbers' field containing the list of numbers.\n"
    "Example: {\"numbers\": [2, 5, 8]}"
)

FORMAT_WITH_PROBABILITY_PROMPT = (
    "Return your answer as a JSON object with a 'numbers' field and a 'probabilities' field.\n"
    "Example: {\"numbers\": [2, 5, 8], \"probabilities\": [0.1, 0.2, 0.7]}"
) 