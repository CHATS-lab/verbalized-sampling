QUESTION = "Write a creative story about potatoes. The story should be at least 5 sentences long."

SAMPLE_QUESTION = "Write {num_samples} creative stories about potatoes. Each story should be at least 5 sentences long. Return them as a list."

SEQUENCE_PROMPT = "Please provide the stories as a Python list, e.g., ['Story 1...', 'Story 2...']."

FORMAT_WITHOUT_PROBABILITY_PROMPT = (
    "Return your answer as a JSON object with a 'stories' field containing the list of stories.\n"
    "Example: {\"stories\": [\"Once upon a time...\", \"In a faraway land...\"]}"
)

FORMAT_WITH_PROBABILITY_PROMPT = (
    "Return your answer as a JSON object with a 'stories' field and a 'probabilities' field.\n"
    "Example: {\"stories\": [\"Once upon a time...\", \"In a faraway land...\"], \"probabilities\": [0.5, 0.5]}"
) 