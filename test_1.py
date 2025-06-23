import json
from pydantic import BaseModel, Field
from verbalized_sampling.llms import get_model

json_schema = {
    "type": "json_schema",  # Required for OpenRouter
    "json_schema": {
        "name": "string_list_schema",
        "schema": {
            "type": "object",
            "properties": {
                "responses": {
                    "type": "array",
                    "description": "A list of strings",
                    "items": {
                        "type": "string",
                        "description": "An individual string in the list"
                    }
                }
            },
            "required": [
                "responses"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
}

def main():
    model_config = {
        "temperature": 1.0,
        "top_p": 1.0,
    }
    model = get_model("openai/gpt-4.1", method="direct", config=model_config, strict_json=True)
    
    system_prompt = """
    Generate 5 responses to the input prompt.

    Return your output as a Python list of strings, in the following format:
    ["response1", "response2", "response3", ...]
    Output ONLY the list, no explanations or extra text.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "What was the title of Critical Role's 33rd one-shot episode that aired on August 29, 2019?"}
    ]
    
    # The chat method expects a list of message lists (one per conversation)
    # and returns a list of responses
    response = model.chat([messages], schema=json_schema)
    print(response)

if __name__ == "__main__":
    main()