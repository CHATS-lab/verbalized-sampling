import os
from typing import Any, Dict, Generator, List
import json

from google import genai
from pydantic import BaseModel

from verbalized_sampling.llms.base import BaseLLM

# --- Google Model Mapping ---
GEMINI_MODELS_MAPPING = {
    "gemini-1.5-pro": "gemini-1.5-pro-latest",
    "gemini-1.5-flash": "gemini-1.5-flash-latest",
    "gemini-1.0-pro": "gemini-1.0-pro",
    "google/gemini-1.5-pro": "gemini-1.5-pro-latest",
    "google/gemini-1.5-flash": "gemini-1.5-flash-latest",
    "google/gemini-1.0-pro": "gemini-1.0-pro",
}


class GoogleLLM(BaseLLM):
    """
    A class for interacting with Google's Generative AI models (e.g., Gemini).
    """
    def __init__(self, model_name: str, config: Dict[str, Any], num_workers: int = 1, strict_json: bool = False):
        """
        Initializes the GoogleLLM client.

        Args:
            model_name (str): The name of the Google model to use.
            config (Dict[str, Any]): Configuration parameters for the model.
            num_workers (int): The number of parallel workers (for compatibility with BaseLLM).
            strict_json (bool): Whether to enforce strict JSON output.
        """
        super().__init__(model_name, config, num_workers, strict_json)

        if model_name in GEMINI_MODELS_MAPPING:
            self.model_name = GEMINI_MODELS_MAPPING[model_name]

        # Configure the Google AI client using the API key from environment variables
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

        # Prepare the generation config by translating standard keys
        self.generation_config = self._prepare_config(config)

        # Initialize the GenerativeModel client
        self.client = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=self.generation_config
        )

    def _prepare_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Translates a generic config to the Google `generation_config` format."""
        prepared_config = config.copy()

        # Translate keys from OpenAI nomenclature to Google's
        if 'max_tokens' in prepared_config:
            prepared_config['max_output_tokens'] = prepared_config.pop('max_tokens')
        if 'top_p' in prepared_config:
            prepared_config['top_p'] = prepared_config.pop('top_p')
        if 'temperature' in prepared_config:
            prepared_config['temperature'] = prepared_config.pop('temperature')
        
        # 'n' or 'best_of' for multiple generations is not directly supported in the same way,
        # so we don't translate it. The parent class handles multiple calls if needed.
        prepared_config.pop('n', None)
        prepared_config.pop('best_of', None)
        
        return prepared_config
        
    def _remap_roles(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Remaps role names from 'assistant' to 'model' for the Google API."""
        remapped_messages = []
        for msg in messages:
            new_msg = msg.copy()
            if new_msg.get('role') == 'assistant':
                new_msg['role'] = 'model'
            remapped_messages.append(new_msg)
        return remapped_messages

    def _chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Sends a chat request to the Google model and returns the text response.
        """
        # Google's API uses 'model' instead of 'assistant' for the response role.
        history = self._remap_roles(messages)
        
        response = self.client.generate_content(history)
        
        try:
            response_text = response.text
            response_text = response_text.replace("\n", "")
            if response_text.startswith('"') and response_text.endswith('"'):
                response_text = response_text[1:-1]
            return response_text
        except Exception:
            # Handle cases where the response might be blocked or empty
            print(f"Warning: Could not extract text from response. Parts: {response.parts}")
            return ""

    def _parse_response_with_schema(self, response: str) -> List[Dict[str, Any]]:
        """
        Parses the JSON response string based on expected structures.
        This logic is kept similar to the OpenAILLM implementation for consistency.
        """
        try:
            if isinstance(response, str):
                # Clean up common markdown fences for JSON
                if response.strip().startswith("```json"):
                    response = response.strip()[7:-4]

                parsed = json.loads(response)

                # Handle double-escaped JSON strings
                if isinstance(parsed, str):
                    parsed = json.loads(parsed)

                if "responses" in parsed and isinstance(parsed["responses"], list):
                    result = []
                    for resp in parsed["responses"]:
                        if isinstance(resp, dict):
                            text = resp.get("text")
                            if "probability" in resp:
                                prob = resp.get("probability")
                            elif "confidence" in resp:
                                prob = resp.get("confidence")
                            elif "perplexity" in resp:
                                prob = resp.get("perplexity")
                            elif "nll" in resp:
                                prob = resp.get("nll")
                            else:
                                prob = 1.0
                            if text is not None:
                                result.append({"response": text, "probability": prob})
                        elif isinstance(resp, str):
                            result.append({"response": resp, "probability": 1.0})
                    return result
                elif "text" in parsed:
                    return [{
                        "response": parsed["text"],
                        "probability": parsed.get("probability", 1.0)
                    }]
                elif "response" in parsed:
                    return [{
                        "response": parsed["response"],
                        "probability": parsed.get("probability", 1.0)
                    }]
                
                # Fallback for unexpected but valid JSON
                return [{"response": str(parsed), "probability": 1.0}]

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response: {e}")
            # If parsing fails, return the raw string as a single response
            return [{"response": response, "probability": 1.0}]
        except Exception as e:
            print(f"Error parsing response with schema: {e}")
            return [{"response": response, "probability": 1.0}]

    def _chat_with_format(self, messages: List[Dict[str, str]], schema: BaseModel) -> List[Dict[str, Any]]:
        """
        Sends a chat request and asks for a JSON response that fits the schema.
        """
        # Add instruction for JSON output to the last user message.
        # This is a robust way to ensure the model adheres to the format.
        formatted_messages = self._remap_roles(messages)
        formatted_messages[-1]['parts'][-1] += "\n\nPlease format your response as a JSON object."

        # Create a specific client for this request to set the JSON mime type
        json_client = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={
                **self.generation_config,
                "response_mime_type": "application/json",
            }
        )
        
        try:
            response = json_client.generate_content(formatted_messages)
            parsed_response = self._parse_response_with_schema(response.text)
            return parsed_response
        except Exception as e:
            print(f"Error during structured generation: {e}")
            # Fallback to a simple chat call if structured generation fails
            print("Falling back to standard chat generation...")
            fallback_response = self._chat(messages)
            return [{"response": fallback_response, "probability": 1.0}]