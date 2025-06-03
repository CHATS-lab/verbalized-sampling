from .base import BaseTask
from verbalized_sampling.prompts.story import (
    QUESTION,
    SAMPLE_QUESTION,
    SEQUENCE_PROMPT,
    FORMAT_WITHOUT_PROBABILITY_PROMPT,
    FORMAT_WITH_PROBABILITY_PROMPT,
)

class StoryTask(BaseTask):
    """Task for generating creative stories."""
    
    def __init__(self, format: str = "direct"):
        self.format = format
        self.prompt_factory = {
            "direct": lambda question: question,
            "seq": lambda question: question + "\n\n" + SEQUENCE_PROMPT,
            "structure": lambda question: question + "\n\n" + FORMAT_WITHOUT_PROBABILITY_PROMPT,
            "structure_with_prob": lambda question: question + "\n\n" + FORMAT_WITH_PROBABILITY_PROMPT,
        }
    
    def get_prompt(self, num_samples: int = 1) -> str:
        """Get the prompt for the task."""
        if self.format == "direct":
            return QUESTION
        prompt = SAMPLE_QUESTION.format(num_samples=num_samples)
        return self.prompt_factory[self.format](prompt)
    
    def parse_response(self, response: str) -> any:
        import json
        if self.format in ["structure", "structure_with_prob"]:
            try:
                if isinstance(response, str):
                    json_block = "```json"
                    code_block = "```"
                    if json_block in response:
                        content = response[response.find(json_block) + len(json_block):]
                        if code_block in content:
                            content = content[:content.find(code_block)]
                    elif code_block in response:
                        content = response[response.find(code_block) + len(code_block):]
                        if code_block in content:
                            content = content[:content.rfind(code_block)]
                    else:
                        content = response
                    content = content.strip()
                    start = content.find('{')
                    end = content.rfind('}') + 1
                    if start != -1 and end != 0:
                        content = content[start:end]
                    content = content.replace(',\n    ]', '\n    ]')
                    content = content.replace(',\n    }', '\n    }')
                    content = content.replace(',\n]', '\n]')
                    content = content.replace(',\n}', '\n}')
                    return json.loads(content)
                return response
            except json.JSONDecodeError:
                return None
        elif self.format == "seq":
            try:
                if isinstance(response, str):
                    start = response.find('[')
                    end = response.rfind(']') + 1
                    if start != -1 and end != 0:
                        try:
                            return json.loads(response[start:end])
                        except json.JSONDecodeError:
                            pass
            except ValueError:
                pass
            return None
        elif self.format == "direct":
            if isinstance(response, str):
                return response.strip()
            return None 