import copy
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, Union, List

from pydantic import BaseModel


def escape_special_characters(
    text: Union[str, Generator[str, None, None]],
) -> Union[str, Generator[str, None, None]]:
    rules = lambda x: x.replace("$", "\$").replace("*", "\*")
    if isinstance(text, Generator):
        for chunk in text:
            yield rules(chunk)
    else:
        return rules(text)


def unescape_special_characters(text: str) -> str:
    rules = lambda x: x.replace("\$", "$").replace("\*", "*")
    return rules(text)


class LM_Agent(ABC):
    def __init__(
        self,
        model_name: str,
        sim_type: str,
        config: dict,
    ):
        self.sim_type = sim_type # ["sampling", "chat"]
        self.model_name = model_name
        self.config = config


    def chat(self, messages) -> Union[str, Generator]:
        if self.sim_type == "sampling":
            response = self._chat_with_format(messages)
        else:
            response = self._chat(messages)
        return response


    @abstractmethod
    def _chat_with_format(self, messages) -> List[Dict[str, Any]]:
        pass


    @abstractmethod
    def _chat(self, messages) -> str:
        pass
