from typing import Dict, Type
from .base import BaseTask
from .creativity.story import CreativeStoryTask
from .creativity.book import BookTask
from .creativity.poem import PoemTask
from .creativity.speech import SpeechTask
from .bias.rand_num import RandomNumberTask
from .bias.state_name import StateNameTask
from enum import Enum

class Task(str, Enum):
    RANDOM_NUM = "rand_num"
    CREATIVE_STORY = "creative_story"
    BOOK = "book"
    POEM = "poem"
    SPEECH = "speech"
    STATE_NAME = "state_name"

TASK_REGISTRY: Dict[str, Type[BaseTask]] = {
    "rand_num": RandomNumberTask,
    "creative_story": CreativeStoryTask,
    "book": BookTask,
    "poem": PoemTask,
    "speech": SpeechTask,
    "state_name": StateNameTask,
}

def get_task(task_name: Task, **kwargs) -> BaseTask:
    """Get a task instance by name."""
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Task {task_name} not supported. Available tasks: {list(TASK_REGISTRY.keys())}")
    return TASK_REGISTRY[task_name](**kwargs)