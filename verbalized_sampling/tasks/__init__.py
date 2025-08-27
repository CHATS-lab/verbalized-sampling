"""
Task definitions for verbalized sampling experiments.
"""

from typing import Dict, Type
from .base import BaseTask
from .creativity.story import CreativeStoryTask
from .creativity.book import BookTask
from .creativity.poem import PoemTask
from .creativity.speech import SpeechTask
from .creativity.joke import JokeTask
from .bias.rand_num import RandomNumberTask
from .bias.state_name import StateNameTask
from .fact.simple_qa import SimpleQATask
from .synthetic_data.gsm8k import GSM8KTask
from .synthetic_data.livecodebench import LiveCodeBenchTask
from .synthetic_data.synthetic_negative import SyntheticNegativeTask
from enum import Enum

class Task(str, Enum):
    """Available tasks for verbalized sampling experiments.
    
    Each task represents a different type of generation task that can be used
    to evaluate LLM sampling methods.
    """
    
    RANDOM_NUM = "rand_num"
    """Random number generation task.
    
    Generates random numbers within a specified range. Used to test basic
    sampling capabilities and uniformity of distribution.
    """
    
    CREATIVE_STORY = "creative_story"
    """Creative story generation task.
    
    Generates creative stories based on prompts. Tests narrative coherence
    and creativity in longer-form text generation.
    """
    
    BOOK = "book"
    """Book continuation task.
    
    Generates continuations of book excerpts. Tests long-form narrative
    coherence and style consistency.
    """
    
    POEM = "poem"
    """Poetry generation task.
    
    Generates poems based on starting lines. Tests creative expression
    and adherence to poetic forms.
    """
    
    SPEECH = "speech"
    """Speech generation task.
    
    Generates speeches based on opening sentences. Tests rhetorical
    effectiveness and persuasive writing.
    """
    
    STATE_NAME = "state_name"
    """State name generation task.
    
    Generates names for fictional states/countries. Tests creative
    naming and world-building capabilities.
    """

    RAND_NUM = "rand_num"
    """Random number generation task.
    
    Generates random numbers within a specified range. Used to test basic
    sampling capabilities and uniformity of distribution.
    """
    
    JOKE = "joke"
    """Joke generation task.
    
    Generates jokes based on prompts. Tests humor and creative
    wordplay capabilities.
    """

    SIMPLE_QA = "simple_qa"
    """Simple QA task.
    
    Generates answers to the SimpleQA dataset from OpenAI. Tests basic
    reasoning and factual knowledge capabilities.
    """

    GSM8K = "gsm8k"
    """GSM8K task.
    
    Generates answers to the GSM8K dataset from OpenAI.
    """

    LIVECODEBENCH = "livecodebench"
    """LiveCodeBench task.
    
    Generates answers to the LiveCodeBench dataset from OpenAI.
    """

    SYNTHETIC_NEGATIVE = "synthetic_negative"
    """Synthetic negative task.
    
    Generates negative synthetic data.
    """


TASK_REGISTRY: Dict[str, Type[BaseTask]] = {
    # creativity
    "creative_story": CreativeStoryTask,
    "book": BookTask,
    "poem": PoemTask,
    "speech": SpeechTask,
    "joke": JokeTask,
    # bias
    "rand_num": RandomNumberTask,
    "state_name": StateNameTask,
    # fact
    "simple_qa": SimpleQATask,
    # synthetic data
    "gsm8k": GSM8KTask,
    "livecodebench": LiveCodeBenchTask,
    "synthetic_negative": SyntheticNegativeTask,
}

def get_task(task_name: Task, **kwargs) -> BaseTask:
    """Get a task instance by name."""
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Task {task_name} not supported. Available tasks: {list(TASK_REGISTRY.keys())}")
    return TASK_REGISTRY[task_name](**kwargs)