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
from .synthetic_data.amc_aime import AMCAndAIMEMathTask
from .synthetic_data.livecodebench import LiveCodeBenchTask
from .synthetic_data.synthetic_negative import SyntheticNegativeTask
from .safety.safety import SafetyTask
from .math.math_task import MathTask
from .dialogue.persuasion import PersuasionTask
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

    AMCAndAIMEMathTask = "amc_aime_math"
    """AMC and AIME math task.
    
    Generates answers to the AMC and AIME math dataset from OpenAI.
    """

    LIVECODEBENCH = "livecodebench"
    """LiveCodeBench task.
    
    Generates answers to the LiveCodeBench dataset from OpenAI.
    """

    SYNTHETIC_NEGATIVE = "synthetic_negative"
    """Synthetic negative task.

    Generates negative synthetic data.
    """

    SAFETY = "safety"
    """Safety evaluation task.

    Evaluates model safety using potentially harmful prompts from HarmBench.
    Tests the model's ability to refuse unsafe requests while remaining helpful.
    """

    # Math tasks
    MATH = "math_math"
    """MATH dataset task.

    Solves problems from the MATH dataset with LaTeX formatting.
    Tests mathematical reasoning across various difficulty levels.
    """

    AIME = "math_aime"
    """AIME competition task.

    Solves American Invitational Mathematics Examination problems.
    Tests advanced competition-level mathematical problem solving.
    """

    AMC = "math_amc"
    """AMC competition task.

    Solves American Mathematics Competitions problems.
    Tests intermediate-level mathematical problem solving.
    """

    MINERVA = "math_minerva"
    """MINERVA dataset task.

    Solves physics and advanced mathematics problems.
    Tests scientific reasoning and advanced mathematical concepts.
    """

    OLYMPIAD_BENCH = "math_olympiad_bench"
    """Olympiad Bench task.

    Solves mathematical olympiad competition problems.
    Tests advanced problem-solving and mathematical creativity.
    """

    PERSUASION_DIALOGUE = "persuasion_dialogue"
    """PersuasionForGood dialogue simulation task.

    Simulates multi-turn persuasive dialogues where a persuader tries
    to convince a persuadee to donate to charity. Tests dialogue coherence,
    persuasion effectiveness, and realistic conversation simulation.
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
    "amc_aime_math": AMCAndAIMEMathTask,
    "livecodebench": LiveCodeBenchTask,
    "synthetic_negative": SyntheticNegativeTask,
    # safety
    "safety": SafetyTask,
    # math
    "math_math": lambda **kwargs: MathTask(dataset="math", **kwargs),
    "math_aime": lambda **kwargs: MathTask(dataset="aime", **kwargs),
    "math_amc": lambda **kwargs: MathTask(dataset="amc", **kwargs),
    "math_minerva": lambda **kwargs: MathTask(dataset="minerva", **kwargs),
    "math_olympiad_bench": lambda **kwargs: MathTask(dataset="olympiad_bench", **kwargs),
    # dialogue
    "persuasion_dialogue": PersuasionTask,
}

def get_task(task_name: Task, **kwargs) -> BaseTask:
    """Get a task instance by name."""
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Task {task_name} not supported. Available tasks: {list(TASK_REGISTRY.keys())}")
    return TASK_REGISTRY[task_name](**kwargs)