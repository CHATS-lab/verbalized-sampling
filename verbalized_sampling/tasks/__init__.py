from typing import Dict, Type
from .base import BaseTask
from .rand_num import RandomNumberTask

TASK_REGISTRY: Dict[str, Type[BaseTask]] = {
    "rand_num": RandomNumberTask,
    # Add more tasks here as they are implemented
}

def get_task(task_name: str, **kwargs) -> BaseTask:
    """Get a task instance by name."""
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Task {task_name} not supported. Available tasks: {list(TASK_REGISTRY.keys())}")
    return TASK_REGISTRY[task_name](**kwargs) 