"""
Dialogue simulation tasks for verbalized sampling experiments.

This module contains tasks for multi-turn dialogue simulation, particularly
focused on persuasive dialogue scenarios like donation conversations.
"""

from .base import DialogueTask
from .persuasion import PersuasionTask

__all__ = ["DialogueTask", "PersuasionTask"]