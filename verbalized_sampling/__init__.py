# Copyright 2025 CHATS-Lab. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Verbalized Sampling: Ask for a distribution, not a sample.

This package provides two APIs:

1. Simple API (NEW in v0.2):
   - verbalize() - One-liner to get a distribution
   - DiscreteDist - Distribution object with .argmax() and .sample()
   - Item - Single candidate with probability

2. Research API (for paper reproduction):
   - Pipeline - Full experimental framework
   - Method, Task, ExperimentConfig
"""

# Simple API (NEW in v0.2)
from .api import select, verbalize
from .selection import DiscreteDist, Item

# Research API (existing, unchanged)
from .cli import app as cli_app

__all__ = [
    # Simple API
    "verbalize",
    "select",
    "Item",
    "DiscreteDist",
    # Research API
    "cli_app",
]

__version__ = "0.2.0"
