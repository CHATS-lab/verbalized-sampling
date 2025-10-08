#!/usr/bin/env python3
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

"""
Quick Start - Simple API (mirrors plan.md §0)

This example demonstrates the basic one-liner usage of verbalized sampling.
Learn a tiny distribution from the model and select via argmax or sample.
"""

from verbalized_sampling import verbalize


def main():
    print("=" * 80)
    print("Quick Start: Verbalized Sampling in One Line")
    print("=" * 80)
    print()

    # Learn a tiny distribution from the model
    dist = verbalize(
        "Write an opening line for a mystery novel",
        k=5,
        tau=0.12,
        temperature=0.9,
        seed=42,
    )

    # Quick view of items & normalized masses
    print(dist.to_markdown())
    print()

    # Deterministic top item
    best = dist.argmax()
    print(f"Best (argmax): {best.text}")
    print()

    # Seeded weighted sample
    choice = dist.sample(seed=7)
    print(f"Sampled (seed=7): {choice.text}")
    print()

    # Show trace metadata
    print("Trace metadata:")
    print(f"  Model: {dist.trace.get('model', 'unknown')}")
    print(f"  Latency: {dist.trace.get('latency_ms', 0):.0f}ms")
    print(f"  Tau final: {dist.trace.get('tau_final', 'N/A')}")
    print(f"  Tau relaxed: {dist.trace.get('tau_relaxed', False)}")
    print()


if __name__ == "__main__":
    main()
