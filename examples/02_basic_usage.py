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
Basic Usage - Exploring verbalize() parameters

This example shows different ways to call verbalize() and work with distributions.
"""

from verbalized_sampling import select, verbalize


def example_simple_prompt():
    """Example 1: Simple string prompt."""
    print("=" * 80)
    print("Example 1: Simple String Prompt")
    print("=" * 80)
    print()

    dist = verbalize("Write a haiku about coding", k=3, temperature=0.8)

    print(f"Generated {len(dist)} items:")
    for i, item in enumerate(dist, 1):
        print(f"{i}. (p={item.p:.3f}) {item.text}")
    print()


def example_messages_format():
    """Example 2: Using messages format for multi-turn."""
    print("=" * 80)
    print("Example 2: Messages Format (Multi-turn)")
    print("=" * 80)
    print()

    messages = [
        {"role": "system", "content": "You are a creative writing assistant."},
        {"role": "user", "content": "Write a short horror story opening."},
    ]

    dist = verbalize(messages=messages, k=5, tau=0.15, temperature=0.9)
    print(dist.to_markdown())
    print()


def example_weight_modes():
    """Example 3: Different weight modes."""
    print("=" * 80)
    print("Example 3: Weight Modes (elicited, uniform, softmax)")
    print("=" * 80)
    print()

    prompt = "Name a programming language"

    # Elicited weights (use model's probabilities)
    dist_elicited = verbalize(prompt, k=5, weight_mode="elicited", seed=42)
    print("Elicited weights:")
    for item in dist_elicited[:3]:
        print(f"  {item.p:.3f} - {item.text}")
    print()

    # Uniform weights (ignore model's probabilities)
    dist_uniform = verbalize(prompt, k=5, weight_mode="uniform", seed=42)
    print("Uniform weights:")
    for item in dist_uniform[:3]:
        print(f"  {item.p:.3f} - {item.text}")
    print()

    # Softmax weights (smooth model's probabilities)
    dist_softmax = verbalize(prompt, k=5, weight_mode="softmax", seed=42)
    print("Softmax weights:")
    for item in dist_softmax[:3]:
        print(f"  {item.p:.3f} - {item.text}")
    print()


def example_selection_methods():
    """Example 4: Different selection strategies."""
    print("=" * 80)
    print("Example 4: Selection Methods (argmax vs sample)")
    print("=" * 80)
    print()

    dist = verbalize("Write a greeting", k=5, temperature=0.8, seed=100)

    # Method 1: Direct on distribution
    best = dist.argmax()
    print(f"Argmax (highest probability): {best.text}")
    print()

    # Method 2: Using select() helper
    sampled = select(dist, strategy="sample", seed=123)
    print(f"Sample (weighted random): {sampled.text}")
    print()

    # Method 3: Multiple samples
    print("Five random samples (notice variety):")
    for i in range(5):
        item = dist.sample(seed=200 + i)
        print(f"  {i+1}. {item.text}")
    print()


def example_serialization():
    """Example 5: Serialization and deserialization."""
    print("=" * 80)
    print("Example 5: Serialization (to_dict / from_dict)")
    print("=" * 80)
    print()

    # Generate distribution
    dist = verbalize("Write a one-sentence joke", k=3, seed=42)

    # Serialize to dict
    data = dist.to_dict()
    print("Serialized to dict:")
    print(f"  Items: {len(data['items'])}")
    print(f"  Trace keys: {list(data['trace'].keys())}")
    print()

    # Reconstruct
    from verbalized_sampling import DiscreteDist

    reconstructed = DiscreteDist.from_dict(data)
    print("Reconstructed distribution:")
    print(f"  Length: {len(reconstructed)}")
    print(f"  Top item: {reconstructed.argmax().text}")
    print()


def example_provider_selection():
    """Example 6: Provider and model selection."""
    print("=" * 80)
    print("Example 6: Provider Selection")
    print("=" * 80)
    print()

    # Auto-detect provider based on API keys
    dist_auto = verbalize("Say hello", k=3, provider="auto")
    print(f"Auto provider used: {dist_auto.trace.get('provider', 'unknown')}")
    print(f"Model: {dist_auto.trace.get('model', 'unknown')}")
    print()

    # Explicit provider and model
    dist_explicit = verbalize("Say hello", k=3, provider="openai", model="gpt-4o-mini")
    print(f"Explicit provider: {dist_explicit.trace.get('provider', 'unknown')}")
    print(f"Model: {dist_explicit.trace.get('model', 'unknown')}")
    print()


def example_item_metadata():
    """Example 7: Inspecting item metadata."""
    print("=" * 80)
    print("Example 7: Item Metadata (repairs, raw probabilities)")
    print("=" * 80)
    print()

    dist = verbalize("Write a color name", k=5, tau=0.1, seed=50)

    print("Item metadata:")
    for i, item in enumerate(dist, 1):
        print(f"{i}. {item.text}")
        print(f"   p (normalized): {item.p:.3f}")
        print(f"   p_raw: {item.meta.get('p_raw', 'N/A')}")
        print(f"   p_clipped: {item.meta.get('p_clipped', 'N/A')}")
        print(f"   repairs: {item.meta.get('repairs', [])}")
        print(f"   idx_orig: {item.meta.get('idx_orig', 'N/A')}")
        print()


if __name__ == "__main__":
    example_simple_prompt()
    example_messages_format()
    example_weight_modes()
    example_selection_methods()
    example_serialization()
    example_provider_selection()
    example_item_metadata()
