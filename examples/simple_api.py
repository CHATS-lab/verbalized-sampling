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

"""Simple API examples - Quick start with verbalize()."""

from verbalized_sampling import verbalize


def example_basic():
    """Example 1: Basic usage."""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)

    dist = verbalize(
        "Write an opening line for a mystery novel", k=5, tau=0.12, temperature=0.9, seed=42
    )

    print(dist.to_markdown())
    print()

    best = dist.argmax()
    choice = dist.sample(seed=7)

    print(f"Best (argmax): {best.text}")
    print(f"Sampled (seed=7): {choice.text}")
    print()


def example_transforms():
    """Example 2: Functional transforms."""
    print("=" * 60)
    print("Example 2: Functional Transforms")
    print("=" * 60)

    dist = verbalize("Say hello in different ways", k=5, temperature=0.8)

    # Filter to short responses
    short_dist = dist.filter_items(lambda it: len(it.text) < 50)
    print(f"Filtered to {len(short_dist)} short items (< 50 chars)")
    print(short_dist.to_markdown())
    print()

    # Map to uppercase
    upper_dist = dist.map(str.upper)
    print(f"Mapped to uppercase:")
    print(upper_dist.to_markdown(max_items=3))
    print()

    # Reweight based on length
    rewighted_dist = dist.reweight(lambda it: 1.0 / (len(it.text) + 1))
    print(f"Reweighted by inverse length:")
    print(rewighted_dist.to_markdown(max_items=3))
    print()


def example_inspect_trace():
    """Example 3: Inspect trace metadata."""
    print("=" * 60)
    print("Example 3: Inspect Trace")
    print("=" * 60)

    dist = verbalize("Name a color", k=3, temperature=0.7, seed=42)

    trace = dist.trace
    print(f"Model: {trace['model']}")
    print(f"Provider: {trace['provider']}")
    print(f"Latency: {trace['latency_ms']:.0f}ms")
    print(f"Temperature: {trace['temperature']}")
    print(f"Tau (final): {trace['tau_final']}")
    print(f"Tau relaxed: {trace['tau_relaxed']}")
    print(f"Weight mode: {trace['weight_mode']}")
    print(f"Seed: {trace['seed']}")
    print()


def example_serialization():
    """Example 4: Serialization."""
    print("=" * 60)
    print("Example 4: Serialization")
    print("=" * 60)

    dist = verbalize("Name a fruit", k=3, temperature=0.7)

    # Convert to dict (JSON-serializable)
    data = dist.to_dict()
    print(f"Serialized to dict with {len(data['items'])} items")

    # Reconstruct from dict
    from verbalized_sampling import DiscreteDist

    dist_restored = DiscreteDist.from_dict(data)
    print(f"Restored distribution:")
    print(dist_restored.to_markdown())
    print()


def example_different_providers():
    """Example 5: Different providers."""
    print("=" * 60)
    print("Example 5: Different Providers")
    print("=" * 60)

    # Auto-detect provider from API keys
    dist_auto = verbalize("Write a haiku about code", k=3, provider="auto")
    print(f"Auto-detected provider: {dist_auto.trace['provider']}")
    print()

    # Explicitly specify provider
    # dist_openai = verbalize("Write a haiku", k=3, provider="openai", model="gpt-4o")
    # dist_anthropic = verbalize("Write a haiku", k=3, provider="anthropic")
    # dist_google = verbalize("Write a haiku", k=3, provider="google")


def example_weight_modes():
    """Example 6: Different weight normalization modes."""
    print("=" * 60)
    print("Example 6: Weight Modes")
    print("=" * 60)

    # Elicited (default) - use model's probabilities
    dist_elicited = verbalize("Name a number from 1-5", k=5, weight_mode="elicited", seed=42)
    print("Elicited weights:")
    print(dist_elicited.to_markdown())
    print()

    # Uniform - ignore model's weights, treat all equally
    dist_uniform = verbalize("Name a number from 1-5", k=5, weight_mode="uniform", seed=42)
    print("Uniform weights:")
    print(dist_uniform.to_markdown())
    print()

    # Softmax - apply softmax to model's weights
    dist_softmax = verbalize("Name a number from 1-5", k=5, weight_mode="softmax", seed=42)
    print("Softmax weights:")
    print(dist_softmax.to_markdown())
    print()


if __name__ == "__main__":
    # Run examples
    # Note: Requires OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY

    import sys

    if len(sys.argv) > 1:
        example_name = sys.argv[1]
        if example_name == "basic":
            example_basic()
        elif example_name == "transforms":
            example_transforms()
        elif example_name == "trace":
            example_inspect_trace()
        elif example_name == "serialize":
            example_serialization()
        elif example_name == "providers":
            example_different_providers()
        elif example_name == "weights":
            example_weight_modes()
        else:
            print(f"Unknown example: {example_name}")
            print("Available: basic, transforms, trace, serialize, providers, weights")
    else:
        # Run basic example by default
        example_basic()
