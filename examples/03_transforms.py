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
Transforms - Functional operations on distributions (plan.md §13.4)

This example demonstrates map, filter_items, and reweight operations.
All transforms preserve invariants (Σp=1, descending order).
"""

from verbalized_sampling import verbalize


def example_map():
    """Example 1: Map - Transform text while preserving probabilities."""
    print("=" * 80)
    print("Example 1: map() - Transform text")
    print("=" * 80)
    print()

    dist = verbalize("Write a greeting", k=5, temperature=0.8, seed=42)

    print("Original:")
    for item in dist[:3]:
        print(f"  {item.p:.3f} - {item.text}")
    print()

    # Transform: uppercase
    uppercased = dist.map(lambda it: it.text.upper())
    print("After map(uppercase):")
    for item in uppercased[:3]:
        print(f"  {item.p:.3f} - {item.text}")
    print()

    # Transform: add prefix
    prefixed = dist.map(lambda it: f">>> {it.text.strip()}")
    print("After map(add prefix):")
    for item in prefixed[:3]:
        print(f"  {item.p:.3f} - {item.text}")
    print()


def example_filter():
    """Example 2: Filter - Remove items and renormalize."""
    print("=" * 80)
    print("Example 2: filter_items() - Remove and renormalize")
    print("=" * 80)
    print()

    dist = verbalize("Write a product name (1-2 words)", k=8, temperature=0.9, seed=100)

    print(f"Original ({len(dist)} items):")
    for item in dist:
        print(f"  {item.p:.3f} - {item.text} (length={len(item.text)})")
    print()

    # Filter: keep only short responses
    short = dist.filter_items(lambda it: len(it.text) < 20)
    print(f"After filter (length < 20): {len(short)} items")
    for item in short:
        print(f"  {item.p:.3f} - {item.text}")
    print()

    # Verify renormalization
    total = sum(item.p for item in short)
    print(f"Total probability after filter: {total:.6f}")
    print()


def example_reweight():
    """Example 3: Reweight - Recompute probabilities and renormalize."""
    print("=" * 80)
    print("Example 3: reweight() - Recompute masses")
    print("=" * 80)
    print()

    dist = verbalize("Write a city name", k=6, temperature=0.8, seed=200)

    print("Original probabilities:")
    for item in dist:
        print(f"  {item.p:.3f} - {item.text} (length={len(item.text)})")
    print()

    # Reweight: favor longer city names
    reweighted = dist.reweight(lambda it: len(it.text))
    print("After reweight(by length):")
    for item in reweighted:
        print(f"  {item.p:.3f} - {item.text} (length={len(item.text)})")
    print()

    # Reweight: use raw probabilities
    raw_reweighted = dist.reweight(lambda it: it.meta.get("p_clipped", 0.1))
    print("After reweight(use p_clipped):")
    for item in raw_reweighted:
        print(f"  {item.p:.3f} - {item.text}")
    print()


def example_chained_transforms():
    """Example 4: Chained transforms (mirrors plan.md §13.4)."""
    print("=" * 80)
    print("Example 4: Chained Transforms")
    print("=" * 80)
    print()

    dist = verbalize("Write a sentence about the ocean", k=8, temperature=0.9, seed=300)

    print(f"Original ({len(dist)} items):")
    for item in dist[:3]:
        print(f"  {item.p:.3f} - {item.text[:60]}...")
    print()

    # Chain: clean → filter → reweight
    cleaned = (
        dist.map(lambda it: it.text.strip())  # Remove whitespace
        .filter_items(lambda it: len(it.text) < 100)  # Keep short
        .reweight(lambda it: it.meta.get("p_raw", 0.1))  # Use raw weights
    )

    print(f"After chain ({len(cleaned)} items):")
    for item in cleaned:
        print(f"  {item.p:.3f} - {item.text[:60]}...")
    print()

    # Verify invariants
    total = sum(item.p for item in cleaned)
    print(f"Total probability: {total:.6f}")
    print(f"Descending order: {all(cleaned[i].p >= cleaned[i+1].p for i in range(len(cleaned)-1))}")
    print()


def example_custom_scoring():
    """Example 5: Custom scoring function."""
    print("=" * 80)
    print("Example 5: Custom Scoring with Reweight")
    print("=" * 80)
    print()

    dist = verbalize("Write a motivational quote", k=6, temperature=0.8, seed=400)

    print("Original:")
    for item in dist:
        print(f"  {item.p:.3f} - {item.text[:50]}...")
    print()

    # Custom score: favor quotes with exclamation marks
    def excitement_score(it):
        base = it.p
        exclamation_bonus = it.text.count("!") * 0.1
        return base + exclamation_bonus

    excited = dist.reweight(excitement_score)
    print("After reweight(excitement score):")
    for item in excited:
        count = item.text.count("!")
        print(f"  {item.p:.3f} - {item.text[:50]}... (! count: {count})")
    print()


def example_transform_metadata():
    """Example 6: Transforms preserve and extend metadata."""
    print("=" * 80)
    print("Example 6: Transform Metadata")
    print("=" * 80)
    print()

    dist = verbalize("Write an animal name", k=4, seed=500)

    print("Original item metadata:")
    for item in dist[:2]:
        print(f"  {item.text}: meta keys = {list(item.meta.keys())}")
    print()

    # Filter adds 'renormalized' flag
    filtered = dist.filter_items(lambda it: len(it.text) > 3)
    print("After filter_items, metadata:")
    for item in filtered[:2]:
        print(f"  {item.text}: renormalized = {item.meta.get('renormalized', False)}")
    print()

    # Reweight adds 'reweighted' flag
    reweighted = dist.reweight(lambda it: 1.0)
    print("After reweight, metadata:")
    for item in reweighted[:2]:
        print(f"  {item.text}: reweighted = {item.meta.get('reweighted', False)}")
    print()


if __name__ == "__main__":
    example_map()
    example_filter()
    example_reweight()
    example_chained_transforms()
    example_custom_scoring()
    example_transform_metadata()
