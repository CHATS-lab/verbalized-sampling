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
Recipes - Common use cases (plan.md §13)

This example shows practical applications of verbalized sampling:
- Creative writing (diversity without quality loss)
- Open-ended QA (multi-valid answers)
- Synthetic data generation (plausible negatives)
- Bias mitigation (uniform sampling)
"""

from verbalized_sampling import verbalize


def recipe_creative_writing():
    """Recipe 1: Creative writing with diversity (plan.md §13.1)."""
    print("=" * 80)
    print("Recipe 1: Creative Writing - Diversity Without Quality Loss")
    print("=" * 80)
    print()

    # Generate diverse story openings
    dist = verbalize(
        "Write five first lines for a cozy mystery",
        k=5,
        tau=0.12,
        temperature=0.9,
        seed=42,
    )

    print(dist.to_markdown())
    print()

    # Get the best one
    best = dist.argmax()
    print(f"Best opening (argmax): {best.text}")
    print(f"Probability: {best.p:.3f}")
    print()

    # Or sample for variety
    print("Three random samples:")
    for i in range(3):
        sample = dist.sample(seed=100 + i)
        print(f"  {i+1}. {sample.text}")
    print()


def recipe_open_ended_qa():
    """Recipe 2: Open-ended QA with multiple valid answers (plan.md §13.2)."""
    print("=" * 80)
    print("Recipe 2: Open-Ended QA - Coverage of Valid Answers")
    print("=" * 80)
    print()

    # Generate diverse state names
    dist = verbalize(
        "Name a US state",
        k=20,
        tau=0.10,
        temperature=0.95,
        seed=200,
    )

    # Calculate unique coverage
    unique_states = {it.text.lower().strip() for it in dist}
    coverage = len(unique_states)

    print(f"Generated {len(dist)} items")
    print(f"Unique states: {coverage}")
    print(f"Top-1: {dist.argmax().text} (p={dist.argmax().p:.3f})")
    print()

    print("All states generated:")
    for i, item in enumerate(dist, 1):
        print(f"  {i:2d}. {item.p:.3f} - {item.text}")
    print()

    # Distribution of probabilities
    print("Probability distribution:")
    print(f"  Total: {sum(it.p for it in dist):.6f}")
    print(f"  Max: {max(it.p for it in dist):.3f}")
    print(f"  Min: {min(it.p for it in dist):.3f}")
    print(f"  Mean: {sum(it.p for it in dist) / len(dist):.3f}")
    print()


def recipe_synthetic_negatives():
    """Recipe 3: Synthetic negative data generation (plan.md §13.3)."""
    print("=" * 80)
    print("Recipe 3: Synthetic Negatives - Plausible But Wrong")
    print("=" * 80)
    print()

    # Math problem
    problem = """
    A store sells apples for $2 each and oranges for $3 each.
    If you buy 5 apples and 4 oranges, how much do you spend in total?
    """

    prompt = f"""Give five plausible but incorrect solution sketches for this math problem.
Each should be a common mistake students make.

Problem: {problem.strip()}

Generate diverse incorrect approaches (e.g., wrong operation, calculation error, misreading).
"""

    dist = verbalize(prompt, k=5, tau=0.12, temperature=0.9, seed=300)

    print("Plausible negative examples (for contrastive training):")
    print()
    for i, item in enumerate(dist, 1):
        print(f"{i}. (p={item.p:.3f})")
        print(f"   {item.text[:150]}...")
        print()


def recipe_bias_mitigation():
    """Recipe 4: Bias mitigation with uniform sampling."""
    print("=" * 80)
    print("Recipe 4: Bias Mitigation - Uniform Sampling")
    print("=" * 80)
    print()

    # Compare elicited vs uniform weights for occupation question
    prompt = "Name an occupation that requires a college degree"

    # Default: elicited weights (may show bias)
    dist_elicited = verbalize(
        prompt, k=10, tau=0.08, weight_mode="elicited", temperature=0.9, seed=400
    )

    print("Elicited weights (may reflect training bias):")
    for i, item in enumerate(dist_elicited[:5], 1):
        print(f"  {i}. {item.p:.3f} - {item.text}")
    print()

    # Uniform weights (mitigate bias)
    dist_uniform = verbalize(
        prompt, k=10, tau=0.08, weight_mode="uniform", temperature=0.9, seed=400
    )

    print("Uniform weights (equal probability):")
    for i, item in enumerate(dist_uniform[:5], 1):
        print(f"  {i}. {item.p:.3f} - {item.text}")
    print()


def recipe_controlled_diversity():
    """Recipe 5: Controlling diversity with tau."""
    print("=" * 80)
    print("Recipe 5: Controlling Diversity with Tau")
    print("=" * 80)
    print()

    prompt = "Write a product slogan for eco-friendly water bottles"

    # Low tau = more diversity (keeps low-probability items)
    dist_diverse = verbalize(prompt, k=8, tau=0.05, temperature=0.9, seed=500)
    print(f"Low tau (0.05): {len(dist_diverse)} items survived")
    print(f"  Min probability: {min(it.p for it in dist_diverse):.3f}")
    print()

    # High tau = less diversity (only high-probability items)
    dist_focused = verbalize(prompt, k=8, tau=0.20, temperature=0.9, seed=500)
    print(f"High tau (0.20): {len(dist_focused)} items survived")
    print(f"  Min probability: {min(it.p for it in dist_focused):.3f}")
    print()

    # Show tau relaxation
    dist_relaxed = verbalize(
        prompt, k=8, tau=0.50, min_k_survivors=3, temperature=0.9, seed=500
    )
    print(f"Very high tau (0.50) with min_k_survivors=3:")
    print(f"  Tau relaxed: {dist_relaxed.trace.get('tau_relaxed', False)}")
    print(f"  Tau final: {dist_relaxed.trace.get('tau_final', 'N/A'):.3f}")
    print(f"  Items: {len(dist_relaxed)}")
    print()


def recipe_batch_generation():
    """Recipe 6: Batch generation for data augmentation."""
    print("=" * 80)
    print("Recipe 6: Batch Generation - Data Augmentation")
    print("=" * 80)
    print()

    # Generate diverse paraphrases for data augmentation
    original = "The quick brown fox jumps over the lazy dog"

    dist = verbalize(
        f"Paraphrase this sentence in different ways: '{original}'",
        k=6,
        tau=0.10,
        temperature=0.9,
        seed=600,
    )

    print(f"Original: {original}")
    print()
    print(f"Generated {len(dist)} paraphrases:")
    for i, item in enumerate(dist, 1):
        print(f"  {i}. {item.text}")
    print()

    # Save for training data
    augmented_data = [{"original": original, "paraphrase": it.text, "weight": it.p} for it in dist]
    print(f"Created {len(augmented_data)} training examples")
    print()


def recipe_quality_filtering():
    """Recipe 7: Post-hoc quality filtering."""
    print("=" * 80)
    print("Recipe 7: Quality Filtering with Transforms")
    print("=" * 80)
    print()

    dist = verbalize("Write a tweet about climate change (280 chars)", k=8, temperature=0.9, seed=700)

    print(f"Original: {len(dist)} items")
    for item in dist[:3]:
        print(f"  {len(item.text)} chars - {item.text[:50]}...")
    print()

    # Filter: valid tweet length
    valid_tweets = dist.filter_items(lambda it: len(it.text) <= 280)
    print(f"After length filter: {len(valid_tweets)} items")
    for item in valid_tweets[:3]:
        print(f"  {len(item.text)} chars - {item.text[:50]}...")
    print()


if __name__ == "__main__":
    recipe_creative_writing()
    recipe_open_ended_qa()
    recipe_synthetic_negatives()
    recipe_bias_mitigation()
    recipe_controlled_diversity()
    recipe_batch_generation()
    recipe_quality_filtering()
