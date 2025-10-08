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

"""Selection and distribution classes for verbalized sampling."""

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union


@dataclass(frozen=True)
class Item:
    """A single candidate with normalized probability and metadata.

    Attributes:
        text: The candidate response text
        p: Normalized probability in [0, 1]
        meta: Metadata dictionary containing:
            - p_raw: Original elicited weight (may be %, >1, malformed)
            - p_clipped: Weight after repair/clipping to [0, 1]
            - repairs: List of repair operations applied
            - idx_orig: Original index in the response
            - provider_meta: Provider-specific metadata (tokens, latency, etc.)
    """

    text: str
    p: float
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate probability is in [0, 1] with small tolerance."""
        # Make meta mutable by creating a copy
        object.__setattr__(self, "meta", dict(self.meta))

        # Validate probability (allow small floating point tolerance)
        if not (0 <= self.p <= 1 + 1e-9):
            raise ValueError(f"Probability must be in [0, 1], got {self.p}")

    def __repr__(self) -> str:
        """Concise representation."""
        text_preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f'Item(p={self.p:.3f}, text="{text_preview}")'


class DiscreteDist(Sequence[Item]):
    """A discrete distribution over text candidates.

    Invariants:
        - Probabilities sum to 1.0 ± 1e-6
        - Items are sorted descending by probability
        - All probabilities are in [0, 1]
    """

    def __init__(self, items: List[Item], trace: Dict[str, Any]):
        """Create a distribution from items and trace metadata.

        Args:
            items: List of Item objects (must be sorted descending by p)
            trace: Metadata dict containing model, tokens, latency, etc.
        """
        self._items = list(items)  # Defensive copy
        self._trace = dict(trace)  # Defensive copy
        self._validate()

    def _validate(self):
        """Ensure invariants: Σp=1.0±ε and descending order."""
        if not self._items:
            return  # Empty distribution is valid

        # Check probabilities sum to 1
        total = sum(it.p for it in self._items)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Probabilities must sum to 1.0, got {total:.6f}")

        # Check descending order
        for i in range(len(self._items) - 1):
            if self._items[i].p < self._items[i + 1].p - 1e-9:
                raise ValueError(
                    f"Items must be sorted descending by p: "
                    f"item[{i}].p={self._items[i].p} < item[{i+1}].p={self._items[i+1].p}"
                )

    # Sequence protocol
    def __getitem__(self, i: int) -> Item:
        """Get item by index."""
        return self._items[i]

    def __len__(self) -> int:
        """Number of items in distribution."""
        return len(self._items)

    def __repr__(self) -> str:
        """Concise representation."""
        return f"DiscreteDist(k={len(self)}, Σp={sum(self.p):.3f}, model={self._trace.get('model', 'unknown')})"

    @property
    def items(self) -> List[Item]:
        """All items (sorted descending by p)."""
        return self._items

    @property
    def p(self) -> List[float]:
        """List of probabilities [item.p for item in items]."""
        return [it.p for it in self._items]

    # Selection methods
    def argmax(self) -> Item:
        """Return highest-probability item (deterministic).

        Returns:
            The first item (highest probability)

        Raises:
            ValueError: If distribution is empty
        """
        if not self._items:
            raise ValueError("Cannot argmax on empty distribution")
        return self._items[0]

    def sample(self, seed: Optional[int] = None) -> Item:
        """Sample an item weighted by probability.

        Args:
            seed: Random seed for reproducibility (optional)

        Returns:
            A sampled item

        Raises:
            ValueError: If distribution is empty
        """
        if not self._items:
            raise ValueError("Cannot sample from empty distribution")

        rng = random.Random(seed) if seed is not None else random
        return rng.choices(self._items, weights=self.p, k=1)[0]

    # Functional transforms (all return new DiscreteDist)
    def map(self, fn: Callable[[Item], str]) -> "DiscreteDist":
        """Map a function over item texts, preserving probabilities.

        Args:
            fn: Function that takes an Item and returns a new text string

        Returns:
            New DiscreteDist with transformed texts
        """
        new_items = [Item(text=fn(it), p=it.p, meta=it.meta) for it in self._items]
        return DiscreteDist(new_items, self._trace)

    def filter_items(self, pred: Callable[[Item], bool]) -> "DiscreteDist":
        """Filter items by predicate and renormalize.

        Args:
            pred: Predicate function that takes an Item and returns bool

        Returns:
            New DiscreteDist with filtered items (renormalized)

        Raises:
            ValueError: If filter removes all items
        """
        filtered = [it for it in self._items if pred(it)]
        if not filtered:
            raise ValueError("Filter removed all items")

        # Renormalize
        total = sum(it.p for it in filtered)
        renormed = [
            Item(text=it.text, p=it.p / total, meta={**it.meta, "renormalized": True})
            for it in filtered
        ]
        return DiscreteDist(renormed, self._trace)

    def reweight(self, fn: Callable[[Item], float]) -> "DiscreteDist":
        """Recompute weights using a function and renormalize.

        Args:
            fn: Function that takes an Item and returns a new weight

        Returns:
            New DiscreteDist with reweighted items (renormalized and re-sorted)

        Raises:
            ValueError: If all new weights are zero or negative
        """
        new_weights = [fn(it) for it in self._items]
        total = sum(new_weights)
        if total <= 1e-12:
            raise ValueError("All new weights are zero or negative")

        renormed = [
            Item(text=it.text, p=w / total, meta={**it.meta, "reweighted": True})
            for it, w in zip(self._items, new_weights)
        ]

        # Re-sort descending
        renormed.sort(key=lambda it: -it.p)
        return DiscreteDist(renormed, self._trace)

    # Serialization
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict.

        Returns:
            Dict with 'items' and 'trace' keys
        """
        return {
            "items": [{"text": it.text, "p": it.p, "meta": it.meta} for it in self._items],
            "trace": self._trace,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiscreteDist":
        """Reconstruct distribution from dict.

        Args:
            data: Dict from to_dict()

        Returns:
            Reconstructed DiscreteDist
        """
        items = [Item(text=it["text"], p=it["p"], meta=it["meta"]) for it in data["items"]]
        return cls(items, data["trace"])

    def to_markdown(self, max_items: Optional[int] = None) -> str:
        """Format as pretty markdown table.

        Args:
            max_items: Maximum number of items to show (None = show all)

        Returns:
            Markdown-formatted string
        """
        lines = ["# verbalized-sampling"]

        # Build header line
        t = self._trace
        header_parts = [
            f"k={len(self)}",
            f"τ={t.get('tau_final', 'N/A')}",
        ]
        if t.get("tau_relaxed"):
            header_parts.append("(relaxed)")

        header_parts.extend(
            [
                f"Σp={sum(self.p):.3f}",
                f"model={t.get('model', 'unknown')}",
            ]
        )

        lines.append("  ".join(header_parts))
        lines.append("")

        # Show items
        items_to_show = self._items[:max_items] if max_items else self._items
        for i, it in enumerate(items_to_show, 1):
            repairs_str = str(it.meta.get("repairs", []))
            text_preview = it.text[:70] + "..." if len(it.text) > 70 else it.text
            lines.append(f'{i}. {it.p:.3f}  "{text_preview}"  {repairs_str}')

        if max_items and len(self._items) > max_items:
            lines.append(f"... ({len(self._items) - max_items} more)")

        return "\n".join(lines)

    @property
    def trace(self) -> Dict[str, Any]:
        """Metadata about the distribution generation.

        Returns:
            Dict containing:
                - model: Model name
                - provider: Provider name
                - latency_ms: Generation latency
                - tau_final: Final tau value used
                - tau_relaxed: Whether tau was relaxed
                - seed: Random seed used
                - etc.
        """
        return self._trace


# ======================== Postprocessing Functions ========================


def repair_weight(raw: Any) -> Tuple[float, List[str]]:
    """Repair a raw weight value to [0, 1], tracking all operations.

    Args:
        raw: Raw weight value (may be str, float, int, malformed)

    Returns:
        Tuple of (repaired_weight, list_of_repair_operations)
    """
    repairs = []
    value = raw

    # Handle string inputs
    if isinstance(value, str):
        value = value.strip()

        # Handle percentages ("70%" -> 0.70)
        if value.endswith("%"):
            repairs.append("percentage")
            try:
                value = float(value.rstrip("%")) / 100.0
            except ValueError:
                repairs.append("invalid")
                return (0.0, repairs)

        # Try parsing as float
        try:
            value = float(value)
        except ValueError:
            repairs.append("invalid")
            return (0.0, repairs)

    # Handle numeric types
    try:
        value = float(value)
    except (TypeError, ValueError):
        repairs.append("invalid")
        return (0.0, repairs)

    # Check for invalid numeric values
    if math.isnan(value) or math.isinf(value):
        repairs.append("invalid")
        return (0.0, repairs)

    # Clip negatives to zero
    if value < 0:
        repairs.append("negative")
        value = 0.0

    # Clip > 1 to 1
    if value > 1:
        repairs.append("clip>1")
        value = 1.0

    return (value, repairs)


def postprocess_responses(
    parsed_responses: List[Dict[str, Any]],
    tau: float,
    min_k_survivors: int,
    weight_mode: Literal["elicited", "softmax", "uniform"],
    seed: Optional[int],
) -> Tuple[List[Item], Dict[str, Any]]:
    """Postprocess parsed responses into normalized Item list.

    Pipeline: Extract & Repair → Filter by tau → Normalize → Sort → Create Items

    Args:
        parsed_responses: List of dicts with 'text' and 'probability' keys
        tau: Probability threshold for filtering
        min_k_survivors: Minimum number of survivors (tau relaxation)
        weight_mode: How to normalize weights ("elicited", "softmax", "uniform")
        seed: Random seed for tie-breaking in stable sort

    Returns:
        Tuple of (items_list, metadata_dict)
    """
    metadata = {}

    # Step 1: Extract and repair weights
    candidates = []
    for idx, resp in enumerate(parsed_responses):
        response = resp.get("response", "")
        raw_p = resp.get("probability", 1.0)

        p_clipped, repairs = repair_weight(raw_p)
        candidates.append(
            {
                "response": response,
                "p_raw": raw_p,
                "p_clipped": p_clipped,
                "repairs": repairs,
                "idx_orig": idx,
            }
        )

    # Step 2: Filter by tau (with relaxation if needed)
    survivors = [c for c in candidates if c["p_clipped"] >= tau]
    tau_relaxed = False

    if len(survivors) < min_k_survivors:
        # Relax tau: sort by p_clipped descending, take top min_k_survivors
        tau_relaxed = True
        candidates.sort(key=lambda c: -c["p_clipped"])
        survivors = candidates[:min_k_survivors]
        tau_final = survivors[-1]["p_clipped"] if survivors else 0.0
    else:
        tau_final = tau

    metadata["tau_relaxed"] = tau_relaxed
    metadata["tau_final"] = tau_final
    metadata["num_filtered"] = len(candidates) - len(survivors)

    if not survivors:
        metadata["error"] = "No survivors after filtering"
        return ([], metadata)

    # Step 3: Normalize weights
    if weight_mode == "uniform":
        # Equal weights
        weights = [1.0 / len(survivors)] * len(survivors)
    elif weight_mode == "softmax":
        # Softmax over clipped weights
        max_p = max(c["p_clipped"] for c in survivors)
        exp_weights = [math.exp(c["p_clipped"] - max_p) for c in survivors]
        total_exp = sum(exp_weights)
        weights = [w / total_exp for w in exp_weights]
    else:  # elicited
        # Use elicited weights, renormalize
        total_p = sum(c["p_clipped"] for c in survivors)
        weights = [c["p_clipped"] / total_p for c in survivors]

    metadata["weight_mode"] = weight_mode

    # Step 4: Stable sort descending by weight
    indexed_survivors = list(zip(survivors, weights, range(len(weights))))
    # Stable sort: primary key = -weight, secondary key = original_index
    indexed_survivors.sort(key=lambda x: (-x[1], x[2]))

    # Step 5: Create Item objects
    items = []
    for cand, weight, _ in indexed_survivors:
        meta = {
            "p_raw": cand["p_raw"],
            "p_clipped": cand["p_clipped"],
            "repairs": cand["repairs"],
            "idx_orig": cand["idx_orig"],
        }
        items.append(Item(text=cand["response"], p=weight, meta=meta))

    return (items, metadata)
