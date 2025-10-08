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

"""Main verbalize() API - thin wrapper over existing infrastructure."""

import os
import time
from typing import Any, Dict, List, Literal, Optional, Sequence

from .selection import DiscreteDist, Item, postprocess_responses
from .llms import get_model
from .methods import Method, ResponseParser
from .methods.prompt import BasePromptTemplate, TaskType, CreativityPromptTemplate
from .methods.schema import get_schema


def verbalize(
    prompt: Optional[str] = None,
    *,
    messages: Optional[Sequence[Dict[str, Any]]] = None,
    # Core knobs
    k: int = 5,
    tau: float = 0.12,
    temperature: float = 0.9,
    # Provider/model
    provider: Literal["auto", "openai", "anthropic", "google"] = "auto",
    model: Optional[str] = None,
    # Robustness
    min_k_survivors: int = 3,
    retries: int = 2,
    # Weight handling
    weight_mode: Literal["elicited", "softmax", "uniform"] = "elicited",
    probability_definition: str = "explicit",
    probability_tuning: float = -1,
    # Determinism
    seed: Optional[int] = None,
    # Advanced
    use_strict_json: bool = True,
    num_workers: int = 1,
    **provider_kwargs,
) -> DiscreteDist:
    """Elicit k weighted candidates → DiscreteDist (filtered, normalized, ordered).

    This function provides a simple one-liner interface to verbalized sampling.
    It reuses the existing LLM infrastructure and adds ergonomic wrappers.

    Args:
        prompt: User prompt string (mutually exclusive with messages)
        messages: Chat messages format (mutually exclusive with prompt)
        k: Number of candidates to generate
        tau: Probability threshold for filtering (0.0-1.0)
        temperature: LLM sampling temperature
        provider: Provider name or "auto" to detect from API keys
        model: Model name (auto-selected if None)
        min_k_survivors: Minimum items after filtering (relaxes tau if needed)
        retries: Number of retry attempts on failure
        weight_mode: How to normalize weights ("elicited"|"softmax"|"uniform")
        probability_definition: Probability field type (see methods.schema)
        probability_tuning: Probability tuning parameter (see methods.prompt)
        seed: Random seed for determinism
        use_strict_json: Use structured JSON output mode
        num_workers: Number of parallel workers (default 1 for simple API)
        **provider_kwargs: Additional kwargs for LLM provider

    Returns:
        DiscreteDist with filtered, normalized, sorted items

    Raises:
        ValueError: If both or neither of prompt/messages provided
        RuntimeError: If generation fails after retries

    Example:
        >>> dist = verbalize("Write a haiku", k=3, tau=0.15)
        >>> print(dist.to_markdown())
        >>> best = dist.argmax()
        >>> print(best.text)
    """
    # Input validation
    if (prompt is None) == (messages is None):
        raise ValueError("Exactly one of 'prompt' or 'messages' must be provided")

    if prompt is not None:
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = list(messages)

    # Auto-select model if not provided
    if model is None:
        model = _auto_select_model(provider)

    # Build config
    config = {
        "temperature": temperature,
        "seed": seed,
        "max_tokens": provider_kwargs.get("max_tokens", 8192),
        "top_p": provider_kwargs.get("top_p", 1.0),
    }

    # Add any additional provider kwargs
    for key, value in provider_kwargs.items():
        if key not in ["max_tokens", "top_p"]:
            config[key] = value

    llm = get_model(
        model_name=model,
        method=Method.VS_STANDARD,
        config=config,
        num_workers=num_workers,
        strict_json=use_strict_json,
    )

    base_template = CreativityPromptTemplate()
    standard_prompt = base_template.get_standard_prompt(num_samplings=k)
    format_prompt = base_template.get_format_prompt(
        method="vs_standard",
        num_samplings=k,
        probability_definition=probability_definition,
        probability_tuning=probability_tuning,
    )

    # Combine prompts
    vs_system_prompt = f"{standard_prompt}\n{format_prompt}"
    full_messages = [{"role": "system", "content": vs_system_prompt}, *messages]

    # Get schema (REUSE existing schema builder!)
    schema = get_schema(Method.VS_STANDARD, use_tools=False, probability_definition=probability_definition)

    # Generate with retries
    responses = None
    latency_ms = 0

    for attempt in range(retries + 1):
        try:
            start_time = time.time()

            # Call LLM (REUSE existing LLM interface!)
            if use_strict_json:
                responses = llm._chat_with_format(full_messages, schema)
            else:
                response_str = llm._chat(full_messages)
                # Parse manually using existing parser
                responses = ResponseParser.parse_structure_with_probability(response_str)

            latency_ms = (time.time() - start_time) * 1000

            # Validate we got responses
            if not responses:
                raise ValueError("No responses returned from LLM")

            # Validate we got at least min_k_survivors
            if len(responses) < min_k_survivors and attempt < retries:
                raise ValueError(
                    f"Expected at least {min_k_survivors} items, got {len(responses)}"
                )

            break  # Success

        except Exception as e:
            if attempt == retries:
                raise RuntimeError(f"Failed after {retries} retries: {e}") from e

            # Continue with same prompt for retry
            continue

    print(f"responses: {responses}")
    # Postprocess (NEW: filter/normalize/order)
    items, transform_meta = postprocess_responses(
        responses, tau=tau, min_k_survivors=min_k_survivors, weight_mode=weight_mode, seed=seed
    )

    # Build trace
    trace = {
        "model": llm.model_name,
        "provider": _infer_provider(model),
        "latency_ms": latency_ms,
        "k": k,
        "tau": tau,
        "temperature": temperature,
        "seed": seed,
        "use_strict_json": use_strict_json,
        "probability_definition": probability_definition,
        **transform_meta,
    }

    return DiscreteDist(items, trace)


def select(
    dist: DiscreteDist,
    strategy: Literal["argmax", "sample"] = "sample",
    seed: Optional[int] = None,
) -> Item:
    """Helper for selecting from a distribution.

    Args:
        dist: DiscreteDist to select from
        strategy: Selection strategy ("argmax" or "sample")
        seed: Random seed for sample strategy

    Returns:
        Selected Item
    """
    if strategy == "argmax":
        return dist.argmax()
    else:
        return dist.sample(seed=seed)


def _auto_select_model(provider: str) -> str:
    """Auto-select default model for provider.

    Args:
        provider: Provider name ("auto", "openai", "anthropic", "google")

    Returns:
        Model name string

    Raises:
        ValueError: If provider is "auto" and no API keys found
    """
    if provider == "openai" or (provider == "auto" and os.getenv("OPENAI_API_KEY")):
        return "gpt-4.1"
    elif provider == "anthropic" or (provider == "auto" and os.getenv("ANTHROPIC_API_KEY")):
        return "claude-3-7-sonnet"
    elif provider == "google" or (provider == "auto" and os.getenv("GOOGLE_API_KEY")):
        return "gemini-2.5-flash"
    elif provider == "auto":
        # Check all possible API keys
        if os.getenv("OPENAI_API_KEY"):
            return "gpt-4.1"
        elif os.getenv("ANTHROPIC_API_KEY"):
            return "claude-3-7-sonnet"
        elif os.getenv("GOOGLE_API_KEY"):
            return "gemini-2.5-flash"
        else:
            raise ValueError(
                "No API key found. Set one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY"
            )
    else:
        # Unknown provider, default to GPT
        return "gpt-4.1"


def _infer_provider(model: str) -> str:
    """Infer provider name from model string.

    Args:
        model: Model name

    Returns:
        Provider name
    """
    model_lower = model.lower()
    if "gpt" in model_lower or "o3" in model_lower or "o4" in model_lower:
        return "openai"
    elif "claude" in model_lower:
        return "anthropic"
    elif "gemini" in model_lower:
        return "google"
    else:
        return "unknown"