# Examples

This directory contains comprehensive examples for using verbalized sampling in your projects.

## Quick Navigation

### Simple API (Recommended for most users)

Start here if you want to quickly add diversity to your LLM applications:

1. **[01_quick_start.py](01_quick_start.py)** - Your first verbalized sampling call (30 seconds)
   - One-liner usage: `verbalize()` → `dist.argmax()` / `dist.sample()`
   - Mirrors [plan.md](plan.md) §0

2. **[02_basic_usage.py](02_basic_usage.py)** - Exploring all parameters
   - Simple prompts vs messages format
   - Weight modes: elicited, uniform, softmax
   - Selection strategies: argmax vs sample
   - Provider selection and model configuration
   - Serialization: `to_dict()` / `from_dict()`
   - Inspecting metadata and repairs

3. **[03_transforms.py](03_transforms.py)** - Functional operations
   - `dist.map()` - Transform text while preserving probabilities
   - `dist.filter_items()` - Remove items and renormalize
   - `dist.reweight()` - Recompute probabilities
   - Chaining transforms: `dist.map().filter_items().reweight()`
   - Custom scoring functions

4. **[04_recipes.py](04_recipes.py)** - Common use cases
   - Creative writing (diversity without quality loss)
   - Open-ended QA (coverage of valid answers)
   - Synthetic negative data (plausible mistakes)
   - Bias mitigation (uniform sampling)
   - Controlled diversity (tuning tau)
   - Batch generation for data augmentation
   - Quality filtering with transforms

### Research API (For paper experiments)

Use these if you're replicating paper results or running systematic comparisons:

5. **[05_research_api.py](05_research_api.py)** - Experimental framework
   - `run_quick_comparison()` - Compare methods with metrics
   - Task-based generation (Task.JOKE, Task.STORY, etc.)
   - Chain-of-thought reasoning (Method.VS_COT)
   - Method comparison (DIRECT, VS_STANDARD, VS_MULTI, etc.)
   - Evaluation metrics (diversity, self-BLEU, entropy, etc.)

For full experiment replication, see **[scripts/EXPERIMENTS.md](../scripts/EXPERIMENTS.md)**.

## Running Examples

### Prerequisites

```bash
# Install the package
pip install verbalized-sampling

# Set API keys
export OPENAI_API_KEY="your_key"
# or
export ANTHROPIC_API_KEY="your_key"
```

### Run Examples

```bash
# Simple API examples
python examples/01_quick_start.py
python examples/02_basic_usage.py
python examples/03_transforms.py
python examples/04_recipes.py

# Research API examples
python examples/05_research_api.py
```

## Example Output

### Quick Start (01_quick_start.py)

```
Quick Start: Verbalized Sampling in One Line
================================================================================

# verbalized-sampling
k=5  τ=0.12  Σp=1.000  model=gpt-4o

1. 0.350  "The clock struck thirteen in a house that swore it couldn't count."  []
2. 0.240  "Rain wrote its alibi on the window."                                 []
3. 0.210  "The missing key sang in the wrong pocket."                           []
4. 0.120  "Every light in the village blinked once."                            []
5. 0.080  "The footprints ended at the river, then began again upstream."      []

Best (argmax): The clock struck thirteen in a house that swore it couldn't count.
Sampled (seed=7): The missing key sang in the wrong pocket.
```

### Transforms (03_transforms.py)

```python
# Chain transforms
cleaned = (dist
    .map(lambda it: it.text.strip())           # Clean whitespace
    .filter_items(lambda it: len(it.text) < 100)  # Keep short
    .reweight(lambda it: it.meta["p_raw"])     # Use raw weights
)
# Result: New DiscreteDist with invariants preserved (Σp=1, descending order)
```

### Recipes (04_recipes.py)

```python
# Creative writing with diversity
dist = verbalize(
    "Write five first lines for a cozy mystery",
    k=5, tau=0.12, temperature=0.9
)
best = dist.argmax()  # Highest quality
sample = dist.sample()  # Weighted random for variety

# Open-ended QA with coverage
dist = verbalize("Name a US state", k=20, tau=0.10)
unique_states = {it.text.lower() for it in dist}
print(f"Coverage: {len(unique_states)}/{len(dist)}")
```

## Key Concepts

### Distribution-First Mental Model

Instead of asking for **one sample**, ask for a **distribution**:

```python
# Old way: single sample
response = model.generate("Write a joke")

# New way: distribution
dist = verbalize("Write a joke", k=5)
best = dist.argmax()      # Deterministic top choice
varied = dist.sample()    # Weighted random for variety
```

### Invariants Preserved

All `DiscreteDist` objects maintain:
- **Σp = 1.0 ± 1e-6** (probabilities sum to 1)
- **Descending order** (sorted by probability)
- **All probabilities in [0, 1]**

Transforms (`map`, `filter_items`, `reweight`) automatically preserve these invariants.

### Tau (τ) - The Diversity Knob

- **Low tau (0.05-0.10)**: More diversity, keeps low-probability items
- **Medium tau (0.12)**: Balanced (default)
- **High tau (0.20+)**: Less diversity, only high-probability items

```python
diverse = verbalize(prompt, k=8, tau=0.05)   # ~8 items survive
focused = verbalize(prompt, k=8, tau=0.20)   # ~3-4 items survive
```

### Weight Modes

- **`elicited`** (default): Use model's probabilities (if valid)
- **`uniform`**: Equal weights (bias mitigation)
- **`softmax`**: Smooth model's probabilities

```python
# Respect model's confidence
dist_elicited = verbalize(prompt, weight_mode="elicited")

# Ignore model's bias
dist_uniform = verbalize(prompt, weight_mode="uniform")
```

## From Plan to Production

This package implements the design in [plan.md](plan.md):

| Plan Section | Example File |
|--------------|--------------|
| §0 Quick start | [01_quick_start.py](01_quick_start.py) |
| §2 API surface | [02_basic_usage.py](02_basic_usage.py) |
| §13 Recipes | [04_recipes.py](04_recipes.py) |
| §13.4 Transforms | [03_transforms.py](03_transforms.py) |

## Additional Resources

- **[Main README](../README.md)** - Package overview and installation
- **[Plan](plan.md)** - Complete API design and semantics
- **[EXPERIMENTS.md](../scripts/EXPERIMENTS.md)** - Paper experiment replication
- **[Paper](https://arxiv.org/abs/2510.01171)** - Research paper

## Common Patterns

### Pattern 1: Creative Generation

```python
dist = verbalize("Generate creative content", k=5, tau=0.12, temperature=0.9)
best = dist.argmax()  # Quality-focused
```

### Pattern 2: Coverage for QA

```python
dist = verbalize("Answer: what are valid options?", k=20, tau=0.08)
all_options = [it.text for it in dist]  # Diverse coverage
```

### Pattern 3: Synthetic Data

```python
dist = verbalize("Generate training examples", k=10, tau=0.10)
dataset = [{"text": it.text, "weight": it.p} for it in dist]
```

### Pattern 4: Post-hoc Filtering

```python
dist = verbalize(prompt, k=10)
filtered = dist.filter_items(lambda it: meets_quality_bar(it.text))
best = filtered.argmax()
```

## Questions?

- Check the [main README](../README.md) for installation and setup
- See [plan.md](plan.md) for complete API reference
- Read [EXPERIMENTS.md](../scripts/EXPERIMENTS.md) for research use cases
- Open an issue on [GitHub](https://github.com/chats-lab/verbalized-sampling)
