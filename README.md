<p align="center">
  <img src="./assets/teaser.png" width=90% alt="Verbalized Sampling" />
</p>

[![PyPI - Version](https://img.shields.io/pypi/v/verbalized-sampling.svg)](https://pypi.org/project/verbalized-sampling/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/verbalized-sampling.svg)](https://pypi.org/project/verbalized-sampling/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)

[Installation](#installation) | [Quick Start](#quick-start) | [Tasks](#supported-tasks) | [Citation](#citation)

---

## Updates
* 🎉 01/10/2025: We release our paper, code and package. Check the release page for more details.

## Introduction

**Verbalized Sampling (VS)** is a prompting strategy that mitigates mode collapse in Large Language Models by explicitly requesting responses with associated probabilities. This framework is:

* **Training-Free**: Works with any LLM without fine-tuning—simply apply VS prompts to unlock diversity.
* **Model-Agnostic**: Compatible with GPT, Claude, Gemini, and open models like Llama and Qwen.
* **Measurable Impact**: Achieves 2-3x diversity improvement in creative writing while maintaining quality.
* **Versatile Applications**: Supports creative writing, synthetic data generation, bias mitigation.
* **Complete Framework**: Includes task implementations, evaluation metrics, and reproducible experiments from our paper.
* **Easy to Use**: Simple CLI and Python API for running experiments and comparing methods.

## Installation

```bash
# Lightweight install (API-based models only)
pip install -e .

# With GPU support for local models (vLLM, torch, transformers)
pip install -e ".[gpu]"

# Development install
pip install -e ".[dev]"

# Complete install
pip install -e ".[gpu,dev]"
```

### API Keys Setup
```bash
export OPENAI_API_KEY="your_openai_key"
export OPENROUTER_API_KEY="your_openrouter_key"
```

## Quick Start

### Command Line Interface

```bash
# List available tasks and methods
verbalize list-tasks
verbalize list-methods

# Run an experiment
verbalize run \
    --task joke \
    --model "gpt-4.1" \
    --methods "vs_standard direct vs_cot vs_multi" \
    --num-responses 50

# Run quick test (TODO add this support to the CLI)
verbalize run \
    --task joke \
    --prompt "Write a joke about the weather." \
    --model "gpt-4.1" \
    --methods "direct vs_standard sequence vs_multi" \
    --num-responses 50 \
    --metrics "diversity length ngram joke_quality"

verbalize dialogue \
  --persuader-model "gpt-4.1" \
  --persuadee-model "gpt-4.1" \
  --method direct \
  --num-conversations 5 \
  --num-samplings 4 \
  --max-turns 10 \
  --word-limit 160 \
  --temperature 0.7 \
  --top-p 0.9 \
  --max-tokens 500 \
  --response-selection probability \
  --evaluate \
  --output-file results/dialogue/persuasion_vs_standard.jsonl

```

### Python API

```python
from verbalized_sampling.pipeline import run_quick_comparison
from verbalized_sampling.tasks import Task
from verbalized_sampling.prompts import Method

# Run a quick comparison
results = run_quick_comparison(
    task=Task.JOKE,
    methods=[Method.DIRECT, Method.VS_STANDARD],
    model_name="anthropic/claude-sonnet-4",
    metrics=["diversity", "length", "ngram"],
    num_responses=50,
)

print(f"VS Diversity: {results['VS_STANDARD']['diversity']:.2f}")
print(f"Direct Diversity: {results['DIRECT']['diversity']:.2f}")
```

## VS Method Variants

| Method | Description | Use Case |
|--------|-------------|----------|
| **VS-Standard** | Basic VS with probability estimates | General diversity improvement |
| **VS-CoT** | VS + Chain-of-Thought reasoning | Complex reasoning tasks |
| **VS-Multi** | VS across multiple conversation turns | Multi-turn dialogues |

### Example Usage

```python
from verbalized_sampling.tasks import get_task, Task
from verbalized_sampling.prompts import Method

# Create a task
task = get_task(Task.STORY, num_prompts=10, random_seed=42)

# Generate diverse responses
vs_prompt = task.get_prompt(Method.VS_STANDARD, num_samples=5, prompt_index=0)
responses = model.generate(vs_prompt)
parsed = task.parse_response(Method.VS_STANDARD, responses)
# Returns: [{"response": "...", "probability": 0.15}, ...]

# Chain-of-thought reasoning
cot_prompt = task.get_prompt(Method.VS_COT, num_samples=3)
cot_responses = model.generate(cot_prompt)
parsed_cot = task.parse_response(Method.VS_COT, cot_responses)
# Returns: [{"reasoning": "...", "response": "...", "probability": 0.22}, ...]
```

<details>
<summary><h2 style="display: inline;">Tasks</h2></summary>

The framework supports various task types from our paper experiments:

### Creative Writing (§5)
- **Poetry Continuation**: Continue poems with diverse styles and themes
- **Story Generation**: Generate creative stories with varied plots and characters
- **Joke Writing**: Create humorous content with different comedic approaches

### Synthetic Data Generation (§7)
- **Math Problems**: Generate diverse competition-level math questions (GSM8K, AMC/AIME, LiveCodeBench)
- **Negative Examples**: Create incorrect solutions for robust training

### Bias Mitigation (Appendix)
- **Random Number Generation**: Achieve uniform sampling vs. mode-collapsed outputs
- **Geographic Bias**: Mitigate location-based biases in state/country naming

### Knowledge & Safety (Appendix)
- **Open-ended QA**: Diverse factual responses while maintaining accuracy
- **Safety Evaluation**: Preserve refusal rates for harmful content

</details>

<details>

<summary><h2 style="display: inline;">Evaluation Framework</h2></summary>

### Diversity Metrics
```python
from verbalized_sampling.evals import get_evaluator

# Semantic diversity using embeddings
diversity_eval = get_evaluator("diversity")
result = diversity_eval.evaluate(prompts, responses)
print(f"Semantic Diversity: {result.overall_metrics['average_similarity']:.3f}")

# Creativity assessment (TTCT framework)
ttct_eval = get_evaluator("ttct")
creativity_result = ttct_eval.evaluate(prompts, responses)
print(f"Creativity Score: {creativity_result.overall_metrics['overall']['creativity_score']:.1f}/5")
```

### Supported Models

- **Closed Models**: GPT-4.1, Claude-3.5-Sonnet, Gemini-2.5-Pro
- **Open Models**: Llama-3.1-70B, Qwen3-235B
- **Reasoning Models**: OpenAI o3, DeepSeek-R1
- **Local Models**: Via vLLM integration (requires `[gpu]` install)

</details>

## Reproducing Paper Results

Run experiments in the order presented in our paper:

```bash
# Main experiments (Sections 5-7)
python scripts/tasks/run_poem.py --method vs_standard
python scripts/tasks/run_story.py --method vs_standard
python scripts/tasks/run_jokes.py --method vs_standard

# Synthetic data generation
python scripts/tasks/run_positive_gsm8k.py --method vs_standard
python scripts/tasks/run_positive_amc_aime.py --method vs_standard

# Bias mitigation
python scripts/tasks/run_rng.py --method vs_standard
python scripts/tasks/run_state_name.py --method vs_standard

# Safety evaluation
python scripts/tasks/run_safety.py --method vs_standard
```

## Key Results

Our experiments demonstrate consistent improvements across tasks and models:

- **Creative Writing**: 2-3x diversity improvement while maintaining quality
- **Bias Mitigation**: Uniform sampling (KL divergence: 0.027 vs 0.926 for direct)
- **Emergent Scaling**: Larger models show greater benefits from VS
- **Safety**: Preserved refusal rates for harmful content
- **Tunable Diversity**: Control output diversity via probability thresholds

## Repository Structure

```
verbalized_sampling/           # Main package
├── tasks/                     # Task implementations
│   ├── creativity/           # Creative writing tasks
│   ├── synthetic_data/       # Data generation tasks
│   ├── bias/                # Bias mitigation tasks
│   └── safety/              # Safety evaluation
├── prompts/                  # VS method implementations
├── llms/                     # Model interfaces
├── evals/                    # Evaluation metrics
└── cli.py                    # Command line interface

scripts/tasks/                 # Experimental scripts
├── run_poem.py               # Poetry experiments
├── run_story.py              # Story generation
├── run_jokes.py              # Joke writing
├── run_positive_*.py         # Synthetic data generation
├── run_rng.py                # Random number generation
├── run_state_name.py         # Geographic bias
└── run_safety.py             # Safety evaluation
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Code formatting and linting
black .
isort .
ruff check .
mypy .

# Run tests
pytest
```

## Citation

If you use Verbalized Sampling in your research, please cite our paper:

```bibtex
@article{zhang2025verbalized,
  title={Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity},
  author={Zhang, Jiayi and Yu, Simon and Chong, Derek and Sicilia, Anthony and Tomz, Michael R and Manning, Christopher D and Shi, Weiyan},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.