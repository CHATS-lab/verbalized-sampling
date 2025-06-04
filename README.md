# Verbalized Sampling

## Summary
This repository contains experiments for testing how verbalized sampling can reduce and mitigate mode collapse in LLMs on tasks that require simulation of popularity and diversity. The project includes:

- Random number/sequence generation
    - Evaluated using chi-square tests compared with Python's random package
- Story generation
    - Evaluated using n-gram metrics and sentence transformer metrics

## Structure

- `verbalized_sampling/`: Main package directory
    - `tasks/`: Task implementations
        - `base.py`: Base task interface
        - `rand_num.py`: Random number generation task
        - `story.py`: Story generation task
    - `llms/`: Language model interfaces
        - `base.py`: Base LLM interface
        - `vllm.py`: vLLM implementation
        - `openrouter.py`: OpenRouter implementation
    - `analysis/`: Analysis modules
        - `plots.py`: Histogram plotting
        - `metrics.py`: Statistical metrics (chi-square, etc.)
    - `prompts/`: Task-specific prompts
    - `cli.py`: Command-line interface using Typer
- `scripts/`: Shell scripts for batch jobs
- `output/`: Model outputs and response files
- `analysis/`: Analysis results
    - `plots/`: Generated plots
    - `metrics/`: Statistical metrics

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Running Experiments

You can run experiments using the CLI:

```bash
python -m verbalized_sampling run-experiment \
    --task rand_num \
    --model-name meta-llama/Llama-3.1-70B-Instruct \
    --format structure \
    --temperature 0.7 \
    --top-p 0.9 \
    --num-responses 3 \
    --num-samples 1 \
    --output-file responses.jsonl
```

### Analyzing Results

To analyze experiment results:

```bash
python -m verbalized_sampling analyze-results \
    --target-dir output/meta-llama_Llama-3.3-70B-Instruct \
    --output-dir analysis \
    --sizes 1 3 5 10 50
```

This will:
1. Generate histograms in `analysis/plots/`
2. Calculate chi-square metrics in `analysis/metrics/`

### Available Options

- `--task`: Task to run (rand_num or story)
- `--model-name`: Model name to use
- `--format`: Sampling format (direct, seq, structure, structure_with_prob)
- `--temperature`: Sampling temperature (default: 0.7)
- `--top-p`: Top-p sampling parameter (default: 0.9)
- `--num-responses`: Number of responses to generate (default: 3)
- `--num-samples`: Number of samples per response (default: 1)
- `--num-workers`: Number of parallel workers (default: 128)
- `--output-file`: Output file path (default: responses.jsonl)
- `--use-vllm`: Whether to use vLLM (default: False)

### Scripts

All `.sh` scripts are in the `scripts/` directory. Run them as needed:

```bash
bash scripts/run.sh
``` 

### TODO
- LLMs
    <!-- - Add back the parallel workers -->
    - handle multi-turn conversations
- Tasks
    - Refine for the story task
    - New tasks for creativity index
- Prompts
    - prompt factory
- Generation methods
    - Direct
    - Sequence
    - Structure
    - Structure with probability
    - Multi-turn conversation