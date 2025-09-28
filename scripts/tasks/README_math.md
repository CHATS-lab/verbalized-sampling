# Math Task Scripts

This directory contains scripts for running math task experiments, similar to the joke generation scripts.

## Scripts

### `run_math_simple.py` - Basic Math Generation
Simple script for standard math generation, similar to `run_jokes_local.py`.

**Usage:**
```bash
# Edit the script to set your model and task
python run_math_simple.py
```

**Configuration in script:**
```python
model = "Qwen/Qwen3-4B-Base"  # Change this
task = Task.MATH              # Or Task.AIME, Task.AMC
```

### `run_math_local.py` - Full Math Experiments
Comprehensive script for testing multiple methods and models.

**Usage:**
```bash
python run_math_local.py
```

Tests multiple sampling methods:
- `DIRECT` - Standard generation
- `DIRECT_COT` - Chain of thought
- `SEQUENCE` - Sequential sampling
- `STRUCTURE_WITH_PROB` - Structured sampling with probabilities
- `CHAIN_OF_THOUGHT` - Structured CoT

## Integration with Bash Scripts

The bash experiment runner also supports pipeline mode:

```bash
# Run pipeline-based generation
./run_math_experiments.sh --experiment pipeline_simple
```

## Output

All scripts generate results in the `generated_data/` directory with:
- Response files (JSONL format)
- Evaluation results
- HTML reports (`pipeline_report.html`)

## Quick Start

1. **Start vLLM server:**
   ```bash
   vllm serve Qwen/Qwen3-4B-Base --port 8000
   ```

2. **Run simple math generation:**
   ```bash
   cd scripts/tasks
   python run_math_simple.py
   ```

3. **Check results:**
   ```bash
   open generated_data/math_simple/Qwen_Qwen3-4B-Base_math_math/pipeline_report.html
   ```

## Customization

### Change Model
Edit the script:
```python
model = "Qwen/Qwen3-4B-Thinking-2507"
```

### Change Dataset
```python
task = Task.AIME  # AIME competition problems
task = Task.AMC   # AMC competition problems
```

### Adjust Sample Size
```python
num_prompts=50,  # Test on 50 problems
```

### Change Method
```python
method=Method.DIRECT_COT,  # Use chain-of-thought
```