# Math Model Testing Guide

This directory contains scripts to test Qwen models on mathematical reasoning tasks.

## Files

- `test_math.py` - Main testing script for evaluating models on math datasets
- `run_single_model_test.py` - Interactive helper for testing single models
- `README_math_testing.md` - This guide

## Quick Start

### 1. Start vLLM Server

First, start a vLLM server with one of the Qwen models:

```bash
# For the base model
vllm serve Qwen/Qwen3-4B-Base --port 8000

# OR for the thinking model
vllm serve Qwen/Qwen3-4B-Thinking-2507 --port 8000
```

### 2. Run Tests

#### Option A: Interactive Runner (Recommended)
```bash
cd synthetic_data
python run_single_model_test.py
```

This will guide you through testing with an interactive menu.

#### Option B: Direct Script
```bash
# Test single model on MATH dataset (20 samples)
python test_math.py --models Qwen/Qwen3-4B-Base --datasets math --num_samples 20

# Test on multiple datasets
python test_math.py --models Qwen/Qwen3-4B-Base --datasets math amc aime --num_samples 10

# Use custom vLLM port
python test_math.py --models Qwen/Qwen3-4B-Base --base_url http://localhost:8001
```

## Available Datasets

| Dataset | Size | Description |
|---------|------|-------------|
| `math` | 500 | MATH dataset - LaTeX problems, difficulty 1-5 |
| `aime` | 30 | American Invitational Mathematics Examination |
| `amc` | 83 | American Mathematics Competitions |
| `minerva` | 272 | Physics and advanced math problems |
| `olympiad_bench` | 675 | Mathematical olympiad problems |

## Testing Strategy

### For Quick Testing (5-10 minutes)
```bash
python test_math.py --models Qwen/Qwen3-4B-Base --datasets math --num_samples 10
```

### For Comprehensive Evaluation (30-60 minutes)
```bash
python test_math.py --models Qwen/Qwen3-4B-Base --datasets math aime amc --num_samples 50
```

### For Full Evaluation (2-3 hours)
```bash
python test_math.py --models Qwen/Qwen3-4B-Base --datasets math aime amc minerva olympiad_bench --num_samples 100
```

## Comparing Models

Since vLLM serves one model at a time, test each model separately:

### Test Model 1
```bash
# Start server with first model
vllm serve Qwen/Qwen3-4B-Base --port 8000

# Run test
python test_math.py --models Qwen/Qwen3-4B-Base --datasets math amc --num_samples 20 --output results_base
```

### Test Model 2
```bash
# Stop previous server (Ctrl+C), then start with second model
vllm serve Qwen/Qwen3-4B-Thinking-2507 --port 8000

# Run test
python test_math.py --models Qwen/Qwen3-4B-Thinking-2507 --datasets math amc --num_samples 20 --output results_thinking
```

## Output

The script generates:
- Console output with real-time progress and results
- JSON file with detailed results (`{output_prefix}_{timestamp}.json`)

### Example Output
```
🚀 Starting Math Model Evaluation
Models: ['Qwen/Qwen3-4B-Base']
Datasets: ['math', 'amc']
Samples per dataset: 20

🤖 Testing model: Qwen/Qwen3-4B-Base

🧮 Testing Qwen/Qwen3-4B-Base on MATH dataset...
✅ math: 0.750 (15/20) avg_time: 2.34s
✅ amc: 0.600 (12/20) avg_time: 1.87s

📊 FINAL RESULTS SUMMARY
Model                              Dataset         Accuracy   Problems   Avg Time
--------------------------------------------------------------------------------
Qwen/Qwen3-4B-Base                 math            0.750      20         2.34s
Qwen/Qwen3-4B-Base                 amc             0.600      20         1.87s

📈 MODEL COMPARISON (Average Accuracy)
Qwen/Qwen3-4B-Base                 0.675 (40 total problems)

💾 Detailed results saved to: results_base_20240101_123456.json
```

## Troubleshooting

### vLLM Server Issues
```bash
# Check if server is running
curl http://localhost:8000/v1/models

# Common vLLM start command
vllm serve Qwen/Qwen3-4B-Base --host 0.0.0.0 --port 8000 --max-model-len 4096
```

### Memory Issues
- Reduce `--num_samples` for testing
- Use smaller datasets (`amc` has only 83 problems vs `olympiad_bench` with 675)
- Add vLLM memory optimization flags: `--gpu-memory-utilization 0.8`

### Import Errors
```bash
# Make sure you're in the right conda environment
conda activate verbalize

# Install missing dependencies
pip install math_verify
```

### Model Loading Issues
- Ensure the model name exactly matches what vLLM expects
- For Hugging Face models, use format: `organization/model-name`
- Check vLLM logs for model loading errors

## Script Parameters

### test_math.py
- `--models`: Model names (one at a time for vLLM)
- `--datasets`: Which math datasets to test
- `--num_samples`: Number of problems per dataset
- `--seed`: Random seed for reproducible sampling
- `--base_url`: vLLM server URL
- `--output`: Output file prefix

### Expected Performance

Based on typical results for 4B models:
- **MATH dataset**: 20-40% accuracy (difficulty varies 1-5)
- **AMC**: 30-50% accuracy (competition problems)
- **AIME**: 10-30% accuracy (harder competition problems)

The Thinking model should generally outperform the Base model on reasoning tasks.