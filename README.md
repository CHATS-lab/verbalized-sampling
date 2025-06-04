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

### ✅ Completed Features

#### LLMs
- ✅ Parallel workers support
- ✅ Multi-turn conversation handling
- ✅ Structured output with JSON validation
- ✅ OpenRouter and vLLM integration
- ✅ Embedding model support

#### Tasks  
- ✅ Story task refinement
- ✅ New tasks for creativity evaluation (Book, Poem, Speech)

#### Prompts
- ✅ Comprehensive prompt factory
- ✅ 8 different sampling methods
- ✅ Robust response parsing
- ✅ Multi-turn conversation support

#### Generation Methods
- ✅ Direct generation
- ✅ Sequential sampling
- ✅ Structured JSON output
- ✅ Structured with probabilities
- ✅ Multi-turn conversations
- ✅ Chain-of-thought reasoning
- ✅ Self-reflection sampling
- ✅ Temperature-based sampling

#### Evaluation Methods
- ✅ Diversity evaluation with embeddings
- ✅ TTCT creativity assessment
- ✅ Creativity Index (overlap detection)

## Framework Capabilities

### 🎯 Tasks
The framework supports multiple task types with reproducible sampling:

- **Random Number Task** (`rand_num`) - Generate random numbers
- **Creative Story Task** (`creative_story`) - Generate creative stories
- **📚 Book Task** (`book`) - Novel/book continuations (100 prompts from literary works)
- **🎭 Poem Task** (`poem`) - Poetry generation (247 prompts with starting lines)
- **🎤 Speech Task** (`speech`) - Speech generation (235 prompts with opening sentences)

**Key Features:**
- ✅ Reproducible sampling with `random_seed` parameter
- ✅ Configurable `sample_size` for dataset subsampling
- ✅ Multiple prompt access via `prompt_index`
- ✅ Task metadata and statistics
- ✅ Integration with all sampling methods

### 🔄 Sampling Methods
Comprehensive set of verbalized sampling approaches:

#### Core Methods
- **Direct** (`direct`) - Use prompts as-is
- **Sequence** (`sequence`) - Generate multiple responses in Python list format
- **Structure** (`structure`) - JSON format with response field only
- **Structure with Probability** (`structure_with_prob`) - JSON with response and probability fields

#### Advanced Methods  
- **Multi-turn** (`multi_turn`) - Conversational multi-turn sampling
- **Chain of Thought** (`chain_of_thought`) - Include reasoning process with responses
- **Self-Reflection** (`self_reflection`) - Responses with quality reflection and confidence scores
- **Temperature Sampling** (`temperature_sampling`) - Varying creativity levels across responses

**Features:**
- ✅ Robust parsing for each response format
- ✅ Error handling with fallbacks
- ✅ Structured JSON output validation
- ✅ Multi-turn conversation support
- ✅ Backward compatibility

### 📊 Evaluation Methods
Multi-dimensional assessment of generated content:

#### Diversity Assessment
- **Diversity Evaluator** (`diversity`)
  - Semantic similarity using OpenAI embeddings
  - Pairwise similarity analysis
  - Vocabulary richness metrics
  - Supports `text-embedding-3-small/large`

#### Creativity Assessment  
- **TTCT Evaluator** (`ttct`) - Torrance Tests of Creative Thinking
  - **Fluency**: Meaningful and relevant responses (1-5 scale)
  - **Flexibility**: Distinct categories and conceptual shifts (1-5 scale)
  - **Originality**: Statistical rarity and uniqueness (1-5 scale)
  - **Elaboration**: Detail and descriptive richness (1-5 scale)
  - LLM-as-a-judge with comprehensive rubrics

#### Originality Assessment
- **Creativity Index Evaluator** (`creativity_index`)
  - Measures overlap with pretraining data
  - **Exact matching**: Uses Infini-gram API for n-gram detection
  - **Semantic matching**: Earth Mover Distance with embeddings
  - Creativity Index = 1 - coverage (higher = more creative)

**Features:**
- ✅ JSON serializable results
- ✅ Save/load evaluation results
- ✅ Detailed span-level analysis
- ✅ Cost tracking for API usage
- ✅ Parallel processing support

### 🤖 Supported LLMs

#### Generation Models
- **OpenRouter** - Access to Claude, Gemini, and other models
  - `claude-3-opus`, `claude-3.5-sonnet`, `claude-3.5-haiku`
  - `gemini-2.0-flash`, `gemini-2.5-flash`, `gemini-2.5-pro`
- **vLLM** - Local model serving with OpenAI-compatible API
- **Structured Output** - JSON schema validation for applicable methods

#### Embedding Models  
- **OpenAI Embeddings** - For diversity evaluation
  - `text-embedding-3-small` (default)
  - `text-embedding-3-large`
  - `text-embedding-ada-002`

**Features:**
- ✅ Parallel processing with configurable workers
- ✅ Structured output support
- ✅ Cost tracking and rate limiting
- ✅ Error handling and retries

## Quick Start Examples

### Basic Task Usage
```python
from verbalized_sampling.tasks import get_task, Task
from verbalized_sampling.prompts import Method

# Create reproducible task
task = get_task(Task.BOOK, sample_size=10, random_seed=42)

# Generate with different methods
direct_prompt = task.get_prompt(Method.DIRECT, prompt_index=0)
structured_prompt = task.get_prompt(Method.STRUCTURE_WITH_PROB, num_samples=5, prompt_index=0)
```

### Comprehensive Evaluation
```python
from verbalized_sampling.evals import get_evaluator

# Evaluate with multiple metrics
diversity_eval = get_evaluator("diversity")
ttct_eval = get_evaluator("ttct") 
creativity_eval = get_evaluator("creativity_index")

# Run evaluations
diversity_result = diversity_eval.evaluate(prompts, responses)
ttct_result = ttct_eval.evaluate(prompts, responses)
creativity_result = creativity_eval.evaluate(prompts, responses)

# Access results
print(f"Diversity: {diversity_result.overall_metrics['average_similarity']:.3f}")
print(f"TTCT Creativity: {ttct_result.overall_metrics['overall']['creativity_score']:.1f}/5")
print(f"Originality Index: {creativity_result.overall_metrics['average_creativity_index']:.3f}")
```

### Advanced Sampling
```python
# Chain of thought reasoning
cot_prompt = task.get_prompt(Method.CHAIN_OF_THOUGHT, num_samples=3)
cot_response = model.generate(cot_prompt)
parsed_cot = task.parse_response(Method.CHAIN_OF_THOUGHT, cot_response)
# Returns: [{"reasoning": "...", "response": "..."}]

# Self-reflection with confidence
reflection_prompt = task.get_prompt(Method.SELF_REFLECTION, num_samples=3)
reflection_response = model.generate(reflection_prompt)
parsed_reflection = task.parse_response(Method.SELF_REFLECTION, reflection_response)
# Returns: [{"response": "...", "reflection": "...", "confidence": 0.85}]
```

## Data Statistics

| Task | Prompts Available | Domain | Format |
|------|------------------|--------|---------|
| Book | 100 | Literary continuations | Novel prompts from published works |
| Poem | 247 | Poetry | Starting lines from various poems |
| Speech | 235 | Rhetoric | Opening sentences from historical speeches |
| Creative Story | Dynamic | General creativity | Prompt factory generated |
| Random Number | N/A | Simple generation | Numeric task |

## Installation & Setup

### Basic Requirements
```bash
pip install torch numpy nltk transformers sacremoses unidecode tqdm requests
```

### API Keys
```bash
export OPENAI_API_KEY="your_openai_key"
export OPENROUTER_API_KEY="your_openrouter_key"
```

### Optional: Creativity Index Setup
For exact matching (requires Infini-gram API access):
```python
evaluator = get_evaluator("creativity_index", method="exact")
```

For semantic matching (requires embedding table):
```python
from verbalized_sampling.evals.creativity_index import create_embedding_table
create_embedding_table("meta-llama/Meta-Llama-3-8B-Instruct", "embeddings.pkl")
evaluator = get_evaluator("creativity_index", method="semantic", embed_table_path="embeddings.pkl")
```

## Demo Scripts

```bash
# Test all evaluation methods
python verbalized_sampling/examples/evaluation_demo.py

# Test all tasks and sampling methods  
python verbalized_sampling/examples/tasks_demo.py
```

### 🚧 Future Enhancements

- **Additional Tasks**: Scientific writing, code generation, dialogue
- **Advanced Sampling**: Tree-of-thought, iterative refinement
- **Evaluation Extensions**: Factual accuracy, coherence metrics
- **Model Support**: Additional LLM providers, local model optimization
- **Analysis Tools**: Statistical significance testing, correlation analysis
