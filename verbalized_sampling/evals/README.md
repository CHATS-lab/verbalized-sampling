# Verbalized Sampling Evaluation Framework

This directory contains evaluators for assessing the quality and creativity of generated text using various methodologies.

## Available Evaluators

### 1. DiversityEvaluator (`diversity`)

Measures response diversity using semantic embeddings and cosine similarity.

**Key Metrics:**
- Average, min, max, and std deviation of pairwise similarities
- Pairwise similarity details for all response pairs
- Vocabulary richness and response length statistics

**Usage:**
```python
from verbalized_sampling.evals import get_evaluator

evaluator = get_evaluator("diversity", model_name="text-embedding-3-small")
result = evaluator.evaluate(prompts, responses)
print(f"Average similarity: {result.overall_metrics['average_similarity']}")
```

**Requirements:**
- OpenAI API key for embedding models
- `OPENAI_API_KEY` environment variable

### 2. TTCTEvaluator (`ttct` or `quality`)

Measures creativity using the Torrance Tests of Creative Thinking (TTCT) framework with LLM-as-a-judge.

**Key Metrics (1-5 scale):**
- **Fluency**: Meaningful and relevant responses
- **Flexibility**: Distinct categories and conceptual shifts
- **Originality**: Statistical rarity and uniqueness
- **Elaboration**: Detail and descriptive richness

**Usage:**
```python
evaluator = get_evaluator("ttct", judge_model="gpt-4-turbo")
result = evaluator.evaluate(prompts, responses)
print(f"Creativity score: {result.overall_metrics['overall']['creativity_score']}")
```

**Requirements:**
- Access to a powerful language model for judging (GPT-4, Claude, etc.)
- Appropriate API keys configured

### 3. CreativityIndexEvaluator (`creativity_index`)

Measures creativity by analyzing overlap with pretraining data using exact or semantic matching.

**Key Metrics:**
- **Creativity Index**: 1 - coverage (higher = more creative)
- **Coverage**: Percentage of text matching reference corpus
- **Match Rate**: Proportion of responses with detected matches
- **Matched Spans**: Detailed information about overlapping text

**Usage:**
```python
# Exact matching (requires Infini-gram API)
evaluator = get_evaluator("creativity_index", method="exact", corpus="v4_rpj_llama_s4")

# Semantic matching (requires embedding table)
evaluator = get_evaluator("creativity_index", method="semantic", 
                         embed_table_path="path/to/embeddings.pkl")

result = evaluator.evaluate(prompts, responses)
print(f"Creativity index: {result.overall_metrics['average_creativity_index']}")
```

**Requirements:**
- For exact matching: Infini-gram API access
- For semantic matching: Pre-computed embedding similarity table

## Installation and Setup

### Basic Requirements
```bash
pip install torch numpy nltk transformers sacremoses unidecode tqdm
```

### API Keys
Set up the following environment variables:
```bash
export OPENAI_API_KEY="your_openai_key"
export OPENROUTER_API_KEY="your_openrouter_key"  # For alternative models
```

### For Creativity Index (Optional)
1. **Exact Matching**: Requires Infini-gram API access
2. **Semantic Matching**: Create embedding table:
```python
from verbalized_sampling.evals.creativity_index import create_embedding_table
create_embedding_table("meta-llama/Meta-Llama-3-8B-Instruct", "embeddings.pkl")
```

## Quick Start

```python
from verbalized_sampling.evals import get_evaluator

# Your generated responses
prompts = ["Tell me a creative story"] * 5
responses = [
    "Once upon a time in a magical forest...",
    "The spaceship landed on an unknown planet...",
    "A detective found a mysterious clue...",
    "The artist painted with colors unseen...",
    "Time traveled backwards that Tuesday..."
]

# 1. Evaluate diversity
diversity_eval = get_evaluator("diversity")
diversity_result = diversity_eval.evaluate(prompts, responses)

# 2. Evaluate creativity (TTCT)
ttct_eval = get_evaluator("ttct", judge_model="gpt-4-turbo")
ttct_result = ttct_eval.evaluate(prompts, responses)

# 3. Evaluate originality (Creativity Index)
creativity_eval = get_evaluator("creativity_index", method="exact")
creativity_result = creativity_eval.evaluate(prompts, responses)

# Print results
print("DIVERSITY METRICS:")
print(f"  Average similarity: {diversity_result.overall_metrics['average_similarity']:.3f}")

print("CREATIVITY METRICS (TTCT):")
print(f"  Overall creativity: {ttct_result.overall_metrics['overall']['creativity_score']:.1f}/5")

print("ORIGINALITY METRICS:")
print(f"  Creativity index: {creativity_result.overall_metrics['average_creativity_index']:.3f}")
```

## Saving and Loading Results

All evaluators support saving and loading results:

```python
# Save results
evaluator.save_results(result, "evaluation_results.json")

# Load results
loaded_result = evaluator.load_results("evaluation_results.json")
```

## Demo Script

Run the demo to see all evaluators in action:
```bash
python verbalized_sampling/examples/evaluation_demo.py
```

## Methodology References

1. **Diversity Evaluation**: Cosine similarity of semantic embeddings
2. **TTCT Framework**: Torrance, E. P. (1966). Torrance Tests of Creative Thinking
3. **Creativity Index**: [AI as Humanity's Salieri: Quantifying Linguistic Creativity of Language Models via Systematic Attribution of Machine Text against Web Text](https://arxiv.org/abs/2410.04265)

## Customization

### Custom Evaluators
Extend the `BaseEvaluator` class to create custom evaluators:

```python
from verbalized_sampling.evals.base import BaseEvaluator, EvalResult

class MyCustomEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__("my_custom")
    
    def compute_instance_metric(self, prompt: str, response: str) -> Dict[str, float]:
        # Your per-response metrics
        return {"my_metric": score}
    
    def aggregate_metrics(self, instance_metrics: List[Dict[str, float]]) -> Dict[str, Any]:
        # Your aggregated metrics
        return {"average_metric": np.mean([m["my_metric"] for m in instance_metrics])}
```

### Configuration Options

Each evaluator accepts various configuration parameters:

```python
# Diversity evaluator options
get_evaluator("diversity", 
              model_name="text-embedding-3-large",  # Embedding model
              device="cuda")                         # Device selection

# TTCT evaluator options  
get_evaluator("ttct",
              judge_model="claude-3.5-sonnet",      # Judge model
              temperature=0.7)                      # Sampling temperature

# Creativity Index options
get_evaluator("creativity_index",
              method="exact",                       # "exact" or "semantic"
              min_ngram=5,                         # Minimum n-gram size
              threshold=0.95,                      # Similarity threshold
              corpus="v4_rpj_llama_s4")           # Reference corpus
```

## Troubleshooting

### Common Issues

1. **Missing API Keys**: Ensure environment variables are set
2. **Model Access**: Some models require authentication or special access
3. **Memory Issues**: Large embedding tables require significant RAM
4. **Rate Limits**: API calls may be rate-limited

### Performance Tips

1. **Batch Processing**: Process multiple responses together when possible
2. **Caching**: Save intermediate results (embeddings, evaluations)
3. **Parallel Processing**: Use `num_workers` parameter for concurrent evaluation
4. **Model Selection**: Choose appropriate model sizes for your use case

## Contributing

To add new evaluators:
1. Create a new file in `verbalized_sampling/evals/`
2. Extend `BaseEvaluator`
3. Add to `__init__.py` registry
4. Add tests and documentation 