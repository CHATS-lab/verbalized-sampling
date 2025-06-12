# Verbalized Sampling Tasks

This directory contains task implementations for the verbalized sampling framework. Tasks define the prompts and parsing logic for different types of text generation experiments.

## Available Tasks

### 1. Random Number Task (`rand_num`)
Generates random numbers within a specified range.

```python
from verbalized_sampling.tasks import get_task, Task

task = get_task(Task.RANDOM_NUM)
prompt = task.get_prompt(Method.DIRECT)
```

### 2. Creative Story Task (`creative_story`)
Generates creative stories using the existing prompt factory system.

```python
task = get_task(Task.CREATIVE_STORY)
prompt = task.get_prompt(Method.STRUCTURE_WITH_PROB, num_samples=5)
```

### 3. Book Task (`book`) 🆕
Generates novel/book continuations from literary prompts.

**Data Source**: `data/book.txt` (100 prompts)
**Format**: Novel continuation prompts from published works

```python
# Basic usage
task = get_task(Task.BOOK, num_prompts=10, random_seed=42)
prompt = task.get_prompt(Method.DIRECT, prompt_index=0)

# With structured output
prompt = task.get_prompt(Method.STRUCTURE_WITH_PROB, num_samples=3, prompt_index=0)
```

### 4. Poem Task (`poem`) 🆕
Generates poems from starting line prompts.

**Data Source**: `data/poem.txt` (247 prompts)
**Format**: Poetry prompts with starting lines from various poems

```python
# Sample 5 prompts with specific seed for reproducibility
task = get_task(Task.POEM, num_prompts=5, random_seed=123)

# Get different starting lines
for i in range(len(task.get_prompts())):
    prompt = task.get_prompt(Method.DIRECT, prompt_index=i)
```

### 5. Speech Task (`speech`) 🆕
Generates speeches from starting sentence prompts.

**Data Source**: `data/speech.txt` (235 prompts)
**Format**: Speech prompts with opening sentences from historical speeches

```python
# Large sample for extensive experiments
task = get_task(Task.SPEECH, num_prompts=50, random_seed=456)
prompt = task.get_prompt(Method.SEQUENCE, num_samples=3, prompt_index=10)
```

## Key Features

### Reproducible Sampling
All new tasks support reproducible random sampling through the `random_seed` parameter:

```python
# Same results every time
task1 = get_task(Task.BOOK, num_prompts=5, random_seed=42)
task2 = get_task(Task.BOOK, num_prompts=5, random_seed=42)
assert task1.get_prompts() == task2.get_prompts()  # True

# Different results
task3 = get_task(Task.BOOK, num_prompts=5, random_seed=999)
assert task1.get_prompts() != task3.get_prompts()  # True
```

### Flexible Sample Sizes
Control how many prompts to sample from each dataset:

```python
# Sample 10 prompts randomly
task = get_task(Task.POEM, num_prompts=10, random_seed=42)

# Load all available prompts
task = get_task(Task.POEM, num_prompts=0)  # or omit num_prompts

# If num_prompts > available prompts, loads all prompts
task = get_task(Task.SPEECH, num_prompts=10000)  # loads all 235 prompts
```

### Multiple Prompt Access
Access individual prompts by index:

```python
task = get_task(Task.BOOK, num_prompts=5, random_seed=42)

# Iterate through all sampled prompts
for i in range(len(task.get_prompts())):
    prompt = task.get_prompt(Method.DIRECT, prompt_index=i)
    # Process prompt...
```

### Task Metadata
Get information about the task and loaded data:

```python
task = get_task(Task.POEM, num_prompts=10, random_seed=42)
metadata = task.get_metadata()

print(metadata)
# {
#     "task_type": "poem",
#     "total_prompts": 10,
#     "num_prompts": 10,
#     "random_seed": 42,
#     "description": "Poetry generation task with starting line prompts"
# }
```

## Supported Methods

All tasks support the standard verbalized sampling methods:

- **DIRECT**: Use prompts as-is
- **SEQUENCE**: Generate multiple responses in sequence
- **STRUCTURE**: Structured JSON output without probabilities
- **STRUCTURE_WITH_PROB**: Structured JSON output with probabilities

Example with different methods:

```python
task = get_task(Task.SPEECH, num_prompts=3, random_seed=42)

# Direct usage
direct_prompt = task.get_prompt(Method.DIRECT, prompt_index=0)

# Structured with probabilities
structured_prompt = task.get_prompt(
    Method.STRUCTURE_WITH_PROB, 
    num_samples=5, 
    prompt_index=0
)
```

## Data Format

The data files follow these formats:

### book.txt
```
Please write a few paragraphs for a novel starting with the following prompt: [PROMPT_TEXT]
```

### poem.txt
```
Please write a poem starting with the following line: [STARTING_LINE]
```

### speech.txt
```
Please write a speech starting with the following sentence: [OPENING_SENTENCE]
```

## Usage Examples

### Basic Experiment Setup
```python
from verbalized_sampling.tasks import get_task, Task
from verbalized_sampling.prompts import Method

# Set up reproducible experiment
task = get_task(Task.BOOK, num_prompts=20, random_seed=42)

prompts = []
for i in range(len(task.get_prompts())):
    prompt = task.get_prompt(Method.STRUCTURE_WITH_PROB, num_samples=5, prompt_index=i)
    prompts.append(prompt)

# Use prompts with your model...
```

### Comparative Analysis
```python
# Compare different domains
book_task = get_task(Task.BOOK, num_prompts=10, random_seed=42)
poem_task = get_task(Task.POEM, num_prompts=10, random_seed=42)
speech_task = get_task(Task.SPEECH, num_prompts=10, random_seed=42)

# Generate responses for each domain and compare
```

### Batch Processing
```python
def process_task(task_name, num_prompts=50, random_seed=42):
    task = get_task(task_name, num_prompts=num_prompts, random_seed=random_seed)
    results = []
    
    for i in range(len(task.get_prompts())):
        prompt = task.get_prompt(Method.STRUCTURE_WITH_PROB, num_samples=3, prompt_index=i)
        # Process with your model
        response = your_model.generate(prompt)
        parsed_response = task.parse_response(Method.STRUCTURE_WITH_PROB, response)
        results.append(parsed_response)
    
    return results

# Process all domains
book_results = process_task(Task.BOOK)
poem_results = process_task(Task.POEM)
speech_results = process_task(Task.SPEECH)
```

## Error Handling

The tasks include proper error handling for common issues:

```python
task = get_task(Task.BOOK, num_prompts=5, random_seed=42)

# Handle index out of range
try:
    prompt = task.get_prompt(Method.DIRECT, prompt_index=100)
except ValueError as e:
    print(f"Index error: {e}")

# Handle missing data files
try:
    task = get_task(Task.BOOK)
except FileNotFoundError as e:
    print(f"Data file error: {e}")
```

## Demo Script

Run the demo to see all tasks in action:

```bash
python verbalized_sampling/examples/tasks_demo.py
```

The demo shows:
- Data statistics for all tasks
- Different method usage examples
- Reproducibility demonstration
- Error handling examples
- Best practices

## Integration with Evaluation Framework

The tasks work seamlessly with the evaluation framework:

```python
from verbalized_sampling.tasks import get_task, Task
from verbalized_sampling.evals import get_evaluator

# Generate responses
task = get_task(Task.POEM, num_prompts=20, random_seed=42)
responses = []
prompts = []

for i in range(len(task.get_prompts())):
    prompt = task.get_prompt(Method.DIRECT, prompt_index=i)
    response = your_model.generate(prompt)
    prompts.append(prompt)
    responses.append(response)

# Evaluate with multiple metrics
diversity_eval = get_evaluator("diversity")
ttct_eval = get_evaluator("ttct")
creativity_eval = get_evaluator("creativity_index")

diversity_result = diversity_eval.evaluate(prompts, responses)
ttct_result = ttct_eval.evaluate(prompts, responses)
creativity_result = creativity_eval.evaluate(prompts, responses)
```

## Best Practices

1. **Always set random_seed** for reproducible experiments
2. **Use appropriate num_prompts** based on your computational resources
3. **Iterate through all prompts** when possible for comprehensive evaluation
4. **Handle parsing errors** gracefully, especially for structured methods
5. **Save task metadata** along with results for experiment tracking
6. **Use consistent method/num_samples** across comparisons

## Adding New Tasks

To add a new task:

1. Create a new file in `verbalized_sampling/tasks/`
2. Extend `BaseTask` class
3. Implement required methods: `get_prompt()`, `parse_response()`
4. Add optional methods: `get_metadata()`, `get_prompts()`
5. Register in `__init__.py`
6. Add data file in `data/` directory if needed

Example skeleton:

```python
from .base import BaseTask
from typing import Any, List
from verbalized_sampling.prompts import Method

class MyNewTask(BaseTask):
    def __init__(self, num_prompts: int = 1, random_seed: int = 42):
        self.num_prompts = num_prompts
        self.random_seed = random_seed
        # Load your data...
    
    def get_prompt(self, method: Method, num_samples: int = 1, **kwargs) -> str:
        # Return formatted prompt
        pass
    
    def parse_response(self, method: Method, response: str) -> Any:
        # Parse model response
        pass
``` 