# Math Task Updates

## Changes Made

### 1. Fixed Hardcoded Path (`math_task.py`)

**Before:**
```python
self.data_path = f"/Users/simonyu/local/local_orby/verbalize-sampling/data/math/{dataset}"
```

**After:**
```python
import pathlib
current_file = pathlib.Path(__file__)
project_root = current_file.parent.parent.parent.parent  # Go up to verbalize-sampling root
self.data_path = project_root / "data" / "math" / dataset
```

### 2. Simplified Prompt Format (`math_task.py`)

**Before (with chat template):**
```python
prompt_text = (
    f"<|im_start|>user\nQuestion: {question}"
    "\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
    "<|im_start|>assistant\n"
)
```

**After (template applied during inference):**
```python
prompt_text = f"Question: {question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
```

### 3. Updated Test Script (`test_math.py`)

- Removed `_get_instruction_for_dataset` method (no longer needed)
- Updated prompt format to match the new simplified approach
- Cleaned up unused imports

### 4. Integrated with PromptFactory System (`prompt.py`, `factory.py`)

**Added:**
- New `TaskType.MATH` enum value
- `MathPromptTemplate` class with all required prompt methods
- Math task mappings in PromptFactory (`math_math`, `math_aime`, etc. → `TaskType.MATH`)
- Registry entry for `MathPromptTemplate`

## Key Benefits

1. **Portable**: No hardcoded paths - works from any location
2. **Clean**: Let the model/tokenizer handle chat templates during inference
3. **Consistent**: Unified prompt format across all math datasets
4. **Simple**: Cleaner code with fewer dependencies
5. **Integrated**: Full integration with existing PromptFactory system for different sampling methods

## Usage

The scripts work exactly the same as before:

```bash
# Start vLLM server
vllm serve Qwen/Qwen3-4B-Base --port 8000

# Run experiments
./run_math_experiments.sh --quick
```

All the math task functionality remains the same - only the internal implementation was improved.