import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple

# Global model cache to implement singleton pattern
_MODEL_CACHE = {}

def load_model(model_name: str, use_4bit: bool = False, force_reload: bool = False) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a language model and its tokenizer based on the model name.
    Uses a singleton pattern to avoid reloading models.
    
    Args:
        model_name: String identifier for the model to load
                   (e.g., "meta-llama/Llama-2-7b-hf", "distilgpt2")
        use_4bit: Whether to use 4-bit quantization (for larger models)
                  Set to False for smaller models like DistilGPT2
        force_reload: If True, forces reloading the model even if it's in cache
    
    Returns:
        Tuple containing (model, tokenizer)
        
    Example:
        >>> # Load a large model with quantization
        >>> model, tokenizer = load_model("meta-llama/Llama-2-7b-hf", use_4bit=True)
        >>> 
        >>> # Load a small model without quantization
        >>> small_model, small_tokenizer = load_model("distilgpt2", use_4bit=False)
    """
    # Create a cache key based on model name and quantization setting
    cache_key = f"{model_name}_{use_4bit}"
    
    # Check if model is already loaded
    if not force_reload and cache_key in _MODEL_CACHE:
        print(f"Using cached model: {model_name}")
        return _MODEL_CACHE[cache_key]
    
    try:
        print(f"Loading model: {model_name}")
        
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        
        # Load the model with appropriate optimizations based on model size
        if use_4bit:
            # Configuration for larger models
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,  # Use bfloat16 for efficiency
                device_map="auto",           # Automatically use available devices
                load_in_4bit=True,           # Quantize to 4-bit for memory efficiency
            )
        else:
            # Configuration for smaller models (no quantization needed)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,   # Use float16 for smaller models
                device_map="auto",           # Automatically use available devices
            )
        
        # Store in cache
        _MODEL_CACHE[cache_key] = (model, tokenizer)
        
        print(f"Successfully loaded {model_name}")
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        raise Exception(f"Failed to load model: {e}")


def unload_model(model_name: str = None, use_4bit: bool = None):
    """
    Unload a model from the cache and free up memory.
    
    Args:
        model_name: String identifier for the model to unload (if None, unloads all models)
        use_4bit: Whether the model was loaded with 4-bit quantization
    """
    global _MODEL_CACHE
    
    if model_name is None:
        # Unload all models
        for key, (model, tokenizer) in list(_MODEL_CACHE.items()):
            del model
            del tokenizer
        _MODEL_CACHE = {}
        print("All models unloaded from cache")
    else:
        # Unload specific model
        cache_key = f"{model_name}_{use_4bit}"
        
        if cache_key in _MODEL_CACHE:
            model, tokenizer = _MODEL_CACHE[cache_key]
            del model
            del tokenizer
            del _MODEL_CACHE[cache_key]
            print(f"Model {model_name} unloaded from cache")
        else:
            print(f"Model {model_name} not found in cache")
    
    # Try to clear CUDA cache
    try:
        torch.cuda.empty_cache()
        print("CUDA cache cleared")
    except:
        pass


