import os
from typing import Dict, Type
from .base import BaseLLM
from .vllm import VLLMOpenAI
from .openrouter import OpenRouterLLM
from verbalized_sampling.methods import Method, is_method_structured
from .embed import get_embedding_model
from .litellm import LiteLLM

__all__ = ["get_model", "get_embedding_model"]

LLM_REGISTRY: Dict[str, Type[BaseLLM]] = {
    "vllm": VLLMOpenAI,
    "openrouter": OpenRouterLLM,
    "litellm": LiteLLM,
}

def get_model(model_name: str, 
              method: Method,
              config: dict, 
              use_vllm: bool = False,
              num_workers: int = 128,
              strict_json: bool = False) -> BaseLLM:
    """Get a model instance."""
    if "claude" in model_name:
        if os.environ.get("ANTHROPIC_API_KEY") is None:
            print("ANTHROPIC_API_KEY is not set, falling back to openrouter")
            model_class = LLM_REGISTRY["openrouter"]
        else:
            model_class = LLM_REGISTRY["litellm"]
    else:
        model_class = LLM_REGISTRY["vllm" if use_vllm else "openrouter"]
    return model_class(model_name=model_name, 
                       config=config, 
                       num_workers=num_workers,
                       strict_json=strict_json
                       ) 