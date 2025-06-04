from typing import Dict, Type
from .base import BaseLLM
from .vllm import VLLMOpenAI
from .openrouter import OpenRouterLLM
from verbalized_sampling.prompts import Method, is_method_structured
from .embed import get_embedding_model

__all__ = ["get_model", "get_embedding_model"]

LLM_REGISTRY: Dict[str, Type[BaseLLM]] = {
    "vllm": VLLMOpenAI,
    "openrouter": OpenRouterLLM,
}

def get_model(model_name: str, 
              method: Method,
              config: dict, 
              use_vllm: bool = False,
              num_workers: int = 128) -> BaseLLM:
    """Get a model instance."""
    model_class: Type[BaseLLM] = LLM_REGISTRY["vllm" if use_vllm else "openrouter"]
    is_structured = is_method_structured(method)
    return model_class(model_name=model_name, 
                       config=config, 
                       num_workers=num_workers,
                       is_structured=is_structured
                       ) 