from typing import Dict, Type
from .base import BaseLLM
from .vllm import VLLMOpenAI
from .openrouter import OpenRouterLLM

LLM_REGISTRY: Dict[str, Type[BaseLLM]] = {
    "vllm": VLLMOpenAI,
    "openrouter": OpenRouterLLM,
}

def get_model(model_name: str, sim_type: str, config: dict, use_vllm: bool = False) -> BaseLLM:
    """Get a model instance."""
    model_class = LLM_REGISTRY["vllm" if use_vllm else "openrouter"]
    return model_class(model_name=model_name, sim_type=sim_type, config=config) 