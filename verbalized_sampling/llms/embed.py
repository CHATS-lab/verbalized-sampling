from typing import Dict, Any, Optional
from dataclasses import dataclass
from openai import OpenAI
import os
import signal
from functools import wraps
import time

@dataclass
class EmbeddingResponse:
    """Container for embedding response."""
    embedding: list[float]
    cost: float

class TimeoutError(Exception):
    """Custom timeout exception."""
    pass

def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Embedding request timed out")

def with_timeout(timeout_seconds: int):
    """Decorator to add timeout to function calls."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Set up the timeout signal
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            
            try:
                result = func(*args, **kwargs)
                signal.alarm(0)  # Cancel the alarm
                return result
            except TimeoutError:
                signal.alarm(0)  # Cancel the alarm
                raise TimeoutError(f"Function call timed out after {timeout_seconds} seconds")
            except Exception as e:
                signal.alarm(0)  # Cancel the alarm
                raise e
            finally:
                # Restore the old signal handler
                signal.signal(signal.SIGALRM, old_handler)
        return wrapper
    return decorator

class OpenAIEmbeddingModel:
    """OpenAI embedding model implementation."""
    
    def __init__(self, model_name: str = "text-embedding-3-small", timeout: int = 128):
        """Initialize the embedding model.
        
        Args:
            model_name: The name of the OpenAI embedding model to use.
                      Defaults to "text-embedding-3-small".
            timeout: Timeout in seconds for embedding requests. Defaults to 128.
        """
        self.model_name = model_name
        self.timeout = timeout
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            timeout=timeout  # Set OpenAI client timeout as well
        )
        
        # Cost per 1K tokens for different models
        self.cost_per_1k_tokens = {
            "text-embedding-3-small": 0.00002,  # $0.00002 per 1K tokens
            "text-embedding-3-large": 0.00013,  # $0.00013 per 1K tokens
            "text-embedding-ada-002": 0.0001,   # $0.0001 per 1K tokens
        }
    
    @with_timeout(128)  # 128 seconds timeout
    def _get_embedding_with_timeout(self, text: str) -> EmbeddingResponse:
        """Internal method to get embedding with timeout protection."""
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text
        )
        
        # Calculate cost based on token count
        token_count = response.usage.total_tokens
        cost = (token_count / 1000) * self.cost_per_1k_tokens.get(self.model_name, 0.00002)
        
        return EmbeddingResponse(
            embedding=response.data[0].embedding,
            cost=cost
        )
    
    def _get_embedding_direct(self, text: str) -> EmbeddingResponse:
        """Internal method to get embedding without timeout protection."""
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text
        )
        
        # Calculate cost based on token count
        token_count = response.usage.total_tokens
        cost = (token_count / 1000) * self.cost_per_1k_tokens.get(self.model_name, 0.00002)
        
        return EmbeddingResponse(
            embedding=response.data[0].embedding,
            cost=cost
        )
    
    def get_embedding(self, text: str, max_retries: int = 3) -> EmbeddingResponse:
        """Get embedding for a text with retry logic."""
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return self._get_embedding_direct(text)
                
            except Exception as e:
                last_exception = e
                if attempt < max_retries and ("timeout" in str(e).lower() or "timed out" in str(e).lower()):
                    print(f"Embedding request failed (attempt {attempt + 1}/{max_retries + 1}): {e}. Retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    print(f"Embedding request failed: {e}")
                    raise e
        
        raise last_exception

def get_embedding_model(model_name: str = "text-embedding-3-small", timeout: int = 128) -> OpenAIEmbeddingModel:
    """Get an embedding model instance.
    
    Args:
        model_name: The name of the OpenAI embedding model to use.
                   Defaults to "text-embedding-3-small".
        timeout: Timeout in seconds for embedding requests. Defaults to 128.
                   
    Returns:
        An instance of OpenAIEmbeddingModel.
    """
    return OpenAIEmbeddingModel(model_name=model_name, timeout=timeout)
