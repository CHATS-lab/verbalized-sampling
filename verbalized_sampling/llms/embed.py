from typing import Dict, Any, Optional
from dataclasses import dataclass
from openai import OpenAI
import os

@dataclass
class EmbeddingResponse:
    """Container for embedding response."""
    embedding: list[float]
    cost: float

class OpenAIEmbeddingModel:
    """OpenAI embedding model implementation."""
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        """Initialize the embedding model.
        
        Args:
            model_name: The name of the OpenAI embedding model to use.
                      Defaults to "text-embedding-3-small".
        """
        self.model_name = model_name
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Cost per 1K tokens for different models
        self.cost_per_1k_tokens = {
            "text-embedding-3-small": 0.00002,  # $0.00002 per 1K tokens
            "text-embedding-3-large": 0.00013,  # $0.00013 per 1K tokens
            "text-embedding-ada-002": 0.0001,   # $0.0001 per 1K tokens
        }
    
    def get_embedding(self, text: str) -> EmbeddingResponse:
        """Get embedding for a text.
        
        Args:
            text: The text to get embedding for.
            
        Returns:
            EmbeddingResponse containing the embedding vector and cost.
        """
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

def get_embedding_model(model_name: str = "text-embedding-3-small") -> OpenAIEmbeddingModel:
    """Get an embedding model instance.
    
    Args:
        model_name: The name of the OpenAI embedding model to use.
                   Defaults to "text-embedding-3-small".
                   
    Returns:
        An instance of OpenAIEmbeddingModel.
    """
    return OpenAIEmbeddingModel(model_name=model_name)
