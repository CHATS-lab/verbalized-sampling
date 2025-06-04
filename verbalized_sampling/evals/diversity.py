from typing import Dict, List, Any, Optional
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os

from .base import BaseEvaluator, EvalResult
from verbalized_sampling.llms import get_embedding_model

class DiversityEvaluator(BaseEvaluator):
    """Evaluator for measuring response diversity using embeddings and cosine similarity."""
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        super().__init__("diversity")
        self.model_name = model_name
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.embedding_model = get_embedding_model(model_name)
    
    def compute_embedding(self, text: str) -> tuple[np.ndarray, float]:
        """Compute embedding for a text."""
        response = self.embedding_model.get_embedding(text)
        return np.array(response.embedding), response.cost
    
    def compute_instance_metric(self, prompt: str, response: str) -> Dict[str, float]:
        """Compute diversity metrics for a single response."""
        return {
            "response_length": len(response.split()),
            "unique_words": len(set(response.split())),
            "vocabulary_richness": len(set(response.split())) / len(response.split())
        }
    
    def aggregate_metrics(self, instance_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Compute diversity metrics across all responses."""
        if len(instance_metrics) <= 1:
            return {
                "average_similarity": 0.0,
                "min_similarity": 0.0,
                "max_similarity": 0.0,
                "std_similarity": 0.0,
                "average_response_length": np.mean([m["response_length"] for m in instance_metrics]),
                "average_unique_words": np.mean([m["unique_words"] for m in instance_metrics]),
                "average_vocabulary_richness": np.mean([m["vocabulary_richness"] for m in instance_metrics])
            }
        
        # Get all responses
        responses = [m["response"] for m in instance_metrics]
        
        # Compute embeddings in parallel
        embeddings_list = []
        total_cost = 0.0
        
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
            futures = [executor.submit(self.compute_embedding, response) for response in responses]
            
            with tqdm(total=len(responses), desc="Computing embeddings") as pbar:
                for future in as_completed(futures):
                    embedding, cost = future.result()
                    embeddings_list.append(embedding)
                    total_cost += cost
                    pbar.update(1)
        
        # Convert to tensor and compute similarities
        embeddings = torch.from_numpy(np.array(embeddings_list)).float().to(self.device)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        similarity_matrix = torch.mm(embeddings, embeddings.t())
        
        # Get upper triangle indices (excluding diagonal)
        indices = torch.triu_indices(len(responses), len(responses), offset=1)
        similarities = similarity_matrix[indices[0], indices[1]].cpu().numpy()
        
        return {
            "average_similarity": float(similarities.mean()),
            "min_similarity": float(similarities.min()),
            "max_similarity": float(similarities.max()),
            "std_similarity": float(similarities.std()),
            "average_response_length": np.mean([m["response_length"] for m in instance_metrics]),
            "average_unique_words": np.mean([m["unique_words"] for m in instance_metrics]),
            "average_vocabulary_richness": np.mean([m["vocabulary_richness"] for m in instance_metrics]),
            "total_cost": total_cost
        }
    
    def evaluate(self, 
                prompts: List[str], 
                responses: List[str],
                metadata: Optional[Dict[str, Any]] = None) -> EvalResult:
        """Evaluate diversity of responses."""
        if metadata is None:
            metadata = {}
            
        # Add model information to metadata
        metadata.update({
            "embedding_model": self.model_name,
            "device": str(self.device)
        })
        
        return super().evaluate(prompts, responses, metadata)
