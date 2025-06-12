from typing import Dict, List, Any, Optional, Union
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
import json
import ast
from .base import BaseEvaluator, EvalResult
from verbalized_sampling.llms import get_embedding_model

class DiversityEvaluator(BaseEvaluator):
    """Evaluator for measuring response diversity using embeddings and cosine similarity."""
    
    def __init__(self, embed_model: str = "text-embedding-3-small", num_workers: int = 128):
        super().__init__("diversity", num_workers)
        self.embed_model = embed_model
        # Check for CUDA first, then MPS, then fall back to CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.embedding_model = get_embedding_model(embed_model)
    
    def compute_embedding(self, text: str) -> tuple[np.ndarray, float]:
        """Compute embedding for a text."""
        response = self.embedding_model.get_embedding(text)
        return np.array(response.embedding), response.cost
    
    def compute_instance_metric(self, prompt: str, response: Dict) -> Dict[str, float]:
        """Compute diversity metrics for a single response."""
        response_text = response.get('text', response)

        word_count_list = []
        unique_words_list = []
        vocabulary_richness_list = []
        
        words = response_text.split()
        word_count = len(words)
        unique_words = len(set(words))
        
        # Calculate vocabulary richness safely
        vocabulary_richness = unique_words / word_count if word_count > 0 else 0.0
        word_count_list.append(word_count)
        unique_words_list.append(unique_words)
        vocabulary_richness_list.append(vocabulary_richness)
        
        return {
            "response_length": word_count,
            "unique_words": unique_words,
            "vocabulary_richness": vocabulary_richness,
            "response": response_text,
            "prompt": prompt
        }
    
    def aggregate_metrics(self, instance_metrics: List[Dict[str, float]]) -> Dict[str, Any]:
        """Compute diversity metrics across all responses."""
        
        if len(instance_metrics) <= 1:
            return {
                "average_diversity": 0.0,
                "min_diversity": 0.0,
                "max_diversity": 0.0,
                "std_diversity": 0.0,
                "average_response_length": 0.0,
                "average_unique_words": 0.0,
                "average_vocabulary_richness": 0.0,
                "pairwise_diversities": []
            }
        
        # Group responses by prompt for intra-class diversity calculation
        prompt_groups = {}
        for i, m in enumerate(instance_metrics):
            prompt = m["prompt"]
            if prompt not in prompt_groups:
                prompt_groups[prompt] = []
            prompt_groups[prompt].append((i, m["response"]))
        
        # Get all responses for embedding computation
        all_responses = [m["response"] for m in instance_metrics]
        
        # Compute embeddings in parallel
        embeddings_list = []
        total_cost = 0.0
        
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
            futures = [executor.submit(self.compute_embedding, response) for response in all_responses]
            
            with tqdm(total=len(all_responses), desc="Computing embeddings") as pbar:
                for future in as_completed(futures):
                    embedding, cost = future.result()
                    embeddings_list.append(embedding)
                    total_cost += cost
                    pbar.update(1)
        
        # Convert to tensor and compute similarities
        print(f"Running with {self.embed_model} on {self.device}")
        embeddings = torch.from_numpy(np.array(embeddings_list)).float().to(self.device)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        similarity_matrix = torch.mm(embeddings, embeddings.t())
        
        # Calculate intra-class (same prompt) pairwise diversities (1 - similarity)
        all_diversities = []
        pairwise_diversities = []
        
        for prompt, indices_responses in prompt_groups.items():
            if len(indices_responses) > 1:  # Need at least 2 responses for diversity
                indices = [idx for idx, _ in indices_responses]
                responses = [resp for _, resp in indices_responses]
                
                # Get all pairs within this prompt group
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        idx_i, idx_j = indices[i], indices[j]
                        similarity = float(similarity_matrix[idx_i, idx_j].cpu().numpy())
                        diversity = 1.0 - similarity  # Convert similarity to diversity
                        all_diversities.append(diversity)
                        
                        # Store detailed pairwise information
                        pairwise_diversities.append(diversity)
        
        # Calculate statistics from intra-class diversities
        if all_diversities:
            diversities_array = np.array(all_diversities)
            metrics = {
                "average_diversity": float(diversities_array.mean()),
                "min_diversity": float(diversities_array.min()),
                "max_diversity": float(diversities_array.max()),
                "std_diversity": float(diversities_array.std()),
                "average_response_length": float(np.mean([m["response_length"] for m in instance_metrics])),
                "average_unique_words": float(np.mean([m["unique_words"] for m in instance_metrics])),
                "average_vocabulary_richness": float(np.mean([m["vocabulary_richness"] for m in instance_metrics])),
                "total_cost": float(total_cost),
                "pairwise_diversities": pairwise_diversities
            }
        else:
            # No valid pairs found (all prompts have only 1 response)
            metrics = {
                "average_diversity": 0.0,
                "min_diversity": 0.0,
                "max_diversity": 0.0,
                "std_diversity": 0.0,
                "average_response_length": float(np.mean([m["response_length"] for m in instance_metrics])),
                "average_unique_words": float(np.mean([m["unique_words"] for m in instance_metrics])),
                "average_vocabulary_richness": float(np.mean([m["vocabulary_richness"] for m in instance_metrics])),
                "total_cost": float(total_cost),
                "pairwise_diversities": []
            }
        
        # Ensure all values are Python native types
        return {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in metrics.items()}
    
    def evaluate(self, 
                prompts: List[str], 
                responses: List[str],
                metadata: Optional[Dict[str, Any]] = None) -> EvalResult:
        """Evaluate diversity of responses."""
        if metadata is None:
            metadata = {}
            
        # Add model information to metadata
        metadata.update({
            "embedding_model": self.embed_model,
            "device": str(self.device)
        })
        
        return super().evaluate(prompts, responses, metadata)

    def save_results(self, result: EvalResult, output_path: str):
        """Save evaluation results to a file."""
        # Convert to dictionary first
        result_dict = {
            "instance_metrics": result.instance_metrics,
            "overall_metrics": {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                for k, v in result.overall_metrics.items()
            },
            "metadata": result.metadata
        }
        
        with open(output_path, 'w') as f:
            json.dump(result_dict, f, indent=2)

    @classmethod
    def load_results(cls, input_path: str) -> EvalResult:
        """Load evaluation results from a file."""        
        with open(input_path, 'r') as f:
            result_dict = json.load(f)
        
        return EvalResult(**result_dict)