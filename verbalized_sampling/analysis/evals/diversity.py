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
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity


class DiversityEvaluator(BaseEvaluator):
    """Evaluator for measuring response diversity using embeddings and cosine similarity."""

    
    instance_plot_metrics = [
        ("pairwise_diversities", "violin"),
        ("vocabulary_richness", "histogram"),
        ("response_length", "histogram"),
        ("unique_words", "histogram")
    ]
    aggregate_plot_metrics = [
        "avg_diversity",
        "min_diversity",
        "max_diversity",
        "avg_response_length",
        "avg_unique_words",
        "avg_vocabulary_richness",
    ]
    key_plot_metrics = [
        ("avg_diversity", "Diversity (Pairwise)"),
    ]
    
    def __init__(self, embed_model: str = "text-embedding-3-small", num_workers: int = 128, num_responses_per_prompt: int = 1):
        super().__init__("diversity", num_workers)
        self.embed_model = embed_model
        self.num_responses_per_prompt = num_responses_per_prompt

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
        try:
            response = self.embedding_model.get_embedding(text)
            return np.array(response.embedding), response.cost
        except Exception as e:
            print(f"Error computing embedding for text: {text}")
            print(f"Error: {e}")
            return np.array([]), 0.0
    
    def compute_instance_metric(self, prompt: str, response: Dict) -> Dict[str, float]:
        """Compute diversity metrics for a single response."""
        response_text = response.get('text', response)
        if isinstance(response_text, dict):
            response_text = str(response_text)

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
                "avg_diversity": 0.0, "std_diversity": 0.0,
                "min_diversity": 0.0, "max_diversity": 0.0,
                "avg_response_length": 0.0, "std_response_length": 0.0,
                "avg_unique_words": 0.0, "std_unique_words": 0.0,
                "avg_vocabulary_richness": 0.0, "std_vocabulary_richness": 0.0,
            }
        
        # 1. Group responses by same prompts
        prompt_groups = {}
        for i, m in enumerate(instance_metrics):
            prompt = m["prompt"]
            if prompt not in prompt_groups:
                prompt_groups[prompt] = []
            prompt_groups[prompt].append((i, m["response"]))
        
        # print(f"Number of prompt groups: {len(prompt_groups)}")
        # for prompt, indices_responses in prompt_groups.items():
        #     print(f"Prompt group size: {len(indices_responses)}")
        
        # 2. Calculate embeddings per prompt group
        def compute_prompt_group_embeddings(prompt_group_data):
            """Compute embeddings for all responses in a prompt group."""
            prompt, indices_responses = prompt_group_data
            embeddings = []
            costs = []
            
            for idx, response in indices_responses:
                embedding, cost = self.compute_embedding(response)
                embeddings.append(embedding)
                costs.append(cost)
            
            return prompt, embeddings, costs, indices_responses
        
        # 3. Calculate pairwise cosine similarity using ThreadPoolExecutor
        all_diversities = []
        pairwise_diversities = []
        total_cost = 0.0
        
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
            # Submit embedding computation tasks for each prompt group
            future_to_prompt = {
                executor.submit(compute_prompt_group_embeddings, (prompt, indices_responses)): prompt 
                for prompt, indices_responses in prompt_groups.items()
            }
            
            # Process results as they complete
            with tqdm(total=len(prompt_groups), desc="Computing embeddings by prompt group") as pbar:
                for future in as_completed(future_to_prompt):
                    prompt, embeddings, costs, indices_responses = future.result()
                    total_cost += sum(costs)
                    
                    # Calculate pairwise similarities within this prompt group
                    if len(embeddings) > 1:  # Need at least 2 responses for diversity
                        embeddings_array = np.array(embeddings)
                        embeddings_normalized = normalize(embeddings_array, norm='l2', axis=1)
                        similarity_matrix = cosine_similarity(embeddings_normalized)
                        
                        # Verify similarity matrix properties
                        if not np.allclose(np.diag(similarity_matrix), 1.0, rtol=1e-5):
                            print(f"Warning: Self-similarities are not exactly 1.0 for prompt group")
                            
                        # Ensure similarities are in valid range [-1, 1]
                        similarity_matrix = np.clip(similarity_matrix, -1.0, 1.0)
                        
                        # Calculate pairwise diversities within this prompt group
                        # No double counting
                        for i in range(len(indices_responses)):
                            for j in range(i + 1, len(indices_responses)):
                                similarity = float(similarity_matrix[i, j])
                                
                                # Convert similarity to diversity score (0 to 1)
                                # diversity = (1 - similarity) / 2 # old metric

                                # diversity = np.clip(1 - similarity, 0.0, 1.0) # new metric
                                diversity = 1 - np.clip(similarity, 0.0, 1.0) # new metric
                                # diversity = similarity
                                
                                pairwise_diversities.append(diversity)
                                all_diversities.append(diversity)
                    
                    pbar.update(1)
        
        # Calculate statistics from intra-class diversities
        if all_diversities:
            from .base import calculate_stats
            diversity_stats = calculate_stats(all_diversities)
            
            # Calculate stats for instance-level metrics
            response_length_stats = calculate_stats([m["response_length"] for m in instance_metrics], self.num_responses_per_prompt)
            unique_words_stats = calculate_stats([m["unique_words"] for m in instance_metrics], self.num_responses_per_prompt)
            vocabulary_richness_stats = calculate_stats([m["vocabulary_richness"] for m in instance_metrics], self.num_responses_per_prompt)
            
            metrics = {
                # Diversity metrics
                "avg_diversity": diversity_stats["mean"],
                "std_diversity": diversity_stats["std"],
                "min_diversity": diversity_stats["min"],
                "max_diversity": diversity_stats["max"],
                
                # Instance-level metrics with stats
                "avg_response_length": response_length_stats["mean"],
                "std_response_length": response_length_stats["std"],
                "avg_unique_words": unique_words_stats["mean"],
                "std_unique_words": unique_words_stats["std"],
                "avg_vocabulary_richness": vocabulary_richness_stats["mean"],
                "std_vocabulary_richness": vocabulary_richness_stats["std"],
                
                "total_cost": float(total_cost),
            }
        else:
            # No valid pairs found (all prompts have only 1 response)
            response_length_stats = calculate_stats([m["response_length"] for m in instance_metrics])
            unique_words_stats = calculate_stats([m["unique_words"] for m in instance_metrics])
            vocabulary_richness_stats = calculate_stats([m["vocabulary_richness"] for m in instance_metrics])
            
            metrics = {
                "avg_diversity": 0.0, "std_diversity": 0.0,
                "min_diversity": 0.0, "max_diversity": 0.0,
                "avg_response_length": response_length_stats["mean"],
                "std_response_length": response_length_stats["std"],
                "avg_unique_words": unique_words_stats["mean"],
                "std_unique_words": unique_words_stats["std"],
                "avg_vocabulary_richness": vocabulary_richness_stats["mean"],
                "std_vocabulary_richness": vocabulary_richness_stats["std"],
                "total_cost": float(total_cost),
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