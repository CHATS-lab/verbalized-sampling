from typing import Dict, List, Any, Optional
import numpy as np
from collections import defaultdict
from .base import BaseEvaluator, EvalResult
import ast

class NgramEvaluator(BaseEvaluator):
    """Evaluator for measuring ROUGE-L scores across responses with the same prompt."""
    
    instance_plot_metrics = [
        ("pairwise_rouge_l_scores", "violin"),
        ("response_length", "histogram")
    ]
    aggregate_plot_metrics = [
        "average_rouge_l"
    ]
    key_plot_metrics = [
        ("average_rouge_l", "N-gram (ROUGE-L)"),
    ]
    
    def __init__(self, num_workers: int = 128):
        super().__init__("ngram", num_workers)
    
    def _longest_common_subsequence(self, text1: str, text2: str) -> int:
        """Compute the length of the longest common subsequence between two texts."""
        words1 = text1.split()
        words2 = text2.split()
        
        # Create a matrix to store LCS lengths
        m, n = len(words1), len(words2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Fill the dp table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if words1[i-1] == words2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def _compute_rouge_l(self, candidate: str, reference: str) -> float:
        """Compute ROUGE-L score between candidate and reference texts."""
        lcs_length = self._longest_common_subsequence(candidate, reference)
        candidate_length = len(candidate.split())
        reference_length = len(reference.split())
        
        if candidate_length == 0 or reference_length == 0:
            return 0.0
        
        # ROUGE-L precision and recall
        precision = lcs_length / candidate_length
        recall = lcs_length / reference_length
        
        # F1 score (harmonic mean of precision and recall)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
    
    def compute_instance_metric(self, prompt: Any, response: Dict) -> Dict[str, float]:
        """Compute ROUGE-L metrics for a single response."""
        response_text = response.get('text', '')
        if isinstance(response_text, dict):
            response_text = str(response_text)

        return {
            "prompt": prompt,
            "response": response_text,
            "response_length": len(response_text.split())
        }
    
    def aggregate_metrics(self, instance_metrics: List[Dict[str, float]]) -> Dict[str, Any]:
        """Compute ROUGE-L metrics across all responses."""
        if len(instance_metrics) <= 1:
            return {
                "average_rouge_l": 0.0,
                "min_rouge_l": 0.0,
                "max_rouge_l": 0.0,
                "std_rouge_l": 0.0,
                "average_response_length": 0.0,
                "pairwise_rouge_l_scores": []
            }
        
        # Group responses by prompt
        prompt_groups = defaultdict(list)
        for i, m in enumerate(instance_metrics):
            prompt_groups[m["prompt"]].append((i, m["response"]))
        
        # Calculate metrics for each prompt group
        all_rouge_l_scores = []
        pairwise_scores = []
        
        for prompt, responses in prompt_groups.items():
            if len(responses) > 1:
                # Calculate pairwise ROUGE-L scores
                for i in range(len(responses)):
                    for j in range(i + 1, len(responses)):
                        score = self._compute_rouge_l(responses[i][1], responses[j][1])
                        pairwise_scores.append(score)
                        all_rouge_l_scores.append(score)
        
        # Calculate overall statistics
        if all_rouge_l_scores:
            scores_array = np.array(all_rouge_l_scores)
            metrics = {
                "average_rouge_l": float(scores_array.mean()),
                "min_rouge_l": float(scores_array.min()),
                "max_rouge_l": float(scores_array.max()),
                "std_rouge_l": float(scores_array.std()),
                "average_response_length": float(np.mean([m["response_length"] for m in instance_metrics])),
                "pairwise_rouge_l_scores": pairwise_scores
            }
        else:
            metrics = {
                "average_rouge_l": 0.0,
                "min_rouge_l": 0.0,
                "max_rouge_l": 0.0,
                "std_rouge_l": 0.0,
                "average_response_length": float(np.mean([m["response_length"] for m in instance_metrics])),
                "pairwise_rouge_l_scores": []
            }
        
        return metrics
    
    def evaluate(self, prompts: List[str], responses: List[str], 
                metadata: Optional[Dict[str, Any]] = None) -> EvalResult:
        """Evaluate ROUGE-L scores for responses."""
        if metadata is None:
            metadata = {}
            
        metadata.update({
            "evaluation_type": "rouge_l",
            "num_responses": len(responses)
        })
        
        return super().evaluate(prompts, responses, metadata)
