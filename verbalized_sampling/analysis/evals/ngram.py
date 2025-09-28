from typing import Dict, List, Any, Optional
import numpy as np
from collections import defaultdict
from .base import BaseEvaluator, EvalResult
import ast

def _get_ngrams(words: List[str], n: int) -> List[tuple]:
    """Return a list of n-grams (as tuples) from a list of words."""
    if len(words) < n:
        return []
    return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]

def _proportion_unique_ngrams(text: str, n: int) -> float:
    """Compute the proportion of unique n-grams in the text."""
    words = text.split()
    ngrams = _get_ngrams(words, n)
    total = len(ngrams)
    if total == 0:
        return 0.0
    unique = len(set(ngrams))
    return unique / total

class NgramEvaluator(BaseEvaluator):
    """Evaluator for measuring ROUGE-L scores and distinct-n (average of unique n-gram proportions for n=1,2,3) across responses with the same prompt."""
    
    instance_plot_metrics = [
        ("pairwise_rouge_l_scores", "violin"),
        ("distinct_n", "histogram"),
        ("response_length", "histogram"),
    ]
    aggregate_plot_metrics = [
        "avg_rouge_l",
        "avg_response_length",
        "avg_distinct_n",
    ]
    key_plot_metrics = [
        ("avg_rouge_l", "N-gram (ROUGE-L)"),
        ("avg_distinct_n", "Distinct-n (avg of 1,2,3)"),
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
        """Compute ROUGE-L and distinct-n (average of unique n-gram proportions for n=1,2,3) for a single response."""
        response_text = response.get('text', '')
        if isinstance(response_text, dict):
            response_text = str(response_text)

        distinct_1 = _proportion_unique_ngrams(response_text, 1)
        distinct_2 = _proportion_unique_ngrams(response_text, 2)
        distinct_3 = _proportion_unique_ngrams(response_text, 3)
        distinct_n = np.mean([distinct_1, distinct_2, distinct_3])

        return {
            "prompt": prompt,
            "response": response_text,
            "response_length": len(response_text.split()),
            "distinct_n": distinct_n,
        }
    
    def aggregate_metrics(self, instance_metrics: List[Dict[str, float]]) -> Dict[str, Any]:
        """Compute ROUGE-L and distinct-n metrics across all responses."""
        if len(instance_metrics) <= 1:
            return {
                "avg_rouge_l": 0.0,
                "min_rouge_l": 0.0,
                "max_rouge_l": 0.0,
                "std_rouge_l": 0.0,
                "avg_response_length": 0.0,
                "std_response_length": 0.0, 
                "pairwise_rouge_l_scores": [],
                "avg_distinct_n": 0.0,
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
            avg_rouge_l = float(scores_array.mean())
            min_rouge_l = float(scores_array.min())
            max_rouge_l = float(scores_array.max())
            std_rouge_l = float(scores_array.std())
        else:
            avg_rouge_l = 0.0
            min_rouge_l = 0.0
            max_rouge_l = 0.0
            std_rouge_l = 0.0

        avg_response_length = float(np.mean([m["response_length"] for m in instance_metrics]))
        std_response_length = float(np.std([m["response_length"] for m in instance_metrics]))
        avg_distinct_n = float(np.mean([m["distinct_n"] for m in instance_metrics]))

        metrics = {
            "avg_rouge_l": avg_rouge_l,
            "min_rouge_l": min_rouge_l,
            "max_rouge_l": max_rouge_l,
            "std_rouge_l": std_rouge_l,
            "avg_response_length": avg_response_length,
            "std_response_length": std_response_length,
            "pairwise_rouge_l_scores": pairwise_scores,
            "avg_distinct_n": avg_distinct_n,
        }
        
        return metrics
    
    def evaluate(self, prompts: List[str], responses: List[str], 
                metadata: Optional[Dict[str, Any]] = None) -> EvalResult:
        """Evaluate ROUGE-L and distinct-n scores for responses."""
        if metadata is None:
            metadata = {}
            
        metadata.update({
            "evaluation_type": "rouge_l",
            "num_responses": len(responses)
        })
        
        return super().evaluate(prompts, responses, metadata)
