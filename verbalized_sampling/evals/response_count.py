from typing import List, Dict, Any, Optional
import json
import ast
from collections import Counter
from .base import BaseEvaluator, EvalResult
import numpy as np
from scipy.stats import chisquare, entropy

class ResponseCountEvaluator(BaseEvaluator):
    """Evaluator that counts the number of responses in the text field of responses."""
    
    instance_plot_metrics = [
        ("response_count", "histogram"),
    ]
    aggregate_plot_metrics = [
        "average_kl_divergence",
        "average_chi_square"
    ]
    key_plot_metrics = [
        ("average_kl_divergence", "KL Divergence"),
        ("average_chi_square", "Chi-square"),
    ]

    def __init__(self, name: str = "response_count", num_workers: int = 128):
        super().__init__(name=name, num_workers=num_workers)
        with open("data/state_name.json", "r") as f:
            self.gt_data = json.load(f)
    
    def compute_instance_metric(self, prompt: str, response: str) -> Dict[str, int]:
        """Compute the count of responses."""
        if isinstance(response, str):
            response = ast.literal_eval(response)

        list_of_responses = [
            response.get('text', response) if isinstance(response, dict) else response
        ]
        list_of_responses = [
            response.lower().rstrip().rstrip('.') for response in list_of_responses
        ]

        # get the gt response count and clean prompt
        prompt_clean = prompt.replace('\n', '')
        gt_response_count = len(self.gt_data[prompt_clean]['answers'])
        
        return [
            {
                "prompt": prompt_clean,
                "response": text,
                "response_count": 1,  # Set to 1 as we'll count in aggregate_metrics
                'total_gt_responses': gt_response_count
            } for text in list_of_responses
        ]

    def _calculate_kl_divergence(self, response_distribution: Counter, num_gt_responses: int) -> float:
        """Calculate KL divergence against uniform distribution."""
        if not response_distribution:
            return 0.0
            
        observed_counts = np.array(list(response_distribution.values()))
        total_responses = sum(observed_counts)
        observed_probs = observed_counts / total_responses
        uniform_probs = np.full_like(observed_probs, 1.0 / num_gt_responses)
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        observed_probs = observed_probs + epsilon
        observed_probs = observed_probs / observed_probs.sum()  # Renormalize
        
        # Calculate KL divergence: KL(P || Q) = sum(P * log(P / Q))
        kl_divergence = float(np.sum(observed_probs * np.log(observed_probs / uniform_probs)))
        return kl_divergence


    def _generate_uniform_sample(self, n_trials, n_labels, seed):
        """Generate what a truly uniform state selection would look like."""
        if seed:
            np.random.seed(seed)

        # Simulate uniform random selection
        state_selections = np.random.choice(range(n_labels), size=n_trials, replace=True)
        
        # Count frequencies
        unique_states, counts = np.unique(state_selections, return_counts=True)
        
        # Create full array (including states with 0 counts)
        full_counts = np.zeros(n_labels)
        full_counts[unique_states] = counts
        
        return sorted([int(count) for count in full_counts], reverse=True)

    def _calculate_chi_square(self, response_distribution: Counter, num_gt_responses: int) -> float:
        """Calculate chi-square statistic against uniform distribution."""
        if not response_distribution:
            return 0.0
        
        total_responses = sum(response_distribution.values())
        observed_values = list(response_distribution.values())
        
        # Ensure we're using the larger of the two numbers for array size
        max_categories = max(len(response_distribution), num_gt_responses)
        
        # Create observed counts array with proper size
        observed_counts = np.zeros(max_categories)
        observed_counts[:len(observed_values)] = observed_values
        
        # Generate uniform sample with same size
        expected_uniform = self._generate_uniform_sample(total_responses, max_categories, 42)
        # print(expected_uniform)
        # print(observed_counts)
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        expected_uniform = np.array(expected_uniform) + epsilon
        
        # Calculate chi-square
        chi_square_stat, _ = chisquare(observed_counts, expected_uniform)
        return float(chi_square_stat)


    def aggregate_metrics(self, instance_metrics: List[List[Dict[str, int]]]) -> Dict[str, float]:
        """Aggregate metrics across all instances, grouped by prompt and calculate per-prompt stats."""
        if not instance_metrics:
            return {}

        # Flatten the list of lists
        flat_metrics = [item for sublist in instance_metrics for item in sublist]

        # Group by prompt
        prompt_groups = {}
        for metric in flat_metrics:
            prompt = metric["prompt"]
            if prompt not in prompt_groups:
                prompt_groups[prompt] = []
            prompt_groups[prompt].append(metric)

        # Calculate per-prompt stats
        per_prompt_stats = {}
        total_kl_div = 0.0
        total_chi_square = 0.0
        num_prompts = len(prompt_groups)
        
        for prompt, group in prompt_groups.items():
            # Create a fresh counter for each prompt group
            response_distribution = Counter()
            num_responses = 0
            gt_count = group[0]['total_gt_responses'] if group else 0
            
            # Count responses for this prompt group
            for metric in group:
                response_distribution[metric['response']] += 1  # Count each response once
                num_responses += 1
            
            # Calculate uniformity metrics for this prompt
            kl_div = self._calculate_kl_divergence(response_distribution, gt_count)
            chi_square = self._calculate_chi_square(response_distribution, gt_count)
            
            total_kl_div += kl_div
            total_chi_square += chi_square
            
            if response_distribution:
                min_responses_per_category = min(response_distribution.values())
                max_responses_per_category = max(response_distribution.values())
            else:
                min_responses_per_category = 0
                max_responses_per_category = 0
                
            per_prompt_stats[prompt] = {
                "min_responses_per_category": min_responses_per_category,
                "max_responses_per_category": max_responses_per_category,
                "num_responses": num_responses,
                "response_distribution": dict(response_distribution),
                "num_gt_responses": gt_count,
                "kl_divergence": kl_div,
                "chi_square": chi_square
            }
        
        return {
            "per_prompt_stats": per_prompt_stats,
            "average_kl_divergence": total_kl_div / num_prompts if num_prompts > 0 else 0.0,
            "average_chi_square": total_chi_square / num_prompts if num_prompts > 0 else 0.0,
            "num_prompts": num_prompts
        }


    def evaluate(self, prompts: List[str], responses: List[str], 
                metadata: Optional[Dict[str, Any]] = None) -> EvalResult:
        """Evaluate responses for token length."""
        if metadata is None:
            metadata = {}
            
        metadata.update({
            "evaluation_type": "response_count",
            "num_responses": len(responses)
        })
        
        return super().evaluate(prompts, responses, metadata)