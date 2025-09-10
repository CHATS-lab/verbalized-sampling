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
        "average_unique_recall_rate",
        "average_precision",
    ]
    key_plot_metrics = [
        ("average_kl_divergence", "KL Divergence"),
        ("average_unique_recall_rate", "Unique Recall Rate"),
        ("average_precision", "Precision"),
    ]

    def __init__(self, name: str = "response_count", num_workers: int = 128, num_responses_per_prompt: int = 100):
        super().__init__(name=name, num_workers=num_workers)
        with open("data/state_name.json", "r") as f:
            self.gt_data = json.load(f)
        self.num_responses_per_prompt = num_responses_per_prompt


    def check_precision(self, response: str, gt_responses: List[Dict[str, Any]]) -> bool:
        """Check if the response matches any of the ground truth responses.
        
        Args:
            response: The response to check
            gt_responses: List of ground truth responses
            
        Returns:
            bool: True if the response matches any ground truth response
        """
        return any(response == fmt.lower().rstrip().rstrip('.') for gt_resp in gt_responses for fmt in gt_resp)

    
    def compute_instance_metric(self, prompt: str, response: str) -> Dict[str, int]:
        """Compute the count of responses."""
        response_text = response.get('text', response)
        if isinstance(response_text, dict):
            response_text = str(response_text)

        # Process the response text
        response_text = response_text.lower().rstrip().rstrip('.')

        # get the gt response count and clean prompt
        prompt_clean = prompt.replace('\n', '')
        gt_response_count = len(self.gt_data[prompt_clean]['answers'])
        correct_response = self.check_precision(response_text, self.gt_data[prompt_clean]['answers'])
        
        return {
            "prompt": prompt_clean,
            "response": response_text,
            "response_count": 1,  # Set to 1 as we'll count in aggregate_metrics
            'correct_response': 1 if correct_response else 0,
            'total_gt_responses': gt_response_count
        }

    def _calculate_kl_divergence(self, response_distribution: Counter, num_gt_responses: int) -> float:
        """Calculate KL divergence against uniform distribution."""
        if not response_distribution:
            return 0.0
            
        observed_counts = np.array(list(response_distribution.values()))
        # print("Total responses per prompt: ", self.num_responses_per_prompt, sum(observed_counts))
        total_responses = self.num_responses_per_prompt if self.num_responses_per_prompt > sum(observed_counts) else sum(observed_counts)
        observed_probs = observed_counts / total_responses

        uniform_probs = np.full_like(observed_probs, 1.0 / num_gt_responses)
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        observed_probs = observed_probs + epsilon
        observed_probs = observed_probs / observed_probs.sum()  # Renormalize
        
        # Calculate KL divergence: KL(P || Q) = sum(P * log(P / Q))
        kl_divergence = float(np.sum(observed_probs * np.log(observed_probs / uniform_probs)))
        return kl_divergence
    

    def re_group(self, response_distribution: Counter, gt_responses: List[Dict[str, Any]]) -> Counter:
        """Re-group the response distribution to the number of gt responses."""
        regrouped_distribution = Counter()
        
        # For each response in the distribution, find which gt_response group it belongs to
        for response, count in response_distribution.items():
            for gt_response in gt_responses:
                gt_response_clean = [fmt.lower().rstrip().rstrip('.') for fmt in gt_response]
                gt_first_ans = gt_response_clean[0]
                if response in gt_response_clean:
                    regrouped_distribution[gt_first_ans] += count
        
        return regrouped_distribution


    def aggregate_metrics(self, instance_metrics: List[List[Dict[str, int]]]) -> Dict[str, float]:
        """Aggregate metrics across all instances, grouped by prompt and calculate per-prompt stats."""
        if not instance_metrics:
            return {
                "per_prompt_stats": {},
                "average_kl_divergence": 0.0,
                # "average_chi_square": 0.0,
                "average_precision": 0.0,
                "num_prompts": 0
            }

        # Group by prompt
        prompt_groups = {}
        for metric in instance_metrics:
            prompt = metric["prompt"]
            if prompt not in prompt_groups:
                prompt_groups[prompt] = []
            prompt_groups[prompt].append(metric)

        # Calculate per-prompt stats
        per_prompt_stats = {}
        total_kl_div = 0.0
        # total_chi_square = 0.0
        total_precision = 0.0
        total_unique_recall_rate = 0.0
        num_prompts = len(prompt_groups)
        
        for prompt, group in prompt_groups.items():
            # Create a fresh counter for each prompt group
            # print("prompt: ", prompt)
            response_distribution = Counter()
            num_correct_responses = 0
            gt_count = group[0]['total_gt_responses'] if group else 0
            
            # Count responses for this prompt group
            for metric in group:
                # Only count if it's a correct response
                if metric.get('correct_response') == 1:
                    response_distribution[metric['response']] += 1
                    num_correct_responses += 1
            
            # re-group the response distribution to the number of gt responses
            prompt_clean = prompt.replace('\n', '')
            response_distribution = self.re_group(response_distribution, self.gt_data[prompt_clean]['answers'])
            
            # Calculate uniformity metrics for this prompt
            kl_div = self._calculate_kl_divergence(response_distribution, gt_count)
            # chi_square = self._calculate_chi_square(response_distribution, gt_count)
            precision = num_correct_responses / len(group)
            # unique recall rate
            unique_recall_rate = len(set(response_distribution.keys())) / gt_count
            
            total_kl_div += kl_div
            # total_chi_square += chi_square
            total_precision += precision
            total_unique_recall_rate += unique_recall_rate

            if response_distribution:
                min_responses_per_category = min(response_distribution.values())
                max_responses_per_category = max(response_distribution.values())
            else:
                min_responses_per_category = 0
                max_responses_per_category = 0
                
            per_prompt_stats[prompt] = {
                "min_responses_per_category": min_responses_per_category,
                "max_responses_per_category": max_responses_per_category,
                "num_correct_responses": num_correct_responses,
                "total_responses": self.num_responses_per_prompt,
                "response_distribution": dict(response_distribution),
                "num_gt_responses": gt_count,
                "kl_divergence": kl_div,
                # "chi_square": chi_square,
                "precision": precision,
                "unique_recall_rate": unique_recall_rate
            }
        
        
        return {
            "per_prompt_stats": per_prompt_stats,
            "average_kl_divergence": total_kl_div / num_prompts if num_prompts > 0 else 0.0,
            # "average_chi_square": total_chi_square / num_prompts if num_prompts > 0 else 0.0,
            "average_precision": total_precision / num_prompts if num_prompts > 0 else 0.0,
            "average_unique_recall_rate": total_unique_recall_rate / num_prompts if num_prompts > 0 else 0.0,
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


    # def _generate_uniform_sample(self, n_trials, n_labels, seed):
    #     """Generate what a truly uniform state selection would look like."""
    #     if seed:
    #         np.random.seed(seed)

    #     # Simulate uniform random selection
    #     state_selections = np.random.choice(range(n_labels), size=n_trials, replace=True)
        
    #     # Count frequencies
    #     unique_states, counts = np.unique(state_selections, return_counts=True)
        
    #     # Create full array (including states with 0 counts)
    #     full_counts = np.zeros(n_labels)
    #     full_counts[unique_states] = counts
        
    #     return sorted([int(count) for count in full_counts], reverse=True)

    # def _calculate_chi_square(self, response_distribution: Counter, num_gt_responses: int) -> float:
    #     """Calculate chi-square statistic against uniform distribution."""
    #     if not response_distribution:
    #         return 0.0
        
    #     total_responses = self.num_responses_per_prompt # sum(response_distribution.values())
    #     observed_values = list(response_distribution.values())
        
    #     # Create observed counts array with proper size
    #     observed_counts = np.zeros(num_gt_responses)
    #     observed_counts[:len(observed_values)] = observed_values
        
    #     # Generate uniform sample with same size
    #     expected_uniform = self._generate_uniform_sample(total_responses, num_gt_responses, 42)
        
    #     # Ensure both arrays are numpy arrays with the same shape
    #     observed_counts = np.array(observed_counts, dtype=float)
    #     expected_uniform = np.array(expected_uniform, dtype=float)
        
    #     # Add small epsilon to avoid division by zero
    #     epsilon = 1e-10
    #     expected_uniform = expected_uniform + epsilon
        
    #     chi_square_stat, _ = chisquare(observed_counts, expected_uniform)
    #     return float(chi_square_stat)