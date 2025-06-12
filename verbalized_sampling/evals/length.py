from typing import Dict, List, Any, Optional
import tiktoken
from .base import BaseEvaluator, EvalResult
import ast

class LengthEvaluator(BaseEvaluator):
    """Simple evaluator for computing token length of text responses using OpenAI tokenizer."""
    
    def __init__(self, num_workers: int = 128):
        super().__init__("length", num_workers)
        # Get the tokenizer for the specified model
        self.tokenizer = tiktoken.get_encoding("o200k_base")
    
    def compute_instance_metric(self, prompts: Any, responses: Any) -> List[Dict[str, float]]:
        """Compute token length for a single response."""
        if isinstance(responses, str):
            responses = ast.literal_eval(responses)

        list_of_responses = [
            response.get('response', response)
            for response in responses
        ]
        list_of_prompts = [
            response.get('prompt', response)
            for response in responses
        ]

        return [
            {
                "response": response,
                "prompt": prompt,
                "token_length": float(len(self.tokenizer.encode(response)))
            } for response, prompt in zip(list_of_responses, list_of_prompts)
        ]
    
    def aggregate_metrics(self, instance_metrics: List[List[Dict[str, float]]]) -> Dict[str, float]:
        """Aggregate token lengths across all instances."""
        if not instance_metrics:
            return {}

        # Flatten the nested list structure and filter out None values
        flattened_metrics = [metric for instance_list in instance_metrics for metric in instance_list if metric]
        
        token_lengths = [metric["token_length"] for metric in flattened_metrics]
        
        return {
            "mean_token_length": sum(token_lengths) / len(token_lengths),
            "min_token_length": min(token_lengths),
            "max_token_length": max(token_lengths),
            "total_tokens": sum(token_lengths),
            "num_responses": len(flattened_metrics)
        }
    
    def evaluate(self, prompts: List[str], responses: List[str], 
                metadata: Optional[Dict[str, Any]] = None) -> EvalResult:
        """Evaluate responses for token length."""
        if metadata is None:
            metadata = {}
            
        metadata.update({
            "evaluation_type": "token_length",
            "tokenizer_model": self.tokenizer.name,
            "num_responses": len(responses)
        })
        
        return super().evaluate(prompts, responses, metadata)
