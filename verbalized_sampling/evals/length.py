from typing import Dict, List, Any, Optional
import tiktoken
from .base import BaseEvaluator, EvalResult
import ast

class LengthEvaluator(BaseEvaluator):
    """Simple evaluator for computing token length of text responses using OpenAI tokenizer."""
    instance_plot_metrics = [
        ("token_length", "violin")
    ]
    aggregate_plot_metrics = [
        "mean_token_length"
    ]
    key_plot_metrics = [
        ("mean_token_length", "Length (Token)"),
    ]
    
    def __init__(self, num_workers: int = 128):
        super().__init__("length", num_workers)
        # Get the tokenizer for the specified model
        self.tokenizer = tiktoken.get_encoding("o200k_base")
    
    def compute_instance_metric(self, prompt: str, response: str) -> Dict[str, float]:
        """Compute token length for a single response."""

        response_text = response['text']
        return {
            "response": response_text,
            "prompt": prompt,
            "token_length": float(len(self.tokenizer.encode(response_text)))
        }
    
    def aggregate_metrics(self, instance_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate token lengths across all instances."""
        if not instance_metrics:
            return {}
        
        token_lengths = [metric["token_length"] for metric in instance_metrics]
        
        return {
            "mean_token_length": sum(token_lengths) / len(token_lengths),
            "min_token_length": min(token_lengths),
            "max_token_length": max(token_lengths),
            "total_tokens": sum(token_lengths),
            "num_responses": len(instance_metrics)
        }
    
    def evaluate(self, prompts: List[str], responses: List[Dict], 
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
