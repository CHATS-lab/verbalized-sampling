from typing import List, Dict, Any, Optional
import json
from collections import Counter
from .base import BaseEvaluator, EvalResult

class ResponseCountEvaluator(BaseEvaluator):
    """Evaluator that counts the number of responses in the text field of responses."""
    
    def __init__(self, name: str = "response_count", num_workers: int = 128):
        super().__init__(name=name, num_workers=num_workers)
        self.counter = Counter()
    
    def compute_instance_metric(self, prompt: str, response: str) -> Dict[str, float]:
        try:
            # If response is already a list, use it directly
            if isinstance(response, list):
                response_count = len(response)
            else:
                # Try to parse as JSON if it's a string
                response_data = json.loads(response)
                response_count = len(response_data.get('text', []))
            
            self.counter.update([response_count])  # Update counter with the count
            return {
                "response_count": float(response_count)
            }
        except (json.JSONDecodeError, AttributeError, TypeError):
            # If response is not valid JSON or doesn't have expected structure
            return {
                "response_count": 0.0
            }
    
    def aggregate_metrics(self, instance_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        if not instance_metrics:
            return {}
        
        response_counts = [metric["response_count"] for metric in instance_metrics]
        
        return {
            "min_responses": min(response_counts),
            "max_responses": max(response_counts),
            "num_responses": sum(response_counts),
            "response_distribution": dict(self.counter)  # Convert Counter to dict for JSON serialization
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