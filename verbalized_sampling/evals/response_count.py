from typing import List, Dict, Any, Optional
import json
import ast
from collections import Counter
from .base import BaseEvaluator, EvalResult

class ResponseCountEvaluator(BaseEvaluator):
    """Evaluator that counts the number of responses in the text field of responses."""
    
    def __init__(self, name: str = "response_count", num_workers: int = 128):
        super().__init__(name=name, num_workers=num_workers)
        self.counter = Counter()
    
    def compute_instance_metric(self, prompt: str, response: Any) -> Dict[str, float]:
        """Compute the count of responses."""
        response_text = ""
        # Try to parse response as JSON if it's a string
        if isinstance(response, str):
            response = ast.literal_eval(response)
        response_text = response.get('response', response)
        response_text = response_text.replace('.', '')
        print(response_text)

        self.counter.update([response_text])
        print(self.counter[response_text])

        return {
            "response_count": self.counter[response_text]
        }
    
    def aggregate_metrics(self, instance_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics across all instances."""
        if not instance_metrics:
            return {}
        
        response_counts = [metric["response_count"] for metric in instance_metrics]
        
        return {
            "min_responses": min(self.counter.values()),
            "max_responses": max(self.counter.values()),
            "num_responses": len(instance_metrics),
            "response_distribution": dict(self.counter)
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