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
        with open("data/CoverageQA.json", "r") as f: self.gt_data = json.load(f)
    
    def compute_instance_metric(self, prompt: str, response: str) -> Dict[str, int]:
        """Compute the count of responses."""
        if isinstance(response, str):
            response = ast.literal_eval(response)

        list_of_responses = [
            response.get('text', response) if isinstance(response, dict) else response
        ]
        list_of_responses = [
            response.replace('.', '').lower().rstrip() for response in list_of_responses
        ]
        self.counter.update(list_of_responses)
        
        return [
            {
                "response_count": self.counter[text]
            } for text in list_of_responses
        ]
    

    def aggregate_metrics(self, instance_metrics: List[List[Dict[str, int]]]) -> Dict[str, float]:
        """Aggregate metrics across all instances."""
        if not instance_metrics:
            return {}
        
        # print(instance_metrics)
        
        return {
            "min_responses_per_category": min(self.counter.values()),
            "max_responses_per_category": max(self.counter.values()),
            "num_responses": len(instance_metrics) * len(instance_metrics[0]),
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