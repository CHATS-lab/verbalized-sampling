from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

@dataclass
class EvalResult:
    """Container for evaluation results."""
    instance_metrics: List[Dict[str, float]]  # List of metrics for each instance
    overall_metrics: Dict[str, float]  # Aggregated metrics across all instances
    metadata: Dict[str, Any]  # Additional metadata about the evaluation

class BaseEvaluator(ABC):
    """Base class for all evaluators."""
    
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compute_instance_metric(self, prompt: str, response: str) -> Dict[str, float]:
        """Compute metrics for a single instance."""
        pass

    @abstractmethod
    def aggregate_metrics(self, instance_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate instance-level metrics into overall metrics."""
        pass

    def evaluate(self, prompts: List[str], responses: List[str], metadata: Optional[Dict[str, Any]] = None) -> EvalResult:
        """Evaluate a list of prompts and responses."""
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            instance_metrics = list(executor.map(
                lambda x: self.compute_instance_metric(x[0], x[1]),
                zip(prompts, responses)
            ))
        overall_metrics = self.aggregate_metrics(instance_metrics)
        return EvalResult(instance_metrics, overall_metrics, metadata)

    def save_results(self, result: EvalResult, output_path: Union[str, Path]):
        """Save evaluation results to a file."""
        # Implementation details...

    @classmethod
    def load_results(cls, input_path: Union[str, Path]) -> EvalResult:
        """Load evaluation results from a file."""
        # Implementation details...

        