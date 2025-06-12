from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import numpy as np
from tqdm import tqdm

class EvalResultEncoder(json.JSONEncoder):
    """Custom JSON encoder for EvalResult."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'to_dict'):  # Handle EvalResult objects
            return obj.to_dict()
        return super().default(obj)

@dataclass
class EvalResult:
    """Container for evaluation results."""
    instance_metrics: List[Dict[str, float]]  # List of metrics for each instance
    overall_metrics: Dict[str, Any]  # Aggregated metrics across all instances
    metadata: Dict[str, Any]  # Additional metadata about the evaluation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert the result to a JSON string."""
        return json.dumps(self.to_dict(), cls=EvalResultEncoder)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvalResult':
        """Create an EvalResult from a dictionary."""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'EvalResult':
        """Create an EvalResult from a JSON string."""
        return cls.from_dict(json.loads(json_str))

class BaseEvaluator(ABC):
    """Base class for all evaluators."""
    
    def __init__(self, name: str, num_workers: int = 128):
        self.name = name
        self.num_workers = num_workers

    @abstractmethod
    def compute_instance_metric(self, prompt: str, response: str) -> Dict[str, float]:
        """Compute metrics for a single instance."""
        pass

    @abstractmethod
    def aggregate_metrics(self, instance_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate instance-level metrics into overall metrics."""
        pass

    def evaluate(self, prompts: List[str], responses: List[Dict], metadata: Optional[Dict[str, Any]] = None) -> EvalResult:
        """Evaluate a list of prompts and responses.
        
        Args:
            prompts: List of prompts
            responses: List of responses, [{'text': 'response', 'index': 0, 'probability'(optional): 0.5}]
            metadata: Additional metadata about the evaluation
            
        Returns:
        """
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            instance_metrics = list(tqdm(
                executor.map(
                    lambda x: self.compute_instance_metric(x[0], x[1]),
                    zip(prompts, responses)
                ),
                total=len(prompts),
                desc=f"Computing {self.name} metrics"
            ))
        overall_metrics = self.aggregate_metrics(instance_metrics)
        return EvalResult(instance_metrics, overall_metrics, metadata)

    def save_results(self, result: EvalResult, output_path: Union[str, Path]):
        """Save evaluation results to a file."""
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=4, cls=EvalResultEncoder)

    @classmethod
    def load_results(cls, input_path: Union[str, Path]) -> EvalResult:
        """Load evaluation results from a file."""
        # Implementation details...

        