from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import numpy as np
from tqdm import tqdm
from abc import ABC
from scipy import stats

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
    
    @classmethod
    def merge(cls, results: List['EvalResult']) -> 'EvalResult':
        """
        Merge multiple EvalResult objects into a single result.
        
        Args:
            results: List of EvalResult objects to merge
            
        Returns:
            A new EvalResult containing the merged data
            
        Note:
            - Instance metrics are concatenated
            - Overall metrics are recalculated based on the combined instance metrics
            - Metadata is merged, with later results taking precedence for overlapping keys
        """
        if not results:
            raise ValueError("No results provided to merge")
            
        # Merge instance metrics by concatenation
        merged_instance_metrics = []
        for result in results:
            merged_instance_metrics.extend(result.instance_metrics)
            
        # Merge metadata, with later results taking precedence
        merged_metadata = {}
        for result in results:
            merged_metadata.update(result.metadata)
            
        # Recalculate overall metrics based on VS-Multi (vs_multi) instance metrics
        # This is a simplified version - you might want to customize this based on your needs
        merged_overall_metrics = {}
        
        for result in results:
            merged_overall_metrics.update(result.overall_metrics)
        
        return cls(
            instance_metrics=merged_instance_metrics,
            overall_metrics=merged_overall_metrics,
            metadata=merged_metadata
        )
    
    def __add__(self, other: 'EvalResult') -> 'EvalResult':
        """
        Allow merging two EvalResult objects using the + operator.
        
        Args:
            other: Another EvalResult object to merge with
            
        Returns:
            A new EvalResult containing the merged data
        """
        return self.merge([self, other])

class BaseEvaluator(ABC):
    """Base class for all evaluators."""
    
    # Define class variables for plot metrics
    instance_plot_metrics: List[tuple[str, str]] = []  # List of (metric_name, plot_type) tuples
    aggregate_plot_metrics: List[str] = []  # List of metric names for aggregate plots
    key_plot_metrics: List[tuple[str, str]] = []  # List of (metric_name, plot_title) tuples for key plots
    
    def __init__(self, name: str, num_workers: int = 128, num_responses_per_prompt: int = 50):
        self.name = name
        self.num_workers = num_workers
        self.num_responses_per_prompt = num_responses_per_prompt

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
        
        # Use the enhanced aggregation method that includes error statistics
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

def calculate_stats(values: List[float], num_responses_per_prompt: Optional[int] = None) -> Dict[str, float]:
    """
    Calculate basic statistics for a list of values.
    
    Args:
        values: List of numeric values
        
    Returns:
        Dictionary containing mean, std, min, max, etc.
    """
    if not values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "count": 0
        }
    
    values = np.array([float(v) for v in values if v is not None and not np.isnan(v)])
    
    if len(values) == 0:
        return {
            "mean": 0.0,
            "std": 0.0, 
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "count": 0
        }
    
    # print("Counts: ", len(values), num_responses_per_prompt)
    return {
        "mean": float(np.sum(values) / num_responses_per_prompt) if num_responses_per_prompt else float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
        "count": len(values)
    }

        