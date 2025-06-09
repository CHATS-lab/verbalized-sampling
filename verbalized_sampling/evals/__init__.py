from .diversity import DiversityEvaluator
from .quality import TTCTEvaluator
from .creativity_index import CreativityIndexEvaluator
from .length import LengthEvaluator
from .comparison_plots import ComparisonPlotter, plot_evaluation_comparison
from .response_count import ResponseCountEvaluator

__all__ = [
    "DiversityEvaluator", 
    "TTCTEvaluator", 
    "CreativityIndexEvaluator", 
    "LengthEvaluator",
    "ResponseCountEvaluator",
    "ComparisonPlotter", 
    "plot_evaluation_comparison",
]

def get_evaluator(metric: str, **kwargs):
    if metric == "diversity":
        return DiversityEvaluator(**kwargs)
    elif metric == "ttct" or metric == "quality":
        return TTCTEvaluator(**kwargs)
    elif metric == "creativity_index":
        return CreativityIndexEvaluator(**kwargs)
    elif metric == "length":
        return LengthEvaluator(**kwargs)
    elif metric == "response_count":
        return ResponseCountEvaluator(**kwargs)
    else:
        raise ValueError(f"Evaluator {metric} not found. Available evaluators: diversity, ttct, creativity_index, length, response_count")