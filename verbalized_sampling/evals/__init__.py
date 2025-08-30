from .diversity import DiversityEvaluator
from .quality import TTCTEvaluator
from .creativity_index import CreativityIndexEvaluator
from .length import LengthEvaluator
from .response_count import ResponseCountEvaluator
from .creative_writing_v3 import CreativeWritingV3Evaluator
from .ngram import NgramEvaluator
from .base import BaseEvaluator
from .factuality import FactualityEvaluator
from .joke_quality import JokeQualityEvaluator
from .data_quality import SyntheticDataQualityEvaluator

# Import plotting functionality from the new plots module
from ..plots import (
    ComparisonPlotter, 
    plot_evaluation_comparison,
    plot_comparison_chart
)

__all__ = [
    "DiversityEvaluator", 
    "TTCTEvaluator", 
    "CreativityIndexEvaluator", 
    "LengthEvaluator",
    "ResponseCountEvaluator",
    "CreativeWritingV3Evaluator",
    "NgramEvaluator",
    "BaseEvaluator",
    "FactualityEvaluator",
    "JokeQualityEvaluator",
    "SyntheticDataQualityEvaluator",
    # Plotting functionality
    "ComparisonPlotter", 
    "plot_evaluation_comparison",
    "plot_comparison_chart"
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
    elif metric == "creative_writing_v3" or metric == "cwv3":
        return CreativeWritingV3Evaluator(**kwargs)
    elif metric == "ngram":
        return NgramEvaluator(**kwargs)
    elif metric == "factuality":
        return FactualityEvaluator(**kwargs)
    elif metric == "joke_quality":
        return JokeQualityEvaluator(**kwargs)
    elif metric == "synthetic_data_quality":
        return SyntheticDataQualityEvaluator(**kwargs)
    else:
        raise ValueError(f"Evaluator {metric} not found. Available evaluators: diversity, ttct, creativity_index, length, response_count, creative_writing_v3, ngram, factuality")