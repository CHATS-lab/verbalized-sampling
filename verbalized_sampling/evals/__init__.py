from .diversity import DiversityEvaluator
from .quality import TTCTEvaluator
from .creativity_index import CreativityIndexEvaluator
from .length import LengthEvaluator
from .comparison_plots import (
    ComparisonPlotter, 
    plot_evaluation_comparison,
    plot_comparison_chart
)
from .response_count import ResponseCountEvaluator
from .creative_writing_v3 import CreativeWritingV3Evaluator
from .ngram import NgramEvaluator
from .base import BaseEvaluator
from .factuality import FactualityEvaluator
from .joke_quality import JokeQualityEvaluator

__all__ = [
    "DiversityEvaluator", 
    "TTCTEvaluator", 
    "CreativityIndexEvaluator", 
    "LengthEvaluator",
    "ResponseCountEvaluator",
    "CreativeWritingV3Evaluator",
    "ComparisonPlotter", 
    "plot_evaluation_comparison",
    "NgramEvaluator",
    "BaseEvaluator",
    "plot_comparison_chart"
    "FactualityEvaluator",
    "JokeQualityEvaluator"
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
    else:
        raise ValueError(f"Evaluator {metric} not found. Available evaluators: diversity, ttct, creativity_index, length, response_count, creative_writing_v3, ngram, factuality")