from .diversity import DiversityEvaluator
from .quality import TTCTEvaluator
from .creativity_index import CreativityIndexEvaluator

__all__ = ["DiversityEvaluator", "TTCTEvaluator", "CreativityIndexEvaluator"]

def get_evaluator(metric: str, **kwargs):
    if metric == "diversity":
        return DiversityEvaluator(**kwargs)
    elif metric == "ttct" or metric == "quality":
        return TTCTEvaluator(**kwargs)
    elif metric == "creativity_index":
        return CreativityIndexEvaluator(**kwargs)
    else:
        raise ValueError(f"Evaluator {metric} not found. Available evaluators: diversity, ttct, creativity_index")