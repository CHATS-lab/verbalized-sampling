from .diversity import DiversityEvaluator

__all__ = ["DiversityEvaluator"]

def get_evaluator(metric: str, **kwargs):
    if metric == "diversity":
        return DiversityEvaluator(**kwargs)
    else:
        raise ValueError(f"Evaluator {metric} not found")