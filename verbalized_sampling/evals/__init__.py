from .diversity import DiversityEvaluator

__all__ = ["DiversityEvaluator"]

def get_evaluator(metric: str):
    if metric == "diversity":
        return DiversityEvaluator()
    else:
        raise ValueError(f"Evaluator {metric} not found")