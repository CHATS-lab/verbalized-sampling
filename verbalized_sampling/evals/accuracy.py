"""
Accuracy evaluator for math and other exact-answer tasks.

This evaluator computes the accuracy of model responses by comparing
extracted answers against reference answers, using math-specific
answer extraction and evaluation logic.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
from pathlib import Path
from rich.console import Console

from .base import BaseEvaluator, EvalResult
from ..tasks.math_eval import evaluate_math_answer, extract_boxed_answer

console = Console()

@dataclass
class AccuracyMetrics:
    """Metrics for accuracy evaluation."""
    accuracy: float
    correct_count: int
    total_count: int
    per_instance_results: List[Dict[str, Any]]


class AccuracyEvaluator(BaseEvaluator):
    """
    Evaluator for computing accuracy of model responses.

    This evaluator is specifically designed for tasks where there are
    correct/incorrect answers, such as math problems, factual questions, etc.

    For math tasks, it uses the math_eval module to extract boxed answers
    and evaluate them properly.
    """

    def __init__(self, task_type: str = "math", **kwargs):
        """
        Initialize accuracy evaluator.

        Args:
            task_type: Type of task ('math', 'factual', etc.)
                      Determines the evaluation method used
        """
        super().__init__(name="accuracy", **kwargs)
        self.task_type = task_type

    def compute_instance_metric(self, prompt: str, response: str) -> Dict[str, float]:
        """
        Compute metrics for a single instance.
        Note: This method is required by BaseEvaluator but not used in our custom evaluate method.
        """
        # This is a placeholder - we override the evaluate method instead
        return {}

    def aggregate_metrics(self, instance_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Aggregate instance-level metrics into overall metrics.
        Note: This method is required by BaseEvaluator but not used in our custom evaluate method.
        """
        # This is a placeholder - we override the evaluate method instead
        return {}

    def evaluate(self, prompts: List[str], responses: List[Dict],
                 reference_answers: Optional[List[str]] = None,
                 **kwargs) -> EvalResult:
        """
        Evaluate accuracy of responses.

        Args:
            prompts: List of input prompts
            responses: List of model responses (dict format with 'text' key)
            reference_answers: List of correct answers
            **kwargs: Additional arguments (may contain 'metadata' with answers)

        Returns:
            EvalResult with accuracy metrics
        """
        if len(prompts) != len(responses):
            raise ValueError(f"Number of prompts ({len(prompts)}) must match number of responses ({len(responses)})")

        # Extract text from response objects
        response_texts = []
        for response in responses:
            if isinstance(response, dict):
                response_texts.append(response.get('text', str(response)))
            else:
                response_texts.append(str(response))

        # For accuracy evaluation, we need reference answers
        # Since the pipeline doesn't automatically provide them, we need to handle this gracefully
        if reference_answers is None:
            # Try to get answers from metadata or generate a warning
            metadata = kwargs.get('metadata', {})
            if isinstance(metadata, dict) and 'answers' in metadata:
                reference_answers = metadata['answers']
            else:
                # For now, we'll create a placeholder that marks all as incorrect
                # In production, this should be handled by the pipeline providing answers
                console.print("[yellow]Warning: No reference answers provided for accuracy evaluation. All responses will be marked as incorrect.[/yellow]")
                reference_answers = ["UNKNOWN"] * len(response_texts)

        if len(response_texts) != len(reference_answers):
            raise ValueError(f"Number of responses ({len(response_texts)}) must match number of reference answers ({len(reference_answers)})")

        # Evaluate each response
        instance_metrics = []
        correct_count = 0

        for i, (response_text, reference) in enumerate(zip(response_texts, reference_answers)):
            result = self._evaluate_single_response(response_text, reference, i)
            instance_metrics.append(result)

            if result['is_correct']:
                correct_count += 1

        # Calculate overall metrics
        accuracy = correct_count / len(responses) if responses else 0.0

        overall_metrics = {
            'accuracy': accuracy,
            'correct_count': correct_count,
            'total_count': len(responses),
            'error_rate': 1.0 - accuracy,
            'success_rate': accuracy
        }

        return EvalResult(
            overall_metrics=overall_metrics,
            instance_metrics=instance_metrics,
            metadata=kwargs.get('metadata', {})
        )

    def _evaluate_single_response(self, response: str, reference_answer: str, index: int) -> Dict[str, Any]:
        """
        Evaluate a single response against reference answer.

        Args:
            response: Model response
            reference_answer: Correct answer
            index: Response index

        Returns:
            Dictionary with evaluation results for this instance
        """
        try:
            if self.task_type == "math":
                # Use math-specific evaluation
                is_correct = evaluate_math_answer(response, reference_answer, "math")
                extracted_answer = extract_boxed_answer(response)
            else:
                # Simple string comparison for other tasks
                extracted_answer = response.strip()
                is_correct = extracted_answer.lower() == reference_answer.lower().strip()

            return {
                'index': index,
                'is_correct': is_correct,
                'extracted_answer': extracted_answer,
                'reference_answer': reference_answer,
                'response_length': len(response),
                'evaluation_method': self.task_type
            }

        except Exception as e:
            # If evaluation fails, mark as incorrect but include error info
            return {
                'index': index,
                'is_correct': False,
                'extracted_answer': '',
                'reference_answer': reference_answer,
                'response_length': len(response),
                'evaluation_method': self.task_type,
                'error': str(e)
            }

    def save_results(self, result: EvalResult, output_path: Path) -> None:
        """Save evaluation results to file."""
        output_data = {
            'evaluator_type': 'accuracy',
            'task_type': self.task_type,
            'overall_metrics': result.overall_metrics,
            'instance_metrics': result.instance_metrics,
            'summary': {
                'total_evaluated': len(result.instance_metrics),
                'accuracy': result.overall_metrics['accuracy'],
                'correct_count': result.overall_metrics['correct_count']
            }
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

    def load_results(self, input_path: Path) -> EvalResult:
        """Load evaluation results from file."""
        with open(input_path, 'r') as f:
            data = json.load(f)

        return EvalResult(
            overall_metrics=data['overall_metrics'],
            instance_metrics=data['instance_metrics'],
            metadata=data.get('metadata', {})
        )

    def get_metric_names(self) -> List[str]:
        """Get list of metric names this evaluator provides."""
        return ['accuracy', 'correct_count', 'total_count', 'error_rate', 'success_rate']