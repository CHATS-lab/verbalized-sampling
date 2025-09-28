# Joke Quality Evaluator
# LLM-as-Judge evaluation for joke quality

from typing import Dict, List, Any, Optional
import json
import re
from .base import BaseEvaluator, EvalResult
from verbalized_sampling.llms import get_model

SCORE_RANGE_MIN = 0
SCORE_RANGE_MAX = 5

JUDGE_PROMPT = """You will receive:
1. The original joke prompt (may or may not contain a topic).
2. The model-generated joke.

Your task is to evaluate the joke based on three qualitative metrics.

Evaluation rules:
- If the prompt includes a topic (e.g., "octopus," "coffee"), check whether the joke is on-topic and score Relevance from 0–5.
- If the prompt does not include a topic (e.g., "Tell me a joke"), automatically assign Relevance = 5.
- A good joke should use at least one recognizable comedic device (pun, irony, exaggeration, reversal, absurd logic, etc.).
- Assign scores on a 0–5 scale (0 = very poor, 5 = excellent) for each dimension:
  - Relevance (0–5): How well does the joke address the topic (or 5 if no topic given).
  - Comedic Device (0–5): How clearly does the joke use a humor mechanism.
  - Humor Quality (0–5): How funny, witty, or clever is the joke overall.

Output format:
Return a JSON object in the following format:
{{
  "Relevance": <int>,
  "Comedic Device": <int>,
  "Humor Quality": <int>
}}

Input format:
Prompt: {prompt}
Generated joke: {joke}"""


def normalize_score(score: float, min_val: float = SCORE_RANGE_MIN, max_val: float = SCORE_RANGE_MAX) -> float:
    """Normalize score from 0-5 range to 0-1 range."""
    # First clamp the score to be within the original range
    clamped_score = max(min_val, min(max_val, score))
    # Then map from 0-5 to 0-1
    normalized_score = clamped_score / max_val
    return normalized_score


def parse_judge_scores_joke(judge_model_response: str) -> Dict[str, float]:
    """Parse judge scores from the model response."""
    scores = {}
    
    # Try to parse as JSON first
    try:
        # Look for JSON object in the response
        json_match = re.search(r'\{[^}]*\}', judge_model_response)
        if json_match:
            json_str = json_match.group()
            parsed = json.loads(json_str)
            
            # Map the expected keys and normalize scores
            expected_keys = ["Relevance", "Comedic Device", "Humor Quality"]
            for key in expected_keys:
                if key in parsed:
                    try:
                        score = float(parsed[key])
                        normalized_score = normalize_score(score)
                        # Normalize key name for consistency
                        normalized_key = key.lower().replace(' ', '_')
                        scores[normalized_key] = normalized_score
                    except (ValueError, TypeError):
                        continue
            
            return scores
    except json.JSONDecodeError:
        pass
    
    # Fallback: parse using regex patterns
    patterns = [
        r'(?:Relevance|relevance).*?(\d+(?:\.\d+)?)',
        r'(?:Comedic Device|comedic_device|comedic device).*?(\d+(?:\.\d+)?)', 
        r'(?:Humor Quality|humor_quality|humor quality).*?(\d+(?:\.\d+)?)'
    ]
    
    keys = ['relevance', 'comedic_device', 'humor_quality']
    
    for i, pattern in enumerate(patterns):
        match = re.search(pattern, judge_model_response, re.IGNORECASE)
        if match:
            try:
                score = float(match.group(1))
                normalized_score = normalize_score(score)
                scores[keys[i]] = normalized_score
            except (ValueError, TypeError):
                continue
    
    return scores


class JokeQualityEvaluator(BaseEvaluator):
    """Evaluator for joke quality using LLM-as-Judge scoring on 3 key metrics."""
    
    instance_plot_metrics = [
        ("relevance", "violin"),
        ("comedic_device", "violin"),
        ("humor_quality", "violin"),
        ("average_score", "violin")
    ]
    
    aggregate_plot_metrics = [
        "avg_score",
    ]
    
    key_plot_metrics = [
        ("avg_score", "Joke Quality (LLM-as-Judge)"),
    ]
    
    def __init__(self, judge_model: str = "anthropic/claude-3.7-sonnet", num_workers: int = 16):
        super().__init__("joke_quality", num_workers=num_workers)
        self.judge_model = get_model(judge_model, method="direct", config={"temperature": 0.0})
        
    def compute_instance_metric(self, prompt: str, response: Dict) -> Dict[str, float]:
        """Compute joke quality metrics for a single prompt-response pair."""
        
        # Create evaluation prompt using the rubric
        evaluation_prompt = JUDGE_PROMPT.format(
            prompt=prompt,
            joke=response['text']
        )
        
        # Get evaluation from judge model
        messages = [{"role": "user", "content": evaluation_prompt}]
        judge_response = self.judge_model._chat(messages)
        
        if judge_response is None:
            return {"average_score": 0.0}
            
        # Parse scores from judge response
        scores = parse_judge_scores_joke(judge_response)
        
        # Calculate and normalize average score
        if scores:
            avg_score = sum(scores.values()) / len(scores)
            scores["average_score"] = avg_score
        else:
            scores["average_score"] = 0.0
        
        return scores
    
    def aggregate_metrics(self, instance_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate instance-level metrics into overall metrics."""
        # Filter out any empty metrics and remove debug fields
        filtered_metrics = []
        for metric in instance_metrics:
            if metric:
                # Create a copy without debug fields
                clean_metric = {k: v for k, v in metric.items() if not k.startswith("_")}
                if clean_metric:  # Only add if there are actual scores
                    filtered_metrics.append(clean_metric)
        
        if not filtered_metrics:
            return {}
        
        from .base import calculate_stats
        
        # Get all unique metric names across all instances
        all_metric_names = set()
        for metric in filtered_metrics:
            all_metric_names.update(metric.keys())
        
        # Calculate averages and std for each metric
        aggregated = {}
        for metric_name in all_metric_names:
            values = [metric.get(metric_name, 0.0) for metric in filtered_metrics if metric_name in metric]
            if values:
                stats = calculate_stats(values)
                # Normalize the aggregated average
                avg_key = f"avg_{metric_name.lower().replace(' ', '_')}"
                std_key = f"std_{metric_name.lower().replace(' ', '_')}"
                aggregated[avg_key] = stats["mean"]
                aggregated[std_key] = stats["std"]

        if aggregated:  # Only calculate average if there are aggregated metrics
            # Calculate overall average from the mean metrics (excluding std metrics)
            mean_metrics = {k: v for k, v in aggregated.items() if k.startswith("avg_")}
            if mean_metrics:
                overall_stats = calculate_stats(list(mean_metrics.values()))
                aggregated["avg_score"] = overall_stats["mean"]
                aggregated["std_score"] = overall_stats["std"]
            else:
                aggregated["avg_score"] = 0.0
                aggregated["std_score"] = 0.0
        else:
            aggregated["avg_score"] = 0.0
            aggregated["std_score"] = 0.0
        
        return aggregated
    
    def evaluate(self, prompts: List[str], responses: List[str], 
                metadata: Optional[Dict[str, Any]] = None) -> EvalResult:
        """Evaluate joke quality responses."""
        if metadata is None:
            metadata = {}
            
        metadata.update({
            "evaluation_framework": "Joke Quality LLM-as-Judge",
            "judge_model": self.judge_model.model_name,
            "num_responses": len(responses),
            "score_range": f"{SCORE_RANGE_MIN}-{SCORE_RANGE_MAX}"
        })
        
        return super().evaluate(prompts, responses, metadata)