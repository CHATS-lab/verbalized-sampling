# Creative Writing v3
# https://eqbench.com/creative_writing.html

from typing import Dict, List, Any, Optional
import json
import re
from .base import BaseEvaluator, EvalResult
from .assets.cwv3_rubrics import JUDGE_RUBRIC
from verbalized_sampling.llms import get_model

SCORE_RANGE_MIN = 0
SCORE_RANGE_MAX = 20

def normalize_score(score: float, min_val: float = SCORE_RANGE_MIN, max_val: float = SCORE_RANGE_MAX) -> float:
    """Normalize score from 0-20 range to 0-1 range."""
    # First clamp the score to be within the original range
    clamped_score = max(min_val, min(max_val, score))
    # Then map from 0-20 to 0-1
    normalized_score = clamped_score / max_val
    return normalized_score

def parse_judge_scores_creative(judge_model_response: str) -> Dict[str, float]:
    """Parse judge scores from the model response."""
    scores = {}

    # First, extract only the [Scores] section to avoid parsing the [Analysis] section
    scores_section = ""
    if "[Scores]" in judge_model_response:
        scores_start = judge_model_response.find("[Scores]")
        scores_section = judge_model_response[scores_start:]
    else:
        # If no [Scores] section found, use the entire response
        scores_section = judge_model_response

    # Parse scores using multiple regex patterns
    # Pattern 1: Metric: Score or Metric: Score X
    score_pattern1 = r'(.*?):\s*(?:Score\s+)?(-?\d+(?:\.\d+)?)'
    # Pattern 2: Metric: [Score]
    score_pattern2 = r'(.*?):\s*\[(-?\d+(?:\.\d+)?)\]'
    
    # Combine both patterns
    matches1 = re.findall(score_pattern1, scores_section)
    matches2 = re.findall(score_pattern2, scores_section)
    
    # Process matches from both patterns
    for matches in [matches1, matches2]:
        for match in matches:
            metric_name = match[0].strip()
            
            # Skip if this is the [Analysis] or [Scores] header
            if metric_name.startswith("[") or metric_name.lower() in ["analysis", "scores"]:
                continue
                
            # Normalize metric name for file system compatibility
            normalized_name = normalize_metric_name(metric_name)
            
            try:
                score = float(match[1])
                # Normalize score to be within 0-20 range
                normalized_score = normalize_score(score)
                scores[normalized_name] = normalized_score
            except ValueError:
                # Skip if score cannot be converted to float
                continue

    return scores

def normalize_metric_name(metric_name: str) -> str:
    """Normalize metric name for file system compatibility and consistency."""
    # Remove special characters and replace with underscores
    normalized = re.sub(r'[^\w\s-]', '', metric_name)
    # Replace spaces and hyphens with underscores
    normalized = re.sub(r'[\s-]+', '_', normalized)
    # Remove multiple consecutive underscores
    normalized = re.sub(r'_+', '_', normalized)
    # Remove leading/trailing underscores
    normalized = normalized.strip('_')
    # Convert to lowercase for consistency
    normalized = normalized.lower()
    return normalized

class CreativeWritingV3Evaluator(BaseEvaluator):
    """Evaluator for creative writing using expert judge scoring on 10 key metrics."""
    instance_plot_metrics = [
        ("imagery_and_descriptive_quality", "violin"),
        ("nuanced_characters", "violin"),
        ("emotionally_complex", "violin"),
        ("elegant_prose", "violin"),
        ("well_earned_lightness_or_darkness", "violin"),
        ("emotionally_engaging", "violin"),
        ("consistent_voicetone_of_writing", "violin"),
        ("sentences_flow_naturally", "violin"),
        ("overall_reader_engagement", "violin"),
        ("Average_Score", "violin")
    ]
    aggregate_plot_metrics = [
        "avg_score",
    ]
    key_plot_metrics = [
        ("avg_score", "Quality (LLM-as-Judge)"),
    ]
    
    def __init__(self, judge_model: str = "anthropic/claude-3.7-sonnet", num_workers: int = 16):
        # Reduce number of workers to 8 to avoid rate limiting since calling claude
        super().__init__("creative_writing_v3", num_workers=16)
        
        self.judge_model = get_model(judge_model, method="direct", config={"temperature": 0.0})
        
    def compute_instance_metric(self, prompt: str, response: Dict) -> Dict[str, float]:
        """Compute creative writing metrics for a single prompt-response pair."""
        
        # Create evaluation prompt using the rubric
        evaluation_prompt = JUDGE_RUBRIC.format(
            writing_prompt=prompt,
            response=response['text']
        )
        
        # Get evaluation from judge model
        messages = [{"role": "user", "content": evaluation_prompt}]
        judge_response = self.judge_model._chat(messages)
        
        if judge_response is None:
            return {"Average_Score": 0.0}
        # Parse scores from judge response
        scores = parse_judge_scores_creative(judge_response)
        
        # Calculate and normalize average score
        if scores:
            avg_score = sum(scores.values()) / len(scores)
            scores["Average_Score"] = avg_score
        else:
            scores["Average_Score"] = 0.0
        
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
        """Evaluate creative writing responses."""
        if metadata is None:
            metadata = {}
            
        metadata.update({
            "evaluation_framework": "Creative Writing V3",
            "judge_model": self.judge_model.model_name,
            "num_responses": len(responses),
            "score_range": f"{SCORE_RANGE_MIN}-{SCORE_RANGE_MAX}"
        })
        
        return super().evaluate(prompts, responses, metadata)