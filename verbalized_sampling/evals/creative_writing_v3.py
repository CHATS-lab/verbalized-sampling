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
                # Add check to ensure score <= 20
                if score <= SCORE_RANGE_MAX:
                    scores[normalized_name] = score
                # If score > 20, it's discarded/ignored
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
        "Average_Score",
    ]
    key_plot_metrics = [
        ("Average_Score", "Quality (LLM-as-Judge)"),
    ]
    
    def __init__(self, judge_model: str = "anthropic/claude-sonnet-4", num_workers: int = 64):
        super().__init__("creative_writing_v3", num_workers=num_workers)
        self.judge_model = get_model(judge_model, method="direct", config={})
        
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
        
        # Parse scores from judge response
        scores = parse_judge_scores_creative(judge_response)
        
        # Add the raw judge response for debugging if needed
        scores["Average_Score"] = sum(scores.values()) / len(scores)
        
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
        
        # Get all unique metric names across all instances
        all_metric_names = set()
        for metric in filtered_metrics:
            all_metric_names.update(metric.keys())
        
        # Calculate averages for each metric
        aggregated = {}
        for metric_name in all_metric_names:
            values = [metric.get(metric_name, 0.0) for metric in filtered_metrics if metric_name in metric]
            if values:
                aggregated[f"avg_{metric_name.lower().replace(' ', '_')}"] = sum(values) / len(values)

        if aggregated:  # Only calculate average if there are aggregated metrics
            aggregated["Average_Score"] = sum(aggregated.values()) / len(aggregated)
        else:
            aggregated["Average_Score"] = 0.0
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