from typing import Dict, List, Any, Optional
import json
from .base import BaseEvaluator, EvalResult
from verbalized_sampling.llms import get_model
from pydantic import BaseModel

class FluencyCriteria(BaseModel):
    score: int
    justification: str

class FlexibilityCriteria(BaseModel):
    score: int
    justification: str

class OriginalityCriteria(BaseModel):
    score: int
    justification: str

class ElaborationCriteria(BaseModel):
    score: int
    justification: str
    
class OverallCriteria(BaseModel):
    creativity_score: float
    normalized_score: float
    
class TTCTCriteria(BaseModel):
    fluency: FluencyCriteria
    flexibility: FlexibilityCriteria
    originality: OriginalityCriteria
    elaboration: ElaborationCriteria
    overall: OverallCriteria
    
class TTCTEvaluator(BaseEvaluator):
    """Comprehensive Torrance Tests of Creative Thinking evaluator in a single LLM call."""
    
    def __init__(self, judge_model: str = "openai/gpt-4.1", num_workers=64):
        super().__init__("ttct", num_workers=num_workers)
        self.judge_model = get_model(judge_model, method="direct", config={}, strict_json=True)
    
    def compute_instance_metric(self, prompt: str, response: str) -> Dict[str, float]:
        
        evaluation_prompt = self._create_evaluation_prompt(prompt, response)
        
        # Get evaluation from judge model
        messages = [{"role": "user", "content": evaluation_prompt}]
        result = self.judge_model._chat(messages)

        try:
            result_in_schema = json.loads(result)
            return result_in_schema
        except json.JSONDecodeError:
            print(f"Error: {result}")
            return None
    
    def _create_evaluation_prompt(self, prompt: str, response: str) -> str:
        
        return f"""You are an expert evaluator using the Torrance Tests of Creative Thinking (TTCT) framework. Evaluate the following responses across four key dimensions of creativity.
REQUEST PROMPT:
{prompt}

RESPONSES TO EVALUATE:
{response}

EVALUATION RUBRICS:

## 1. FLUENCY
**Definition**: Quality and relevance of the response content that demonstrates productive thinking.

**Scoring Criteria (1-5 scale)**:
- **5 (Exceptional)**: Response is highly meaningful, directly relevant, and demonstrates clear understanding. Ideas are well-expressed and coherent.
- **4 (Strong)**: Response is meaningful and relevant with only minor gaps. Good productive content.
- **3 (Adequate)**: Response has some meaningful and relevant elements. Some off-topic or unclear aspects.
- **2 (Limited)**: Response has few meaningful and relevant elements. Largely tangential or poorly developed.
- **1 (Poor)**: Response fails to meet basic relevance and meaning criteria.

**Evaluate**: Assess meaningfulness, relevance to the prompt, and overall coherence of the single response.

## 2. FLEXIBILITY
**Definition**: Breadth of thinking and conceptual diversity demonstrated within the response.

**Scoring Criteria (1-5 scale)**:
- **5 (Exceptional)**: Response demonstrates multiple distinct perspectives, approaches, or conceptual frameworks. Clear evidence of diverse thinking styles.
- **4 (Strong)**: Response shows good variety in approaches or themes with some conceptual diversity.
- **3 (Adequate)**: Response shows moderate variety but tends toward similar concepts or approaches.
- **2 (Limited)**: Response follows a narrow pattern with little conceptual variation.
- **1 (Poor)**: Response demonstrates rigid, single-approach thinking with no conceptual diversity.

**Evaluate**: Identify different themes/approaches within the response, note perspective shifts, assess thinking flexibility.

## 3. ORIGINALITY
**Definition**: Statistical rarity and uniqueness of the response compared to what would be typical or expected.

**Scoring Criteria (1-5 scale)**:
- **5 (Exceptional)**: Highly unique, unexpected response that shows novel connections. Ideas that would be statistically rare.
- **4 (Strong)**: Response contains several uncommon or surprising elements. Good departure from conventional thinking.
- **3 (Adequate)**: Response mixes common and less common ideas. Some original elements but also predictable ones.
- **2 (Limited)**: Response is mostly conventional with occasional less common elements.
- **1 (Poor)**: Highly predictable, common response that most people would generate.

**Evaluate**: Compare against typical responses, assess novelty and unexpectedness, identify unique connections or perspectives.

## 4. ELABORATION
**Definition**: Degree of detail, development, and descriptive richness in the response.

**Scoring Criteria (1-5 scale)**:
- **5 (Exceptional)**: Rich, detailed development with vivid descriptions, specific examples, sensory details, and thorough exploration of ideas.
- **4 (Strong)**: Good level of detail and development. Ideas are expanded with supporting elements and specificity.
- **3 (Adequate)**: Moderate detail. Basic ideas are somewhat developed but could be richer.
- **2 (Limited)**: Minimal detail beyond basic concepts. Ideas are stated but not developed or elaborated.
- **1 (Poor)**: Bare-bones response with little to no elaboration or detail.

**Evaluate**: Assess depth of development, richness of description, use of specific details, examples, and sensory elements.

EVALUATION INSTRUCTIONS:
1. Read the response carefully
2. For each dimension, provide a score (1-5) with clear justification based on the single response
3. Calculate an overall creativity score as weighted average:
   - Fluency: 20%
   - Flexibility: 30% 
   - Originality: 30%
   - Elaboration: 20%

REQUIRED JSON OUTPUT FORMAT:
{{
    "fluency": {{
        "score": <1-5>,
        "justification": "Why this score was assigned..."
    }},
    "flexibility": {{
        "score": <1-5>,
        "justification": "Why this score was assigned..."
    }},
    "originality": {{
        "score": <1-5>,
        "justification": "Why this score was assigned..."
    }},
    "elaboration": {{
        "score": <1-5>,
        "justification": "Why this score was assigned..."
    }},
    "overall": {{
        "creativity_score": <weighted average>,
        "normalized_score": <0-1>,
    }},
}}

Be thorough, specific, and evidence-based in your analysis. Provide concrete examples from the responses to support your scores."""

    def aggregate_metrics(self, instance_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate instance-level metrics into overall metrics."""
        # Filter out any empty metrics
        instance_metrics = [metric for metric in instance_metrics if metric]
        
        if not instance_metrics:
            return {
                "fluency": 0.0,
                "flexibility": 0.0,
                "originality": 0.0,
                "elaboration": 0.0,
                "overall": 0.0
            }
        return {
            "fluency": sum(metric["fluency"]["score"] for metric in instance_metrics) / len(instance_metrics),
            "flexibility": sum(metric["flexibility"]["score"] for metric in instance_metrics) / len(instance_metrics),
            "originality": sum(metric["originality"]["score"] for metric in instance_metrics) / len(instance_metrics),
            "elaboration": sum(metric["elaboration"]["score"] for metric in instance_metrics) / len(instance_metrics),
            "overall": sum(metric["overall"]["creativity_score"] for metric in instance_metrics) / len(instance_metrics)
        }
    
    def evaluate(self, prompts: List[str], responses: List[str], 
                metadata: Optional[Dict[str, Any]] = None) -> EvalResult:
        """Evaluate responses using TTCT framework."""
        if metadata is None:
            metadata = {}
            
        metadata.update({
            "evaluation_framework": "TTCT",
            "judge_model": self.judge_model.model_name,
            "num_responses": len(responses)
        })
        
        return super().evaluate(prompts, responses, metadata)
