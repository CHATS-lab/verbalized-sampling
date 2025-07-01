from typing import Dict, List, Any, Optional
import json
from .base import BaseEvaluator, EvalResult
from verbalized_sampling.llms import get_model
from pydantic import BaseModel
import os
from openai import OpenAI

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
    
# class OverallCriteria(BaseModel):
#     creativity_score: float
#     normalized_score: float
    
class TTCTCriteria(BaseModel):
    fluency: FluencyCriteria
    flexibility: FlexibilityCriteria
    originality: OriginalityCriteria
    elaboration: ElaborationCriteria
    # overall: OverallCriteria
    
class TTCTEvaluator(BaseEvaluator):
    """Comprehensive Torrance Tests of Creative Thinking evaluator in a single LLM call."""
    
    instance_plot_metrics = [
        ("fluency.score", "violin"),
        ("flexibility.score", "violin"),
        ("originality.score", "violin"),
        ("elaboration.score", "violin"),
    ]

    aggregate_plot_metrics = [
        "avg_fluency",
        "avg_flexibility",
        "avg_originality",
        "avg_elaboration",
        "avg_overall",
    ]
    
    key_plot_metrics = [
        ("avg_normalized_overall", "Quality (TTCT)")
    ]

    def __init__(self, judge_model: str = "gpt-4.1", num_workers=64):
        super().__init__("ttct", num_workers=num_workers)
        # self.judge_model = get_model(judge_model, method="direct", config={}, strict_json=True)
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.judge_model = judge_model

    def parse_response(self, response) -> Dict[str, Any]:
        """Parse the response from the Pydantic model into a dictionary format."""
        # print(f"Response: {response}")
        
        parsed_response = {}
        for criteria in ["fluency", "flexibility", "originality", "elaboration"]:
            criteria_obj = getattr(response, criteria)
            parsed_response[criteria] = {
                "score": criteria_obj.score,
                "justification": criteria_obj.justification
            }
        # print(f"Parsed response: {parsed_response}")

        return parsed_response

    def _chat_with_format(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        completion = self.client.beta.chat.completions.parse(
            model=self.judge_model,
            messages=messages,
            temperature=0.1,
            response_format=TTCTCriteria,
        )
        message = completion.choices[0].message

        if message.parsed:
            parsed_response = self.parse_response(message.parsed)
            # print(f"Structured Output Response:\n" + "\n".join(str(resp) for resp in parsed_response))
            return parsed_response
        else:
            raise ValueError(message.refusal)
        
    
    def compute_instance_metric(self, prompt: Any, response: Dict) -> Dict[str, float]:
        evaluation_prompt = self._create_evaluation_prompt(prompt, response)
            
        # Get evaluation from judge model
        messages = [{"role": "user", "content": evaluation_prompt}]
        result = self._chat_with_format(messages)

        try:
            if isinstance(result, str):
                result_in_schema = json.loads(result)
            else:
                result_in_schema = result
        except json.JSONDecodeError:
            print(f"Error: Failed to parse JSON response from judge model: {result}")
            return None

        # Calculate overall creativity score as weighted average
        # Weights as specified in the prompt:
        # Fluency: 25%, Flexibility: 25%, Originality: 25%, Elaboration: 25%
        weights = {
            "fluency": 0.25,
            "flexibility": 0.25,
            "originality": 0.25,
            "elaboration": 0.25
        }
        
        weighted_score = sum(
            result_in_schema[aspect]["score"] * weights[aspect]
            for aspect in weights.keys()
        )
        
        # Normalize to 0-1 range (since scores are 1-5)
        normalized_score = weighted_score / 5
        
        result_in_schema["overall"] = {
            "creativity_score": weighted_score,
            "normalized_score": normalized_score
        }

        return result_in_schema
    
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

Be thorough, specific, and evidence-based in your analysis. Provide concrete examples from the responses to support your scores.

Return the output in JSON format with keys: "fluency", "flexibility", "originality", "elaboration". Each key must include:
- 'score': the score (1-5)
- 'justification': the justification for the score
Output ONLY the JSON object, no explanations or extra text.
"""

    def aggregate_metrics(self, instance_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate instance-level metrics into overall metrics."""
        
        if not instance_metrics:
            return {
                "fluency": 0.0, "std_fluency": 0.0,
                "flexibility": 0.0, "std_flexibility": 0.0,
                "originality": 0.0, "std_originality": 0.0,
                "elaboration": 0.0, "std_elaboration": 0.0,
                "overall": 0.0, "std_overall": 0.0,
                "normalized_overall": 0.0, "std_normalized_overall": 0.0,
            }
        
        from .base import calculate_stats
        
        # Extract values for each metric
        fluency_values = [metric["fluency"]["score"] for metric in instance_metrics]
        flexibility_values = [metric["flexibility"]["score"] for metric in instance_metrics]
        originality_values = [metric["originality"]["score"] for metric in instance_metrics]
        elaboration_values = [metric["elaboration"]["score"] for metric in instance_metrics]
        overall_values = [metric["overall"]["creativity_score"] for metric in instance_metrics]
        normalized_values = [metric["overall"]["normalized_score"] for metric in instance_metrics]
        
        # Calculate stats for each metric
        fluency_stats = calculate_stats(fluency_values)
        flexibility_stats = calculate_stats(flexibility_values)
        originality_stats = calculate_stats(originality_values)
        elaboration_stats = calculate_stats(elaboration_values)
        overall_stats = calculate_stats(overall_values)
        normalized_stats = calculate_stats(normalized_values)
        
        return {
            # Means (backward compatible)
            "avg_fluency": fluency_stats["mean"],
            "avg_flexibility": flexibility_stats["mean"],
            "avg_originality": originality_stats["mean"],
            "avg_elaboration": elaboration_stats["mean"],
            "avg_overall": overall_stats["mean"],
            "avg_normalized_overall": normalized_stats["mean"],
            
            # Standard deviations
            "std_fluency": fluency_stats["std"],
            "std_flexibility": flexibility_stats["std"],
            "std_originality": originality_stats["std"],
            "std_elaboration": elaboration_stats["std"],
            "std_overall": overall_stats["std"],
            "std_normalized_overall": normalized_stats["std"],
        }
    
    def evaluate(self, prompts: List[str], responses: List[Dict], 
                metadata: Optional[Dict[str, Any]] = None) -> EvalResult:
        """Evaluate responses using TTCT framework."""
        if metadata is None:
            metadata = {}
            
        metadata.update({
            "evaluation_framework": "TTCT",
            "judge_model": self.judge_model,
            "num_responses": len(responses)
        })
        
        return super().evaluate(prompts, responses, metadata)
