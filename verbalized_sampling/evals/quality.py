from typing import Dict, List, Any, Optional
import json
from .base import BaseEvaluator, EvalResult
from verbalized_sampling.llms import get_model

class TTCTEvaluator(BaseEvaluator):
    """Comprehensive Torrance Tests of Creative Thinking evaluator in a single LLM call."""
    
    def __init__(self, judge_model: str = "gpt-4-turbo"):
        super().__init__("ttct")
        self.judge_model = get_model(judge_model, method="direct", config={})
    
    def compute_instance_metric(self, prompt: str, response: str) -> Dict[str, float]:
        
        evaluation_prompt = self._create_evaluation_prompt(prompt, response)
        
        # Get evaluation from judge model
        messages = [{"role": "user", "content": evaluation_prompt}]
        result = self.judge_model._chat([messages])[0]

        try:
            parsed_result = json.loads(result)
            return parsed_result
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "error": "Failed to parse judge response",
                "raw_response": result,
                "fluency_score": 0.0,
                "flexibility_score": 0.0,
                "originality_score": 0.0,
                "elaboration_score": 0.0,
                "creativity_score": 0.0
            }
    
    def _create_evaluation_prompt(self, prompt: str, response: str) -> str:
        
        return f"""You are an expert evaluator using the Torrance Tests of Creative Thinking (TTCT) framework. Evaluate the following responses across four key dimensions of creativity.
REQUEST PROMPT:
{prompt}

RESPONSES TO EVALUATE:
{response}

EVALUATION RUBRICS:

## 1. FLUENCY
**Definition**: Total count of meaningful, relevant responses that demonstrate productive thinking.

**Scoring Criteria (1-5 scale)**:
- **5 (Exceptional)**: All responses are highly meaningful, directly relevant, and demonstrate clear understanding. Ideas flow naturally and abundantly.
- **4 (Strong)**: Most responses are meaningful and relevant with minor gaps. Good productive output.
- **3 (Adequate)**: About half the responses are truly meaningful and relevant. Some off-topic or unclear elements.
- **2 (Limited)**: Few responses are both meaningful and relevant. Many are tangential or poorly developed.
- **1 (Poor)**: Very few or no responses meet basic relevance and meaning criteria.

**Evaluate**: Count meaningful responses, assess relevance to any implicit prompt/theme, and rate overall productive fluency.

## 2. FLEXIBILITY
**Definition**: Number of distinct categories, approaches, or conceptual shifts demonstrated across responses.

**Scoring Criteria (1-5 scale)**:
- **5 (Exceptional)**: 4+ distinct conceptual categories/approaches. Clear evidence of perspective shifts, multiple thinking styles.
- **4 (Strong)**: 3-4 distinct categories. Good variety in approaches or themes with some conceptual shifts.
- **3 (Adequate)**: 2-3 categories. Moderate variety but some clustering around similar concepts.
- **2 (Limited)**: 1-2 categories. Most responses follow similar patterns with little conceptual variation.
- **1 (Poor)**: Single category or approach. Responses are very similar in theme and execution.

**Evaluate**: Identify distinct themes/categories, note conceptual shifts, assess thinking flexibility and adaptability.

## 3. ORIGINALITY
**Definition**: Statistical rarity and uniqueness of responses compared to what would be typical or expected.

**Scoring Criteria (1-5 scale)**:
- **5 (Exceptional)**: Highly unique, unexpected responses that show novel connections. Ideas that would be statistically rare.
- **4 (Strong)**: Several uncommon or surprising elements. Good departure from conventional thinking.
- **3 (Adequate)**: Mix of common and less common ideas. Some original elements but also predictable ones.
- **2 (Limited)**: Mostly conventional responses with occasional less common elements.
- **1 (Poor)**: Highly predictable, common responses that most people would generate.

**Evaluate**: Compare against your knowledge of typical responses, assess novelty and unexpectedness, identify unique connections or perspectives.

## 4. ELABORATION
**Definition**: Degree of detail, development, and descriptive richness added beyond the basic concept.

**Scoring Criteria (1-5 scale)**:
- **5 (Exceptional)**: Rich, detailed development with vivid descriptions, specific examples, sensory details, and thorough exploration of ideas.
- **4 (Strong)**: Good level of detail and development. Ideas are expanded with supporting elements and some specificity.
- **3 (Adequate)**: Moderate detail. Basic ideas are somewhat developed but could be richer.
- **2 (Limited)**: Minimal detail beyond basic concepts. Ideas are stated but not developed or elaborated.
- **1 (Poor)**: Bare-bones responses with little to no elaboration or detail.

**Evaluate**: Assess depth of development, richness of description, use of specific details, examples, and sensory elements.

EVALUATION INSTRUCTIONS:
1. Read all responses carefully
2. For each dimension, provide:
   - Detailed analysis and evidence
   - Specific examples from the responses
   - Score (1-5) with clear justification
3. Calculate an overall creativity score as weighted average:
   - Fluency: 20%
   - Flexibility: 30% 
   - Originality: 30%
   - Elaboration: 20%

REQUIRED JSON OUTPUT FORMAT:
{{
    "fluency": {{
        "score": <1-5>,
        "meaningful_responses_count": <number>,
        "total_responses": <number>,
        "analysis": "Detailed analysis with specific examples...",
        "justification": "Why this score was assigned..."
    }},
    "flexibility": {{
        "score": <1-5>,
        "categories_identified": ["category1", "category2", ...],
        "category_count": <number>,
        "conceptual_shifts": ["shift1", "shift2", ...],
        "analysis": "Detailed analysis of variety and shifts...",
        "justification": "Why this score was assigned..."
    }},
    "originality": {{
        "score": <1-5>,
        "unique_elements": ["element1", "element2", ...],
        "commonality_assessment": "common|mixed|uncommon|rare|unique",
        "novel_connections": ["connection1", "connection2", ...],
        "analysis": "Detailed analysis of uniqueness and rarity...",
        "justification": "Why this score was assigned..."
    }},
    "elaboration": {{
        "score": <1-5>,
        "detail_level": "minimal|basic|moderate|rich|exceptional",
        "descriptive_elements": ["element1", "element2", ...],
        "development_quality": "shallow|adequate|good|thorough|exceptional",
        "analysis": "Detailed analysis of richness and development...",
        "justification": "Why this score was assigned..."
    }},
    "overall": {{
        "creativity_score": <weighted average>,
        "normalized_score": <0-1>,
        "strengths": ["strength1", "strength2", ...],
        "areas_for_improvement": ["area1", "area2", ...],
        "overall_assessment": "Comprehensive summary of creative performance..."
    }},
    "response_breakdown": [
        {{
            "response_id": 1,
            "fluency_contribution": "meaningful|unclear|irrelevant",
            "flexibility_category": "category_name",
            "originality_level": "common|uncommon|rare|unique",
            "elaboration_quality": "minimal|basic|moderate|rich|exceptional"
        }},
        ...
    ]
}}

Be thorough, specific, and evidence-based in your analysis. Provide concrete examples from the responses to support your scores."""

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
