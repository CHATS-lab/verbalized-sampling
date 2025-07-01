from typing import Dict, List, Any, Optional
import json
from .base import BaseEvaluator, EvalResult
from verbalized_sampling.llms import get_model
from pydantic import BaseModel
import os
from openai import OpenAI

class FunninessCriteria(BaseModel):
    score: int
    justification: str

class ClevernessCriteria(BaseModel):
    score: int
    justification: str

class OriginalityCriteria(BaseModel):
    score: int
    justification: str

class StructureCriteria(BaseModel):
    score: int
    justification: str
    
class JokeQualityCriteria(BaseModel):
    funniness: FunninessCriteria
    cleverness: ClevernessCriteria
    originality: OriginalityCriteria
    structure: StructureCriteria
    
class JokeQualityEvaluator(BaseEvaluator):
    """Comprehensive joke quality evaluator using humor-specific criteria."""
    
    instance_plot_metrics = [
        ("funniness.score", "violin"),
        ("cleverness.score", "violin"),
        ("originality.score", "violin"),
        ("structure.score", "violin"),
    ]

    aggregate_plot_metrics = [
        "funniness",
        "cleverness",
        "originality",
        "structure",
        "overall",
    ]
    
    key_plot_metrics = [
        ("normalized_overall", "Joke Quality")
    ]

    def __init__(self, judge_model: str = "gpt-4o", num_workers=64):
        super().__init__("joke_quality", num_workers=num_workers)
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.judge_model = judge_model

    def parse_response(self, response) -> Dict[str, Any]:
        """Parse the response from the Pydantic model into a dictionary format."""
        parsed_response = {}
        for criteria in ["funniness", "cleverness", "originality", "structure"]:
            criteria_obj = getattr(response, criteria)
            parsed_response[criteria] = {
                "score": criteria_obj.score,
                "justification": criteria_obj.justification
            }
        return parsed_response

    def _chat_with_format(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        completion = self.client.beta.chat.completions.parse(
            model=self.judge_model,
            messages=messages,
            temperature=0.1,
            response_format=JokeQualityCriteria,
        )
        message = completion.choices[0].message

        if message.parsed:
            parsed_response = self.parse_response(message.parsed)
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

        # Calculate overall joke quality score as weighted average
        # Weights prioritizing core humor elements:
        # Funniness: 40%, Cleverness: 25%, Originality: 20%, Structure: 15%
        weights = {
            "funniness": 0.40,
            "cleverness": 0.25,
            "originality": 0.20,
            "structure": 0.15
        }
        
        weighted_score = sum(
            result_in_schema[aspect]["score"] * weights[aspect]
            for aspect in weights.keys()
        )
        
        # Normalize to 0-1 range (since scores are 1-5)
        normalized_score = weighted_score / 5
        
        result_in_schema["overall"] = {
            "joke_quality_score": weighted_score,
            "normalized_score": normalized_score
        }

        return result_in_schema
    
    def _create_evaluation_prompt(self, prompt: str, response: str) -> str:
        return f"""You are an expert comedy evaluator with deep knowledge of humor theory, joke mechanics, and what makes content genuinely funny. Evaluate the following joke response across four key dimensions of comedy quality.

REQUEST PROMPT:
{prompt}

JOKE RESPONSE TO EVALUATE:
{response}

EVALUATION RUBRICS:

## 1. FUNNINESS
**Definition**: The core measure of humor - how likely this joke is to make people laugh or smile.

**Scoring Criteria (1-5 scale)**:
- **5 (Hilarious)**: Genuinely laugh-out-loud funny. Would make most people laugh or at least smile broadly. Strong comedic impact.
- **4 (Very Funny)**: Clearly humorous with good comedic effect. Would elicit chuckles or smiles from most people.
- **3 (Moderately Funny)**: Mildly amusing. Some people might find it funny, others might just appreciate the attempt.
- **2 (Slightly Funny)**: Barely registers as humorous. Might get a polite smile but lacks real comedic impact.
- **1 (Not Funny)**: Fails to be humorous. No comedic effect, may even be awkward or unfunny.

**Evaluate**: Consider universal appeal, comedic timing, and whether it genuinely evokes laughter or amusement.

## 2. CLEVERNESS
**Definition**: The wit, intelligence, and sophistication demonstrated in the joke's construction and concept.

**Scoring Criteria (1-5 scale)**:
- **5 (Brilliant)**: Demonstrates exceptional wit, wordplay, or intellectual humor. Shows sophisticated understanding of language, concepts, or situations.
- **4 (Very Clever)**: Shows good wit and intelligence. Clear evidence of thoughtful construction or clever observations.
- **3 (Moderately Clever)**: Some clever elements but not consistently witty. Mix of smart and obvious humor.
- **2 (Slightly Clever)**: Minimal wit or intelligence. Mostly obvious or simple humor with occasional clever moments.
- **1 (Not Clever)**: Lacks wit or intelligence. Purely simplistic or obvious humor with no clever elements.

**Evaluate**: Assess wordplay, puns, double meanings, cultural references, intellectual depth, and sophisticated humor techniques.

## 3. ORIGINALITY
**Definition**: How unique, fresh, and unexpected the joke is compared to common or overused jokes.

**Scoring Criteria (1-5 scale)**:
- **5 (Highly Original)**: Completely fresh and unexpected. Novel approach or concept that stands out from typical jokes.
- **4 (Very Original)**: Mostly original with creative elements. Avoids common joke formats while being inventive.
- **3 (Moderately Original)**: Mix of original and familiar elements. Some fresh aspects but may follow predictable patterns.
- **2 (Slightly Original)**: Mostly familiar with minor variations. Based on common joke formats with small twists.
- **1 (Not Original)**: Completely predictable or overused. Relies on tired tropes, clichés, or very common joke formats.

**Evaluate**: Compare against typical jokes in this category, assess uniqueness of approach, and identify novel elements.

## 4. STRUCTURE
**Definition**: The technical quality of joke construction, including setup, punchline, timing, and delivery.

**Scoring Criteria (1-5 scale)**:
- **5 (Excellent Structure)**: Perfect setup and punchline. Excellent timing, clear delivery, and masterful joke mechanics.
- **4 (Good Structure)**: Well-constructed with clear setup and effective punchline. Good timing and delivery.
- **3 (Adequate Structure)**: Basic joke structure present but may have minor issues with timing or delivery.
- **2 (Poor Structure)**: Weak structure with unclear setup or weak punchline. Timing or delivery issues affect impact.
- **1 (No Structure)**: Lacks proper joke structure. Confusing, poorly timed, or missing key elements like setup or punchline.

**Evaluate**: Analyze setup effectiveness, punchline strength, timing, brevity, and overall joke architecture.

EVALUATION INSTRUCTIONS:
1. Read the joke response carefully, considering both the prompt context and the humor attempt
2. For each dimension, provide a score (1-5) with clear justification based on the criteria above
3. Consider the target audience and context when evaluating appropriateness and effectiveness
4. Be specific about what works and what doesn't in your justifications
5. Calculate overall joke quality as weighted average:
   - Funniness: 40% (most important)
   - Cleverness: 25% (wit and intelligence)
   - Originality: 20% (freshness and uniqueness)
   - Structure: 15% (technical construction)

Provide concrete examples from the joke to support your scores. Consider both the immediate comedic impact and the lasting impression.

Return the output in JSON format with keys: "funniness", "cleverness", "originality", "structure". Each key must include:
- 'score': the score (1-5)
- 'justification': detailed justification for the score with specific examples

Output ONLY the JSON object, no explanations or extra text."""

    def aggregate_metrics(self, instance_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate instance-level metrics into overall metrics."""
        
        if not instance_metrics:
            return {
                "funniness": 0.0,
                "cleverness": 0.0,
                "originality": 0.0,
                "structure": 0.0,
                "overall": 0.0,
                "normalized_overall": 0.0,
            }
        return {
            "funniness": sum(metric["funniness"]["score"] for metric in instance_metrics) / len(instance_metrics),
            "cleverness": sum(metric["cleverness"]["score"] for metric in instance_metrics) / len(instance_metrics),
            "originality": sum(metric["originality"]["score"] for metric in instance_metrics) / len(instance_metrics),
            "structure": sum(metric["structure"]["score"] for metric in instance_metrics) / len(instance_metrics),
            "overall": sum(metric["overall"]["joke_quality_score"] for metric in instance_metrics) / len(instance_metrics),
            "normalized_overall": sum(metric["overall"]["normalized_score"] for metric in instance_metrics) / len(instance_metrics),
        }
    
    def evaluate(self, prompts: List[str], responses: List[Dict], 
                metadata: Optional[Dict[str, Any]] = None) -> EvalResult:
        """Evaluate jokes using comedy quality framework."""
        if metadata is None:
            metadata = {}
            
        metadata.update({
            "evaluation_framework": "Joke Quality Assessment",
            "judge_model": self.judge_model,
            "num_responses": len(responses)
        })
        
        return super().evaluate(prompts, responses, metadata)