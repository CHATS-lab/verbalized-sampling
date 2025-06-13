"""
SimpleQA: Measuring short-form factuality in large language models 
Authors: Jason Wei, Nguyen Karina, Hyung Won Chung, Yunxin Joy Jiao, Spencer Papay, Amelia Glaese, John Schulman, William Fedus
https://cdn.openai.com/papers/simpleqa.pdf
""" 

import ast
import json
import os
import random
import re
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel

from .base import BaseEvaluator, EvalResult
from verbalized_sampling.llms import get_model


GRADER_TEMPLATE = """
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and then you will grade a new example.


The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.


The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children.
Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
```
These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'm not sure, i think") are also considered incorrect.


The following are examples of NOT_ATTEMPTED predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
```
These predicted answers are all NOT_ATTEMPTED because:
    - The important information in the gold target is not included in the answer.
    - No statements in the answer contradict the gold target.


Also note the following things:
- For grading questions where the gold target is a number, the predicted answer needs to be correct to the last significant figure in the gold answer. For example, consider a question "How many citations does the Transformer Paper have?" with gold target "120k". 
    - Predicted answers "120k", "124k", and 115k" are all CORRECT. 
    - Predicted answers "100k" and "113k" are INCORRECT. 
    - Predicted answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED because they neither confirm nor contradict the gold target.
- The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
    - For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
- Do not punish predicted answers if they omit information that would be clearly inferred from the question.
    - For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include "California".
    - Consider the question "What award did A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
    - For the question "What is the height of Jason Wei in meters?", the gold target is "1.73 m". The predicted answer "1.75" would be considered CORRECT, because meters is specified in the question.
    - For the question "What is the name of Barack Obama's wife?", the gold target is "Michelle Obama". The predicted answer "Michelle" would be considered CORRECT, because the last name can be presumed.
- Do not punish for typos in people's name if it's clearly the same name. 
    - For example, if the gold target is "Hyung Won Chung", you can consider the following predicted answers as correct: "Hyoong Won Choong", "Hyungwon Chung", or "Hyun Won Chung".


Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
```
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
```

Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Just return the letters "A", "B", or "C", with no text around it.
""".strip()


CHOICE_LETTERS = ["A", "B", "C"]
CHOICE_STRINGS = ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"]
CHOICE_LETTER_TO_STRING = dict(zip(CHOICE_LETTERS, CHOICE_STRINGS))


class FactualityEvaluator(BaseEvaluator):
    def __init__(self, judge_model: str = "openai/gpt-4.1", num_workers=64):
        super().__init__("factuality", num_workers=num_workers)
        self.judge_model = get_model(judge_model, method="direct", config={}, strict_json=True)
        self.df = pd.read_csv("https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv")
        if self.df is None:
            self.df = pd.read_csv("data/simple_qa.csv")

    def get_answer_by_question(self, question):
        question = question.strip()
        for index, row in self.df.iterrows():
            if row["problem"] == question:
                return row["answer"]
        return None

    def grade_sample(self, question: str, target: str, predicted_answer: str) -> str:
        grader_prompt = GRADER_TEMPLATE.format(
            question=question,
            target=target,
            predicted_answer=predicted_answer,
        )
        
        prompt_messages = [
            {"role": "user","content": grader_prompt}
        ]
        judge_response = self.judge_model._chat(prompt_messages)
        
        match = re.search(r"(A|B|C)", judge_response)
        return match.group(0) if match else "C"  # Default to "NOT_ATTEMPTED" if no match


    def compute_instance_metric(self, prompts: Any, responses: Any) -> List[Dict[str, float]]:
        if isinstance(responses, str):
            responses = ast.literal_eval(responses)

        list_of_responses = [
            response.get('response', response)
            for response in responses
        ]
        list_of_questions = [
            response.get('prompt', response)
            for response in responses
        ]

        grade_letter_list = []
        target_list = []
        for question, response in zip(list_of_questions, list_of_responses):
            target = self.get_answer_by_question(question)
            grade_letter = self.grade_sample(question, target, response)
            grade_letter_list.append(grade_letter)
            target_list.append(target)
        
        return [
            {
                'question': question,
                'target': target,
                'predicted_answer': response,
                'grade_letter': grade_letter,
                # Metrics based on grading response
                "is_correct": grade_letter == "A",
                "is_incorrect": grade_letter == "B",
                "is_not_attempted": grade_letter == "C"
            }
            for question, target, response, grade_letter in zip(list_of_questions, target_list, list_of_responses, grade_letter_list)
        ]
    

    def aggregate_metrics(self, instance_metrics: List[List[Dict[str, float]]]) -> Dict[str, float]:
        instance_metrics = [metric for instance_list in instance_metrics for metric in instance_list if metric]
        
        aggregate_metrics = {
            "is_correct": sum(metric["is_correct"] for metric in instance_metrics) / len(instance_metrics),
            "is_incorrect": sum(metric["is_incorrect"] for metric in instance_metrics) / len(instance_metrics),
            "is_not_attempted": sum(metric["is_not_attempted"] for metric in instance_metrics) / len(instance_metrics),
        }
        aggregate_metrics["is_given_attempted"] = aggregate_metrics["is_correct"] + aggregate_metrics["is_incorrect"]
        aggregate_metrics["accuracy_given_attempted"] = (
            aggregate_metrics["is_correct"]
            / aggregate_metrics["is_given_attempted"]
            if aggregate_metrics["is_given_attempted"] > 0
            else 0
        )

        return {
            'num_is_correct': sum(metric['is_correct'] for metric in instance_metrics),
            'num_is_incorrect': sum(metric['is_incorrect'] for metric in instance_metrics),
            'num_is_not_attempted': sum(metric['is_not_attempted'] for metric in instance_metrics),
            'num_responses': len(instance_metrics),
            'accuracy_given_attempted': aggregate_metrics['accuracy_given_attempted'],
            'f1': (
                2 * aggregate_metrics['accuracy_given_attempted'] * aggregate_metrics['is_correct']
                / (aggregate_metrics['accuracy_given_attempted'] + aggregate_metrics['is_correct'])
                if (aggregate_metrics['accuracy_given_attempted'] + aggregate_metrics['is_correct']) > 0
                else 0
            )
        }

    def evaluate(self, prompts: List[str], responses: List[str], metadata: Optional[Dict[str, Any]] = None) -> EvalResult:
        if metadata is None:
            metadata = {}

        metadata.update({
            "evaluation_framework": "Factuality",
            "judge_model": self.judge_model.model_name,
            "num_responses": len(responses)
        })
        
        return super().evaluate(prompts, responses, metadata)