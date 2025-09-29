"""
Linguistic evaluation metrics for dialogue simulation.

This module provides evaluators for linguistic diversity, readability,
and dialogue-specific language features.
"""

import re
import numpy as np
from typing import Dict, List, Any, Optional, Set
from collections import Counter
import nltk

# Optional textstat import for readability metrics
try:
    from textstat import flesch_reading_ease, flesch_kincaid_grade
    HAS_TEXTSTAT = True
except ImportError:
    HAS_TEXTSTAT = False
    # Dummy functions if textstat not available
    def flesch_reading_ease(text):
        return 0.0
    def flesch_kincaid_grade(text):
        return 0.0

from verbalized_sampling.analysis.evals.base import BaseEvaluator, EvalResult as EvaluationResult


class DialogueLinguisticEvaluator(BaseEvaluator):
    """Evaluator for linguistic features in dialogue responses."""

    def __init__(self, **kwargs):
        """Initialize linguistic evaluator."""
        super().__init__(name="dialogue_linguistic", **kwargs)

        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

    def compute_instance_metric(self, prompt: str, response: Any) -> Dict[str, float]:
        """Compute linguistic metrics for a single conversation instance."""
        if isinstance(response, dict) and "turns" in response:
            # Handle structured conversation
            all_turns = [turn.get("text", "") for turn in response["turns"] if turn.get("text")]

            if not all_turns:
                return self._get_empty_metrics()

            # Combine all turns for overall metrics
            combined_text = " ".join(all_turns)
            text_metrics = self._calculate_text_metrics(combined_text)

            # Add turn-level statistics
            turn_lengths = [len(turn.split()) for turn in all_turns]
            text_metrics.update({
                "total_turns": len(all_turns),
                "avg_turn_length": np.mean(turn_lengths) if turn_lengths else 0,
                "std_turn_length": np.std(turn_lengths) if turn_lengths else 0,
            })

        elif isinstance(response, str):
            # Handle single text response
            text_metrics = self._calculate_text_metrics(response)
            text_metrics.update({
                "total_turns": 1,
                "avg_turn_length": text_metrics.get("word_count", 0),
                "std_turn_length": 0,
            })
        else:
            return self._get_empty_metrics()

        return text_metrics

    def aggregate_metrics(self, instance_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate linguistic metrics across all instances."""
        if not instance_metrics:
            return {}

        # Calculate averages for all metrics
        aggregated = {}
        for key in instance_metrics[0].keys():
            values = [m.get(key, 0) for m in instance_metrics]
            aggregated[f"avg_{key}"] = np.mean(values)
            aggregated[f"std_{key}"] = np.std(values)
            aggregated[f"min_{key}"] = min(values)
            aggregated[f"max_{key}"] = max(values)

        # Add total counts
        aggregated["total_conversations"] = len(instance_metrics)
        aggregated["total_turns"] = sum(m.get("total_turns", 0) for m in instance_metrics)

        return aggregated

    def _get_empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics for failed cases."""
        return {
            "distinct_1": 0.0,
            "distinct_2": 0.0,
            "lexical_diversity": 0.0,
            "readability_score": 0.0,
            "grade_level": 0.0,
            "word_count": 0,
            "sentence_count": 0,
            "total_turns": 0,
            "avg_turn_length": 0,
            "std_turn_length": 0,
        }

    def _calculate_conversation_metrics(self,
                                      all_turns: List[str],
                                      persuader_turns: List[str],
                                      persuadee_turns: List[str]) -> Dict[str, Any]:
        """Calculate metrics for a single conversation."""
        metrics = {}

        # Turn-level statistics
        turn_lengths = [len(turn.split()) for turn in all_turns]
        metrics["avg_turn_length"] = np.mean(turn_lengths) if turn_lengths else 0
        metrics["std_turn_length"] = np.std(turn_lengths) if turn_lengths else 0

        # Role-specific turn lengths
        if persuader_turns:
            persuader_lengths = [len(turn.split()) for turn in persuader_turns]
            metrics["avg_persuader_turn_length"] = np.mean(persuader_lengths)
        else:
            metrics["avg_persuader_turn_length"] = 0

        if persuadee_turns:
            persuadee_lengths = [len(turn.split()) for turn in persuadee_turns]
            metrics["avg_persuadee_turn_length"] = np.mean(persuadee_lengths)
        else:
            metrics["avg_persuadee_turn_length"] = 0

        # Linguistic diversity
        if all_turns:
            combined_text = " ".join(all_turns)
            metrics.update(self._calculate_text_metrics(combined_text))

        return metrics

    def _calculate_text_metrics(self, text: str) -> Dict[str, Any]:
        """Calculate linguistic metrics for a single text."""
        if not text.strip():
            return {
                "distinct_1": 0.0,
                "distinct_2": 0.0,
                "lexical_diversity": 0.0,
                "readability_score": 0.0,
                "grade_level": 0.0,
                "word_count": 0,
                "sentence_count": 0
            }

        # Tokenize
        words = re.findall(r'\b\w+\b', text.lower())
        word_count = len(words)

        # Calculate distinct-n metrics
        distinct_1 = self._calculate_distinct_n(words, 1)
        distinct_2 = self._calculate_distinct_n(words, 2)

        # Lexical diversity (TTR)
        unique_words = len(set(words))
        lexical_diversity = unique_words / word_count if word_count > 0 else 0

        # Readability metrics
        try:
            readability_score = flesch_reading_ease(text)
            grade_level = flesch_kincaid_grade(text)
        except:
            readability_score = 0.0
            grade_level = 0.0

        # Sentence count
        sentences = nltk.sent_tokenize(text)
        sentence_count = len(sentences)

        return {
            "distinct_1": distinct_1,
            "distinct_2": distinct_2,
            "lexical_diversity": lexical_diversity,
            "readability_score": readability_score,
            "grade_level": grade_level,
            "word_count": word_count,
            "sentence_count": sentence_count
        }

    def _calculate_corpus_metrics(self, texts: List[str]) -> Dict[str, Any]:
        """Calculate corpus-level linguistic metrics."""
        if not texts:
            return {}

        # Combine all texts
        all_words = []
        total_word_count = 0
        total_sentence_count = 0
        readability_scores = []
        grade_levels = []

        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            all_words.extend(words)
            total_word_count += len(words)

            sentences = nltk.sent_tokenize(text)
            total_sentence_count += len(sentences)

            try:
                readability_scores.append(flesch_reading_ease(text))
                grade_levels.append(flesch_kincaid_grade(text))
            except:
                pass

        # Calculate metrics
        metrics = {
            "total_word_count": total_word_count,
            "total_sentence_count": total_sentence_count,
            "avg_words_per_text": total_word_count / len(texts),
            "avg_sentences_per_text": total_sentence_count / len(texts),
        }

        # Distinct-n for corpus
        if all_words:
            metrics["corpus_distinct_1"] = self._calculate_distinct_n(all_words, 1)
            metrics["corpus_distinct_2"] = self._calculate_distinct_n(all_words, 2)

            # Vocabulary size
            metrics["vocabulary_size"] = len(set(all_words))
            metrics["corpus_lexical_diversity"] = len(set(all_words)) / len(all_words)

        # Average readability
        if readability_scores:
            metrics["avg_readability_score"] = np.mean(readability_scores)
            metrics["std_readability_score"] = np.std(readability_scores)

        if grade_levels:
            metrics["avg_grade_level"] = np.mean(grade_levels)
            metrics["std_grade_level"] = np.std(grade_levels)

        return metrics

    def _calculate_distinct_n(self, words: List[str], n: int) -> float:
        """Calculate distinct-n metric (ratio of unique n-grams to total n-grams)."""
        if len(words) < n:
            return 0.0

        # Generate n-grams
        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = tuple(words[i:i+n])
            ngrams.append(ngram)

        if not ngrams:
            return 0.0

        # Calculate distinct ratio
        unique_ngrams = len(set(ngrams))
        total_ngrams = len(ngrams)

        return unique_ngrams / total_ngrams

    def get_evaluation_description(self) -> str:
        """Get description of this evaluation metric."""
        return "Linguistic diversity and readability evaluation for dialogue responses"

    def get_supported_tasks(self) -> List[str]:
        """Get list of task types this evaluator supports."""
        return ["persuasion_dialogue", "dialogue", "creative_writing"]