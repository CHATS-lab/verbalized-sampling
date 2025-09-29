"""
Donation-specific evaluation metrics for PersuasionForGood dialogues.

This module provides evaluators for measuring donation outcomes,
distribution alignment, and persuasion effectiveness.
"""

import re
import numpy as np
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from scipy import stats

from verbalized_sampling.analysis.evals.base import BaseEvaluator, EvalResult as EvaluationResult


@dataclass
class DonationMetrics:
    """Container for donation-related metrics."""
    donation_rate: float  # Percentage of conversations resulting in donations
    average_donation: float  # Average donation amount
    median_donation: float  # Median donation amount
    total_donated: float  # Total amount donated across all conversations
    donation_distribution: List[float]  # List of all donation amounts
    ks_statistic: Optional[float] = None  # KS test statistic vs ground truth
    ks_pvalue: Optional[float] = None  # KS test p-value
    l1_distance: Optional[float] = None  # L1 distance vs ground truth


class DonationEvaluator(BaseEvaluator):
    """Evaluator for donation outcomes in persuasion dialogues."""

    def __init__(self,
                 ground_truth_path: Optional[str] = None,
                 **kwargs):
        """Initialize donation evaluator.

        Args:
            ground_truth_path: Path to ground truth donation distribution
        """
        super().__init__(name="donation", **kwargs)
        self.ground_truth_path = ground_truth_path
        self.ground_truth_distribution = None

        if ground_truth_path:
            self._load_ground_truth()

    def _load_ground_truth(self) -> None:
        """Load ground truth donation distribution."""
        try:
            with open(self.ground_truth_path, 'r') as f:
                data = json.load(f)
                self.ground_truth_distribution = data.get("donation_amounts", [])
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load ground truth from {self.ground_truth_path}: {e}")
            self.ground_truth_distribution = None

    def compute_instance_metric(self, prompt: str, response: Any) -> Dict[str, float]:
        """Compute donation metrics for a single conversation instance."""
        # Extract donation amount from response
        if isinstance(response, dict):
            # Handle structured conversation result
            if "outcome" in response and "final_donation_amount" in response["outcome"]:
                donation_amount = response["outcome"]["final_donation_amount"]
            else:
                # Try to extract from conversation turns
                donation_amount = self._extract_donation_from_conversation(response)
        elif isinstance(response, str):
            # Handle raw text response
            donation_amount = self._extract_donation_amount(response)
        else:
            # Unknown format, assume no donation
            donation_amount = 0.0

        # Return instance-level metrics
        return {
            "donation_amount": donation_amount,
            "donated": 1.0 if donation_amount > 0 else 0.0,
        }

    def aggregate_metrics(self, instance_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate donation metrics across all instances."""
        if not instance_metrics:
            return {
                "donation_rate": 0.0,
                "average_donation": 0.0,
                "median_donation": 0.0,
                "total_donated": 0.0,
                "num_conversations": 0,
            }

        # Extract donation amounts
        donation_amounts = [m["donation_amount"] for m in instance_metrics]
        donations_made = [m["donated"] for m in instance_metrics]

        # Basic statistics
        donation_rate = np.mean(donations_made)
        average_donation = np.mean(donation_amounts)
        median_donation = np.median(donation_amounts)
        total_donated = sum(donation_amounts)

        metrics = {
            "donation_rate": donation_rate,
            "average_donation": average_donation,
            "median_donation": median_donation,
            "total_donated": total_donated,
            "num_conversations": len(instance_metrics),
            "min_donation": min(donation_amounts),
            "max_donation": max(donation_amounts),
            "std_donation": np.std(donation_amounts),
        }

        # Add distribution comparison metrics if ground truth available
        if self.ground_truth_distribution:
            ks_stat, ks_pval = self._calculate_ks_test(donation_amounts)
            l1_dist = self._calculate_l1_distance(donation_amounts)

            if ks_stat is not None:
                metrics["ks_statistic"] = ks_stat
                metrics["ks_pvalue"] = ks_pval
                metrics["ks_significant"] = ks_pval < 0.05 if ks_pval else False

            if l1_dist is not None:
                metrics["l1_distance"] = l1_dist

        return metrics

    def _calculate_donation_metrics(self, donation_amounts: List[float]) -> DonationMetrics:
        """Calculate donation-related metrics."""
        if not donation_amounts:
            return DonationMetrics(
                donation_rate=0.0,
                average_donation=0.0,
                median_donation=0.0,
                total_donated=0.0,
                donation_distribution=[]
            )

        # Basic statistics
        donations_made = sum(1 for amount in donation_amounts if amount > 0)
        donation_rate = donations_made / len(donation_amounts)
        average_donation = np.mean(donation_amounts)
        median_donation = np.median(donation_amounts)
        total_donated = sum(donation_amounts)

        return DonationMetrics(
            donation_rate=donation_rate,
            average_donation=average_donation,
            median_donation=median_donation,
            total_donated=total_donated,
            donation_distribution=donation_amounts.copy()
        )

    def _calculate_ks_test(self, donation_amounts: List[float]) -> Tuple[float, float]:
        """Calculate Kolmogorov-Smirnov test against ground truth."""
        if not self.ground_truth_distribution or not donation_amounts:
            return None, None

        try:
            ks_stat, p_value = stats.ks_2samp(donation_amounts, self.ground_truth_distribution)
            return ks_stat, p_value
        except Exception as e:
            print(f"Warning: KS test failed: {e}")
            return None, None

    def _calculate_l1_distance(self, donation_amounts: List[float]) -> float:
        """Calculate L1 distance between distributions."""
        if not self.ground_truth_distribution or not donation_amounts:
            return None

        try:
            # Create histograms with same bins
            bins = np.linspace(0, 2, 21)  # 0 to $2 in $0.1 increments
            hist1, _ = np.histogram(donation_amounts, bins=bins, density=True)
            hist2, _ = np.histogram(self.ground_truth_distribution, bins=bins, density=True)

            # Calculate L1 distance
            l1_distance = np.sum(np.abs(hist1 - hist2)) * (bins[1] - bins[0])
            return l1_distance
        except Exception as e:
            print(f"Warning: L1 distance calculation failed: {e}")
            return None

    def _extract_donation_from_conversation(self, conversation: Dict[str, Any]) -> float:
        """Extract donation amount from conversation turns."""
        if "turns" not in conversation:
            return 0.0

        # Look through persuadee turns for donation mentions
        for turn in conversation["turns"]:
            if turn.get("role") == 1:  # Persuadee role
                amount = self._extract_donation_amount(turn.get("text", ""))
                if amount is not None:
                    return amount

        return 0.0

    def _extract_donation_amount(self, text: str) -> float:
        """Extract donation amount from text using regex patterns."""
        if not text:
            return 0.0

        text_lower = text.lower()

        # Pattern for cents
        pattern_cents = r'\b(?:donate|donating|give|giving|do|spare|contribute|contributing)\s*\$?(\d*\.\d{1,2}|\d+)\s*cents\b'
        match_cents = re.search(pattern_cents, text_lower)

        if match_cents:
            match = next((m for m in match_cents.groups() if m is not None), None)
            if match:
                try:
                    amount = float(match.replace('$', '').replace(',', '').strip())
                    if amount >= 1.0:
                        amount = amount / 100  # Convert cents to dollars
                    return min(amount, 2.0)  # Cap at $2
                except ValueError:
                    pass

        # Pattern for dollars
        pattern_dollars = r'\b(?:donate|donating|give|giving|do|spare|contribute|contributing)\s*(?:the whole|all|entire)?\s*\$?(\d*\.\d{1,2}|\d+)\b'
        match_dollars = re.search(pattern_dollars, text_lower)

        if match_dollars:
            match = next((m for m in match_dollars.groups() if m is not None), None)
            if match:
                try:
                    amount = float(match.replace('$', '').replace(',', '').strip())
                    return min(amount, 2.0)  # Cap at $2
                except ValueError:
                    pass

        return 0.0

    def get_evaluation_description(self) -> str:
        """Get description of this evaluation metric."""
        return "Donation outcome evaluation for persuasion dialogues"

    def get_supported_tasks(self) -> List[str]:
        """Get list of task types this evaluator supports."""
        return ["persuasion_dialogue", "dialogue"]