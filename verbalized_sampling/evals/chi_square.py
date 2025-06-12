from typing import List
import numpy as np
from scipy.stats import chi2_contingency
import random

def evaluate_chi_square(responses: List[float], expected_dist: List[float] = None):
    """Evaluate responses using chi-square test."""
    if expected_dist is None:
        # Use uniform distribution as default
        # Legacy code
        # expected_dist = [1/10] * 10  # Assuming numbers are in [0, 9]
        random.seed(42)
        number_selection = random.choices(range(10), size=len(responses), replace=True)
        unique_numbers, counts = np.unique(number_selection, return_counts=True)
        expected_dist = np.zeros(10)
        expected_dist[unique_numbers] = counts
    
    # Create observed distribution
    observed, _ = np.histogram(responses, bins=len(expected_dist), range=(0, len(expected_dist)))
    observed = observed / sum(observed)  # Normalize
    
    # Perform chi-square test
    chi2, p_value = chi2_contingency([observed, expected_dist])[0:2]
    
    return {
        "chi2": chi2,
        "p_value": p_value,
        "observed": observed.tolist(),
        "expected": expected_dist
    } 