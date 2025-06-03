from pathlib import Path
from typing import List
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency

def plot_histograms(target_dir: Path, output: Path, sizes: List[int]):
    """Plot histograms for different sampling sizes."""
    plt.figure(figsize=(15, 10))
    
    for i, size in enumerate(sizes, 1):
        plt.subplot(len(sizes), 1, i)
        
        # Read responses
        responses = []
        for file in target_dir.glob(f"*_size_{size}_*.jsonl"):
            with open(file) as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if isinstance(data, list):
                            responses.extend(data)
                        else:
                            responses.append(data)
                    except json.JSONDecodeError:
                        continue
        
        if not responses:
            continue
        
        # Convert responses to numbers
        numbers = []
        for resp in responses:
            if isinstance(resp, (int, float)):
                numbers.append(resp)
            elif isinstance(resp, dict) and "number" in resp:
                numbers.append(resp["number"])
            elif isinstance(resp, list):
                numbers.extend([x for x in resp if isinstance(x, (int, float))])
        
        if not numbers:
            continue
        
        # Plot histogram
        plt.hist(numbers, bins=50, alpha=0.7, label=f"Size {size}")
        plt.title(f"Distribution for Size {size}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(output)
    plt.close()

def evaluate_chi_square(responses: List[float], expected_dist: List[float] = None):
    """Evaluate responses using chi-square test."""
    if expected_dist is None:
        # Use uniform distribution as default
        expected_dist = [1/10] * 10  # Assuming numbers are in [0, 9]
    
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