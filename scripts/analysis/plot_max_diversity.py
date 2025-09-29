#!/usr/bin/env python3
"""
Script to plot max_diversity values from all diversity_results.json files
found recursively under generated_data/
"""

import os
import json
import glob
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def load_diversity_results():
    """Load all diversity_results.json files and extract max_diversity values."""
    data_dir = "generated_data"
    diversity_files = glob.glob(f"{data_dir}/**/diversity_results.json", recursive=True)

    print(f"Found {len(diversity_files)} diversity result files")

    results = []

    for file_path in diversity_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            max_diversity = data['overall_metrics']['max_diversity']

            # Extract meaningful path information
            path_parts = Path(file_path).parts

            # Find experiment type and model info from path
            experiment_info = ""
            model_info = ""

            for i, part in enumerate(path_parts):
                if 'experiments_final' in part:
                    task_type = part.replace('_experiments_final', '')
                    if i + 1 < len(path_parts):
                        model_info = path_parts[i + 1]
                    if i + 3 < len(path_parts):
                        experiment_info = path_parts[i + 3]
                    break

            results.append({
                'file_path': file_path,
                'max_diversity': max_diversity,
                'task_type': task_type,
                'model': model_info,
                'experiment': experiment_info,
                'full_path': '/'.join(path_parts[-4:])  # Last 4 parts for readability
            })

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    return results


def plot_max_diversity(results):
    """Create histogram of max_diversity values."""
    df = pd.DataFrame(results)

    print(f"Loaded {len(df)} results")
    print(f"Max diversity range: {df['max_diversity'].min():.4f} to {df['max_diversity'].max():.4f}")

    # Create histogram
    plt.figure(figsize=(12, 8))

    # Main histogram
    plt.hist(df['max_diversity'], bins=50, alpha=0.7, edgecolor='black', color='skyblue')
    plt.title('Distribution of Max Diversity Values Across All Experiments', fontsize=16)
    plt.xlabel('Max Diversity', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True, alpha=0.3)

    # Add statistics text box
    stats_text = f"""Statistics:
    Mean: {df['max_diversity'].mean():.4f}
    Median: {df['max_diversity'].median():.4f}
    Std: {df['max_diversity'].std():.4f}
    Min: {df['max_diversity'].min():.4f}
    Max: {df['max_diversity'].max():.4f}
    Count: {len(df)}"""

    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Add vertical lines for mean and median
    plt.axvline(df['max_diversity'].mean(), color='red', linestyle='--', alpha=0.8, label='Mean')
    plt.axvline(df['max_diversity'].median(), color='orange', linestyle='--', alpha=0.8, label='Median')
    plt.legend()

    plt.tight_layout()
    plt.savefig('max_diversity_histogram.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Mean max_diversity: {df['max_diversity'].mean():.4f}")
    print(f"Median max_diversity: {df['max_diversity'].median():.4f}")
    print(f"Std max_diversity: {df['max_diversity'].std():.4f}")

    print(f"\nTop 10 experiments by max_diversity:")
    for i, (_, row) in enumerate(top_experiments.iterrows(), 1):
        print(f"{i:2d}. {row['max_diversity']:.4f} - {row['full_path']}")

    print(f"\nBottom 10 experiments by max_diversity:")
    for i, (_, row) in enumerate(bottom_experiments.iterrows(), 1):
        print(f"{i:2d}. {row['max_diversity']:.4f} - {row['full_path']}")

    # Save detailed results to CSV
    df.to_csv('max_diversity_results.csv', index=False)
    print(f"\nDetailed results saved to: max_diversity_results.csv")

    return df


def main():
    """Main function."""
    print("Loading diversity results...")
    results = load_diversity_results()

    if not results:
        print("No diversity results found!")
        return

    print("Creating plots...")
    df = plot_max_diversity(results)

    print("Analysis complete!")


if __name__ == "__main__":
    main()