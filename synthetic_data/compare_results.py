#!/usr/bin/env python3
"""
Compare results from different model experiments.
"""

import json
import glob
import argparse
from pathlib import Path
from typing import Dict, List
import pandas as pd

def load_result_file(file_path: str) -> Dict:
    """Load a result JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def extract_model_name(result_data: Dict) -> str:
    """Extract model name from result data."""
    if 'results' in result_data:
        # Get first model from results
        models = list(result_data['results'].keys())
        return models[0] if models else "unknown"
    return "unknown"

def create_comparison_table(result_files: List[str]) -> pd.DataFrame:
    """Create a comparison table from multiple result files."""
    comparison_data = []

    for file_path in result_files:
        result_data = load_result_file(file_path)
        if not result_data:
            continue

        model_name = extract_model_name(result_data)

        # Extract results for each dataset
        if 'results' in result_data and model_name in result_data['results']:
            model_results = result_data['results'][model_name]

            for dataset, dataset_results in model_results.items():
                if 'error' not in dataset_results:
                    comparison_data.append({
                        'Model': model_name,
                        'Dataset': dataset,
                        'Accuracy': dataset_results.get('accuracy', 0),
                        'Correct': dataset_results.get('correct_count', 0),
                        'Total': dataset_results.get('total_problems', 0),
                        'Avg_Time': dataset_results.get('avg_inference_time', 0),
                        'File': Path(file_path).name
                    })

    return pd.DataFrame(comparison_data)

def print_comparison(df: pd.DataFrame):
    """Print formatted comparison table."""
    if df.empty:
        print("No valid results found.")
        return

    print("🔍 MODEL COMPARISON RESULTS")
    print("=" * 80)

    # Overall comparison by model
    model_summary = df.groupby('Model').agg({
        'Accuracy': 'mean',
        'Total': 'sum',
        'Avg_Time': 'mean'
    }).round(3)

    print("\n📊 OVERALL PERFORMANCE BY MODEL")
    print("-" * 50)
    print(f"{'Model':<35} {'Avg Accuracy':<12} {'Total Problems':<15} {'Avg Time':<10}")
    print("-" * 50)
    for model, data in model_summary.iterrows():
        print(f"{model:<35} {data['Accuracy']:<12.3f} {data['Total']:<15} {data['Avg_Time']:<10.2f}s")

    # Detailed breakdown by dataset
    print("\n📈 DETAILED BREAKDOWN BY DATASET")
    print("-" * 80)
    print(f"{'Model':<30} {'Dataset':<15} {'Accuracy':<10} {'Correct/Total':<12} {'Avg Time':<10}")
    print("-" * 80)

    for _, row in df.iterrows():
        correct_total = f"{row['Correct']}/{row['Total']}"
        print(f"{row['Model']:<30} {row['Dataset']:<15} {row['Accuracy']:<10.3f} {correct_total:<12} {row['Avg_Time']:<10.2f}s")

    # Best performance per dataset
    print("\n🏆 BEST PERFORMANCE PER DATASET")
    print("-" * 50)
    best_per_dataset = df.loc[df.groupby('Dataset')['Accuracy'].idxmax()]
    for _, row in best_per_dataset.iterrows():
        print(f"{row['Dataset']:<15} {row['Model']:<30} {row['Accuracy']:.3f}")

def main():
    parser = argparse.ArgumentParser(description="Compare math model experiment results")
    parser.add_argument("--results_dir", default="math_results",
                       help="Directory containing result JSON files")
    parser.add_argument("--pattern", default="*.json",
                       help="File pattern to match")
    parser.add_argument("--export",
                       help="Export comparison to CSV file")

    args = parser.parse_args()

    # Find result files
    search_pattern = f"{args.results_dir}/{args.pattern}"
    result_files = glob.glob(search_pattern)

    if not result_files:
        print(f"No result files found matching: {search_pattern}")
        print(f"Make sure to run experiments first:")
        print(f"  ./run_math_experiments.sh")
        return 1

    print(f"Found {len(result_files)} result files:")
    for f in result_files:
        print(f"  📄 {f}")
    print()

    # Create comparison
    comparison_df = create_comparison_table(result_files)

    if comparison_df.empty:
        print("No valid results to compare.")
        return 1

    # Print comparison
    print_comparison(comparison_df)

    # Export if requested
    if args.export:
        comparison_df.to_csv(args.export, index=False)
        print(f"\n💾 Comparison exported to: {args.export}")

    return 0

if __name__ == "__main__":
    exit(main())