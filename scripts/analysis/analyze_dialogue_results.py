#!/usr/bin/env python3
"""
Dialogue simulation results analysis script.

This script analyzes dialogue simulation results and generates
comprehensive visualizations and evaluation reports.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add verbalized_sampling to path
script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))

from verbalized_sampling.analysis.dialogue_plots import (
    plot_donation_distribution,
    plot_conversation_flow,
    plot_linguistic_metrics,
    plot_method_comparison,
    create_dialogue_report
)
from verbalized_sampling.evals.dialogue import DonationEvaluator, DialogueLinguisticEvaluator


def analyze_single_experiment(results_file: Path,
                             evaluation_file: Optional[Path] = None,
                             ground_truth_file: Optional[Path] = None,
                             output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Analyze results from a single dialogue experiment."""
    if output_dir is None:
        output_dir = results_file.parent / "analysis"

    output_dir.mkdir(exist_ok=True)

    print(f"🔍 Analyzing dialogue results from {results_file}")

    # Load results
    with open(results_file, 'r') as f:
        results = [json.loads(line) for line in f]

    print(f"   Loaded {len(results)} conversations")

    # Load evaluation results if available
    evaluation_results = {}
    if evaluation_file and evaluation_file.exists():
        with open(evaluation_file, 'r') as f:
            evaluation_results = json.load(f)

    # Run evaluation if not available
    if not evaluation_results:
        print("   Running evaluation metrics...")

        # Initialize evaluators
        ground_truth_path = str(ground_truth_file) if ground_truth_file else None
        donation_evaluator = DonationEvaluator(ground_truth_path=ground_truth_path)
        linguistic_evaluator = DialogueLinguisticEvaluator()

        # Compute metrics
        donation_metrics = []
        linguistic_metrics = []

        for result in results:
            donation_metric = donation_evaluator.compute_instance_metric("", result)
            donation_metrics.append(donation_metric)

            linguistic_metric = linguistic_evaluator.compute_instance_metric("", result)
            linguistic_metrics.append(linguistic_metric)

        # Aggregate
        evaluation_results = {
            "donation_metrics": donation_evaluator.aggregate_metrics(donation_metrics),
            "linguistic_metrics": linguistic_evaluator.aggregate_metrics(linguistic_metrics)
        }

        # Save evaluation results
        eval_output = output_dir / "evaluation_results.json"
        with open(eval_output, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        print(f"   💾 Evaluation results saved to {eval_output}")

    # Load ground truth if available
    ground_truth = None
    if ground_truth_file and ground_truth_file.exists():
        with open(ground_truth_file, 'r') as f:
            ground_truth_data = json.load(f)
            ground_truth = ground_truth_data.get("donation_amounts", [])

    # Generate visualizations
    print("   📊 Generating visualizations...")

    plot_donation_distribution(
        results,
        output_dir / "donation_distribution.png",
        ground_truth=ground_truth,
        title="Donation Amount Distribution"
    )

    plot_conversation_flow(
        results,
        output_dir / "conversation_flow.png"
    )

    if evaluation_results:
        plot_linguistic_metrics(
            results,
            evaluation_results,
            output_dir / "linguistic_metrics.png"
        )

    # Generate summary report
    summary = create_analysis_summary(results, evaluation_results, ground_truth)
    summary_file = output_dir / "analysis_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(summary)

    print(f"   ✅ Analysis complete! Results saved to {output_dir}")
    print(f"   📋 Summary report: {summary_file}")

    return evaluation_results


def analyze_method_comparison(results_files: List[Path],
                            method_names: Optional[List[str]] = None,
                            output_dir: Optional[Path] = None) -> None:
    """Analyze and compare results across different methods."""
    if output_dir is None:
        output_dir = Path("method_comparison")

    output_dir.mkdir(exist_ok=True)

    print(f"🔬 Comparing methods across {len(results_files)} experiments")

    # Load all results
    results_by_method = {}

    for i, results_file in enumerate(results_files):
        method_name = method_names[i] if method_names and i < len(method_names) else f"Method_{i+1}"

        with open(results_file, 'r') as f:
            results = [json.loads(line) for line in f]

        results_by_method[method_name] = results
        print(f"   Loaded {len(results)} conversations for {method_name}")

    # Generate comparison plots
    print("   📊 Generating comparison visualizations...")

    plot_method_comparison(
        results_by_method,
        output_dir / "method_comparison.png",
        title="Method Comparison Analysis"
    )

    # Generate detailed comparison report
    comparison_report = create_comparison_report(results_by_method)
    report_file = output_dir / "comparison_report.txt"
    with open(report_file, 'w') as f:
        f.write(comparison_report)

    print(f"   ✅ Method comparison complete! Results saved to {output_dir}")
    print(f"   📋 Comparison report: {report_file}")


def create_analysis_summary(results: List[Dict[str, Any]],
                           evaluation_results: Dict[str, Any],
                           ground_truth: Optional[List[float]] = None) -> str:
    """Create a text summary of the analysis."""
    summary = []
    summary.append("=" * 60)
    summary.append("DIALOGUE SIMULATION ANALYSIS SUMMARY")
    summary.append("=" * 60)
    summary.append("")

    # Basic statistics
    summary.append(f"Total Conversations: {len(results)}")

    # Extract donations
    donations = []
    turn_counts = []

    for result in results:
        if "outcome" in result and "final_donation_amount" in result["outcome"]:
            donations.append(result["outcome"]["final_donation_amount"])
        else:
            donations.append(0.0)

        turn_counts.append(len(result.get("turns", [])))

    # Donation statistics
    donation_rate = sum(1 for d in donations if d > 0) / len(donations)
    avg_donation = sum(donations) / len(donations)

    summary.append(f"Average Conversation Length: {sum(turn_counts) / len(turn_counts):.1f} turns")
    summary.append(f"Donation Rate: {donation_rate:.1%}")
    summary.append(f"Average Donation Amount: ${avg_donation:.2f}")
    summary.append(f"Total Amount Donated: ${sum(donations):.2f}")
    summary.append("")

    # Evaluation metrics
    if evaluation_results:
        donation_metrics = evaluation_results.get("donation_metrics", {})
        linguistic_metrics = evaluation_results.get("linguistic_metrics", {})

        if donation_metrics:
            summary.append("DONATION EVALUATION:")
            for key, value in donation_metrics.items():
                if key in ['donation_rate', 'average_donation', 'median_donation']:
                    if 'rate' in key:
                        summary.append(f"  {key.replace('_', ' ').title()}: {value:.1%}")
                    else:
                        summary.append(f"  {key.replace('_', ' ').title()}: ${value:.2f}")
            summary.append("")

        if linguistic_metrics:
            summary.append("LINGUISTIC EVALUATION:")
            key_metrics = ['avg_lexical_diversity', 'avg_distinct_1', 'avg_distinct_2', 'total_turns']
            for key in key_metrics:
                if key in linguistic_metrics:
                    value = linguistic_metrics[key]
                    if key == 'total_turns':
                        summary.append(f"  Total Turns: {int(value)}")
                    else:
                        clean_key = key.replace('avg_', '').replace('_', ' ').title()
                        summary.append(f"  {clean_key}: {value:.3f}")
            summary.append("")

    # Ground truth comparison
    if ground_truth:
        summary.append("GROUND TRUTH COMPARISON:")
        gt_donation_rate = sum(1 for d in ground_truth if d > 0) / len(ground_truth)
        gt_avg_donation = sum(ground_truth) / len(ground_truth)

        summary.append(f"  Generated Donation Rate: {donation_rate:.1%}")
        summary.append(f"  Ground Truth Donation Rate: {gt_donation_rate:.1%}")
        summary.append(f"  Rate Difference: {abs(donation_rate - gt_donation_rate):.1%}")
        summary.append(f"  Generated Avg Donation: ${avg_donation:.2f}")
        summary.append(f"  Ground Truth Avg Donation: ${gt_avg_donation:.2f}")
        summary.append("")

    return "\\n".join(summary)


def create_comparison_report(results_by_method: Dict[str, List[Dict[str, Any]]]) -> str:
    """Create a comparison report across methods."""
    report = []
    report.append("=" * 60)
    report.append("METHOD COMPARISON REPORT")
    report.append("=" * 60)
    report.append("")

    # Collect statistics for all methods
    method_stats = {}

    for method, results in results_by_method.items():
        donations = []
        turn_counts = []

        for result in results:
            if "outcome" in result and "final_donation_amount" in result["outcome"]:
                donations.append(result["outcome"]["final_donation_amount"])
            else:
                donations.append(0.0)

            turn_counts.append(len(result.get("turns", [])))

        method_stats[method] = {
            'conversations': len(results),
            'donation_rate': sum(1 for d in donations if d > 0) / len(donations),
            'avg_donation': sum(donations) / len(donations),
            'avg_turns': sum(turn_counts) / len(turn_counts)
        }

    # Create comparison table
    report.append(f"{'Method':<15} {'Convs':<8} {'Don.Rate':<10} {'Avg Don.':<10} {'Avg Turns':<10}")
    report.append("-" * 60)

    for method, stats in method_stats.items():
        report.append(f"{method:<15} {stats['conversations']:<8} {stats['donation_rate']:.1%:<10} ${stats['avg_donation']:.2f:<9} {stats['avg_turns']:.1f:<10}")

    report.append("")

    # Find best performing method
    best_donation_rate = max(method_stats.items(), key=lambda x: x[1]['donation_rate'])
    best_avg_donation = max(method_stats.items(), key=lambda x: x[1]['avg_donation'])

    report.append("PERFORMANCE SUMMARY:")
    report.append(f"  Highest Donation Rate: {best_donation_rate[0]} ({best_donation_rate[1]['donation_rate']:.1%})")
    report.append(f"  Highest Average Donation: {best_avg_donation[0]} (${best_avg_donation[1]['avg_donation']:.2f})")
    report.append("")

    return "\\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Analyze dialogue simulation results")

    subparsers = parser.add_subparsers(dest='command', help='Analysis commands')

    # Single experiment analysis
    single_parser = subparsers.add_parser('single', help='Analyze single experiment')
    single_parser.add_argument('results_file', type=Path,
                              help='Path to dialogue results JSONL file')
    single_parser.add_argument('--evaluation-file', type=Path,
                              help='Path to evaluation results JSON file')
    single_parser.add_argument('--ground-truth-file', type=Path,
                              help='Path to ground truth donation distribution')
    single_parser.add_argument('--output-dir', type=Path,
                              help='Output directory for analysis')

    # Method comparison
    compare_parser = subparsers.add_parser('compare', help='Compare multiple methods')
    compare_parser.add_argument('results_files', nargs='+', type=Path,
                               help='Paths to dialogue results JSONL files')
    compare_parser.add_argument('--method-names', nargs='+',
                               help='Names for the methods (optional)')
    compare_parser.add_argument('--output-dir', type=Path,
                               help='Output directory for comparison')

    args = parser.parse_args()

    if args.command == 'single':
        analyze_single_experiment(
            args.results_file,
            evaluation_file=args.evaluation_file,
            ground_truth_file=args.ground_truth_file,
            output_dir=args.output_dir
        )
    elif args.command == 'compare':
        analyze_method_comparison(
            args.results_files,
            method_names=args.method_names,
            output_dir=args.output_dir
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()