"""
Visualization utilities for dialogue simulation experiments.

This module provides plotting functions specifically for dialogue
simulation results, including donation distributions, linguistic
metrics, and conversation flow analysis.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
import pandas as pd

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_donation_distribution(results: List[Dict[str, Any]],
                             output_path: Path,
                             ground_truth: Optional[List[float]] = None,
                             title: str = "Donation Amount Distribution") -> None:
    """Plot distribution of donation amounts."""
    # Extract donation amounts
    donations = []
    for result in results:
        if "outcome" in result and "final_donation_amount" in result["outcome"]:
            donations.append(result["outcome"]["final_donation_amount"])
        else:
            donations.append(0.0)

    plt.figure(figsize=(12, 6))

    # Create subplot layout
    if ground_truth:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

    # Plot main distribution
    bins = np.linspace(0, 2.0, 21)  # $0 to $2 in $0.1 increments
    ax1.hist(donations, bins=bins, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Donation Amount ($)')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'{title}\nTotal: {len(donations)} conversations, Donation Rate: {np.mean([d > 0 for d in donations]):.1%}')
    ax1.grid(True, alpha=0.3)

    # Add statistics text
    stats_text = f'Mean: ${np.mean(donations):.2f}\nMedian: ${np.median(donations):.2f}\nStd: ${np.std(donations):.2f}'
    ax1.text(0.7, 0.8, stats_text, transform=ax1.transAxes,
             bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))

    # Compare with ground truth if provided
    if ground_truth:
        ax2.hist([donations, ground_truth], bins=bins, alpha=0.7,
                label=['Generated', 'Ground Truth'], edgecolor='black', linewidth=0.5)
        ax2.set_xlabel('Donation Amount ($)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Comparison with Ground Truth')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_conversation_flow(results: List[Dict[str, Any]],
                          output_path: Path,
                          title: str = "Conversation Flow Analysis") -> None:
    """Plot conversation flow metrics."""
    # Extract conversation statistics
    turn_counts = []
    donation_by_turns = defaultdict(list)

    for result in results:
        turns = result.get("turns", [])
        num_turns = len(turns)
        turn_counts.append(num_turns)

        donation = 0.0
        if "outcome" in result and "final_donation_amount" in result["outcome"]:
            donation = result["outcome"]["final_donation_amount"]

        donation_by_turns[num_turns].append(donation)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Turn count distribution
    ax1.hist(turn_counts, bins=range(min(turn_counts), max(turn_counts) + 2),
             alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Number of Turns')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Conversation Lengths')
    ax1.grid(True, alpha=0.3)

    # 2. Donation rate by conversation length
    turn_nums = sorted(donation_by_turns.keys())
    donation_rates = []
    avg_donations = []

    for turns in turn_nums:
        donations = donation_by_turns[turns]
        rate = np.mean([d > 0 for d in donations])
        avg = np.mean(donations)
        donation_rates.append(rate)
        avg_donations.append(avg)

    ax2.bar(turn_nums, donation_rates, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Number of Turns')
    ax2.set_ylabel('Donation Rate')
    ax2.set_title('Donation Rate by Conversation Length')
    ax2.grid(True, alpha=0.3)

    # 3. Average donation by conversation length
    ax3.bar(turn_nums, avg_donations, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax3.set_xlabel('Number of Turns')
    ax3.set_ylabel('Average Donation ($)')
    ax3.set_title('Average Donation by Conversation Length')
    ax3.grid(True, alpha=0.3)

    # 4. Turn length distribution
    all_turn_lengths = []
    persuader_lengths = []
    persuadee_lengths = []

    for result in results:
        turns = result.get("turns", [])
        for turn in turns:
            text = turn.get("text", "")
            length = len(text.split())
            all_turn_lengths.append(length)

            if turn.get("role") == 0:  # Persuader
                persuader_lengths.append(length)
            elif turn.get("role") == 1:  # Persuadee
                persuadee_lengths.append(length)

    if persuader_lengths and persuadee_lengths:
        ax4.hist([persuader_lengths, persuadee_lengths],
                bins=30, alpha=0.7, label=['Persuader', 'Persuadee'],
                edgecolor='black', linewidth=0.5)
        ax4.set_xlabel('Turn Length (words)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Turn Length Distribution by Role')
        ax4.legend()
    else:
        ax4.hist(all_turn_lengths, bins=30, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax4.set_xlabel('Turn Length (words)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Overall Turn Length Distribution')

    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_linguistic_metrics(results: List[Dict[str, Any]],
                           evaluation_results: Dict[str, Any],
                           output_path: Path,
                           title: str = "Linguistic Diversity Analysis") -> None:
    """Plot linguistic diversity metrics."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Extract per-conversation metrics
    lexical_diversity = []
    distinct_1 = []
    distinct_2 = []
    readability_scores = []

    # Note: This would need actual per-instance metrics
    # For now, we'll use the aggregated metrics from evaluation_results
    linguistic_metrics = evaluation_results.get("linguistic_metrics", {})

    # 1. Metrics overview bar plot
    metrics_names = []
    metrics_values = []

    for key, value in linguistic_metrics.items():
        if key.startswith('avg_') and not key.endswith('_turn_length'):
            clean_name = key.replace('avg_', '').replace('_', ' ').title()
            metrics_names.append(clean_name)
            metrics_values.append(value)

    if metrics_names:
        bars = ax1.bar(metrics_names, metrics_values, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax1.set_ylabel('Score')
        ax1.set_title('Average Linguistic Metrics')
        ax1.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')

    # 2. Turn length statistics
    turn_stats = {}
    if 'avg_avg_turn_length' in linguistic_metrics:
        turn_stats['Average'] = linguistic_metrics['avg_avg_turn_length']
    if 'avg_std_turn_length' in linguistic_metrics:
        turn_stats['Std Dev'] = linguistic_metrics['avg_std_turn_length']

    if turn_stats:
        ax2.bar(turn_stats.keys(), turn_stats.values(), alpha=0.7, edgecolor='black', linewidth=0.5)
        ax2.set_ylabel('Words per Turn')
        ax2.set_title('Turn Length Statistics')

    # 3. Conversation-level summary
    conv_stats = {
        'Total Conversations': linguistic_metrics.get('total_conversations', 0),
        'Total Turns': linguistic_metrics.get('total_turns', 0),
        'Avg Turns/Conv': linguistic_metrics.get('total_turns', 0) / max(linguistic_metrics.get('total_conversations', 1), 1)
    }

    ax3.bar(conv_stats.keys(), conv_stats.values(), alpha=0.7, edgecolor='black', linewidth=0.5)
    ax3.set_ylabel('Count')
    ax3.set_title('Conversation Summary Statistics')
    ax3.tick_params(axis='x', rotation=45)

    # 4. Diversity comparison (placeholder - would need multiple methods)
    diversity_metrics = []
    diversity_values = []

    for metric in ['distinct_1', 'distinct_2', 'lexical_diversity']:
        avg_key = f'avg_{metric}'
        if avg_key in linguistic_metrics:
            diversity_metrics.append(metric.replace('_', '-').title())
            diversity_values.append(linguistic_metrics[avg_key])

    if diversity_metrics:
        ax4.bar(diversity_metrics, diversity_values, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax4.set_ylabel('Score')
        ax4.set_title('Diversity Metrics Comparison')
        ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_method_comparison(results_by_method: Dict[str, List[Dict[str, Any]]],
                          output_path: Path,
                          title: str = "Method Comparison") -> None:
    """Compare results across different sampling methods."""
    methods = list(results_by_method.keys())

    # Collect metrics
    donation_rates = []
    avg_donations = []
    avg_turns = []
    avg_diversity = []

    for method in methods:
        results = results_by_method[method]

        # Donation metrics
        donations = []
        turn_counts = []

        for result in results:
            if "outcome" in result and "final_donation_amount" in result["outcome"]:
                donations.append(result["outcome"]["final_donation_amount"])
            else:
                donations.append(0.0)

            turn_counts.append(len(result.get("turns", [])))

        donation_rates.append(np.mean([d > 0 for d in donations]))
        avg_donations.append(np.mean(donations))
        avg_turns.append(np.mean(turn_counts))

        # Placeholder for diversity (would need actual calculation)
        avg_diversity.append(0.5)  # Placeholder

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Donation rates
    bars1 = ax1.bar(methods, donation_rates, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Donation Rate')
    ax1.set_title('Donation Rate by Method')
    ax1.tick_params(axis='x', rotation=45)

    for bar, rate in zip(bars1, donation_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{rate:.1%}', ha='center', va='bottom')

    # 2. Average donations
    bars2 = ax2.bar(methods, avg_donations, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Average Donation ($)')
    ax2.set_title('Average Donation by Method')
    ax2.tick_params(axis='x', rotation=45)

    for bar, amt in zip(bars2, avg_donations):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'${amt:.2f}', ha='center', va='bottom')

    # 3. Average turns
    bars3 = ax3.bar(methods, avg_turns, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax3.set_ylabel('Average Turns per Conversation')
    ax3.set_title('Conversation Length by Method')
    ax3.tick_params(axis='x', rotation=45)

    for bar, turns in zip(bars3, avg_turns):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{turns:.1f}', ha='center', va='bottom')

    # 4. Combined metrics radar chart (placeholder)
    ax4.text(0.5, 0.5, 'Diversity Metrics\n(Would show lexical diversity,\ndistinct-n scores, etc.)',
             ha='center', va='center', transform=ax4.transAxes,
             bbox=dict(boxstyle="round", facecolor='lightgray', alpha=0.5))
    ax4.set_title('Linguistic Diversity (Placeholder)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_dialogue_report(results_file: Path,
                          evaluation_file: Optional[Path] = None,
                          ground_truth_file: Optional[Path] = None,
                          output_dir: Optional[Path] = None) -> Path:
    """Create a comprehensive dialogue analysis report with plots."""
    if output_dir is None:
        output_dir = results_file.parent

    output_dir.mkdir(exist_ok=True)

    # Load data
    with open(results_file, 'r') as f:
        results = [json.loads(line) for line in f]

    evaluation_results = {}
    if evaluation_file and evaluation_file.exists():
        with open(evaluation_file, 'r') as f:
            evaluation_results = json.load(f)

    ground_truth = None
    if ground_truth_file and ground_truth_file.exists():
        with open(ground_truth_file, 'r') as f:
            ground_truth_data = json.load(f)
            ground_truth = ground_truth_data.get("donation_amounts", [])

    # Create plots
    plot_donation_distribution(
        results,
        output_dir / "donation_distribution.png",
        ground_truth=ground_truth
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

    print(f"📊 Dialogue analysis plots saved to {output_dir}")
    return output_dir