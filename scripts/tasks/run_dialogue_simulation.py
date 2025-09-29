#!/usr/bin/env python3
"""
PersuasionForGood dialogue simulation experiment script.

This script runs multi-turn persuasive dialogue simulations using
verbalized sampling methods to evaluate dialogue diversity and
persuasion effectiveness.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List
import sys

# Add verbalized_sampling to path
script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))

from verbalized_sampling.tasks.dialogue.persuasion import PersuasionTask
from verbalized_sampling.llms import get_model
from verbalized_sampling.methods import Method
from verbalized_sampling.evals.dialogue import DonationEvaluator, DialogueLinguisticEvaluator


def setup_models(persuader_model_name: str,
                 persuadee_model_name: str,
                 method: Method,
                 config: Dict[str, Any]) -> tuple:
    """Set up LLM models for persuader and persuadee."""

    # Create model instances
    persuader_model = get_model(
        model_name=persuader_model_name,
        method=Method.DIRECT,  # Persuader always uses direct
        config=config,
        num_workers=1,
        strict_json=False
    )

    persuadee_model = get_model(
        model_name=persuadee_model_name,
        method=method,
        config=config,
        num_workers=1,
        strict_json=(method in [Method.VS_STANDARD, Method.VS_COT, Method.SEQUENCE])
    )

    return persuader_model, persuadee_model


def run_dialogue_experiment(args) -> List[Dict[str, Any]]:
    """Run dialogue simulation experiment."""

    # Model configuration
    config = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }

    # Set up models
    persuader_model, persuadee_model = setup_models(
        args.persuader_model,
        args.persuadee_model,
        Method(args.method),
        config
    )

    # Create dialogue task
    task = PersuasionTask(
        persuader_model=persuader_model,
        persuadee_model=persuadee_model,
        method=Method(args.method),
        max_turns=args.max_turns,
        word_limit=args.word_limit,
        num_samplings=args.num_samplings,
        sampling_method=args.method,
        dataset_path=args.dataset_path,
        corpus_path=args.corpus_path
    )

    # Set response selection strategy
    task.response_selection = args.response_selection

    # Run experiment
    print(f"🔬 Running dialogue simulation experiment...")
    print(f"   Persuader Model: {args.persuader_model}")
    print(f"   Persuadee Model: {args.persuadee_model}")
    print(f"   Method: {args.method}")
    print(f"   Max Turns: {args.max_turns}")
    print(f"   Conversations: {args.num_conversations}")
    print(f"   Response Selection: {args.response_selection}")

    conversations = task.run_experiment(num_conversations=args.num_conversations)

    # Process results
    results = []
    total_turns = 0
    total_donations = 0

    for conv in conversations:
        # Convert conversation to serializable format
        result = {
            "conversation_id": conv.conversation_id,
            "persuader_persona": conv.persuader_persona,
            "persuadee_persona": conv.persuadee_persona,
            "turns": [
                {
                    "role": turn.role.value,
                    "text": turn.text,
                    "turn_number": turn.turn_number,
                    "metadata": turn.metadata
                }
                for turn in conv.turns
            ],
            "outcome": conv.outcome,
            "method": args.method,
            "persuader_model": args.persuader_model,
            "persuadee_model": args.persuadee_model,
            "config": config
        }
        results.append(result)

        # Track statistics
        total_turns += len(conv.turns)
        if conv.outcome and conv.outcome.get("final_donation_amount", 0) > 0:
            total_donations += 1

    # Print progress update
    if not args.quiet:
        print(f"   ✅ Completed {len(conversations)} conversations")
        print(f"   📊 Total turns: {total_turns}, Avg turns/conv: {total_turns/len(conversations):.1f}")
        print(f"   💰 Donations: {total_donations}/{len(conversations)} ({total_donations/len(conversations)*100:.1f}%)")

    return results


def evaluate_results(results: List[Dict[str, Any]],
                    ground_truth_path: Optional[str] = None,
                    quiet: bool = False) -> Dict[str, Any]:
    """Evaluate dialogue simulation results using donation and linguistic metrics."""
    if not results:
        return {}

    # Initialize evaluators
    donation_evaluator = DonationEvaluator(ground_truth_path=ground_truth_path)
    linguistic_evaluator = DialogueLinguisticEvaluator()

    # Compute instance-level metrics
    donation_metrics = []
    linguistic_metrics = []

    for result in results:
        # Compute donation metrics
        donation_metric = donation_evaluator.compute_instance_metric("", result)
        donation_metrics.append(donation_metric)

        # Compute linguistic metrics
        linguistic_metric = linguistic_evaluator.compute_instance_metric("", result)
        linguistic_metrics.append(linguistic_metric)

    # Aggregate metrics
    aggregated_donation = donation_evaluator.aggregate_metrics(donation_metrics)
    aggregated_linguistic = linguistic_evaluator.aggregate_metrics(linguistic_metrics)

    # Combine all metrics
    evaluation_results = {
        "donation_metrics": aggregated_donation,
        "linguistic_metrics": aggregated_linguistic
    }

    if not quiet:
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)

        # Donation metrics
        print("📊 Donation Metrics:")
        if aggregated_donation:
            print(f"   Donation Rate: {aggregated_donation.get('donation_rate', 0):.1%}")
            print(f"   Average Donation: ${aggregated_donation.get('average_donation', 0):.2f}")
            print(f"   Total Donated: ${aggregated_donation.get('total_donated', 0):.2f}")

            if 'ks_statistic' in aggregated_donation:
                print(f"   KS Test vs Ground Truth: {aggregated_donation['ks_statistic']:.3f}")
                if aggregated_donation.get('ks_significant', False):
                    print("   ⚠️  Distribution significantly different from ground truth")
                else:
                    print("   ✅ Distribution similar to ground truth")

        # Linguistic metrics
        print("\n🔤 Linguistic Metrics:")
        if aggregated_linguistic:
            print(f"   Avg Lexical Diversity: {aggregated_linguistic.get('avg_lexical_diversity', 0):.3f}")
            print(f"   Avg Distinct-1: {aggregated_linguistic.get('avg_distinct_1', 0):.3f}")
            print(f"   Avg Turn Length: {aggregated_linguistic.get('avg_avg_turn_length', 0):.1f} words")
            print(f"   Total Conversations: {aggregated_linguistic.get('total_conversations', 0)}")
            print(f"   Total Turns: {aggregated_linguistic.get('total_turns', 0)}")

    return evaluation_results


def save_results(results: List[Dict[str, Any]], output_file: Path, evaluation_results: Optional[Dict[str, Any]] = None):
    """Save experiment results."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    print(f"✅ Results saved to {output_file}")

    # Save evaluation results if provided
    if evaluation_results:
        eval_file = output_file.parent / f"{output_file.stem}_evaluation.json"
        with open(eval_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        print(f"📊 Evaluation results saved to {eval_file}")


def print_summary(results: List[Dict[str, Any]]):
    """Print experiment summary."""
    if not results:
        print("No results to summarize.")
        return

    total_conversations = len(results)
    total_turns = sum(len(r["turns"]) for r in results)
    avg_turns = total_turns / total_conversations if total_conversations > 0 else 0

    # Calculate donation statistics
    donation_amounts = [r["outcome"]["final_donation_amount"] for r in results]
    avg_donation = sum(donation_amounts) / len(donation_amounts) if donation_amounts else 0
    donations_made = sum(1 for amount in donation_amounts if amount > 0)
    donation_rate = donations_made / len(donation_amounts) if donation_amounts else 0

    print("\n" + "="*50)
    print("EXPERIMENT SUMMARY")
    print("="*50)
    print(f"Total Conversations: {total_conversations}")
    print(f"Average Turns per Conversation: {avg_turns:.1f}")
    print(f"Total Donation Rate: {donation_rate:.1%}")
    print(f"Average Donation Amount: ${avg_donation:.2f}")
    print(f"Conversations with Donations: {donations_made}/{total_conversations}")

    if results:
        method = results[0]["method"]
        print(f"Method Used: {method}")
        print(f"Persuader Model: {results[0]['persuader_model']}")
        print(f"Persuadee Model: {results[0]['persuadee_model']}")


def main():
    parser = argparse.ArgumentParser(description="Run PersuasionForGood dialogue simulation")

    # Model configuration
    parser.add_argument("--persuader-model", default="gpt-4.1-mini",
                       help="Model for persuader role")
    parser.add_argument("--persuadee-model", default="gpt-4.1-mini",
                       help="Model for persuadee role")
    parser.add_argument("--method", default="vs_standard",
                       choices=["direct", "vs_standard", "vs_cot", "vs_multi", "sequence"],
                       help="Sampling method for persuadee")

    # Experiment configuration
    parser.add_argument("--max-turns", type=int, default=10,
                       help="Maximum turns per conversation")
    parser.add_argument("--word-limit", type=int, default=160,
                       help="Word limit per response")
    parser.add_argument("--num-samplings", type=int, default=4,
                       help="Number of response samples for VS methods")
    parser.add_argument("--num-conversations", type=int, default=5,
                       help="Number of conversations to simulate")

    # Data paths
    parser.add_argument("--dataset-path", type=str, default=None,
                       help="Path to PersuasionForGood dataset")
    parser.add_argument("--corpus-path", type=str, default=None,
                       help="Path to corpus for persona generation")

    # Model parameters
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9,
                       help="Top-p parameter")
    parser.add_argument("--max-tokens", type=int, default=500,
                       help="Maximum tokens per response")

    # Response selection strategy
    parser.add_argument("--response-selection", default="probability",
                       choices=["probability", "random"],
                       help="Response selection strategy for VS methods")

    # Evaluation options
    parser.add_argument("--evaluate", action="store_true",
                       help="Run evaluation metrics on results")
    parser.add_argument("--ground-truth-path", type=str, default=None,
                       help="Path to ground truth donation distribution for evaluation")

    # Output
    parser.add_argument("--output-file", type=str, default="dialogue_results.jsonl",
                       help="Output file for results")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress verbose output")

    args = parser.parse_args()

    try:
        # Run experiment
        results = run_dialogue_experiment(args)

        # Run evaluation if requested
        evaluation_results = None
        if args.evaluate:
            if not args.quiet:
                print(f"\n🔍 Running evaluation metrics...")
            evaluation_results = evaluate_results(
                results,
                ground_truth_path=args.ground_truth_path,
                quiet=args.quiet
            )

        # Save results
        output_path = Path(args.output_file)
        save_results(results, output_path, evaluation_results)

        # Print summary
        if not args.quiet:
            print_summary(results)

    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()