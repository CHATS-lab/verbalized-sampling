#!/usr/bin/env python3
"""
Demonstration script for using the various evaluators in the verbalized sampling framework.

This script shows how to use:
1. DiversityEvaluator - for measuring response diversity using embeddings
2. TTCTEvaluator - for measuring creativity using Torrance Tests framework
3. CreativityIndexEvaluator - for measuring overlap with pretraining data
"""

import json
from pathlib import Path
from verbalized_sampling.evals import get_evaluator

def demo_diversity_evaluation():
    """Demo the diversity evaluator."""
    print("=" * 60)
    print("DIVERSITY EVALUATION DEMO")
    print("=" * 60)
    
    # Sample responses about potatoes
    prompts = ["Tell me a story about potatoes"] * 5
    responses = [
        "Once upon a time, in a small village, there lived a magical potato that could grant wishes.",
        "The potato farmer worked hard every day to grow the finest potatoes in the region.",
        "Deep underground, the potatoes whispered secrets to each other about the soil above.",
        "A young chef discovered that these particular potatoes had an extraordinary flavor.",
        "The potatoes grew so large that year that the whole town celebrated with a festival."
    ]
    
    # Create and run evaluator
    evaluator = get_evaluator("diversity", model_name="text-embedding-3-small")
    result = evaluator.evaluate(prompts, responses)
    
    print(f"Average similarity: {result.overall_metrics['average_similarity']:.3f}")
    print(f"Min similarity: {result.overall_metrics['min_similarity']:.3f}")
    print(f"Max similarity: {result.overall_metrics['max_similarity']:.3f}")
    print(f"Number of pairwise comparisons: {len(result.overall_metrics['pairwise_similarities'])}")
    
    # Show most and least similar pairs
    similarities = result.overall_metrics['pairwise_similarities']
    most_similar = max(similarities, key=lambda x: x['similarity'])
    least_similar = min(similarities, key=lambda x: x['similarity'])
    
    print(f"\nMost similar pair (similarity: {most_similar['similarity']:.3f}):")
    print(f"  Response 1: {most_similar['response1'][:80]}...")
    print(f"  Response 2: {most_similar['response2'][:80]}...")
    
    print(f"\nLeast similar pair (similarity: {least_similar['similarity']:.3f}):")
    print(f"  Response 1: {least_similar['response1'][:80]}...")
    print(f"  Response 2: {least_similar['response2'][:80]}...")

def demo_ttct_evaluation():
    """Demo the TTCT (quality) evaluator."""
    print("\n" + "=" * 60)
    print("TTCT CREATIVITY EVALUATION DEMO")
    print("=" * 60)
    
    # Sample creative responses
    prompts = ["Generate creative uses for a paperclip"] * 3
    responses = [
        "A paperclip can be used as a bookmark, a zipper pull, or a tiny sculpture wire.",
        "Transform it into a lock pick, a phone stand, a cable organizer, or even jewelry.",
        "Use it as a reset button tool, a keychain, a mini coat hanger for dolls, or a drawing compass."
    ]
    
    try:
        # Create and run evaluator (using a mock model for demo)
        print("Note: This demo uses a simplified evaluation for demonstration purposes.")
        print("In practice, you'd need access to a powerful language model for judging.")
        
        # For demo purposes, we'll create mock results
        mock_results = {
            "fluency": {
                "score": 4.2,
                "analysis": "Most responses are clear and relevant with good productive output.",
                "justification": "All responses directly address the prompt with meaningful suggestions."
            },
            "flexibility": {
                "score": 3.8,
                "categories_identified": ["tools", "accessories", "art", "technology"],
                "category_count": 4,
                "analysis": "Good variety across different use categories.",
                "justification": "Responses span multiple conceptual domains."
            },
            "originality": {
                "score": 3.5,
                "unique_elements": ["lock pick", "mini coat hanger", "drawing compass"],
                "commonality_assessment": "mixed",
                "analysis": "Mix of common and less conventional uses.",
                "justification": "Some expected uses but also creative applications."
            },
            "elaboration": {
                "score": 3.0,
                "detail_level": "moderate",
                "analysis": "Basic concepts stated but could use more development.",
                "justification": "Ideas are listed but not extensively detailed."
            },
            "overall": {
                "creativity_score": 3.6,
                "normalized_score": 0.72,
                "strengths": ["good flexibility", "practical applications"],
                "areas_for_improvement": ["add more detail", "explore unconventional uses"]
            }
        }
        
        print(f"Fluency Score: {mock_results['fluency']['score']}/5")
        print(f"Flexibility Score: {mock_results['flexibility']['score']}/5")
        print(f"Originality Score: {mock_results['originality']['score']}/5")
        print(f"Elaboration Score: {mock_results['elaboration']['score']}/5")
        print(f"Overall Creativity Score: {mock_results['overall']['creativity_score']}/5")
        
        print(f"\nCategories identified: {', '.join(mock_results['flexibility']['categories_identified'])}")
        print(f"Unique elements: {', '.join(mock_results['originality']['unique_elements'])}")
        
    except Exception as e:
        print(f"Error running TTCT evaluation: {e}")
        print("This requires a configured language model for judging.")

def demo_creativity_index_evaluation():
    """Demo the creativity index evaluator."""
    print("\n" + "=" * 60)
    print("CREATIVITY INDEX EVALUATION DEMO")
    print("=" * 60)
    
    responses = [
        "The quick brown fox jumps over the lazy dog.",  # Common phrase
        "Quantum entanglement provides fascinating insights into the nature of reality.",  # Technical
        "In a world where purple elephants dance on rainbow bridges, creativity knows no bounds."  # Creative
    ]
    prompts = ["Write something interesting"] * len(responses)
    
    try:
        # Note: This requires API access to Infini-gram
        print("Note: This demo shows the creativity index concept.")
        print("Full functionality requires Infini-gram API access.")
        
        # Mock results for demonstration
        mock_results = {
            "average_creativity_index": 0.75,
            "std_creativity_index": 0.15,
            "average_coverage": 0.25,
            "total_responses": 3,
            "responses_with_matches": 1,
            "match_rate": 0.33,
            "method": "exact"
        }
        
        print(f"Average Creativity Index: {mock_results['average_creativity_index']:.3f}")
        print(f"Average Coverage (overlap): {mock_results['average_coverage']:.3f}")
        print(f"Match Rate: {mock_results['match_rate']:.1%}")
        print(f"Responses with matches: {mock_results['responses_with_matches']}/{mock_results['total_responses']}")
        
        print("\nInterpretation:")
        print("- Higher Creativity Index = Less overlap with training data = More creative")
        print("- Lower Coverage = Less text matches found in reference corpus")
        print("- The common phrase likely has lower creativity index due to frequent occurrence")
        
    except Exception as e:
        print(f"Error running creativity index evaluation: {e}")
        print("This requires Infini-gram API access for full functionality.")

def demo_save_and_load_results():
    """Demo saving and loading evaluation results."""
    print("\n" + "=" * 60)
    print("SAVE/LOAD RESULTS DEMO")
    print("=" * 60)
    
    # Create a simple diversity evaluation result
    prompts = ["Tell a story"] * 3
    responses = [
        "A tale of adventure and mystery.",
        "Once upon a time in a distant land.",
        "The story began on a dark and stormy night."
    ]
    
    try:
        evaluator = get_evaluator("diversity")
        result = evaluator.evaluate(prompts, responses)
        
        # Save results
        output_path = Path("demo_results.json")
        evaluator.save_results(result, output_path)
        print(f"Results saved to {output_path}")
        
        # Load results
        loaded_result = evaluator.load_results(output_path)
        print(f"Results loaded successfully!")
        print(f"Loaded {len(loaded_result.instance_metrics)} instance metrics")
        print(f"Average similarity: {loaded_result.overall_metrics['average_similarity']:.3f}")
        
        # Clean up
        output_path.unlink()
        print("Demo file cleaned up.")
        
    except Exception as e:
        print(f"Error in save/load demo: {e}")

def main():
    """Run all evaluation demos."""
    print("VERBALIZED SAMPLING EVALUATION FRAMEWORK DEMO")
    print("=" * 80)
    
    try:
        demo_diversity_evaluation()
    except Exception as e:
        print(f"Diversity evaluation demo failed: {e}")
    
    try:
        demo_ttct_evaluation()
    except Exception as e:
        print(f"TTCT evaluation demo failed: {e}")
    
    try:
        demo_creativity_index_evaluation()
    except Exception as e:
        print(f"Creativity index evaluation demo failed: {e}")
    
    try:
        demo_save_and_load_results()
    except Exception as e:
        print(f"Save/load demo failed: {e}")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print("\nTo use these evaluators in your own code:")
    print("1. from verbalized_sampling.evals import get_evaluator")
    print("2. evaluator = get_evaluator('diversity')  # or 'ttct' or 'creativity_index'")
    print("3. result = evaluator.evaluate(prompts, responses)")
    print("4. print(result.overall_metrics)")

if __name__ == "__main__":
    main() 