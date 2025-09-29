import os
import json
import pandas as pd
from collections import defaultdict

def calculate_coverage_comparison(data_dir="generated_data/openended_qa_general"):
    """
    Calculate coverage comparison between sequence and structure_with_prob methods.
    
    For each model and prompt, compare if structure_with_prob fully covers sequence.
    """
    results = []
    
    # Get all model directories
    for model_dir in os.listdir(data_dir):
        model_path = os.path.join(data_dir, model_dir)
        if not os.path.isdir(model_path):
            continue
            
        print(f"Processing model: {model_dir}")
        
        # Get evaluation path
        eval_path = os.path.join(model_path, "evaluation")
        if not os.path.exists(eval_path):
            continue
        
        # Look for sequence and VS-Standard (vs_standard) methods
        sequence_file = None
        structure_file = None
        
        for method_dir in os.listdir(eval_path):
            method_path = os.path.join(eval_path, method_dir)
            if not os.path.isdir(method_path):
                continue
                
            json_file = os.path.join(method_path, "response_count_results.json")
            if not os.path.exists(json_file):
                continue
                
            if 'sequence' in method_dir and 'strict' in method_dir:
                sequence_file = json_file
                print(f"  Found sequence method: {method_dir}")
            elif 'vs_standard' in method_dir and 'strict' in method_dir:
                structure_file = json_file
                print(f"  Found structure_with_prob method: {method_dir}")
        
        # Skip if we don't have both methods
        if not sequence_file or not structure_file:
            print(f"  Skipping {model_dir}: missing sequence or structure_with_prob data")
            continue
            
        try:
            # Load sequence data
            with open(sequence_file, 'r') as f:
                sequence_data = json.load(f)
            
            # Load VS-Standard (vs_standard) data
            with open(structure_file, 'r') as f:
                structure_data = json.load(f)
            
            # Extract per-prompt stats
            sequence_stats = sequence_data.get('overall_metrics', {}).get('per_prompt_stats', {})
            structure_stats = structure_data.get('overall_metrics', {}).get('per_prompt_stats', {})
            
            # Compare coverage for each prompt
            for prompt in sequence_stats.keys():
                if prompt not in structure_stats:
                    print(f"    Warning: Prompt not found in structure_with_prob: {prompt}")
                    continue
                
                sequence_responses = set(sequence_stats[prompt].get('response_distribution', {}).keys())
                structure_responses = set(structure_stats[prompt].get('response_distribution', {}).keys())
                
                # Check if VS-Standard (vs_standard) fully covers sequence
                structure_covers_sequence = sequence_responses.issubset(structure_responses)
                
                # Check if sequence fully covers VS-Standard (vs_standard) (reverse)
                sequence_covers_structure = structure_responses.issubset(sequence_responses)
                
                # Calculate coverage metrics
                sequence_unique = len(sequence_responses)
                structure_unique = len(structure_responses)
                intersection = len(sequence_responses.intersection(structure_responses))
                
                # Coverage ratios in both directions
                structure_to_sequence_ratio = intersection / sequence_unique if sequence_unique > 0 else 0
                sequence_to_structure_ratio = intersection / structure_unique if structure_unique > 0 else 0
                
                results.append({
                    'model': model_dir,
                    'prompt': prompt,
                    'sequence_unique_responses': sequence_unique,
                    'structure_unique_responses': structure_unique,
                    'intersection_count': intersection,
                    'structure_covers_sequence': structure_covers_sequence,
                    'sequence_covers_structure': sequence_covers_structure,
                    'structure_to_sequence_ratio': structure_to_sequence_ratio,
                    'sequence_to_structure_ratio': sequence_to_structure_ratio,
                    'sequence_responses': sequence_responses,
                    'structure_responses': structure_responses
                })
                
        except Exception as e:
            print(f"    Error processing {model_dir}: {e}")
            continue
    
    return pd.DataFrame(results)

def print_coverage_comparison_summary(df):
    """Print summary statistics for coverage comparison."""
    print("\n" + "="*80)
    print("COVERAGE COMPARISON: structure_with_prob vs sequence")
    print("="*80)
    
    print(f"\nTotal number of model-prompt combinations: {len(df)}")
    print(f"Number of unique models: {df['model'].nunique()}")
    print(f"Number of unique prompts: {df['prompt'].nunique()}")
    
    # Overall statistics for both directions
    total_structure_covers_sequence = df['structure_covers_sequence'].sum()
    total_sequence_covers_structure = df['sequence_covers_structure'].sum()
    total_combinations = len(df)
    
    structure_percentage = (total_structure_covers_sequence / total_combinations) * 100 if total_combinations > 0 else 0
    sequence_percentage = (total_sequence_covers_structure / total_combinations) * 100 if total_combinations > 0 else 0
    
    print(f"\nOverall Statistics:")
    print(f"  Total combinations: {total_combinations}")
    print(f"  Structure covers sequence: {total_structure_covers_sequence} ({structure_percentage:.2f}%)")
    print(f"  Sequence covers structure: {total_sequence_covers_structure} ({sequence_percentage:.2f}%)")
    
    # Statistics by model
    print(f"\nCoverage by Model:")
    model_stats = df.groupby('model').agg({
        'structure_covers_sequence': ['sum', 'count', 'mean'],
        'sequence_covers_structure': ['sum', 'mean'],
        'structure_to_sequence_ratio': ['mean', 'std'],
        'sequence_to_structure_ratio': ['mean', 'std']
    }).round(4)
    
    model_stats.columns = [
        'structure_covers_count', 'total_prompts', 'structure_covers_percentage',
        'sequence_covers_count', 'sequence_covers_percentage',
        'avg_structure_to_sequence_ratio', 'std_structure_to_sequence_ratio',
        'avg_sequence_to_structure_ratio', 'std_sequence_to_structure_ratio'
    ]
    model_stats['structure_covers_percentage'] *= 100
    model_stats['sequence_covers_percentage'] *= 100
    
    for model in model_stats.index:
        stats = model_stats.loc[model]
        print(f"  {model}:")
        print(f"    Structure covers sequence: {stats['structure_covers_count']:.0f}/{stats['total_prompts']:.0f} ({stats['structure_covers_percentage']:.2f}%)")
        print(f"    Sequence covers structure: {stats['sequence_covers_count']:.0f}/{stats['total_prompts']:.0f} ({stats['sequence_covers_percentage']:.2f}%)")
        print(f"    Avg structure→sequence ratio: {stats['avg_structure_to_sequence_ratio']:.4f} ± {stats['std_structure_to_sequence_ratio']:.4f}")
        print(f"    Avg sequence→structure ratio: {stats['avg_sequence_to_structure_ratio']:.4f} ± {stats['std_sequence_to_structure_ratio']:.4f}")
    
    # Statistics by prompt category
    df['prompt_category'] = df['prompt'].apply(lambda x: x.split('.')[0].split('Name a ')[1] if 'Name a ' in x else 'Other')
    
    print(f"\nCoverage by Prompt Category:")
    category_stats = df.groupby('prompt_category').agg({
        'structure_covers_sequence': ['sum', 'count', 'mean'],
        'sequence_covers_structure': ['sum', 'mean'],
        'structure_to_sequence_ratio': ['mean', 'std'],
        'sequence_to_structure_ratio': ['mean', 'std']
    }).round(4)
    
    category_stats.columns = [
        'structure_covers_count', 'total_prompts', 'structure_covers_percentage',
        'sequence_covers_count', 'sequence_covers_percentage',
        'avg_structure_to_sequence_ratio', 'std_structure_to_sequence_ratio',
        'avg_sequence_to_structure_ratio', 'std_sequence_to_structure_ratio'
    ]
    category_stats['structure_covers_percentage'] *= 100
    category_stats['sequence_covers_percentage'] *= 100
    
    for category in category_stats.index:
        stats = category_stats.loc[category]
        print(f"  {category}:")
        print(f"    Structure covers sequence: {stats['structure_covers_count']:.0f}/{stats['total_prompts']:.0f} ({stats['structure_covers_percentage']:.2f}%)")
        print(f"    Sequence covers structure: {stats['sequence_covers_count']:.0f}/{stats['total_prompts']:.0f} ({stats['sequence_covers_percentage']:.2f}%)")
        print(f"    Avg structure→sequence ratio: {stats['avg_structure_to_sequence_ratio']:.4f} ± {stats['std_structure_to_sequence_ratio']:.4f}")
        print(f"    Avg sequence→structure ratio: {stats['avg_sequence_to_structure_ratio']:.4f} ± {stats['std_sequence_to_structure_ratio']:.4f}")
    
    # Detailed breakdown for each model
    print(f"\nDetailed Model Breakdown:")
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        print(f"\n  {model}:")
        
        structure_covers_prompts = model_df[model_df['structure_covers_sequence'] == True]
        sequence_covers_prompts = model_df[model_df['sequence_covers_structure'] == True]
        
        print(f"    Structure covers sequence prompts ({len(structure_covers_prompts)}/{len(model_df)}):")
        for _, row in structure_covers_prompts.iterrows():
            prompt_short = row['prompt'].split('.')[0][:50] + "..."
            print(f"      - {prompt_short}")
        
        print(f"    Sequence covers structure prompts ({len(sequence_covers_prompts)}/{len(model_df)}):")
        for _, row in sequence_covers_prompts.iterrows():
            prompt_short = row['prompt'].split('.')[0][:50] + "..."
            print(f"      - {prompt_short}")
    
    return model_stats, category_stats

def main():
    print("Starting coverage comparison analysis...")
    
    # Calculate coverage comparison
    df = calculate_coverage_comparison()
    
    if df.empty:
        print("No data found!")
        return
    
    # Print summary statistics
    model_stats, category_stats = print_coverage_comparison_summary(df)
    
    # Save detailed results
    output_file = "coverage_comparison_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")
    
    # Save summary statistics
    model_stats.to_csv("model_coverage_stats.csv")
    category_stats.to_csv("category_coverage_stats.csv")
    print(f"Model statistics saved to: model_coverage_stats.csv")
    print(f"Category statistics saved to: category_coverage_stats.csv")
    
    print("\nCoverage comparison analysis complete!")

if __name__ == "__main__":
    main()
