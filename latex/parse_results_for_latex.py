import json
import os
import numpy as np
from pathlib import Path

def load_metric(model_dir, method, metric_file, metric_key):
    """Load a specific metric from a results file"""
    file_path = os.path.join(model_dir, "evaluation", method, metric_file)
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data.get('overall_metrics', {}).get(metric_key, None)
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None

def get_model_results(model_dir, model_name):
    """Extract all metrics for a model"""
    methods = {
        "Baseline": "direct (samples=1)",
        "Sequence": "sequence [strict] (samples=5)", 
        "Multi-turn": "multi_turn [strict] (samples=5)",
        "Standard": "structure_with_prob [strict] (samples=5)",
        "CoT": "chain_of_thought [strict] (samples=5)",
        "Combined": "combined [strict] (samples=5)"
    }
    
    results = {"model": model_name}
    
    for method_name, method_dir in methods.items():
        # Get diversity (higher is better)
        diversity = load_metric(model_dir, method_dir, "diversity_results.json", "avg_diversity")
        
        # Get Rouge-L (lower is better - convert to percentage and multiply by 100)
        rouge_l = load_metric(model_dir, method_dir, "ngram_results.json", "avg_rouge_l")
        
        # Get quality score (convert from 0-1 scale to 0-100 scale)
        quality = load_metric(model_dir, method_dir, "creative_writing_v3_results.json", "avg_score")
        
        results[method_name] = {
            "diversity": diversity * 100 if diversity is not None else None,  # Convert to percentage
            "rouge_l": rouge_l * 100 if rouge_l is not None else None,        # Convert to percentage  
            "quality": quality * 100 if quality is not None else None         # Convert to 0-100 scale
        }
    
    return results

def calculate_improvements(baseline, method):
    """Calculate percentage improvement over baseline"""
    improvements = {}
    
    if baseline["diversity"] is not None and method["diversity"] is not None:
        improvements["diversity"] = ((method["diversity"] - baseline["diversity"]) / baseline["diversity"]) * 100
    
    if baseline["rouge_l"] is not None and method["rouge_l"] is not None:
        # For Rouge-L, lower is better, so improvement is negative change
        improvements["rouge_l"] = ((baseline["rouge_l"] - method["rouge_l"]) / baseline["rouge_l"]) * 100
    
    if baseline["quality"] is not None and method["quality"] is not None:
        improvements["quality"] = ((method["quality"] - baseline["quality"]) / baseline["quality"]) * 100
    
    return improvements

def format_metric(value, is_best=False):
    """Format metric value for LaTeX table"""
    if value is None:
        return "N/A"
    
    formatted = f"{value:.1f}"
    if is_best:
        formatted = f"\\textbf{{{formatted}}}"
    
    return formatted

def generate_latex_table():
    """Generate LaTeX table from all model results"""
    
    # Model directory mapping
    models = {
        "Claude-4-Sonnet": "anthropic_claude-4-sonnet",
        "Claude-3.7-Sonnet": "anthropic_claude-3.7-sonnet", 
        "Gemini-2.5-Pro": "google_gemini-2.5-pro",
        "Gemini-2.5-Flash": "google_gemini-2.5-flash",
        "GPT-4.1": "openai_gpt-4.1",
        "GPT-4.1-Mini": "openai_gpt-4.1-mini",
        "GPT-o3": "openai_o3",
        "Llama-3.1-70B": "meta-llama_Llama-3.1-70B-Instruct",
        "Llama-3.1-8B": "meta-llama_Llama-3.1-8B-Instruct",
        "DeepSeek-R1": "deepseek_deepseek-r1-0528"
    }
    
    base_dir = "poem_experiments_final"
    all_results = {}
    
    # Collect results for all models
    for model_name, model_dir_name in models.items():
        model_path = os.path.join(base_dir, model_dir_name, f"{model_dir_name}_poem")
        if os.path.exists(model_path):
            results = get_model_results(model_path, model_name)
            all_results[model_name] = results
            print(f"Processed {model_name}")
        else:
            print(f"Warning: Directory not found for {model_name}: {model_path}")
    
    # Generate LaTeX table
    print("\n" + "="*80)
    print("LATEX TABLE DATA")
    print("="*80)
    
    for model_name, results in all_results.items():
        print(f"\n{model_name}:")
        print("-" * 40)
        
        baseline = results.get("Baseline")
        if baseline is None:
            print("No baseline data available")
            continue
            
        # Find best values across all methods for highlighting
        all_diversity = [results[method]["diversity"] for method in ["Baseline", "Sequence", "Multi-turn", "Standard", "CoT", "Combined"] 
                        if results.get(method) and results[method]["diversity"] is not None]
        all_rouge_l = [results[method]["rouge_l"] for method in ["Baseline", "Sequence", "Multi-turn", "Standard", "CoT", "Combined"] 
                      if results.get(method) and results[method]["rouge_l"] is not None]
        all_quality = [results[method]["quality"] for method in ["Baseline", "Sequence", "Multi-turn", "Standard", "CoT", "Combined"] 
                      if results.get(method) and results[method]["quality"] is not None]
        
        best_diversity = max(all_diversity) if all_diversity else None
        best_rouge_l = min(all_rouge_l) if all_rouge_l else None  # Lower is better for Rouge-L
        best_quality = max(all_quality) if all_quality else None
        
        # Print baseline
        if baseline["diversity"] is not None:
            print(f"& Baseline & {format_metric(baseline['diversity'], baseline['diversity'] == best_diversity)} & {format_metric(baseline['rouge_l'], baseline['rouge_l'] == best_rouge_l)} & {format_metric(baseline['quality'], baseline['quality'] == best_quality)} \\\\")
        
        # Print other methods
        for method in ["Sequence", "Multi-turn"]:
            data = results.get(method)
            if data and any(v is not None for v in data.values()):
                print(f"& {method} & {format_metric(data['diversity'], data['diversity'] == best_diversity)} & {format_metric(data['rouge_l'], data['rouge_l'] == best_rouge_l)} & {format_metric(data['quality'], data['quality'] == best_quality)} \\\\")
        
        # Print Verbalized Sampling methods
        print("& \\textbf{Verbalized Sampling} \\\\")
        for method, display_name in [("Standard", "$\\hookrightarrow$ Standard"), ("CoT", "$\\hookrightarrow$ CoT"), ("Combined", "$\\hookrightarrow$ Combined")]:
            data = results.get(method)
            if data and any(v is not None for v in data.values()):
                print(f"& {display_name} & {format_metric(data['diversity'], data['diversity'] == best_diversity)} & {format_metric(data['rouge_l'], data['rouge_l'] == best_rouge_l)} & {format_metric(data['quality'], data['quality'] == best_quality)} \\\\")
        
        # Calculate improvements for the best VS method (find the one with best overall performance)
        vs_methods = ["Standard", "CoT", "Combined"]
        best_vs_method = None
        best_vs_score = -float('inf')
        
        for method in vs_methods:
            data = results.get(method)
            if data and all(v is not None for v in data.values()):
                # Simple scoring: normalize each metric and sum (diversity + quality - rouge_l)
                score = (data["diversity"] / 100) + (data["quality"] / 100) - (data["rouge_l"] / 100)
                if score > best_vs_score:
                    best_vs_score = score
                    best_vs_method = method
        
        if best_vs_method and baseline:
            improvements = calculate_improvements(baseline, results[best_vs_method])
            improvement_str = []
            if "diversity" in improvements:
                improvement_str.append(f"+{improvements['diversity']:.1f}%")
            if "rouge_l" in improvements: 
                improvement_str.append(f"-{improvements['rouge_l']:.1f}%")
            if "quality" in improvements:
                improvement_str.append(f"+{improvements['quality']:.1f}%")
            
            print(f"% Best VS method ({best_vs_method}) improvements: {', '.join(improvement_str)}")
        
        print("\\midrule")

    # Also generate a summary table with key statistics
    print(f"\n\nSUMMARY STATISTICS:")
    print("-" * 40)
    
    # Calculate average improvements across all models for best VS method
    total_diversity_imp = []
    total_rouge_l_imp = []
    total_quality_imp = []
    
    for model_name, results in all_results.items():
        baseline = results.get("Baseline")
        if not baseline:
            continue
            
        # Find best VS method for this model
        vs_methods = ["Standard", "CoT", "Combined"]
        best_vs_method = None
        best_vs_score = -float('inf')
        
        for method in vs_methods:
            data = results.get(method)
            if data and all(v is not None for v in data.values()):
                score = (data["diversity"] / 100) + (data["quality"] / 100) - (data["rouge_l"] / 100)
                if score > best_vs_score:
                    best_vs_score = score
                    best_vs_method = method
        
        if best_vs_method:
            improvements = calculate_improvements(baseline, results[best_vs_method])
            if "diversity" in improvements:
                total_diversity_imp.append(improvements["diversity"])
            if "rouge_l" in improvements:
                total_rouge_l_imp.append(improvements["rouge_l"]) 
            if "quality" in improvements:
                total_quality_imp.append(improvements["quality"])
    
    if total_diversity_imp:
        print(f"Average Diversity Improvement: +{np.mean(total_diversity_imp):.1f}% (±{np.std(total_diversity_imp):.1f}%)")
    if total_rouge_l_imp:
        print(f"Average Rouge-L Improvement: -{np.mean(total_rouge_l_imp):.1f}% (±{np.std(total_rouge_l_imp):.1f}%)")
    if total_quality_imp:
        print(f"Average Quality Improvement: +{np.mean(total_quality_imp):.1f}% (±{np.std(total_quality_imp):.1f}%)")

if __name__ == "__main__":
    generate_latex_table()