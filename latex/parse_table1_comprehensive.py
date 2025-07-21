#!/usr/bin/env python3

import json
import os
import sys
from pathlib import Path

# Try to import numpy, use basic statistics if not available
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Warning: numpy not available, using basic statistics")

def mean(values):
    return sum(values) / len(values) if values else 0

def std(values):
    if not values or len(values) < 2:
        return 0
    m = mean(values)
    return (sum((x - m) ** 2 for x in values) / (len(values) - 1)) ** 0.5

def load_metric(model_dir, method, metric_file, metric_key):
    """Load a specific metric from a results file"""
    file_path = os.path.join(model_dir, "evaluation", method, metric_file)
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data.get('overall_metrics', {}).get(metric_key, None)
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None

def get_creativity_results(base_dir, task_name):
    """Extract creativity results (poem, joke)"""
    models = {
        "Claude-4-Sonnet": "anthropic_claude-4-sonnet",
        "Claude-3.7-Sonnet": "anthropic_claude-3.7-sonnet", 
        "Gemini-2.5-Pro": "google_gemini-2.5-pro",
        "Gemini-2.5-Flash": "google_gemini-2.5-flash",
        "GPT-4.1": "openai_gpt-4.1",
        "GPT-4.1-Mini": "openai_gpt-4.1-mini",
        "Llama-3.1-70B": "meta-llama_Llama-3.1-70B-Instruct",
        "DeepSeek-R1": "deepseek_deepseek-r1-0528",
        "GPT-o3": "openai_o3",
    }
    
    methods = {
        "Direct": "direct (samples=1)",
        "VS-Combined": "combined [strict] (samples=5)"
    }
    
    results = {}
    
    for model_name, model_dir_name in models.items():
        model_path = os.path.join(base_dir, model_dir_name, f"{model_dir_name}_{task_name}")
        
        if not os.path.exists(model_path):
            continue
            
        results[model_name] = {}
        
        for method_name, method_dir in methods.items():
            # Get diversity (higher is better) - try both metric keys
            diversity_avg = load_metric(model_path, method_dir, "diversity_results.json", "avg_diversity")
            if diversity_avg is None:
                diversity_avg = load_metric(model_path, method_dir, "diversity_results.json", "average_diversity")
            
            if task_name == "poem":
                # Get quality score (0-1 scale)
                quality_avg = load_metric(model_path, method_dir, "creative_writing_v3_results.json", "avg_score")
            elif task_name == "joke":
                # Get joke quality score
                quality_avg = load_metric(model_path, method_dir, "joke_quality_results.json", "avg_normalized_overall")
            
            results[model_name][method_name] = {
                "diversity": diversity_avg * 100 if diversity_avg is not None else None,
                "quality": quality_avg * 100 if quality_avg is not None else None
            }
    
    return results

def get_bias_results(base_dir="method_results_bias"):
    """Extract bias results (state_name)"""
    # Only have Llama-3.1-70B results for bias
    model_path = os.path.join(base_dir, "meta-llama_Llama-3.1-70B-Instruct_state_name")
    
    if not os.path.exists(model_path):
        return {}
        
    methods = {
        "Direct": "direct (samples=1)",
        "VS-Combined": "combined [strict] (samples=20)"
    }
    
    results = {"Llama-3.1-70B": {}}
    
    for method_name, method_dir in methods.items():
        # Get unique recall (higher is better)
        unique_recall = load_metric(model_path, method_dir, "response_count_results.json", "unique_recall")
        
        # Get KL divergence (lower is better) 
        kl_divergence = load_metric(model_path, method_dir, "response_count_results.json", "kl_divergence")
        
        results["Llama-3.1-70B"][method_name] = {
            "unique_recall": unique_recall * 100 if unique_recall is not None else None,
            "kl_divergence": kl_divergence if kl_divergence is not None else None
        }
    
    return results

def get_commonsense_results(base_dir="method_results_simple_qa"):
    """Extract commonsense results (simple_qa)"""
    # Only have Llama-3.1-70B results for commonsense
    model_path = os.path.join(base_dir, "meta-llama_Llama-3.1-70B-Instruct_simple_qa")
    
    if not os.path.exists(model_path):
        return {}
        
    methods = {
        "Direct": "direct (samples=1)",
        "VS-Combined": "combined [strict] (samples=5)"
    }
    
    results = {"Llama-3.1-70B": {}}
    
    for method_name, method_dir in methods.items():
        # Get accuracy (higher is better)
        accuracy = load_metric(model_path, method_dir, "factuality_results.json", "accuracy")
        
        results["Llama-3.1-70B"][method_name] = {
            "accuracy": accuracy * 100 if accuracy is not None else None
        }
    
    return results

def format_metric(value, is_percentage=True):
    """Format metric value for LaTeX table"""
    if value is None:
        return "---"
    
    if is_percentage:
        return f"{value:.1f}\\%"
    else:
        return f"{value:.2f}"

def calculate_improvement(baseline, verbalized):
    """Calculate improvement percentage"""
    if baseline is None or verbalized is None:
        return None
    
    if baseline == 0:
        return None
        
    return ((verbalized - baseline) / baseline) * 100

def generate_table1_latex():
    """Generate LaTeX Table 1 with comprehensive results"""
    
    print("Loading creativity results...")
    poem_results = get_creativity_results("poem_experiments_final", "poem")
    joke_results = get_creativity_results("joke_experiments_final", "joke")
    
    
    print("Loading bias results...")
    bias_results = get_bias_results()
    
    print("Loading commonsense results...")
    commonsense_results = get_commonsense_results()
    
    # Models to include in Table 1
    target_models = ["GPT-4.1", "GPT-4.1-Mini", "Claude-4-Sonnet", "Claude-3.7-Sonnet", "Gemini-2.5-Pro"]
    
    output_file = "table1_comprehensive.tex"
    
    with open(output_file, 'w') as f:
        f.write("% Table 1: Comprehensive Results Summary\n")
        f.write("% Generated automatically from experimental results\n\n")
        
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Performance comparison between baseline and verbalized sampling methods across different task categories.}\n")
        f.write("\\label{tab:comprehensive_results}\n")
        f.write("\\resizebox{\\textwidth}{!}{%\n")
        f.write("\\begin{tabular}{l|l|c|c|c||c|c|c||c|c||c|c}\n")
        f.write("\\toprule\n")
        
        # Table header
        f.write("& & \\multicolumn{3}{c||}{\\textbf{Creativity}} & \\multicolumn{3}{c||}{\\textbf{Bias}} & \\multicolumn{2}{c||}{\\textbf{Simulation}} & \\multicolumn{2}{c}{\\textbf{Commonsense}} \\\\\n")
        f.write("\\cmidrule(lr){3-5} \\cmidrule(lr){6-8} \\cmidrule(lr){9-10} \\cmidrule(lr){11-12}\n")
        f.write("\\textbf{Model} & \\textbf{Method} & \\textbf{Poem} & \\textbf{Joke} & \\textbf{Avg} & \\textbf{State} & \\textbf{Rand} & \\textbf{Avg} & \\textbf{Dial} & \\textbf{Avg} & \\textbf{QA} & \\textbf{Avg} \\\\\n")
        f.write("& & Div↑ & Div↑ & Div↑ & UR↑ & UR↑ & UR↑ & KS↓ & KS↓ & Acc↑ & Acc↑ \\\\\n")
        f.write("\\midrule\n")
        
        # Generate rows for each model
        for model_name in target_models:
            f.write(f"\\multirow{{3}}{{*}}{{{model_name}}}\n")
            
            # Baseline row
            poem_div_baseline = poem_results.get(model_name, {}).get("Direct", {}).get("diversity")
            joke_div_baseline = joke_results.get(model_name, {}).get("Direct", {}).get("diversity")
            
            # Average creativity baseline
            creativity_baseline = None
            if poem_div_baseline is not None and joke_div_baseline is not None:
                creativity_baseline = (poem_div_baseline + joke_div_baseline) / 2
            
            # For now, use placeholder values for missing tasks
            f.write(f"& Baseline & {format_metric(poem_div_baseline)} & {format_metric(joke_div_baseline)} & {format_metric(creativity_baseline)} & --- & --- & --- & --- & --- & --- & --- \\\\\n")
            
            # Verbalized row
            poem_div_verbalized = poem_results.get(model_name, {}).get("VS-Combined", {}).get("diversity")
            joke_div_verbalized = joke_results.get(model_name, {}).get("VS-Combined", {}).get("diversity")
            
            # Average creativity verbalized
            creativity_verbalized = None
            if poem_div_verbalized is not None and joke_div_verbalized is not None:
                creativity_verbalized = (poem_div_verbalized + joke_div_verbalized) / 2
            
            f.write(f"& Verbalized & {format_metric(poem_div_verbalized)} & {format_metric(joke_div_verbalized)} & {format_metric(creativity_verbalized)} & --- & --- & --- & --- & --- & --- & --- \\\\\n")
            
            # Gap row
            poem_gap = calculate_improvement(poem_div_baseline, poem_div_verbalized)
            joke_gap = calculate_improvement(joke_div_baseline, joke_div_verbalized)
            creativity_gap = calculate_improvement(creativity_baseline, creativity_verbalized)
            
            f.write(f"& Gap & {format_metric(poem_gap, True) if poem_gap is not None else '---'} & {format_metric(joke_gap, True) if joke_gap is not None else '---'} & {format_metric(creativity_gap, True) if creativity_gap is not None else '---'} & --- & --- & --- & --- & --- & --- & --- \\\\\n")
            
            if model_name != target_models[-1]:
                f.write("\\midrule\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("}\n")
        f.write("\\end{table}\n")
    
    print(f"✅ Generated comprehensive Table 1 in {output_file}")
    
    # Generate summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    
    # Calculate overall improvements
    all_poem_improvements = []
    all_joke_improvements = []
    
    for model_name in target_models:
        poem_baseline = poem_results.get(model_name, {}).get("Direct", {}).get("diversity")
        poem_verbalized = poem_results.get(model_name, {}).get("VS-Combined", {}).get("diversity")
        joke_baseline = joke_results.get(model_name, {}).get("Direct", {}).get("diversity")
        joke_verbalized = joke_results.get(model_name, {}).get("VS-Combined", {}).get("diversity")
        
        if poem_baseline is not None and poem_verbalized is not None:
            improvement = calculate_improvement(poem_baseline, poem_verbalized)
            if improvement is not None:
                all_poem_improvements.append(improvement)
        
        if joke_baseline is not None and joke_verbalized is not None:
            improvement = calculate_improvement(joke_baseline, joke_verbalized)
            if improvement is not None:
                all_joke_improvements.append(improvement)
    
    if all_poem_improvements:
        if HAS_NUMPY:
            print(f"📊 Poem Diversity Improvements: {np.mean(all_poem_improvements):.1f}% ± {np.std(all_poem_improvements):.1f}%")
        else:
            print(f"📊 Poem Diversity Improvements: {mean(all_poem_improvements):.1f}% ± {std(all_poem_improvements):.1f}%")
    if all_joke_improvements:
        if HAS_NUMPY:
            print(f"📊 Joke Diversity Improvements: {np.mean(all_joke_improvements):.1f}% ± {np.std(all_joke_improvements):.1f}%")
        else:
            print(f"📊 Joke Diversity Improvements: {mean(all_joke_improvements):.1f}% ± {std(all_joke_improvements):.1f}%")
    
    # Print data availability summary
    print("\n" + "="*50)
    print("DATA AVAILABILITY SUMMARY")
    print("="*50)
    
    print(f"📈 Poem results: {len(poem_results)} models")
    print(f"📈 Joke results: {len(joke_results)} models") 
    print(f"📈 Bias results: {len(bias_results)} models")
    print(f"📈 Commonsense results: {len(commonsense_results)} models")
    
    print("\nNOTE: Table 1 currently shows creativity results (poem, joke) for main models.")
    print("Additional task results (bias, simulation, commonsense) can be added as data becomes available.")

if __name__ == "__main__":
    generate_table1_latex()