import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import os
import json
from statsmodels.stats.weightstats import ttost_ind
plt.style.use('seaborn-v0_8')
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

# Set your equivalence margin here!
equiv_margin = 0.03  # This is a common margin for accuracy/proportion; adjust as needed.
METHOD_MAP = {
    "direct": ("Direct", "direct"),
    "direct_cot": ("CoT", "cot"),
    "sequence": ("Sequence", "sequence"),
    "multi_turn": ("Multi-turn", "multi_turn"),
    "vs_standard": ("VS-Standard", "structure_with_prob"),
    "vs_cot": ("VS-CoT", "chain_of_thought"),
    "vs_combined": ("VS-Combined", "combined"),
}

def generate_latex_factual_table(metrics_values, all_model_names, all_methods, metric_labels):
    """
    Generate a LaTeX table summarizing Top@1 and Pass@K accuracy (mean±std) for each method, 
    and Equivalence test results (p-values) for VS variants vs. best baseline.
    """
    # Table order and display names
    table_methods = [
        "direct",
        "direct_cot",
        "sequence",
        "multi_turn",
        "vs_standard",
        "vs_cot",
        "vs_combined"
    ]
    display_names = [METHOD_MAP[m][0] for m in table_methods]
    metrics = ["first_response_accuracy", "pass_at_k_accuracy"]
    baseline_methods = ["direct", "direct_cot", "sequence", "multi_turn"]
    
    # Calculate means and stds for all methods
    means = {m: {} for m in table_methods}
    stds = {m: {} for m in table_methods}
    for m in table_methods:
        for metric in metrics:
            vals = []
            for model_name in all_model_names:
                if model_name in metrics_values:
                    for method_name, method_data in metrics_values[model_name].items():
                        if METHOD_MAP[m][1] in method_name:
                            vals.extend(method_data[metric])
            means[m][metric] = np.mean(vals) if vals else np.nan
            stds[m][metric] = np.std(vals) if vals else np.nan
    
    # Find best baseline for each metric
    best_baseline = {}
    for metric in metrics:
        baseline_means = []
        for m in baseline_methods:
            baseline_means.append(means[m][metric])
        best_baseline_idx = np.nanargmax(baseline_means)
        best_baseline[metric] = baseline_methods[best_baseline_idx]
    
    # Compute equivalence test results for VS methods vs best baseline
    equiv_results = {m: {metric: '--' for metric in metrics} for m in table_methods}
    equiv_detailed = {m: {metric: {} for metric in metrics} for m in table_methods}
    
    for metric in metrics:
        best_baseline_method = best_baseline[metric]
        
        # Get best baseline data
        baseline_data = []
        for model_name in all_model_names:
            if model_name in metrics_values:
                for method_name, method_data in metrics_values[model_name].items():
                    if METHOD_MAP[best_baseline_method][1] in method_name:
                        baseline_data.extend(method_data[metric])
        
        # Test VS methods against best baseline
        for m in ["vs_standard", "vs_cot", "vs_combined"]:
            vs_data = []
            for model_name in all_model_names:
                if model_name in metrics_values:
                    for method_name, method_data in metrics_values[model_name].items():
                        if METHOD_MAP[m][1] in method_name:
                            vs_data.extend(method_data[metric])
            
            if len(vs_data) > 1 and len(baseline_data) > 1:
                try:
                    # Perform TOST test
                    result = ttost_ind(vs_data, baseline_data, -equiv_margin, equiv_margin, usevar='unequal')
                    
                    # Extract results - ttost_ind returns different structures in different versions
                    if hasattr(result, 'pvalue'):
                        pvalue = result.pvalue
                        statistic = getattr(result, 'statistic', 'N/A')
                    elif len(result) >= 2:
                        pvalue = result[0]
                        statistic = result[1] if len(result) > 1 else 'N/A'
                    else:
                        pvalue = result
                        statistic = 'N/A'
                    
                    # Store detailed results
                    equiv_detailed[m][metric] = {
                        'pvalue': pvalue,
                        'statistic': statistic,
                        'vs_mean': np.mean(vs_data),
                        'baseline_mean': np.mean(baseline_data),
                        'vs_std': np.std(vs_data),
                        'baseline_std': np.std(baseline_data),
                        'n_vs': len(vs_data),
                        'n_baseline': len(baseline_data)
                    }
                    
                    # Format for table display
                    if pvalue < 0.001:
                        equiv_results[m][metric] = "p<0.001"
                    elif pvalue < 0.01:
                        equiv_results[m][metric] = f"p={pvalue:.3f}"
                    elif pvalue < 0.05:
                        equiv_results[m][metric] = f"p={pvalue:.3f}*"
                    else:
                        equiv_results[m][metric] = f"p={pvalue:.3f}"
                    
                except Exception as e:
                    print(f"Error in equivalence test for {m}, {metric}: {e}")
                    equiv_results[m][metric] = "Error"
                    equiv_detailed[m][metric] = {'error': str(e)}
            else:
                equiv_results[m][metric] = '--'
                equiv_detailed[m][metric] = {'insufficient_data': True}

    # Build LaTeX table
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lcc|cc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Top@1 Accuracy & Pass@K Accuracy & Top@1 TOST p-val & Pass@K TOST p-val \\")
    lines.append(r"\midrule")
    
    for m, disp in zip(table_methods, display_names):
        row = []
        
        # Check if this method is the best baseline for any metric
        is_best_top1 = (m == best_baseline["first_response_accuracy"])
        is_best_passk = (m == best_baseline["pass_at_k_accuracy"])
        
        for i, metric in enumerate(metrics):
            mean = means[m][metric]
            std = stds[m][metric]
            cell = f"{mean:.3f}$_{{\pm{std:.3f}}}$" if not np.isnan(mean) and not np.isnan(std) else "--"
            
            # Bold if this is the best baseline for this metric
            if (i == 0 and is_best_top1) or (i == 1 and is_best_passk):
                cell = f"\\textbf{{{cell}}}"
            
            row.append(cell)
        
        # Equiv. results for VS methods only
        if m in ["vs_standard", "vs_cot", "vs_combined"]:
            p1 = equiv_results[m]["first_response_accuracy"]
            p2 = equiv_results[m]["pass_at_k_accuracy"]
        else:
            p1 = p2 = '--'
        
        # Bold method name if it's best for any metric
        if is_best_top1 or is_best_passk:
            disp = f"\\textbf{{{disp}}}"
        
        lines.append(f"{disp} & {row[0]} & {row[1]} & {p1} & {p2} \\\\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\vspace{-0.5em}")
    lines.append(r"\caption{Top@1 and Pass@K accuracy ($\mu_{\pm\sigma}$) for each method, and TOST equivalence test p-values (* indicates equivalence at $\alpha=0.05$) for VS variants vs. best baseline. Equivalence margin = " + f"{equiv_margin}" + r".}")
    lines.append(r"\label{tab:compact_all_in_one}")
    lines.append(r"\end{table}")
    
    latex_table = '\n'.join(lines)
    print(latex_table)
    
    # Print detailed summary
    print(f"\n" + "="*80)
    print("DETAILED EQUIVALENCE TEST RESULTS")
    print("="*80)
    print(f"Equivalence margin: ±{equiv_margin}")
    print(f"Best baselines:")
    print(f"  Top@1 Accuracy: {METHOD_MAP[best_baseline['first_response_accuracy']][0]} ({means[best_baseline['first_response_accuracy']]['first_response_accuracy']:.4f})")
    print(f"  Pass@K Accuracy: {METHOD_MAP[best_baseline['pass_at_k_accuracy']][0]} ({means[best_baseline['pass_at_k_accuracy']]['pass_at_k_accuracy']:.4f})")
    
    for metric in metrics:
        print(f"\n{metric.replace('_', ' ').title()}:")
        print(f"  Best baseline: {METHOD_MAP[best_baseline[metric]][0]}")
        
        for m in ["vs_standard", "vs_cot", "vs_combined"]:
            if metric in equiv_detailed[m] and 'pvalue' in equiv_detailed[m][metric]:
                details = equiv_detailed[m][metric]
                vs_mean = details['vs_mean']
                baseline_mean = details['baseline_mean']
                pvalue = details['pvalue']
                diff = vs_mean - baseline_mean
                
                equiv_status = "EQUIVALENT" if pvalue < 0.05 else "NOT EQUIVALENT"
                
                print(f"    {METHOD_MAP[m][0]}:")
                print(f"      VS mean: {vs_mean:.4f} ± {details['vs_std']:.4f} (n={details['n_vs']})")
                print(f"      Baseline mean: {baseline_mean:.4f} ± {details['baseline_std']:.4f} (n={details['n_baseline']})")
                print(f"      Difference: {diff:+.4f}")
                print(f"      TOST p-value: {pvalue:.6f}")
                print(f"      Result: {equiv_status}")
            elif metric in equiv_detailed[m]:
                if 'error' in equiv_detailed[m][metric]:
                    print(f"    {METHOD_MAP[m][0]}: ERROR - {equiv_detailed[m][metric]['error']}")
                elif 'insufficient_data' in equiv_detailed[m][metric]:
                    print(f"    {METHOD_MAP[m][0]}: Insufficient data")
                else:
                    print(f"    {METHOD_MAP[m][0]}: No results")


def main():
    folder = "method_results_simple_qa"
    task_name = "simple_qa"
    all_model_names = [
        "gpt-4.1-mini",
        "gpt-4.1",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "meta-llama_Llama-3.1-70B-Instruct",
        "deepseek-r1",
        "o3",
        "claude-4-sonnet",
    ]
    metrics = ["first_response_accuracy", "pass_at_k_accuracy"]
    metric_labels = {
        "first_response_accuracy": "Top@1 Accuracy ↑",
        "pass_at_k_accuracy": "Pass@K Accuracy ↑"
    }
    all_methods = [
        "direct",
        "direct_cot",
        "sequence",
        "multi_turn",
        "vs_standard",
        "vs_cot",
        "vs_combined"
    ]
    
    metrics_values = {}
    for model_dir in os.listdir(folder):
        if not model_dir.endswith(f"_{task_name}"):
            continue
        model_name = model_dir.replace(f"_{task_name}", "")
        evaluation_dir = Path(folder) / model_dir / "evaluation"
        if not evaluation_dir.exists():
            print(f"Warning: No evaluation directory found for {model_name}")
            continue
        
        for method_dir in evaluation_dir.iterdir():
            if not method_dir.is_dir():
                continue
            method_name = method_dir.name
            results_file = method_dir / "factuality_results.json"
            if not results_file.exists():
                print(f"Warning: No results file found for {model_name} - {method_name}")
                continue
            
            try:
                with open(results_file, "r") as f:
                    data = json.load(f)
                aggregate_metrics = data.get("overall_metrics", {})
                
                # Fix method name mapping
                for key, (display_name, internal_name) in METHOD_MAP.items():
                    if internal_name == method_name:
                        method_name = display_name
                        break
                
                if model_name not in metrics_values:
                    metrics_values[model_name] = {}
                if method_name not in metrics_values[model_name]:
                    metrics_values[model_name][method_name] = {metric: [] for metric in metrics}
                
                for metric in metrics:
                    if metric in aggregate_metrics:
                        metrics_values[model_name][method_name][metric].append(aggregate_metrics[metric])
            except Exception as e:
                print(f"Error reading {results_file}: {e}")

    print(len(metrics_values))
    print(metrics_values)
    
    # generate_latex_factual_table(metrics_values, all_model_names, all_methods, metric_labels)


if __name__ == "__main__":
    main()