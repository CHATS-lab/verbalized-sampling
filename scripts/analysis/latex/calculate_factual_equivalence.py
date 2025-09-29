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
equiv_margin = 0.03 # 0.03, This is a common margin for accuracy/proportion; adjust as needed.

# Bootstrap parameters
N_BOOTSTRAP = 1000  # Number of bootstrap samples
BOOTSTRAP_CONFIDENCE = 0.95  # Confidence level for bootstrap CI
METHOD_MAP = {
    "direct": ("Direct", "direct"),
    "direct_cot": ("CoT", "direct_cot"),
    "sequence": ("Sequence", "sequence"),
    "multi_turn": ("Multi-turn", "multi_turn"),
    "vs_standard": ("VS-Standard", "vs_standard"),
    "vs_cot": ("VS-CoT", "vs_cot"),
    "vs_combined": ("VS-Combined", "vs_multi"),
}

def bootstrap_equivalence_test(sample1, sample2, equiv_margin, n_bootstrap=N_BOOTSTRAP, confidence_level=BOOTSTRAP_CONFIDENCE, random_seed=42):
    """
    Perform bootstrap-based equivalence test using the Two One-Sided Tests (TOST) approach.
    
    Args:
        sample1: First sample (VS method)
        sample2: Second sample (baseline method)
        equiv_margin: Equivalence margin (delta)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        random_seed: Random seed for reproducibility
    
    Returns:
        dict: Contains p-value, confidence interval, and test details
    """
    np.random.seed(random_seed)
    
    sample1 = np.array(sample1)
    sample2 = np.array(sample2)
    
    if len(sample1) < 2 or len(sample2) < 2:
        return {
            'pvalue': np.nan,
            'equivalent': False,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'error': 'Insufficient sample size'
        }
    
    # Observed difference
    observed_diff = np.mean(sample1) - np.mean(sample2)
    
    # Bootstrap resampling
    bootstrap_diffs = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        boot_sample1 = np.random.choice(sample1, size=len(sample1), replace=True)
        boot_sample2 = np.random.choice(sample2, size=len(sample2), replace=True)
        
        # Calculate difference
        boot_diff = np.mean(boot_sample1) - np.mean(boot_sample2)
        bootstrap_diffs.append(boot_diff)
    
    bootstrap_diffs = np.array(bootstrap_diffs)
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_diffs, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))
    
    # TOST equivalence test using bootstrap
    # H0: |difference| >= equiv_margin (not equivalent)
    # H1: |difference| < equiv_margin (equivalent)
    
    # Two one-sided tests:
    # Test 1: H0: diff <= -equiv_margin vs H1: diff > -equiv_margin
    # Test 2: H0: diff >= equiv_margin vs H1: diff < equiv_margin
    
    # For bootstrap, we check if the confidence interval is entirely within [-equiv_margin, equiv_margin]
    equivalent = (ci_lower > -equiv_margin) and (ci_upper < equiv_margin)
    
    # Calculate p-value approximation using bootstrap distribution
    # P-value is the probability that |bootstrap_diff| >= equiv_margin
    p_upper = np.mean(bootstrap_diffs >= equiv_margin)  # P(diff >= margin)
    p_lower = np.mean(bootstrap_diffs <= -equiv_margin)  # P(diff <= -margin)
    
    # TOST p-value is the maximum of the two one-sided p-values
    # But we want the probability of being within the equivalence region
    # So we use 1 - max(p_upper, p_lower) as an approximation
    p_value = max(p_upper, p_lower)
    
    # Alternative: use the proportion of bootstrap samples outside the equivalence region
    outside_equiv_region = np.mean((bootstrap_diffs <= -equiv_margin) | (bootstrap_diffs >= equiv_margin))
    p_value_alt = outside_equiv_region
    
    return {
        'pvalue': p_value_alt,  # Use alternative p-value calculation
        'equivalent': equivalent,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'observed_diff': observed_diff,
        'bootstrap_mean': np.mean(bootstrap_diffs),
        'bootstrap_std': np.std(bootstrap_diffs),
        'n_bootstrap': n_bootstrap
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
    
    # Calculate means and stds for all methods (only using specified models)
    means = {m: {} for m in table_methods}
    stds = {m: {} for m in table_methods}
    for m in table_methods:
        for metric in metrics:
            vals = []
            # Only use models from all_model_names
            if m in metrics_values:
                for model_name in all_model_names:
                    if model_name in metrics_values[m]:
                        vals.extend(metrics_values[m][model_name][metric])
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
    
    # Compute bootstrap equivalence test results for all methods vs best baseline
    equiv_results = {m: {metric: '--' for metric in metrics} for m in table_methods}
    equiv_detailed = {m: {metric: {} for metric in metrics} for m in table_methods}
    
    for metric in metrics:
        best_baseline_method = best_baseline[metric]
        
        # Get best baseline data (only from specified models)
        baseline_data = []
        if best_baseline_method in metrics_values:
            for model_name in all_model_names:
                if model_name in metrics_values[best_baseline_method]:
                    baseline_data.extend(metrics_values[best_baseline_method][model_name][metric])
        
        # Test only VS methods against best baseline using bootstrap
        vs_methods = ["vs_standard", "vs_cot", "vs_combined"]
        
        for m in table_methods:
            if m not in vs_methods:
                # Skip non-VS methods - no equivalence test
                equiv_results[m][metric] = '--'
                equiv_detailed[m][metric] = {'non_vs_method': True}
                continue
                
            method_data = []
            if m in metrics_values:
                for model_name in all_model_names:
                    if model_name in metrics_values[m]:
                        method_data.extend(metrics_values[m][model_name][metric])
            

            
            if len(method_data) > 1 and len(baseline_data) > 1:
                try:
                    # Perform bootstrap equivalence test
                    bootstrap_result = bootstrap_equivalence_test(
                        method_data, baseline_data, equiv_margin,
                        n_bootstrap=N_BOOTSTRAP,
                        confidence_level=BOOTSTRAP_CONFIDENCE
                    )
                    
                    if 'error' in bootstrap_result:
                        equiv_results[m][metric] = "Error"
                        equiv_detailed[m][metric] = bootstrap_result
                    else:
                        pvalue = bootstrap_result['pvalue']
                        
                        # Store detailed results
                        equiv_detailed[m][metric] = {
                            'pvalue': pvalue,
                            'equivalent': bootstrap_result['equivalent'],
                            'ci_lower': bootstrap_result['ci_lower'],
                            'ci_upper': bootstrap_result['ci_upper'],
                            'observed_diff': bootstrap_result['observed_diff'],
                            'bootstrap_mean': bootstrap_result['bootstrap_mean'],
                            'bootstrap_std': bootstrap_result['bootstrap_std'],
                            'method_mean': np.mean(method_data),
                            'baseline_mean': np.mean(baseline_data),
                            'method_std': np.std(method_data),
                            'baseline_std': np.std(baseline_data),
                            'n_method': len(method_data),
                            'n_baseline': len(baseline_data),
                            'n_bootstrap': bootstrap_result['n_bootstrap'],
                            'baseline_method': METHOD_MAP[best_baseline_method][0]
                        }
                        
                        # Format for table display based on equivalence and p-value
                        if bootstrap_result['equivalent']:
                            if pvalue < 0.001:
                                equiv_results[m][metric] = "p<0.001*"
                            elif pvalue < 0.01:
                                equiv_results[m][metric] = f"p={pvalue:.3f}*"
                            elif pvalue < 0.05:
                                equiv_results[m][metric] = f"p={pvalue:.3f}*"
                            else:
                                equiv_results[m][metric] = f"p={pvalue:.3f}*"
                        else:
                            if pvalue < 0.001:
                                equiv_results[m][metric] = "p<0.001"
                            else:
                                equiv_results[m][metric] = f"p={pvalue:.3f}"
                    
                except Exception as e:
                    print(f"Error in bootstrap equivalence test for {m}, {metric}: {e}")
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
    lines.append(r"Method & Top@1 Accuracy & Pass@K Accuracy & Top@1 Bootstrap p-val & Pass@K Bootstrap p-val \\")
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
        
        # Bootstrap equivalence results only for VS methods
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
    lines.append(r"\caption{Top@1 and Pass@K accuracy ($\mu_{\pm\sigma}$) for each method, and bootstrap equivalence test p-values (* indicates equivalence) for VS methods vs. best baseline. Equivalence margin = " + f"{equiv_margin}" + f", bootstrap samples = {N_BOOTSTRAP}" + r".}")
    lines.append(r"\label{tab:compact_all_in_one}")
    lines.append(r"\end{table}")
    
    latex_table = '\n'.join(lines)
    print(latex_table)
    
    # Print detailed summary
    print(f"\n" + "="*80)
    print("DETAILED BOOTSTRAP EQUIVALENCE TEST RESULTS")
    print("="*80)
    print(f"Equivalence margin: ±{equiv_margin}")
    print(f"Bootstrap samples: {N_BOOTSTRAP}")
    print(f"Confidence level: {BOOTSTRAP_CONFIDENCE}")
    print(f"Best baselines:")
    print(f"  Top@1 Accuracy: {METHOD_MAP[best_baseline['first_response_accuracy']][0]} ({means[best_baseline['first_response_accuracy']]['first_response_accuracy']:.4f})")
    print(f"  Pass@K Accuracy: {METHOD_MAP[best_baseline['pass_at_k_accuracy']][0]} ({means[best_baseline['pass_at_k_accuracy']]['pass_at_k_accuracy']:.4f})")
    
    for metric in metrics:
        print(f"\n{metric.replace('_', ' ').title()}:")
        print(f"  Best baseline: {METHOD_MAP[best_baseline[metric]][0]}")
        
        # Only show results for VS methods
        vs_methods = ["vs_standard", "vs_cot", "vs_combined"]
        
        for m in vs_methods:
            if metric in equiv_detailed[m] and 'pvalue' in equiv_detailed[m][metric]:
                details = equiv_detailed[m][metric]
                method_mean = details['method_mean']
                baseline_mean = details['baseline_mean']
                pvalue = details['pvalue']
                diff = details['observed_diff']
                
                # Use bootstrap-specific equivalence determination
                equiv_status = "EQUIVALENT" if details['equivalent'] else "NOT EQUIVALENT"
                
                print(f"    {METHOD_MAP[m][0]} vs {details['baseline_method']}:")
                print(f"      Method mean: {method_mean:.4f} ± {details['method_std']:.4f} (n={details['n_method']})")
                print(f"      Baseline mean: {baseline_mean:.4f} ± {details['baseline_std']:.4f} (n={details['n_baseline']})")
                print(f"      Observed difference: {diff:+.4f}")
                print(f"      Bootstrap mean diff: {details['bootstrap_mean']:+.4f} ± {details['bootstrap_std']:.4f}")
                print(f"      Bootstrap 95% CI: [{details['ci_lower']:+.4f}, {details['ci_upper']:+.4f}]")
                print(f"      Bootstrap p-value: {pvalue:.6f}")
                print(f"      Result: {equiv_status}")
                if details['equivalent']:
                    print(f"      ✓ CI entirely within equivalence region [-{equiv_margin:.3f}, +{equiv_margin:.3f}]")
                else:
                    print(f"      ✗ CI extends outside equivalence region [-{equiv_margin:.3f}, +{equiv_margin:.3f}]")
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
        "anthropic_claude-4-sonnet",
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
        
        # Only process models in our target list
        if model_name not in all_model_names:
            continue
            
        evaluation_dir = Path(folder) / model_dir / "evaluation"
        if not evaluation_dir.exists():
            print(f"Warning: No evaluation directory found for {model_name}")
            continue
        
        for method_dir in evaluation_dir.iterdir():
            if not method_dir.is_dir():
                continue
            method_name = method_dir.name
            method_name = method_name.split()[0]
            results_file = method_dir / "factuality_results.json"
            if not results_file.exists():
                print(f"Warning: No results file found for {model_name} - {method_name}")
                continue
            
            try:
                with open(results_file, "r") as f:
                    data = json.load(f)
                aggregate_metrics = data.get("overall_metrics", {})
                
                # Fix method name mapping - keep the key (not display name) for consistency
                mapped_method_key = None
                for key, (display_name, internal_name) in METHOD_MAP.items():
                    if internal_name == method_name:
                        mapped_method_key = key
                        break
                
                if mapped_method_key is None:
                    continue  # Skip methods not in our mapping
                
                method_name = mapped_method_key
                
                if method_name not in metrics_values:
                    metrics_values[method_name] = {}
                if model_name not in metrics_values[method_name]:
                    metrics_values[method_name][model_name] = {metric: [] for metric in metrics}
                
                for metric in metrics:
                    if metric in aggregate_metrics:
                        metrics_values[method_name][model_name][metric].append(aggregate_metrics[metric])
            except Exception as e:
                print(f"Error reading {results_file}: {e}")

    print(f"Found data for {len(metrics_values)} methods")
    for method, models in metrics_values.items():
        print(f"  {method}: {len(models)} models")
        missing_models = set(all_model_names) - set(models.keys())
        if missing_models:
            print(f"    Missing models: {missing_models}")
    
    print(f"\nExpected methods from all_methods: {all_methods}")
    print(f"Available methods in data: {list(metrics_values.keys())}")
    print(f"Expected models: {all_model_names}")
    print(f"Total expected models: {len(all_model_names)}")
    
    generate_latex_factual_table(metrics_values, all_model_names, all_methods, metric_labels)


if __name__ == "__main__":
    main()