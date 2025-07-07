import json
import os
from pathlib import Path
import numpy as np

def main():
    folder = "method_results_bias"
    task_name = "state_name"
    
    # Define the four metrics we want to analyze
    metrics = ["kl_divergence", "chi_square", "precision", "unique_recall_rate"]
    
    # Collect all data
    metrics_values = {}
    
    # Iterate through all model directories
    for model_dir in os.listdir(folder):
        if not model_dir.endswith(f"_{task_name}"):
            continue
            
        model_name = model_dir.replace(f"_{task_name}", "")
        evaluation_dir = Path(folder) / model_dir / "evaluation"
        
        if not evaluation_dir.exists():
            print(f"Warning: No evaluation directory found for {model_name}")
            continue
            
        # Iterate through all method directories
        for method_dir in evaluation_dir.iterdir():
            if not method_dir.is_dir():
                continue
                
            method_name = method_dir.name
            results_file = method_dir / "response_count_results.json"
            
            if not results_file.exists():
                print(f"Warning: No results file found for {model_name} - {method_name}")
                continue
                
            try:
                with open(results_file, "r") as f:
                    data = json.load(f)
                
                aggregate_metrics = data.get("overall_metrics", {})
                per_prompt_stats = aggregate_metrics.get("per_prompt_stats", {})
                
                # Initialize data structure for this model-method combination
                if model_name not in metrics_values:
                    metrics_values[model_name] = {}
                if method_name not in metrics_values[model_name]:
                    metrics_values[model_name][method_name] = {metric: [] for metric in metrics}
                
                # Collect metric values from all prompts
                for prompt_stats in per_prompt_stats.values():
                    for metric in metrics:
                        if metric in prompt_stats:
                            metrics_values[model_name][method_name][metric].append(prompt_stats[metric])
                
            except Exception as e:
                print(f"Error reading {results_file}: {e}")
    
    # Calculate statistics for each model-method combination
    print("\n" + "=" * 100)
    print("STATISTICS BY MODEL AND METHOD")
    print("=" * 100)
    
    for model_name, model_data in metrics_values.items():
        print(f"\n📊 MODEL: {model_name}")
        print("=" * 80)
        
        for method_name, method_data in model_data.items():
            print(f"\n🔧 METHOD: {method_name}")
            print("-" * 60)
            
            for metric in metrics:
                values = method_data[metric]
                if values:
                    mean_val = np.mean(values)
                    variance_val = np.var(values)
                    std_val = np.std(values)
                    print(f"{metric.replace('_', ' ').title()}: {mean_val:.2f} ± {std_val:.2f} (var: {variance_val:.2f})")
                else:
                    print(f"{metric.replace('_', ' ').title()}: No data available")
    

    
    # # Calculate overall statistics for each model across all methods
    # print("\n" + "=" * 100)
    # print("OVERALL STATISTICS BY MODEL (ACROSS ALL METHODS)")
    # print("=" * 100)
    
    # for model_name, model_data in metrics_values.items():
    #     print(f"\n📊 MODEL: {model_name}")
    #     print("-" * 60)
        
    #     # Collect all values for this model across all methods
    #     model_metrics = {metric: [] for metric in metrics}
        
    #     for method_name, method_data in model_data.items():
    #         for metric in metrics:
    #             if metric in method_data:
    #                 model_metrics[metric].extend(method_data[metric])
        
    #     # Calculate mean and variance for each metric for this model
    #     for metric in metrics:
    #         if model_metrics[metric]:
    #             mean_val = np.mean(model_metrics[metric])
    #             variance_val = np.var(model_metrics[metric])
    #             std_val = np.std(model_metrics[metric])
    #             print(f"{metric.replace('_', ' ').title()}: {mean_val:.4f} ± {std_val:.4f} (var: {variance_val:.4f})")
    #         else:
    #             print(f"{metric.replace('_', ' ').title()}: No data available")


if __name__ == "__main__":
    main() 