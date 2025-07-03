import json
import numpy as np

for model in ["anthropic/claude-4-sonnet", 
              "google/gemini-2.5-pro", 
              "openai/o3",
              "meta-llama/llama-3.1-70b-instruct",
              "google/gemini-2.5-flash",
                "anthropic/claude-3.7-sonnet",
                "deepseek/deepseek-r1-0528",
]:
    print(f"Processing model: {model}")
    for method in ["direct (samples=1)", 
                   "sequence [strict] (samples=5)", 
                   "multi_turn [strict] (samples=5)", 
                   "structure_with_prob [strict] (samples=5)",
                   "combined [strict] (samples=5)"]:
        model_basename = model.replace("/", "_")
        filename = f"poem_experiments_final/{model_basename}/{model_basename}_poem/evaluation/{method}/creative_writing_v3_results.json"
        try:
            with open(filename) as f:
                data = json.load(f)
            instance_metrics = data["instance_metrics"]

            # Collect all metric names except Average_Score
            metric_names = set()
            for inst in instance_metrics:
                metric_names.update(k for k in inst if k != "Average_Score")
            metric_names = sorted(metric_names)

            # Compute averages
            avg_metrics = {}
            for metric in metric_names:
                values = [inst[metric] for inst in instance_metrics if metric in inst]
                avg_metrics[f"avg_{metric}"] = float(np.mean(values)) if values else None
                avg_metrics[f"std_{metric}"] = float(np.std(values)) if values else None

            # Compute average of Average_Score
            average_scores = [inst["Average_Score"] for inst in instance_metrics if "Average_Score" in inst]
            avg_metrics["avg_score"] = float(np.mean(average_scores)) if average_scores else None
            avg_metrics["std_score"] = float(np.std(average_scores)) if average_scores else None

            data['overall_metrics'] = avg_metrics
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)
        except FileNotFoundError:
            print(f"File not found: {filename}")

        filename = f"poem_experiments_final/{model_basename}/{model_basename}_poem/evaluation/{method}/diversity_results.json"
        try:
            with open(filename) as f:
                data = json.load(f)
            
            if "avg_diversity" not in data['overall_metrics']:
                data['overall_metrics']['avg_diversity'] = data['overall_metrics'].get('average_diversity', 0.0)

            # data['overall_metrics'] = avg_metrics
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)
        except FileNotFoundError:
            print(f"File not found: {filename}")

        filename = f"poem_experiments_final/{model_basename}/{model_basename}_poem/evaluation/{method}/length_results.json"
        try:
            with open(filename) as f:
                data = json.load(f)
            
            if "avg_token_length" not in data['overall_metrics']:
                data['overall_metrics']['avg_token_length'] = data['overall_metrics'].get('mean_token_length', 0.0)

            # data['overall_metrics'] = avg_metrics
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)
        except FileNotFoundError:
            print(f"File not found: {filename}")
        
        filename = f"poem_experiments_final/{model_basename}/{model_basename}_poem/evaluation/{method}/ngram_results.json"
        try:
            with open(filename) as f:
                data = json.load(f)
            
            if "avg_rouge_l" not in data['overall_metrics']:
                data['overall_metrics']['avg_rouge_l'] = data['overall_metrics'].get('average_rouge_l', 0.0)

            # data['overall_metrics'] = avg_metrics
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)
        except FileNotFoundError:
            print(f"File not found: {filename}")