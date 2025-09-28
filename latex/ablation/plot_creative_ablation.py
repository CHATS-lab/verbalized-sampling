#!/usr/bin/env python3

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_training_progression(output_dir="latex_figures"):
    """Create line plot showing diversity improvement across training progression"""
    
    # Model directory mapping for training progression
    models = {
        "Base Model": "meta-llama_Llama-3.1-70B",
        "Tulu-SFT-70B": "allenai_Llama-3.1-Tulu-3-70B-SFT", 
        "Tulu-DPO-70B": "allenai_Llama-3.1-Tulu-3-70B-DPO",
        "Tulu-Final-70B": "allenai_Llama-3.1-Tulu-3-70B"
    }
    
    # Methods mapping
    methods_mapping = {
        "Direct": "direct (samples=1)",
        "Sequence": "combined [strict] (samples=5)",
        "Multi-Turn": "structure_with_prob [strict] (samples=5)",
        "Verbalized Sampling": "chain_of_thought [strict] (samples=5)"
    }
    
    # Base model only has direct_base method
    base_model_methods = {
        "Direct": "direct_base (samples=1)"
    }
    
    base_dir = "ablation_data/poem_experiments_test"
    
    # Load diversity results
    results_data = {}
    base_model_score = None
    
    # Separate base model and training stages
    training_models = {k: v for k, v in models.items() if k != "Base Model"}
    
    for method_name in methods_mapping.keys():
        results_data[method_name] = []
    
    # Get base model score separately
    if "Base Model" in models:
        model_path = os.path.join(base_dir, models["Base Model"], f"{models['Base Model']}_poem")
        method_dir = base_model_methods["Direct"]
        file_path = os.path.join(model_path, "evaluation", method_dir, "diversity_results.json")
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            diversity_score = data.get('overall_metrics', {}).get('avg_diversity', None) * 2
            base_model_score = diversity_score * 100 if diversity_score else None
        except (FileNotFoundError, json.JSONDecodeError):
            base_model_score = None
    
    # Print experiment results header
    print(f"\n{'='*60}")
    print(f"TRAINING PROGRESSION DIVERSITY RESULTS")
    print(f"{'='*60}")
    
    # Print base model results
    if base_model_score is not None:
        print(f"\nBase Model (meta-llama_Llama-3.1-70B):")
        print(f"  Direct: {base_model_score:.2f}")
    else:
        print(f"\nBase Model: No data available")
    
    # Load training progression data (excluding base model)
    for stage_name, model_dir in training_models.items():
        model_path = os.path.join(base_dir, model_dir, f"{model_dir}_poem")
        print(f"\n{stage_name}:")
        
        # Training models have all methods
        for method_name, method_dir in methods_mapping.items():
            file_path = os.path.join(model_path, "evaluation", method_dir, "diversity_results.json")
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                diversity_score = data.get('overall_metrics', {}).get('avg_diversity', None) * 2
                final_score = diversity_score * 100 if diversity_score else None
                results_data[method_name].append(final_score)
                print(f"  {method_name}: {final_score:.2f}" if final_score else f"  {method_name}: No data")
            except (FileNotFoundError, json.JSONDecodeError):
                results_data[method_name].append(None)
                print(f"  {method_name}: No data")
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*60}")
    
    for method_name, values in results_data.items():
        valid_values = [v for v in values if v is not None]
        if valid_values:
            avg_score = np.mean(valid_values)
            max_score = max(valid_values)
            min_score = min(valid_values)
            print(f"{method_name}:")
            print(f"  Average: {avg_score:.2f}")
            print(f"  Best: {max_score:.2f}")
            print(f"  Worst: {min_score:.2f}")
            print(f"  Data points: {len(valid_values)}/{len(values)}")
        else:
            print(f"{method_name}: No valid data")
    
    print(f"{'='*60}")
    
    # Set up seaborn style for beautiful plots
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.size': 16,
        'axes.labelsize': 18,
        'axes.titlesize': 20,
        'xtick.labelsize': 30,
        'ytick.labelsize': 20,
        'legend.fontsize': 16,
        'figure.titlesize': 20
    })
    
    # Create the plot with seaborn styling
    plt.figure(figsize=(9, 6))
    ax = plt.gca()
    
    # Set up beautiful colors and markers
    colors = {
        'Direct': '#2E86AB',           # Ocean blue
        'Sequence': '#A23B72',      # Deep magenta
        'Multi-Turn': '#C73E1D',       # Rich red
        'Verbalized Sampling': '#F18F01'          # Vibrant orange
    }
    markers = {'Direct': 'o', 'Sequence': 's', 'Multi-Turn': 'D', 'Verbalized Sampling': '^'}
    
    x_positions = range(len(training_models))
    x_labels = list(training_models.keys())
    
    # Plot each method with larger markers and better styling
    for method_name, values in results_data.items():
        # Filter out None values for plotting
        valid_positions = []
        valid_values = []
        for i, val in enumerate(values):
            if val is not None:
                valid_positions.append(i)
                valid_values.append(val)
        
        if valid_values:  # Only plot if we have data
            ax.plot(valid_positions, valid_values, 
                   color=colors[method_name], marker=markers[method_name], 
                   linewidth=4, markersize=16, label=method_name, alpha=0.9,
                   markeredgecolor='white', markeredgewidth=2.5)
    
    # Add base model as horizontal dotted red line
    if base_model_score is not None:
        ax.axhline(y=base_model_score, color='maroon', linestyle='--', linewidth=3, 
                  alpha=0.8)
        # Add inline annotation for base model
        ax.text(len(x_positions) - 1.45, base_model_score - 0.5, 'Base Model', 
               fontsize=20, fontweight='bold', color='maroon',
               ha='left', va='top')
    
    # Customize the plot with seaborn styling
    ax.set_xlabel('Training Stage', fontsize=18, fontweight='bold')
    ax.set_ylabel('Diversity', fontsize=18, fontweight='bold')
    # ax.set_title('Diversity Improvement Across Training Progression', fontsize=20, fontweight='bold', pad=20)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=22)
    ax.tick_params(axis='y', labelsize=20)
    
    # Add more visible grid
    ax.grid(True, alpha=0.6, linestyle='-', linewidth=0.8, color='gray')
    ax.set_axisbelow(True)  # Put grid behind the plot elements
    
    # Put legend on top
    ax.legend(fontsize=22, loc='upper center', bbox_to_anchor=(0.5, 1.30), ncol=2, frameon=True)
    
    # Set better y-axis limits with padding (include base model score)
    all_values = []
    for values in results_data.values():
        all_values.extend([v for v in values if v is not None])
    if base_model_score is not None:
        all_values.append(base_model_score)
    
    y_min = min(all_values)
    y_max = max(all_values)
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
    
    # Add improvement arrow for Tulu-DPO-70B
    dpo_stage_idx = 1  # Tulu-DPO-70B is the 2nd stage (index 1) in training progression
    
    # Get the diversity values for Direct and Verbalized Sampling at DPO stage
    direct_value = results_data["Direct"][dpo_stage_idx] if len(results_data["Direct"]) > dpo_stage_idx else None
    vs_value = results_data["Verbalized Sampling"][dpo_stage_idx] if len(results_data["Verbalized Sampling"]) > dpo_stage_idx else None
    
    if direct_value is not None and vs_value is not None:
        # Calculate improvement percentage
        improvement_pct = ((vs_value - direct_value) / direct_value) * 100
        
        # Position for the arrow (slightly to the right to avoid blocking other elements)
        arrow_x = dpo_stage_idx + 0.1  # Shift arrow to the right
        arrow_y_start = direct_value
        arrow_y_end = vs_value
        
        # Draw arrow
        ax.annotate('', xy=(arrow_x, arrow_y_end), xytext=(arrow_x, arrow_y_start),
                   arrowprops=dict(arrowstyle='->', lw=3, color='#FF6B6B', alpha=0.8))
        
        # Add improvement percentage text (further to the right)
        text_y = (arrow_y_start + arrow_y_end) / 2
        ax.text(arrow_x + 0.2, text_y, f'+{improvement_pct:.1f}%', 
               fontsize=18, fontweight='bold', color='#FF6B6B',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='#FF6B6B', alpha=0.9))
    
    # Add subtle background color
    ax.set_facecolor('#FAFAFA')
    
    # Save the plot with proper spacing for top legend
    ablation_output_dir = os.path.join(output_dir, "ablation", "training_progression")
    os.makedirs(ablation_output_dir, exist_ok=True)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Make room for top legend
    plt.savefig(f'{ablation_output_dir}/training_progression_diversity.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f'{ablation_output_dir}/training_progression_diversity.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved training progression diversity plot")
    print(f"📁 Results saved to: latex_figures/ablation/training_progression/")
    print(f"📊 Generated files:")
    print(f"  - training_progression_diversity.png/pdf")
    print(f"📈 Data points collected:")
    for method, values in results_data.items():
        valid_count = sum(1 for v in values if v is not None)
        print(f"    {method}: {valid_count}/{len(values)} stages")

if __name__ == "__main__":
    plot_training_progression()