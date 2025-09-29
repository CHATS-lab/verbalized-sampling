import json
import os
import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path

def extract_prob_tuning_from_path(path):
    """Extract prob_tuning value from directory path"""
    # Look for prob_tuning in the path
    match = re.search(r'prob_tuning=([-\d.]+)', path)
    if match:
        value = float(match.group(1))
        # Handle -1 as 1 as requested
        if value == -1:
            return 1.0
        return value
    return None

def load_evaluation_data(base_path):
    """Load evaluation data for all prob_tuning values"""
    data = {
        'direct': {},
        'sequence': {},
        'vs_standard': {},
        'vs_multi': {}
    }
    
    evaluation_path = os.path.join(base_path, 'evaluation')
    
    # Process each directory
    for dir_name in os.listdir(evaluation_path):
        dir_path = os.path.join(evaluation_path, dir_name)
        if not os.path.isdir(dir_path):
            continue

        # Determine method type
        if 'direct' in dir_name:
            method = 'direct'
            prob_tuning = None  # Direct doesn't have prob_tuning
        elif 'sequence' in dir_name:
            method = 'sequence'
            prob_tuning = None  # Sequence doesn't have prob_tuning
        elif 'vs_standard' in dir_name:
            method = 'vs_standard'
            prob_tuning = extract_prob_tuning_from_path(dir_name)
        elif 'vs_multi' in dir_name:
            method = 'vs_multi'
            prob_tuning = extract_prob_tuning_from_path(dir_name)
        else:
            continue
            
        # Load the JSON file
        json_file = os.path.join(dir_path, 'response_count_results.json')
        if os.path.exists(json_file):
            try:
                with open(json_file, 'r') as f:
                    result = json.load(f)
                    
                # Extract metrics
                metrics = result.get('overall_metrics', {})
                data[method][prob_tuning] = {
                    'kl_divergence': metrics.get('average_kl_divergence', 0),
                    'precision': metrics.get('average_precision', 0),
                    'coverage': metrics.get('average_unique_recall_rate', 0)
                }
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                
    return data


def create_line_plots(data, output_path):
    """Create line plots for different metrics, with the legend at the top of the whole image."""
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'font.size': 16,
        'axes.labelsize': 18,
        'axes.titlesize': 20,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'legend.fontsize': 18,
        'axes.linewidth': 0.8,
        'axes.edgecolor': '#666666',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'lines.linewidth': 2.0,
        'lines.markersize': 8,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white'
    })

    method_markers = {
        'vs_standard': 'o',  # Circle
        'vs_multi': 's'  # Square
    }
    method_to_label = {
        'sequence': 'Sequence',
        'vs_standard': 'VS-Standard',
        'vs_multi': 'VS-Multi'
    }
    # Define prob_tuning values (excluding None for direct/sequence)
    prob_tuning_values = []
    for method in ['vs_standard', 'vs_multi']:
        for prob_tuning in data[method].keys():
            if prob_tuning is not None and prob_tuning not in prob_tuning_values:
                prob_tuning_values.append(prob_tuning)
    
    # Sort prob_tuning values
    prob_tuning_values = sorted(prob_tuning_values, reverse=True)
    print(prob_tuning_values)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = ['kl_divergence', 'coverage', 'precision']
    metric_labels = ['KL Divergence ($\\downarrow$)', 'Coverage ($\\uparrow$)', 'Precision ($\\uparrow$)']

    # For collecting legend handles/labels
    legend_handles = []
    legend_labels = []
    legend_added = set()

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        
        # Plot structure_with_prob and VS-Multi (vs_multi)
        for method in ['vs_standard', 'vs_multi']:
            x_values = []
            y_values = []
            
            for prob_tuning in prob_tuning_values:
                if prob_tuning in data[method]:
                    x_values.append(prob_tuning)
                    y_values.append(data[method][prob_tuning][metric])
            
            if x_values and y_values:
                # Sort by x_values to ensure proper line plotting
                sorted_pairs = sorted(zip(x_values, y_values))
                x_sorted, y_sorted = zip(*sorted_pairs)
                
                # Only add label for legend once per method
                label_for_legend = method_to_label[method] if method not in legend_added else None
                line, = ax.plot(
                    x_sorted, y_sorted, method_markers[method] + '-', 
                    label=label_for_legend, linewidth=2, markersize=6
                )
                if method not in legend_added:
                    legend_handles.append(line)
                    legend_labels.append(method_to_label[method])
                    legend_added.add(method)
        
        # # Add direct and sequence as horizontal lines
        # if 'direct' in data and data['direct']:
        #     direct_value = list(data['direct'].values())[0][metric]
        #     ax.axhline(y=direct_value, color='red', linestyle='--', 
        #               label='Direct', linewidth=2, alpha=0.8)
        
        if 'sequence' in data and data['sequence']:
            sequence_value = list(data['sequence'].values())[0][metric]
            # Only add label for legend once
            label_for_legend = method_to_label['sequence'] if 'sequence' not in legend_added else None
            line = ax.axhline(
                y=sequence_value, color='green', linestyle='--', 
                label=label_for_legend, linewidth=2, alpha=0.8
            )
            if 'sequence' not in legend_added:
                legend_handles.append(line)
                legend_labels.append(method_to_label['sequence'])
                legend_added.add('sequence')
        
        # Set log scale for x-axis
        ax.set_xscale('log')
        ax.set_xlabel('Probability Tuning Value')
        ax.set_ylabel(label)
        ax.set_title(f'{label}', fontweight='bold')
        ax.grid(True, alpha=0.3)
        # Remove per-axes legend
        # ax.legend()
        
        # Set x-axis limits from 0.009 to 0.92 (left to right, increasing)
        # ax.set_xlim(0.009, 0.92)
        # x_ticks = [0.01, 0.05, 0.1, 0.5, 0.9]

        ax.set_xlim(0.92, 0.009)
        x_ticks = [0.9, 0.5, 0.1, 0.05, 0.01]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f'{x:.3f}' if x != 1.0 else '1.0' for x in x_ticks], 
                          rotation=30)
    
    # Draw the legend at the top of the whole image
    fig.legend(
        legend_handles, legend_labels, 
        loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=len(legend_labels),
        frameon=False,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # leave space for legend at top
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    # Also save as PDF
    pdf_output_path = output_path.rsplit('.', 1)[0] + '.pdf'
    plt.savefig(pdf_output_path, bbox_inches='tight')
    print(f"Plot also saved to: {pdf_output_path}")
    # plt.show()
    
    return fig




def main():
    # Set the base path
    # model = 'gpt-4.1'
    model = 'google_gemini-2.5-flash'
    base_path = f'bias_experiments_prob_tuning/{model}_state_name'
    
    # Load data
    print("Loading evaluation data...")
    data = load_evaluation_data(base_path)
    # print(data)
    
    # Print loaded data for verification, sorted by prob_tuning if possible
    print("\nLoaded data structure:")
    for method, method_data in data.items():
        print(f"\n{method}:")
        prob_tuning_keys = list(method_data.keys())
        # Check if all prob_tuning keys are not None and can be converted to float
        can_sort = all(
            k is not None and (
                isinstance(k, (int, float)) or (isinstance(k, str) and k.replace('.', '', 1).replace('-', '', 1).isdigit())
            )
            for k in prob_tuning_keys
        )
        if can_sort:
            sorted_keys = sorted(prob_tuning_keys, key=lambda x: float(x))
        else:
            sorted_keys = prob_tuning_keys
        for prob_tuning in sorted_keys:
            metrics = method_data[prob_tuning]
            print(f"  {prob_tuning}: {metrics}")
    
    # Create plots
    output_path = f'openended_qa_prob_tuning/{model}_prob_tuning_line_plot.png'
    os.makedirs('openended_qa_prob_tuning', exist_ok=True)
    print(f"\nCreating line plots...")
    fig = create_line_plots(data, output_path)
    
    print(f"Plot saved to: {output_path}")

if __name__ == "__main__":
    main()
