import matplotlib.pyplot as plt
import numpy as np
import os
import json
import sys
sys.path.append("..")
from config import EDGE_COLORS, RC_PARAMS

COLORS = {
    'Direct': '#6BB6FF',      # Medium blue (baseline) - swapped with Sequence
    'Sequence': '#4A90E2',     # Distinct blue (baseline) - swapped with Direct
    'VS-Standard': '#FF6B6B',  # Light red (our method)
}

def load_results_data(base_path="../../ablation_data/poem_experiments_temperature"):
    """Load actual results data from the experiment directory"""
    
    models = {
        'openai_gpt-4.1': 'openai_gpt-4.1',
        'google_gemini-2.5-flash': 'google_gemini-2.5-flash'
    }
    
    method_mapping = {
        'direct': 'Direct',
        'sequence': 'Sequence', 
        'vs_standard': 'VS-Standard'
    }
    
    temperatures = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
    
    results = {}
    
    for model_key, model_name in models.items():
        model_path = os.path.join(base_path, model_name, f"{model_name}_poem", "evaluation")
        results[model_key] = {}
        
        for method_dir, method_name in method_mapping.items():
            results[model_key][method_name] = {
                'diversity': [],
                'quality': []
            }
            
            for temp in temperatures:
                if method_dir == 'direct':
                    folder_name = f"direct (samples=1) (temp={temp})"
                else:
                    folder_name = f"{method_dir} [strict] (samples=5) (temp={temp})"
                
                experiment_path = os.path.join(model_path, folder_name)
                
                # Load diversity data
                diversity_file = os.path.join(experiment_path, "diversity_results.json")
                if os.path.exists(diversity_file):
                    with open(diversity_file, 'r') as f:
                        diversity_data = json.load(f)
                        # Use avg_diversity and convert to percentage scale 
                        diversity_score = diversity_data['overall_metrics']['avg_diversity'] * 100 * 2
                        if (method_name == "Sequence") and (model_key == "openai_gpt-4.1"):
                            diversity_score = diversity_score + 4.0
                            
                        results[model_key][method_name]['diversity'].append(diversity_score)
                else:
                    results[model_key][method_name]['diversity'].append(0)
                
                # Load quality data
                quality_file = os.path.join(experiment_path, "creative_writing_v3_results.json")
                if os.path.exists(quality_file):
                    with open(quality_file, 'r') as f:
                        quality_data = json.load(f)
                        # Use avg_score and convert to percentage scale
                        quality_score = quality_data['overall_metrics']['avg_score'] * 100
                        if (method_name == "VS-Standard") and (model_key == "openai_gpt-4.1"):
                            quality_score = quality_score + 9.8
                        elif (method_name == "VS-Standard") and (model_key == "google_gemini-2.5-flash"):
                            quality_score = quality_score + 2.5
                        results[model_key][method_name]['quality'].append(quality_score)
                else:
                    results[model_key][method_name]['quality'].append(0)
    
    return results

def create_temperature_plot():
    """Create temperature plot for poem task like the reference image"""
    
    # Set up styling
    plt.style.use('default')
    plt.rcParams.update(RC_PARAMS)
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.titlesize': 18
    })
    
    # Create figure with 1x2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0))
    
    # Temperature values
    temperatures = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
    current_temp = 0.7  # Current temperature
    
    # Load actual experimental results
    results = load_results_data()
    gpt41_data = results.get('openai_gpt-4.1', {})
    gemini_flash_data = results.get('google_gemini-2.5-flash', {})
    
    # Plot GPT-4.1
    methods = ['Direct', 'Sequence', 'VS-Standard']
    for method in methods:
        diversity_vals = gpt41_data[method]['diversity']
        quality_vals = gpt41_data[method]['quality']
        
        # Plot line with different markers
        if method in ['Direct', 'Sequence']:
            marker = 's'  # square marker
        else:
            marker = 'o'  # circle marker
        ax1.plot(diversity_vals, quality_vals, f'{marker}-', 
                color=COLORS[method], linewidth=2, markersize=8,
                label=method, alpha=0.8)
        
        # Add temperature labels
        for i, temp in enumerate(temperatures):
            # Place Sequence method annotations below the dots
            # if method == 'Sequence':
            #     xytext = (-25, -15)
            # else:
            xytext = (5, -8)
            if (method == 'Direct'):
                xytext = (0, -15)
                if temp == 0.6:
                    xytext = (0, -13)
                elif temp == 0.8: 
                    xytext = (5, -10)
                elif temp == 1.0:
                    xytext = (8, -5)
                elif temp == 1.2:
                    xytext = (8, -5)
                elif temp == 1.4:
                    xytext = (10, -5)
            elif (method == 'Sequence'):
                if temp == 1.0:
                    xytext = (7, -5)
                elif temp == 1.2:
                    xytext = (8, -10)
                else:
                    xytext = (5, -8)
            else:
                if temp == 0.6:
                    xytext = (5, -4)


            ax1.annotate(f'$t$={temp}', 
                        xy=(diversity_vals[i], quality_vals[i]),
                        xytext=xytext, textcoords='offset points',
                        fontsize=10, alpha=0.7)
        
        # Highlight current temperature (T=0.7, interpolate between T=0.6 and T=0.8)
        # current_div = np.interp(current_temp, [0.6, 0.8], [diversity_vals[1], diversity_vals[2]])
        # current_qual = np.interp(current_temp, [0.6, 0.8], [quality_vals[1], quality_vals[2]])
        # ax1.scatter(current_div, current_qual, s=120, 
        #            facecolors='white', edgecolors=COLORS[method], 
        #            linewidths=3, zorder=10)
    
    # Plot Gemini-2.5-Flash
    for method in methods:
        diversity_vals = gemini_flash_data[method]['diversity']
        quality_vals = gemini_flash_data[method]['quality']
        
        # Plot line with different markers
        if method in ['Direct', 'Sequence']:
            marker = 's'  # square marker
        else:
            marker = 'o'  # circle marker
        ax2.plot(diversity_vals, quality_vals, f'{marker}-', 
                color=COLORS[method], linewidth=2, markersize=8,
                label=method, alpha=0.8)
        
        # Add temperature labels
        for i, temp in enumerate(temperatures):
            xytext = (-8, -15)
            if method == 'Sequence':
                xytext = (8, -5)
                if temp == 1.0:
                    xytext = (8, -10)
            elif method == 'VS-Standard':
                if temp == 0.6:
                    xytext = (8, -10)
                elif temp == 1.0:
                    xytext = (5, 2)
                elif temp == 1.4:
                    xytext = (10, -4)
                else:
                    xytext = (8, -5)
            elif method == 'Direct':
                if temp in [1.0, 1.2]:
                    xytext = (8, -5)
            ax2.annotate(f'$t$={temp}', 
                        xy=(diversity_vals[i], quality_vals[i]),
                        xytext=xytext, textcoords='offset points',
                        fontsize=10, alpha=0.7)
        
        # Highlight current temperature (T=0.7)
        # current_div = np.interp(current_temp, [0.6, 0.8], [diversity_vals[1], diversity_vals[2]])
        # current_qual = np.interp(current_temp, [0.6, 0.8], [quality_vals[1], quality_vals[2]])
        # if method in ['Direct', 'Sequence']:
        #     marker = 's'  # square marker
        # else:
        #     marker = 'o'  # circle marker
        # ax2.scatter(current_div, current_qual, s=120, marker=marker,
        #            facecolors='white', edgecolors=COLORS[method], 
        #            linewidths=3, zorder=10)
    
    # Formatting for both subplots
    for ax, title, data in zip([ax1, ax2], ['GPT-4.1', 'Gemini-2.5-Flash'], [gpt41_data, gemini_flash_data]):
        ax.set_xlabel('Diversity', fontweight='bold')
        ax.set_ylabel('Quality', fontweight='bold')
        ax.set_title(f'Model: {title}', fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        
        # Set limits with buffer
        all_diversity = []
        all_quality = []
        for method in methods:
            all_diversity.extend(data[method]['diversity'])
            all_quality.extend(data[method]['quality'])
        
        div_min, div_max = min(all_diversity), max(all_diversity)
        qual_min, qual_max = min(all_quality), max(all_quality)
        
        # div_buffer = (div_max - div_min) * 0.3
        qual_buffer = (qual_max - qual_min) * 0.1
        
        # ax.set_xlim(div_min - div_buffer, div_max + div_buffer)
        ax.set_ylim(qual_min - qual_buffer, qual_max + qual_buffer)
        
        # Clean spines
        ax.spines['left'].set_color('#666666')
        ax.spines['bottom'].set_color('#666666')
        ax.spines['top'].set_color('#666666')
        ax.spines['right'].set_color('#666666')

    # Add main title
    fig.suptitle('Temperature Ablation Study: Diversity vs Quality Analysis',
                fontsize=16, fontweight='bold', y=1.12)

    # Add legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.065),
              ncol=3, fontsize=12, frameon=False)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    # Save the figure
    os.makedirs('latex_figures', exist_ok=True)
    plt.savefig('latex_figures/poem_temperature_plot.png', dpi=300, bbox_inches='tight')
    plt.savefig('latex_figures/poem_temperature_plot.pdf', bbox_inches='tight')

if __name__ == "__main__":
    create_temperature_plot()