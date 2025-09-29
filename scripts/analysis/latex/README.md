# LaTeX Results Processing

This directory contains scripts for generating LaTeX tables and publication-quality plots from experimental results.

## Files

- **`generate_latex_tables.py`** - Unified script for parsing experimental results into LaTeX table format
- **`generate_plots.py`** - Unified script for creating publication-quality figures and plots
- **`generate_all_figures.py`** - Master script to run all plotting and table generation scripts
- **`ablation/plot_creative_ablation.py`** - Specialized plotting for training progression analysis
- **`ablation/model_size_ablation.py`** - Model size ablation study comparing large vs small models

## Usage

### Quick Start - Generate Everything

```bash
# Generate all LaTeX tables and figures with one command
python generate_all_figures.py
```

### Individual Scripts

#### Generate LaTeX Tables

```bash
# Generate tables for both poem and story experiments
python generate_latex_tables.py

# Generate tables for specific task
python generate_latex_tables.py --task poem
python generate_latex_tables.py --task story
```

#### Generate Standard Plots

```bash
# Generate plots for both poem and story experiments
python generate_plots.py

# Generate plots for specific task
python generate_plots.py --task poem
python generate_plots.py --task story
```

#### Generate Ablation Studies

```bash
# Generate training progression plots
python ablation/plot_creative_ablation.py

# Generate model size ablation study
python ablation/model_size_ablation.py
```

## Output

### LaTeX Tables
- Formatted LaTeX table rows with standard deviations
- Best values highlighted in bold
- Summary statistics with improvements over baseline
- Multirow formatting for clean presentation

### Plots
- Individual diversity vs quality scatter plots for each model
- Method average bar charts with statistical significance tests
- All models comparison plots
- Publication-ready PDF and PNG formats

### Ablation Studies
- **Training Progression**: Diversity improvements across training stages
- **Model Size Analysis**: 
  - Side-by-side diversity vs quality scatter plots (large vs small models)
  - Pareto efficiency comparison with area-under-curve metrics
  - Method effectiveness analysis by model size
  - Statistical significance tests for size-based improvements
  - **NEW: Cognitive Burden Analysis** - Investigates quality drops from VS methods
  - **Large models**: GPT-4.1, Gemini-2.5-Pro, GPT-o3, Claude-4-Sonnet, Llama-3.1-70B, DeepSeek-R1
  - **Small models**: GPT-4.1-Mini, Gemini-2.5-Flash, Llama-3.1-8B

## Directory Structure Expected

```
poem_experiments_final/
└── [model_name]/
    └── [model_name]_poem/
        └── evaluation/
            └── [method_name]/
                ├── diversity_results.json
                ├── ngram_results.json
                └── creative_writing_v3_results.json

story_experiments_final/
└── [model_name]/
    └── [model_name]_book/
        └── evaluation/
            └── [method_name]/
                ├── diversity_results.json
                ├── ngram_results.json
                └── creative_writing_v3_results.json
```

## Output Directory Structure

All plots are now saved to a standardized `latex_figures/` directory with organized subdirectories:

```
latex_figures/
├── poem/                           # Poem task results
│   ├── individual_models/          # Individual model scatter plots
│   ├── method_averages/            # Method comparison bar charts
│   └── model_comparisons/          # All models comparison plots
├── story/                          # Story task results
│   ├── individual_models/          # Individual model scatter plots
│   ├── method_averages/            # Method comparison bar charts
│   └── model_comparisons/          # All models comparison plots
└── ablation/                       # Ablation studies
    ├── model_size/                 # Model size comparison plots
    └── training_progression/       # Training progression plots
```

### File Organization Benefits:
- **Easy Navigation**: Clear separation by task and analysis type
- **Publication Ready**: Organized structure for LaTeX document inclusion
- **Consistent Naming**: Standardized file naming across all scripts
- **Scalable**: Easy to add new analysis types and tasks