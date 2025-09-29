#!/usr/bin/env python3
"""
Create bar charts showing base-model agreement with preference datasets
as a proxy for cognitive bias. Saves combined and per-dataset panel figures
with confidence intervals.

Usage:
  python latex/appendix/cognitive_bias.py \
      --output-dir latex_figures/appendix \
      --figures both  # options: combined, panels, both

This script is self-contained with the numbers provided in the prompt.
"""

import os
import argparse
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_data() -> Tuple[List[str], List[Dict], Dict[str, Dict[str, Dict[str, float]]]]:
    """Return (models, datasets, data) where data contains mean and CI per model.

    Data schema:
      models: [model_name,...]
      datasets: [ {key, label, source}, ...]
      data: {
        dataset_key: {
          model_name: {"mean": float, "low": float, "high": float}
        }
      }
    """

    models = [
        "Gemma-3-4B",
        "Qwen3-4B",
        "Llama-3.1-8B",
        "Gemma-3-27B",
        "Llama-3.1-70B",
    ]

    datasets = [
        {
            "key": "openai_summarize",
            "label": "OpenAI TL;DR",
            "source": "Human",
        },
        {
            "key": "ultrafeedback",
            "label": "UltraFeedback",
            "source": "GPT-4",
        },
        {
            "key": "helpsteer",
            "label": "NVIDIA HelpSteer-v2",
            "source": "Human",
        },
        {
            "key": "skywork",
            "label": "Skywork Preference",
            "source": "Human + GPT-4",
        },
    ]

    # Agreement rate (%) and CI bounds (%) from the prompt
    data = {
        "openai_summarize": {
            "Gemma-3-4B":   {"mean": 52.4, "low": 50.44, "high": 54.35},
            "Qwen3-4B":     {"mean": 51.6, "low": 49.68, "high": 53.59},
            "Gemma-3-27B":  {"mean": 54.3, "low": 52.32, "high": 56.22},
            "Llama-3.1-8B": {"mean": 54.2, "low": 52.20, "high": 56.11},
            "Llama-3.1-70B": {"mean": 56.4, "low": 54.45, "high": 58.33},
        },
        "ultrafeedback": {
            "Gemma-3-4B":   {"mean": 57.5, "low": 55.27, "high": 59.60},
            "Qwen3-4B":     {"mean": 59.0, "low": 56.83, "high": 61.14},
            "Gemma-3-27B":  {"mean": 60.2, "low": 58.04, "high": 62.32},
            "Llama-3.1-8B": {"mean": 58.0, "low": 55.77, "high": 60.10},
            "Llama-3.1-70B": {"mean": 59.5, "low": 57.33, "high": 61.63},
        },
        "helpsteer": {
            "Gemma-3-4B":   {"mean": 57.8, "low": 56.23, "high": 60.18},
            "Qwen3-4B":     {"mean": 60.8, "low": 58.88, "high": 62.75},
            "Gemma-3-27B":  {"mean": 58.4, "low": 55.38, "high": 60.74},
            "Llama-3.1-8B": {"mean": 56.2, "low": 55.90, "high": 59.66},
            "Llama-3.1-70B": {"mean": 59.8, "low": 58.58, "high": 61.98},
        },
        "skywork": {
            "Gemma-3-4B":   {"mean": 59.6, "low": 57.62, "high": 61.47},
            "Qwen3-4B":     {"mean": 61.7, "low": 59.76, "high": 63.57},
            "Gemma-3-27B":  {"mean": 59.6, "low": 57.62, "high": 61.47},
            "Llama-3.1-8B": {"mean": 58.8, "low": 56.90, "high": 60.75},
            "Llama-3.1-70B": {"mean": 59.6, "low": 57.70, "high": 61.55},
        },
    }

    return models, datasets, data


def _setup_style():
    """Set up elegant matplotlib styling inspired by seaborn"""
    plt.style.use('default')

    # Use a sophisticated color palette inspired by seaborn
    plt.rcParams.update({
        # Typography
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Helvetica'],
        'font.size': 13,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 12,
        'figure.titlesize': 18,

        # Layout and spacing
        'axes.linewidth': 1.2,
        'axes.edgecolor': '#2E2E2E',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.color': '#333333',
        'ytick.color': '#333333',

        # Background and grid
        'figure.facecolor': 'white',
        'axes.facecolor': '#FAFAFA',
        'grid.color': '#E0E0E0',
        'grid.linewidth': 0.8,
        'grid.alpha': 0.6,

        # Error bars
        'errorbar.capsize': 4,
    })


def _compute_global_ylim(datasets: List[Dict], data: Dict[str, Dict[str, Dict[str, float]]]) -> Tuple[float, float]:
    y_min = 49.0  # emphasize above-chance baseline
    y_max = 50.0
    for d in datasets:
        key = d["key"]
        for v in data[key].values():
            y_max = max(y_max, v["high"])
    # pad
    ymax_padded = min(100.0, y_max + 2.0)
    return y_min, ymax_padded


def plot_combined(models: List[str], datasets: List[Dict], data: Dict[str, Dict[str, Dict[str, float]]], output_dir: str) -> str:
    _setup_style()

    n_datasets = len(datasets)
    n_models = len(models)

    fig, ax = plt.subplots(figsize=(14, 7))

    # Sophisticated color palette inspired by seaborn's best palettes
    # Using a curated palette that's colorblind-friendly and elegant
    elegant_colors = [
        '#2E86AB',  # Deep blue
        '#A23B72',  # Magenta
        '#F18F01',  # Orange
        '#C73E1D',  # Red
        '#592E83',  # Purple
    ]
    model_colors = {models[i]: elegant_colors[i] for i in range(len(models))}

    x = np.arange(n_datasets)
    bar_width = 0.14
    group_width = bar_width * n_models

    # Plot bars grouped by dataset, colored by model
    for mi, model in enumerate(models):
        xs = x - group_width/2 + bar_width*(mi + 0.5)
        means = []
        yerr_low = []
        yerr_high = []
        for d in datasets:
            v = data[d["key"]][model]
            means.append(v["mean"])
            yerr_low.append(max(0.0, v["mean"] - v["low"]))
            yerr_high.append(max(0.0, v["high"] - v["mean"]))

        yerr = np.array([yerr_low, yerr_high])
        ax.bar(xs, means, width=bar_width, color=model_colors[model], label=model,
               alpha=0.85, yerr=yerr, capsize=4, linewidth=1,
               edgecolor='white', error_kw={'elinewidth': 2, 'alpha': 0.8})

    # Chance line at 50% with better styling
    # ax.axhline(50, color='#E74C3C', linestyle='--', linewidth=2.5, alpha=0.8, zorder=1)
    # ax.text(n_datasets - 0.1, 50.8, 'Chance (50%)', color='#E74C3C',
            # fontsize=16, ha='right', va='bottom', fontweight='600')

    # Labels and ticks with better formatting
    ax.set_ylabel('Typicality Bias (%)', fontweight='bold', fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels([d["label"] for d in datasets], rotation=0, fontweight='500')

    # Add value labels on top of bars
    for mi, model in enumerate(models):
        xs = x - group_width/2 + bar_width*(mi + 0.5)
        for i, d in enumerate(datasets):
            v = data[d["key"]][model]
            mean_val = v["mean"]
            yerr_high_val = max(0.0, v["high"] - v["mean"])

            # Add value label on top of error bar
            ax.text(xs[i], mean_val + yerr_high_val + 0.5, f'{mean_val:.1f}',
                   ha='center', va='bottom', fontsize=14, fontweight='bold', alpha=0.9)

    ymin, ymax = _compute_global_ylim(datasets, data)
    ax.set_ylim(ymin, ymax)
    ax.grid(axis='y', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    # Enhanced legend above plot, no box
    legend = ax.legend(ncol=min(n_models, 5), loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       frameon=False, fontsize=18)

    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, 'cognitive_bias_combined.png')
    pdf_path = os.path.join(output_dir, 'cognitive_bias_combined.pdf')
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    return png_path


def plot_panels(models: List[str], datasets: List[Dict], data: Dict[str, Dict[str, Dict[str, float]]], output_dir: str) -> str:
    _setup_style()

    n_rows, n_cols = 2, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 10), sharey=True)
    axes = axes.flatten()

    # Use the same sophisticated color palette
    elegant_colors = [
        '#2E86AB',  # Deep blue
        '#A23B72',  # Magenta
        '#F18F01',  # Orange
        '#C73E1D',  # Red
        '#592E83',  # Purple
    ]
    model_colors = {models[i]: elegant_colors[i] for i in range(len(models))}

    ymin, ymax = _compute_global_ylim(datasets, data)

    for idx, d in enumerate(datasets):
        ax = axes[idx]
        key = d["key"]
        label = d["label"]
        source = d["source"]

        xs = np.arange(len(models))
        means = [data[key][m]["mean"] for m in models]
        yerr_low = [max(0.0, data[key][m]["mean"] - data[key][m]["low"]) for m in models]
        yerr_high = [max(0.0, data[key][m]["high"] - data[key][m]["mean"]) for m in models]
        yerr = np.array([yerr_low, yerr_high])

        bars = ax.bar(xs, means, color=[model_colors[m] for m in models],
                      yerr=yerr, capsize=4, alpha=0.85, width=0.75,
                      linewidth=1, edgecolor='white',
                      error_kw={'elinewidth': 2, 'alpha': 0.8})

        # Enhanced chance line
        ax.axhline(50, color='#E74C3C', linestyle='--', linewidth=2, alpha=0.8, zorder=1)

        # Enhanced labels
        ax.set_title(f"{label}", fontweight='bold', fontsize=20, pad=10)
        ax.set_xticks(xs)
        ax.set_xticklabels(models, rotation=25, ha='right', fontweight='500')
        ax.set_ylim(ymin, ymax)
        ax.grid(axis='y', alpha=0.4, zorder=0)
        ax.set_axisbelow(True)


        # Enhanced value labels above bars
        for rect, mean, yerr_low_val, yerr_high_val in zip(bars, means, yerr_low, yerr_high):
            h = rect.get_height()
            if h >= 55 or yerr_high_val >= 1.5:  # only mark notable ones
                ax.text(rect.get_x() + rect.get_width()/2.0, h + yerr_high_val + 0.4,
                        f"{mean:.1f}", ha='center', va='bottom', fontsize=18,
                        fontweight='bold', alpha=0.9)

    # Enhanced common labels
    fig.supylabel('Agreement Rate (%)', x=0.02, fontweight='bold', fontsize=15)

    # Enhanced legend above plots, no box
    handles = [plt.Rectangle((0, 0), 1, 1, color=model_colors[m], alpha=0.85) for m in models]
    legend = fig.legend(handles, models, loc='upper center', ncol=min(5, len(models)),
                       bbox_to_anchor=(0.5, 0.95), frameon=False, fontsize=17)

    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, 'cognitive_bias_panels.png')
    pdf_path = os.path.join(output_dir, 'cognitive_bias_panels.pdf')
    plt.tight_layout(rect=[0, 0.05, 1, 0.94])
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    return png_path


def main():
    parser = argparse.ArgumentParser(description="Plot cognitive bias bar charts with CIs")
    parser.add_argument('--output-dir', type=str, default='latex_figures/appendix', help='Directory to save figures')
    parser.add_argument('--figures', type=str, default='both', choices=['vs_multi', 'panels', 'both'], help='Which figures to generate')

    args = parser.parse_args()

    models, datasets, data = get_data()

    generated = []
    if args.figures in ('vs_multi', 'both'):
        generated.append(plot_combined(models, datasets, data, args.output_dir))
    if args.figures in ('panels', 'both'):
        generated.append(plot_panels(models, datasets, data, args.output_dir))

    print("\n✅ Cognitive bias figures generated.")
    print("📁 Saved to:")
    for p in generated:
        base = os.path.splitext(os.path.basename(p))[0]
        print(f"  - {args.output_dir}/{base}.png")
        print(f"  - {args.output_dir}/{base}.pdf")


if __name__ == '__main__':
    main()
