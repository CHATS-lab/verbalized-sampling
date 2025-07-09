from typing import List
from abc import ABC, abstractmethod
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from .base import ComparisonData

class EvaluatorSpecificPlotter(ABC):
    """Base class for evaluator-specific plotting logic."""
    
    @abstractmethod
    def create_plots(self, comparison_data: List[ComparisonData], output_dir: Path):
        """Create evaluator-specific plots."""
        pass

class ResponseCountPlotter(EvaluatorSpecificPlotter):
    """Creates response count-specific plots."""
    
    def __init__(self, figsize: tuple = (15, 8), colors: List[str] = None):
        self.figsize = figsize
        self.colors = colors or ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    def create_plots(self, comparison_data: List[ComparisonData], output_dir: Path):
        for idx_1, comp in enumerate(comparison_data):
            method_name = comp.name
            method_dir = output_dir / method_name
            method_dir.mkdir(exist_ok=True)
            
            metrics = comp.result.overall_metrics['per_prompt_stats']
            for idx_2, (key, data_point) in enumerate(metrics.items()):
                response_counter = data_point["response_distribution"]
                if not response_counter:
                    continue
                
                self._create_single_response_plot(response_counter, key, idx_2, method_dir)
    
    def _create_single_response_plot(self, response_counter: dict, question: str, idx: int, output_dir: Path):
        sorted_items = sorted(response_counter.items(), key=lambda x: x[1], reverse=True)
        responses, counts = zip(*sorted_items)

        plt.figure(figsize=self.figsize)
        ax = sns.barplot(x=range(len(responses)), y=counts, color=self.colors[0], alpha=0.7)
        
        plt.xticks(range(len(responses)), responses, rotation=45, ha='right')
        plt.xlabel('Response')
        plt.ylabel('Count')
        plt.title(f"Question: {question}")
        plt.ylim(0, max(counts) * 1.1)
        
        # Add value labels on top of bars
        for i, count in enumerate(counts):
            ax.text(i, count, f'{count}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"response_count_distribution_Q{idx}.png", 
                   dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()

class FactualityPlotter(EvaluatorSpecificPlotter):
    """Creates factuality-specific plots."""
    
    def __init__(self, figsize: tuple = (15, 8), colors: List[str] = None):
        self.figsize = figsize
        self.colors = colors or ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    def create_plots(self, comparison_data: List[ComparisonData], output_dir: Path):
        # Collect data from all methods
        all_data = []
        for comp in comparison_data:
            all_data.append({
                'Method': comp.name,
                'Correct': comp.result.overall_metrics['num_is_correct'],
                'Incorrect': comp.result.overall_metrics['num_is_incorrect'],
                'Not attempted': comp.result.overall_metrics['num_is_not_attempted'],
                'Total': comp.result.overall_metrics['num_responses'],
            })
        
        df = pd.DataFrame(all_data)
        self._create_stacked_bar_chart(df, output_dir)
    
    def _create_stacked_bar_chart(self, df: pd.DataFrame, output_dir: Path):
        categories = ['Correct', 'Not attempted', 'Incorrect']
        colors = ['#6C8CFF', '#23233B', '#E6E6E6']
        
        for cat in categories:
            df[cat + ' %'] = df[cat] / df['Total']

        fig, ax = plt.subplots(figsize=(9, 4 + 0.5 * len(df)))
        left = np.zeros(len(df))
        bar_handles = []
        
        for idx, cat in enumerate(categories):
            bar = ax.barh(df['Method'], df[cat + ' %'], left=left, color=colors[idx], 
                        label=cat, height=0.5, edgecolor='none')
            bar_handles.append(bar)
            left += df[cat + ' %']

        # Style plot
        ax.set_xlim(0, 1)
        ax.set_xticks(np.linspace(0, 1, 5))
        ax.set_xticklabels([f"{int(x*100)}%" for x in np.linspace(0, 1, 5)])
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df['Method'], fontsize=11)
        
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis='both', length=0)
        
        ax.legend(bar_handles, categories, loc='upper left', bbox_to_anchor=(0, 1.08),
                 ncol=len(categories), frameon=False, fontsize=10)

        plt.tight_layout()
        plt.savefig(output_dir / "factuality_distribution.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

class GenericPlotter(EvaluatorSpecificPlotter):
    """Creates generic plots for unknown evaluator types."""
    
    def __init__(self, comparison_plotter):
        self.comparison_plotter = comparison_plotter
    
    def create_plots(self, comparison_data: List[ComparisonData], output_dir: Path):
        first_result = comparison_data[0].result
        
        # Try to create plots for all available metrics
        instance_metrics = set()
        overall_metrics = first_result.overall_metrics
        
        # Collect all possible instance metrics
        for instance in first_result.instance_metrics:
            if isinstance(instance, dict):
                instance_metrics.update(instance.keys())
            elif isinstance(instance, (int, float)):
                instance_metrics.add('value')
        
        # Create plots for numeric instance metrics
        for metric in instance_metrics:
            try:
                self.comparison_plotter.compare_distributions(
                    comparison_data, metric,
                    output_dir / f"{metric}_distribution.png",
                    title=f"{metric.replace('_', ' ').title()} Distribution",
                    plot_type="histogram"
                )
            except (ValueError, TypeError):
                continue  # Skip non-numeric metrics
        
        # Create plots for list-type overall metrics
        for metric, value in overall_metrics.items():
            if isinstance(value, (list, tuple)) and value:
                try:
                    self.comparison_plotter.compare_distributions(
                        comparison_data, metric,
                        output_dir / f"{metric}_distribution.png",
                        title=f"{metric.replace('_', ' ').title()} Distribution",
                        plot_type="violin"
                    )
                except (ValueError, TypeError):
                    continue
        
        # Create aggregate metrics plot for scalar values
        numeric_aggregates = []
        for metric, value in overall_metrics.items():
            if isinstance(value, (int, float)):
                numeric_aggregates.append(metric)
        
        if numeric_aggregates:
            self.comparison_plotter.compare_aggregate_metrics(
                comparison_data, numeric_aggregates,
                output_dir / "aggregate_metrics.png",
                title="Aggregate Metrics Comparison"
            ) 