from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from dataclasses import dataclass
from .base import EvalResult

@dataclass
class ComparisonData:
    """Container for comparison data."""
    name: str  # Format name (e.g., "direct", "cot", "verbalized")
    result: EvalResult
    color: Optional[str] = None

class ComparisonPlotter:
    """Plotter for comparing evaluation results across different formats."""
    
    def __init__(self, style: str = "seaborn-v0_8", figsize: tuple = (15, 8)):
        plt.style.use(style)
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def compare_distributions(self, 
                            comparison_data: List[ComparisonData],
                            metric_name: str,
                            output_path: Union[str, Path],
                            title: Optional[str] = None,
                            plot_type: str = "histogram",
                            bins: int = 100,
                            alpha: float = 0.7) -> None:
        """
        Create distribution comparison plots that can handle both instance and overall metrics.
        
        Args:
            comparison_data: List of comparison data
            metric_name: Name of metric to plot (can be in instance_metrics or overall_metrics)
            output_path: Where to save the plot
            title: Plot title
            plot_type: "histogram", "violin", "kde", or "box"
            bins: Number of bins for histogram
            alpha: Transparency level
        """
        
        plt.figure(figsize=self.figsize)
        
        # Extract data for the specified metric
        all_data = []
        labels = []
        colors = []
        
        for i, data in enumerate(comparison_data):
            if data is None:
                continue
            values = self._extract_metric_values(data, metric_name)
            
            if values:
                all_data.append(values)
                labels.append(data.name)
                colors.append(data.color or self.colors[i % len(self.colors)])
        
        if not all_data:
            raise ValueError(f"No data found for metric '{metric_name}'")
        
        # Create the appropriate plot type
        if plot_type == "histogram":
            self._plot_histogram(all_data, labels, colors, bins, alpha)
        elif plot_type == "sns_histogram":
            self._plot_sns_histogram(all_data, labels, colors, bins, alpha)
        elif plot_type == "violin":
            self._plot_violin(all_data, labels, colors)
        elif plot_type == "kde":
            self._plot_kde(all_data, labels, colors)
        elif plot_type == "box":
            self._plot_box(all_data, labels, colors)
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}")
        
        plt.xlabel(metric_name.replace('_', ' ').title())
        plt.title(title or f'Distribution Comparison: {metric_name.replace("_", " ").title()}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _extract_metric_values(self, data: ComparisonData, metric_name: str) -> List[float]:
        """Extract metric values from either instance_metrics or overall_metrics."""
        values = []
        
        # First, try to find in instance_metrics
        for instance in data.result.instance_metrics:
            value = instance
            for key in metric_name.split('.'):
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    value = None
                    break
            if value is not None:
                if isinstance(value, (list, tuple)):
                    values.extend([float(v) for v in value if v is not None])
                else:
                    values.append(float(value))
        
        # If no values found in instance_metrics, try overall_metrics
        if not values:
            overall_value = data.result.overall_metrics
            for key in metric_name.split('.'):
                if isinstance(overall_value, dict) and key in overall_value:
                    overall_value = overall_value[key]
                else:
                    overall_value = None
                    break
            
            if overall_value is not None:
                if isinstance(overall_value, (list, tuple)):
                    values = [float(v) for v in overall_value if v is not None]
                elif isinstance(overall_value, (int, float)):
                    values = [float(overall_value)]
        
        return values
    
    def _plot_histogram(self, all_data: List[List[float]], labels: List[str], 
                       colors: List[str], bins: int, alpha: float):
        """Create histogram plot."""
        plt.hist(all_data, bins=bins, alpha=alpha, label=labels, color=colors)
        plt.ylabel('Frequency')
    
    def _plot_sns_histogram(self, all_data: List[List[float]], labels: List[str], colors: List[str], bins: int, alpha: float):
        """Create histogram plot."""
        sns.histplot(all_data, alpha=alpha, color=colors, kde=True)
        plt.ylabel('Frequency')
    
    def _plot_violin(self, all_data: List[List[float]], labels: List[str], colors: List[str]):
        """Create violin plot similar to your notebook example."""
        # Prepare data for seaborn
        df_data = []
        for i, (data_values, label) in enumerate(zip(all_data, labels)):
            for value in data_values:
                df_data.append({"Value": value, "Method": label})
        
        df = pd.DataFrame(df_data)
        
        # Create color palette
        palette = {label: color for label, color in zip(labels, colors)}
        
        # Create violin plot
        ax = sns.violinplot(
            data=df,
            x="Value", 
            y="Method",
            order=labels,
            palette=palette,
            scale="width",
            # inner=None,
            cut=0,
            alpha=0.6
        )
        
        # Add quartile lines and median points (like your notebook example)
        for i, label in enumerate(labels):
            vals = df[df["Method"] == label]["Value"].values
            if len(vals) > 0:
                q1, median, q3 = np.percentile(vals, [25, 50, 75])
                
                # Draw IQR line
                ax.hlines(y=i, xmin=q1, xmax=q3, color="black", linewidth=6, zorder=3)
                
                # Draw median point
                ax.scatter(median, i, color="white", edgecolor="black", s=50, zorder=4)
    
    def _plot_kde(self, all_data: List[List[float]], labels: List[str], colors: List[str]):
        """Create KDE plot."""
        for data_values, label, color in zip(all_data, labels, colors):
            if len(data_values) > 1:  # Need at least 2 points for KDE
                sns.kdeplot(data_values, label=label, color=color)
        plt.ylabel('Density')
    
    def _plot_box(self, all_data: List[List[float]], labels: List[str], colors: List[str]):
        """Create box plot."""
        box_plot = plt.boxplot(all_data, labels=labels, patch_artist=True, vert=False)  # vert=False rotates the plot
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Make the median lines more bold
        for median in box_plot['medians']:
            median.set_linewidth(3)  # Make median line thicker
            median.set_color('black')  # Make it black for better visibility
        
        plt.xlabel('Value')  # Changed from ylabel to xlabel since we rotated
    
    def compare_instance_metrics(self, 
                               comparison_data: List[ComparisonData],
                               metric_name: str,
                               output_path: Union[str, Path],
                               title: Optional[str] = None,
                               plot_type: str = "histogram",
                               **kwargs) -> None:
        """Backward compatibility method - now uses compare_distributions."""
        self.compare_distributions(
            comparison_data, metric_name, output_path, title, plot_type, **kwargs
        )
    
    def compare_aggregate_metrics(self,
                                comparison_data: List[ComparisonData],
                                metric_names: List[str],
                                output_path: Union[str, Path],
                                title: Optional[str] = None,
                                plot_type: str = "bar",
                                figsize: Optional[tuple] = None,
                                colors: Optional[List[str]] = None,
                                patterns: Optional[List[str]] = None) -> None:
        """Create bar chart comparison of aggregate metrics with improved styling."""
        
        if plot_type not in ["bar", "line"]:
            raise ValueError("plot_type must be 'bar' or 'line'")
        
        # Use provided figsize or default
        fig_size = figsize or self.figsize
        plt.figure(figsize=fig_size)
        
        # Enhanced color palette (similar to your reference image)
        if colors is None:
            colors = ['#8B8B8B', '#B8C5E1', '#5A5A5A', '#7CB8D4', '#003F7F']
        
        # Patterns for additional distinction (like the hatching in your image)
        if patterns is None:
            patterns = ['', '', '', '', '///']
        
        # Prepare data
        format_names = [data.name for data in comparison_data]
        metric_values = {metric: [] for metric in metric_names}
        
        # Extract and process metric values
        for data in comparison_data:
            for metric in metric_names:
                # Navigate nested metrics structure
                value = data.result.overall_metrics
                for key in metric.split('.'):
                    if isinstance(value, dict) and key in value:
                        value = value[key]
                    else:
                        value = 0.0
                        break
                
                # Handle list/tuple values by taking mean
                if isinstance(value, (list, tuple)):
                    value = np.mean(value) if value else 0.0
                
                metric_values[metric].append(float(value) if value is not None else 0.0)
        
        if plot_type == "bar":
            # Setup bar chart parameters
            x = np.arange(len(metric_names))
            width = 0.15
            
            # Create grouped bars with enhanced styling
            for i, (format_name, data) in enumerate(zip(format_names, comparison_data)):
                values = [metric_values[metric][i] for metric in metric_names]
                offset = (i - len(format_names)/2 + 0.5) * width
                
                # Apply visual styling
                color = data.color or colors[i % len(colors)]
                pattern = patterns[i % len(patterns)] if patterns else ''
                
                plt.bar(x + offset, values, width,
                       label=format_name,
                       color=color,
                       alpha=0.8,
                       hatch=pattern,
                       edgecolor='white',
                       linewidth=0.5)
            
            # Apply bar chart styling
            plt.xlabel('Metrics', fontsize=12, fontweight='bold')
            plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
            plt.xticks(x, [metric.replace('_', ' ').title() for metric in metric_names], fontsize=11)
            
            # Position and style legend
            plt.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98),
                      frameon=True, fancybox=True, shadow=True,
                      fontsize=10, framealpha=0.9)
            
            plt.grid(False)
            plt.ylim(0, max(max(metric_values[metric]) for metric in metric_names) * 1.1)
            
        else:  # line plot
            # Define line plot styling elements
            line_styles = ['-', '--', '-.', ':', '-']
            markers = ['o', 's', '^', 'D', 'v']
            
            # Create line plot with enhanced styling
            for i, format_name in enumerate(format_names):
                values = [metric_values[metric][i] for metric in metric_names]
                color = colors[i % len(colors)]
                linestyle = line_styles[i % len(line_styles)]
                marker = markers[i % len(markers)]
                
                plt.plot(metric_names, values,
                        marker=marker, label=format_name,
                        color=color, linewidth=2.5, markersize=8,
                        linestyle=linestyle, markerfacecolor='white',
                        markeredgewidth=2, markeredgecolor=color)
            
            # Apply line plot styling
            plt.xlabel('Metrics', fontsize=12, fontweight='bold')
            plt.ylabel('Score', fontsize=12, fontweight='bold')
            plt.xticks(rotation=45, fontsize=11)
            plt.legend(loc='best', fontsize=10)
            plt.grid(True, alpha=0.3)
        
        # Add title if provided
        if title:
            plt.title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Finalize plot
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
    
    def _create_ngram_plots(self, comparison_data: List[ComparisonData], output_dir: Path):
        """Create ROUGE-L specific plots."""
        # ROUGE-L score distribution
        self.compare_distributions(
            comparison_data, "pairwise_rouge_l_scores",
            output_dir / "rouge_l_scores_distribution.png",
            title="ROUGE-L Score Distribution Comparison",
            plot_type="violin"
        )
        
        # Response length distribution
        self.compare_distributions(
            comparison_data, "response_length",
            output_dir / "response_length_distribution.png",
            title="Response Length Distribution Comparison",
            plot_type="histogram"
        )
        
        # Aggregate metrics summary
        self.compare_aggregate_metrics(
            comparison_data,
            ["average_rouge_l"],
            output_dir / "rouge_l_metrics.png",
            title="ROUGE-L Metrics Comparison",
            plot_type="bar"
        )

    def create_comprehensive_comparison(self,
                                      comparison_data: List[ComparisonData],
                                      output_dir: Union[str, Path],
                                      evaluator_type: str = "auto") -> None:
        """Create a comprehensive set of comparison plots."""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not comparison_data:
            raise ValueError("No comparison data provided")
        
        # Determine evaluator type from first result
        if evaluator_type == "auto":
            first_result = comparison_data[0].result
            if "average_similarity" in first_result.overall_metrics:
                evaluator_type = "diversity"
            elif "fluency" in first_result.overall_metrics:
                evaluator_type = "ttct"
            elif "average_creativity_index" in first_result.overall_metrics:
                evaluator_type = "creativity_index"
            elif "mean_token_length" in first_result.overall_metrics:
                evaluator_type = "length"
            elif "average_ngram_diversity" in first_result.overall_metrics:
                evaluator_type = "ngram"
            else:
                evaluator_type = "generic"
        
        # Create plots based on evaluator type
        if evaluator_type == "diversity":
            self._create_diversity_plots(comparison_data, output_dir)
        elif evaluator_type == "ttct":
            self._create_ttct_plots(comparison_data, output_dir)
        elif evaluator_type == "creativity_index":
            self._create_creativity_index_plots(comparison_data, output_dir)
        elif evaluator_type == "creative_writing_v3":
            self._create_creative_writing_v3_plots(comparison_data, output_dir)
        elif evaluator_type == "length":
            self._create_length_plots(comparison_data, output_dir)
        elif evaluator_type == "ngram":
            self._create_ngram_plots(comparison_data, output_dir)
        else:
            self._create_generic_plots(comparison_data, output_dir)
    
    def _create_creative_writing_v3_plots(self, comparison_data: List[ComparisonData], output_dir: Path):
        """Create creative writing v3-specific plots."""
        # Creative writing v3 distribution
        # Plot distributions for each creative writing metric
        metrics = [
            "imagery_and_descriptive_quality",
            "nuanced_characters", 
            "emotionally_complex",
            "elegant_prose",
            "well_earned_lightness_or_darkness",
            "emotionally_engaging",
            "consistent_voicetone_of_writing",
            "sentences_flow_naturally",
            "overall_reader_engagement",
            "Average_Score"
        ]
        
        for metric in metrics:
            self.compare_distributions(
                comparison_data, metric,
                output_dir / f"{metric}_distribution.png",
                title=f"{metric.replace('_', ' ').title()} Distribution Comparison",
                plot_type="violin"
            )
        
        # Aggregate metrics
        self.compare_aggregate_metrics(
            comparison_data,
            ["creative_writing_v3"],
            output_dir / "creative_writing_v3_metrics.png",
            title="Creative Writing v3 Metrics Comparison",
            plot_type="bar"
        )
    
    def _create_diversity_plots(self, comparison_data: List[ComparisonData], output_dir: Path):
        """Create diversity-specific plots."""
        # Pairwise similarities distribution (from overall_metrics)
        self.compare_distributions(
            comparison_data, "pairwise_diversities",
            output_dir / "pairwise_diversities_distribution.png",
            title="Pairwise Diversity Distribution Comparison",
            plot_type="violin"
        )
        
        # Response length distribution (from instance_metrics)
        self.compare_distributions(
            comparison_data, "response_length",
            output_dir / "response_length_distribution.png",
            title="Response Length Distribution Comparison",
            plot_type="histogram"
        )
        
        # Vocabulary richness distribution (from instance_metrics)
        self.compare_distributions(
            comparison_data, "vocabulary_richness",
            output_dir / "vocabulary_richness_distribution.png",
            title="Vocabulary Richness Distribution Comparison",
            plot_type="histogram"
        )
        
        # Aggregate metrics summary
        self.compare_aggregate_metrics(
            comparison_data,
            ["average_diversity", "min_diversity", "max_diversity", "std_diversity", "average_response_length", "average_unique_words", "average_vocabulary_richness", "total_cost", "pairwise_diversities"],
            output_dir / "diversity_metrics.png",
            title="Diversity Metrics Comparison",
            plot_type="bar"
        )
    
    def _create_ttct_plots(self, comparison_data: List[ComparisonData], output_dir: Path):
        """Create TTCT-specific plots."""
        # Individual TTCT dimensions
        for dimension in ["fluency", "flexibility", "originality", "elaboration"]:
            self.compare_distributions(
                comparison_data, f"{dimension}.score",
                output_dir / f"{dimension}_distribution.png",
                title=f"{dimension.title()} Score Distribution",
                plot_type="violin"
            )
        
        # Aggregate comparison
        self.compare_aggregate_metrics(
            comparison_data,
            ["fluency", "flexibility", "originality", "elaboration", "overall"],
            output_dir / "ttct_metrics.png",
            title="TTCT Dimensions Comparison"
        )
    
    def _create_creativity_index_plots(self, comparison_data: List[ComparisonData], output_dir: Path):
        """Create creativity index-specific plots."""
        # Creativity index distribution
        self.compare_distributions(
            comparison_data, "creativity_index",
            output_dir / "creativity_index_distribution.png",
            title="Creativity Index Distribution Comparison",
            plot_type="violin"
        )
        
        # Aggregate metrics
        self.compare_aggregate_metrics(
            comparison_data,
            ["average_creativity_index", "average_coverage", "match_rate"],
            output_dir / "creativity_index_metrics.png",
            title="Creativity Index Metrics Comparison"
        )
    
    def _create_length_plots(self, comparison_data: List[ComparisonData], output_dir: Path):
        """Create length-specific plots."""
        # Token length distribution
        self.compare_distributions(
            comparison_data, "token_length",
            output_dir / "token_length_distribution.png",
            title="Token Length Distribution Comparison",
            plot_type="violin"
        )
        
        # Aggregate metrics
        self.compare_aggregate_metrics(
            comparison_data,
            ["mean_token_length"],
            output_dir / "length_metrics.png",
            title="Length Metrics Comparison"
        )
    
    def _create_generic_plots(self, comparison_data: List[ComparisonData], output_dir: Path):
        """Create generic plots for unknown evaluator types."""
        first_result = comparison_data[0].result
        
        # Try to create plots for all available metrics
        instance_metrics = set()
        overall_metrics = first_result.overall_metrics
        
        # Collect all possible instance metrics
        for instance in first_result.instance_metrics:
            instance_metrics.update(instance.keys())
        
        # Create plots for numeric instance metrics
        for metric in instance_metrics:
            try:
                self.compare_distributions(
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
                    self.compare_distributions(
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
            self.compare_aggregate_metrics(
                comparison_data, numeric_aggregates,
                output_dir / "aggregate_metrics.png",
                title="Aggregate Metrics Comparison"
            )

    def create_performance_comparison_chart(self,
                                          comparison_data: List[ComparisonData],
                                          task_metrics: Dict[str, List[str]],
                                          output_path: Union[str, Path],
                                          title: Optional[str] = None,
                                          subplot_titles: Optional[List[str]] = None) -> None:
        """Create multi-subplot comparison chart like in your reference image."""
        
        n_tasks = len(task_metrics)
        fig, axes = plt.subplots(1, n_tasks, figsize=(6 * n_tasks, 6))
        
        if n_tasks == 1:
            axes = [axes]
        
        # Enhanced colors for consistency
        colors = ['#8B8B8B', '#B8C5E1', '#5A5A5A', '#7CB8D4', '#003F7F']
        patterns = ['', '', '', '', '///']
        
        format_names = [data.name for data in comparison_data]
        
        for task_idx, (task_name, metrics) in enumerate(task_metrics.items()):
            ax = axes[task_idx]
            
            # Prepare data for this task
            metric_values = {metric: [] for metric in metrics}
            
            for data in comparison_data:
                for metric in metrics:
                    # Extract metric value
                    value = data.result.overall_metrics
                    for key in metric.split('.'):
                        if isinstance(value, dict) and key in value:
                            value = value[key]
                        else:
                            value = 0.0
                            break
                    
                    if isinstance(value, (list, tuple)):
                        value = np.mean(value) if value else 0.0
                    
                    metric_values[metric].append(float(value) if value is not None else 0.0)
            
            # Create bars
            x = np.arange(len(metrics))
            width = 0.15
            
            for i, (format_name, data) in enumerate(zip(format_names, comparison_data)):
                values = [metric_values[metric][i] for metric in metrics]
                offset = (i - len(format_names)/2 + 0.5) * width
                
                color = data.color or colors[i % len(colors)]
                pattern = patterns[i % len(patterns)]
                
                bars = ax.bar(x + offset, values, width,
                             label=format_name if task_idx == 0 else "",  # Only show legend on first subplot
                             color=color,
                             alpha=0.8,
                             hatch=pattern,
                             edgecolor='white',
                             linewidth=0.5)
            
            # Styling
            ax.set_xlabel('Metrics', fontsize=11, fontweight='bold')
            if task_idx == 0:
                ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
            
            ax.set_xticks(x)
            ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], fontsize=10)
            
            # Subplot title
            subplot_title = subplot_titles[task_idx] if subplot_titles else task_name
            ax.set_title(subplot_title, fontsize=12, fontweight='bold')
            
            # Remove grid
            ax.grid(False)
            
            # Set y limits
            ax.set_ylim(0, max(max(metric_values[metric]) for metric in metrics) * 1.1)
        
        # Add shared legend
        if format_names:
            fig.legend(format_names, loc='upper center', bbox_to_anchor=(0.5, 0.02),
                      ncol=len(format_names), fontsize=10, frameon=True)
        
        # Overall title
        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        if title:
            plt.subplots_adjust(top=0.85, bottom=0.15)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()

# Convenience function for easy usage
def plot_evaluation_comparison(results: Dict[str, Union[EvalResult, str, Path]],
                             output_dir: Union[str, Path],
                             evaluator_type: str = "auto",
                             **kwargs) -> None:
    """
    Convenience function to plot evaluation comparisons.
    
    Args:
        results: Dict mapping format names to EvalResult objects or file paths
        output_dir: Directory to save plots
        evaluator_type: Type of evaluator ("diversity", "ttct", "creativity_index", "length", "auto")
        **kwargs: Additional arguments passed to ComparisonPlotter
    """
    
    plotter = ComparisonPlotter(**kwargs)
    comparison_data = []
    
    for name, result in results.items():
        if isinstance(result, (str, Path)):
            # Load from file
            with open(result, 'r') as f:
                import json
                result_dict = json.load(f)
                eval_result = EvalResult.from_dict(result_dict)
        else:
            eval_result = result
        
        comparison_data.append(ComparisonData(name=name, result=eval_result))
    
    plotter.create_comprehensive_comparison(comparison_data, output_dir, evaluator_type) 