from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from dataclasses import dataclass
from .base import EvalResult
from scipy.stats import chisquare
import numpy as np

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
            # print("instance: ", instance)
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
                if isinstance(overall_value, dict) and key in overall_value.keys():
                    overall_value = overall_value[key]
                else:
                    overall_value = None
                    break
            
            if overall_value is not None:
                if isinstance(overall_value, (list, tuple)):
                    values = [float(v) for v in overall_value if v is not None]
                elif isinstance(overall_value, (int, float)):
                    values = [float(overall_value)]
                elif isinstance(overall_value, dict):
                    values = [float(v) for v in overall_value.values() if v is not None]
            
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
            elif "response_distribution" in first_result.overall_metrics:
                evaluator_type = "response_count"
            elif "accuracy_given_attempted" in first_result.overall_metrics:
                evaluator_type = "factuality"
            else:
                raise ValueError(f"Unknown evaluator type: {evaluator_type}")
                evaluator_type = "generic"
        
        if evaluator_type == "response_count":
            self._create_response_count_plots(comparison_data, output_dir)
        elif evaluator_type == "factuality":
            self._create_factuality_plots(comparison_data, output_dir)
        else:
            self._create_comprehensive_comparison(evaluator_type, comparison_data, output_dir)
    
    # def _create_creative_writing_v3_plots(self, comparison_data: List[ComparisonData], output_dir: Path):
    #     """Create creative writing v3-specific plots."""
    #     # Creative writing v3 distribution
    #     # Plot distributions for each creative writing metric
    #     metrics = [
    #         "imagery_and_descriptive_quality",
    #         "nuanced_characters", 
    #         "emotionally_complex",
    #         "elegant_prose",
    #         "well_earned_lightness_or_darkness",
    #         "emotionally_engaging",
    #         "consistent_voicetone_of_writing",
    #         "sentences_flow_naturally",
    #         "overall_reader_engagement",
    #         "Average_Score"
    #     ]
        
    #     for metric in metrics:
    #         self.compare_distributions(
    #             comparison_data, metric,
    #             output_dir / f"{metric}_distribution.png",
    #             title=f"{metric.replace('_', ' ').title()} Distribution Comparison",
    #             plot_type="violin"
    #         )
        
    #     self._create_comprehensive_comparison(evaluator_type, comparison_data, output_dir)

    def _create_comprehensive_comparison(self,
                                    evaluator_type: str,
                                    comparison_data: List[ComparisonData],
                                    output_dir: Union[str, Path]) -> None:
        """Create a comprehensive set of comparison plots."""
        from verbalized_sampling.evals import get_evaluator
        evaluator_class = get_evaluator(evaluator_type)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not comparison_data:
            raise ValueError("No comparison data provided")
        
        # Create instance metric plots
        for metric_name, plot_type in evaluator_class.instance_plot_metrics:
            self.compare_distributions(
                comparison_data,
                metric_name,
                output_dir / f"{metric_name}_distribution.png",
                title=f"{metric_name.replace('_', ' ').title()} Distribution Comparison",
                plot_type=plot_type
            )
        
        # Create aggregate metric plots
        if evaluator_class.aggregate_plot_metrics:
            self.compare_aggregate_metrics(
                comparison_data,
                evaluator_class.aggregate_plot_metrics,
                output_dir / "aggregate_metrics.png",
                title=f"{evaluator_class.name.replace('_', ' ').title()} Metrics Comparison",
                plot_type="bar"
            )

    def _generate_uniform_state_sample(self, n_trials=500, n_states=50, seed=42):
        """Generate what a truly uniform state selection would look like."""
        if seed:
            np.random.seed(seed)
        
        # Simulate uniform random selection
        state_selections = np.random.choice(range(n_states), size=n_trials, replace=True)
        
        # Count frequencies
        unique_states, counts = np.unique(state_selections, return_counts=True)
        
        # Create full array (including states with 0 counts)
        full_counts = np.zeros(n_states)
        full_counts[unique_states] = counts
        
        return sorted([int(count) for count in full_counts], reverse=True)


    def _create_response_count_plots(self, comparison_data: List[ComparisonData], output_dir: Path):
        """Create response count-specific plots."""
        # Extract and sort data
        response_counter = comparison_data[0].result.overall_metrics["response_distribution"]
        sorted_items = sorted(response_counter.items(), key=lambda x: x[1], reverse=True)
        values, labels = zip(*sorted_items)
        
        # Create plot
        plt.figure(figsize=self.figsize)
        ax = sns.barplot(data=pd.DataFrame({'Response Type': labels, 'Count': values}),
                        x='Response Type', y='Count', palette=[self.colors[0]], alpha=0.7)
        
        # Style plot
        plt.xticks(rotation=45)
        plt.xlabel('Name of the State')
        plt.ylabel('Count')
        plt.title('State Name Distribution')
        plt.ylim(0, 500)
        
        # Add value labels
        for i, v in enumerate(values):
            ax.text(i, v, f'{int(v)}', ha='center', va='bottom')
        plt.tight_layout()
        
        # Chi-square test
        total_trials = sum(values)
        values_extended = list(values) + [0] * (50 - len(values))
        expected_frequencies = self._generate_uniform_state_sample(n_trials=total_trials, n_states=50, seed=42)
        chi2_stat, _ = chisquare(f_obs=values_extended, f_exp=expected_frequencies)
        
        # Add test result and save
        plt.text(0.98, 0.98, f'Chi-square: {chi2_stat:.2f}', transform=plt.gca().transAxes,
                va='top', ha='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.savefig(output_dir / "response_count_distribution.png", dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()


    def _create_factuality_plots(self, comparison_data: List[ComparisonData], output_dir: Path):
        """Create factuality-specific plots as horizontal stacked bar chart with percentages."""
        # Create DataFrame from metrics
        df = pd.DataFrame([{
            'Method': 'GPT-4.1 (Structure w Prob)',
            'Correct': comp.result.overall_metrics['num_is_correct'],
            'Incorrect': comp.result.overall_metrics['num_is_incorrect'],
            'Not attempted': comp.result.overall_metrics['num_is_not_attempted'],
            'Total': comp.result.overall_metrics['num_responses'],
        } for comp in comparison_data])

        # Setup plot data
        categories = ['Correct', 'Not attempted', 'Incorrect']
        colors = ['#6C8CFF', '#23233B', '#E6E6E6']
        for cat in categories:
            df[cat + ' %'] = df[cat] / df['Total']

        # Create plot
        fig, ax = plt.subplots(figsize=(9, 4 + 0.5 * len(df)))
        left = np.zeros(len(df))
        bar_handles = []
        
        # Plot bars
        for idx, cat in enumerate(categories):
            bar = ax.barh(df['Method'], df[cat + ' %'], left=left, color=colors[idx], 
                         label=cat, height=0.5, edgecolor='none')
            bar_handles.append(bar)
            left += df[cat + ' %']

        # # Add labels and style
        # for i, (method, correct_pct) in enumerate(zip(df['Method'], df['Correct %'])):
        #     if correct_pct > 0:
        #         ax.text(correct_pct/2, i, f"{method} Correct  {correct_pct*100:.1f}%",
        #                va='center', ha='left', color='black', fontsize=10, fontweight='bold',
        #                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.2'))

        # Style plot
        ax.set_xlim(0, 1)
        ax.set_xticks(np.linspace(0, 1, 5))
        ax.set_xticklabels([f"{int(x*100)}%" for x in np.linspace(0, 1, 5)])
        ax.set_yticklabels(df['Method'], fontsize=11)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis='both', length=0)
        
        # Add legend
        ax.legend(bar_handles, categories, loc='upper left', bbox_to_anchor=(0, 1.08),
                 ncol=len(categories), frameon=False, fontsize=10)

        plt.tight_layout()
        plt.savefig(output_dir / "factuality_distribution.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()


        
    def _create_generic_plots(self, comparison_data: List[ComparisonData], output_dir: Path):
        """Create generic plots for unknown evaluator types."""
        first_result = comparison_data[0].result
        
        # Try to create plots for all available metrics
        instance_metrics = set()
        overall_metrics = first_result.overall_metrics
        
        # Collect all possible instance metrics - now handling list format
        for instance in first_result.instance_metrics:
            if isinstance(instance, dict):
                instance_metrics.update(instance.keys())
            elif isinstance(instance, (int, float)):
                instance_metrics.add('value')  # Add a generic metric name for numeric values
        
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
                                          comparison_data: Dict[str, EvalResult],
                                          key_metric_names: List[tuple[str, str]],
                                          output_path: Union[str, Path],
                                          title: Optional[str] = None,
                                          subplot_titles: Optional[List[str]] = None) -> None:
        """Create multi-subplot comparison chart like in your reference image."""
        
        plt.figure(figsize=(4 * len(key_metric_names), 8))
        
        # Enhanced colors for consistency
        colors = ['#8B8B8B', '#B8C5E1', '#5A5A5A', '#7CB8D4', '#003F7F']
        patterns = ['', '', '', '', '///']
        
        # Get method names from comparison_data keys
        method_names = list(comparison_data.keys())
        
        # Setup bar chart parameters
        x = np.arange(len(key_metric_names))
        width = 0.8 / len(method_names)  # Adjust width based on number of methods
        
        # Create grouped bars
        for i, method_name in enumerate(method_names):
            print(f"Method name: {method_name}")
            eval_result = comparison_data[method_name]
            values = []
            
            # Extract values for each metric
            for metric_name, plot_title in key_metric_names:
                value = eval_result.overall_metrics
                for key in metric_name.split('.'):
                    print(f"Key: {key}")
                    # print(f"Value: {value}")
                    if isinstance(value, dict) and key in value:
                        value = value[key]
                    else:
                        value = 0.0
                        break
                
                # Handle list/tuple values by taking mean
                if isinstance(value, (list, tuple)):
                    value = np.mean(value) if value else 0.0
                
                values.append(float(value) if value is not None else 0.0)
            
            # Calculate offset for grouped bars
            offset = (i - len(method_names)/2 + 0.5) * width
            
            # Create bars for this method
            color = colors[i % len(colors)]
            pattern = patterns[i % len(patterns)] if patterns else ''
            
            plt.bar(x + offset, values, width,
                   label=method_name,
                   color=color,
                   alpha=0.8,
                   hatch=pattern,
                   edgecolor='white',
                   linewidth=0.5)
            
        # Styling
        plt.xlabel('Metrics', fontsize=12, fontweight='bold')
        plt.ylabel('Score', fontsize=12, fontweight='bold')
        plt.xticks(x, [plot_title for _, plot_title in key_metric_names], fontsize=11)
        plt.ylim(0, 1)

        # Add legend
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15),
                  frameon=True, fancybox=True, shadow=True,
                  fontsize=10, framealpha=0.9, ncol=len(method_names))
        
        # Add title if provided
        if title:
            plt.title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.grid(False)
        
        # Set y limits
        all_values = []
        for method_name in method_names:
            eval_result = comparison_data[method_name]
            for metric_name, plot_title in key_metric_names:
                value = eval_result.overall_metrics
                for key in metric_name.split('.'):
                    if isinstance(value, dict) and key in value:
                        value = value[key]
                    else:
                        value = 0.0
                        break
                if isinstance(value, (list, tuple)):
                    value = np.mean(value) if value else 0.0
                all_values.append(float(value) if value is not None else 0.0)
        
        # if all_values:
        #     plt.ylim(0, max(all_values) * 1.1)
        
        # Finalize plot
        plt.tight_layout()
        plt.savefig(output_path / "comparison_chart.png", dpi=300, bbox_inches='tight',
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

def plot_comparison_chart(results: Dict[str, Union[EvalResult, str, Path]],
                          output_path: Union[str, Path],
                          title: Optional[str] = None,
                          **kwargs) -> None:
    """Create a performance comparison chart."""
    plotter = ComparisonPlotter(**kwargs)
    print(f"Starting to plot comparison chart...")
    comparison_data = {} # {exp_name: EvalResult}
    key_metric_names = []
    for metric, results in results.items():
        from verbalized_sampling.evals import get_evaluator
        evaluator_class = get_evaluator(metric)
        if evaluator_class.key_plot_metrics:
            key_metric_names.extend(evaluator_class.key_plot_metrics)
        
        for exp_name, result in results.items():
            if isinstance(result, (str, Path)):
                # Load from file
                with open(result, 'r') as f:
                    import json
                    result_dict = json.load(f)
                    eval_result = EvalResult.from_dict(result_dict)
            else:
                eval_result = result
            if exp_name not in comparison_data:
                comparison_data[exp_name] = eval_result
            else:
                comparison_data[exp_name] = comparison_data[exp_name] + eval_result
    print(f"Number of experiments: {len(comparison_data)}")
    print(f"Plotting comparison chart...")
    plotter.create_performance_comparison_chart(
            comparison_data, 
            key_metric_names,
            output_path,
            title=title)
        