from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from ..analysis.evals.base import EvalResult
from .base import ComparisonData, MetricExtractor
from .factory import PlotterFactory, EvaluatorPlotterFactory

class ComparisonPlotter:
    """Main plotter for comparing evaluation results across different formats."""
    
    def __init__(self, style: str = "seaborn-v0_8", figsize: tuple = (15, 8)):
        plt.style.use(style)
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        # self.colors = ['#FC8EAC', '#A4C8E1']
        self.metric_extractor = MetricExtractor()
    
    def compare_distributions(self, 
                            comparison_data: List[ComparisonData],
                            metric_name: str,
                            output_path: Union[str, Path],
                            title: Optional[str] = None,
                            plot_type: str = "histogram",
                            **kwargs) -> None:
        """Create distribution comparison plots."""
        
        plt.figure(figsize=self.figsize)
        
        # Extract data for the specified metric
        all_data = []
        labels = []
        colors = []
        
        for i, data in enumerate(comparison_data):
            if data is None:
                continue
            values = self.metric_extractor.extract_metric_values(data, metric_name)
            
            if values:
                all_data.append(values)
                labels.append(data.name)
                colors.append(data.color or self.colors[i % len(self.colors)])
        
        if not all_data:
            raise ValueError(f"No data found for metric '{metric_name}'")
        
        # Create the plot using factory
        plotter = PlotterFactory.create_plotter(plot_type, self.colors)
        plotter.plot(all_data, labels, colors, **kwargs)
        
        plt.xlabel(metric_name.replace('_', ' ').title())
        plt.title(title or f'Distribution Comparison: {metric_name.replace("_", " ").title()}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def compare_aggregate_metrics(self,
                                comparison_data: List[ComparisonData],
                                metric_names: List[str],
                                output_path: Union[str, Path],
                                title: Optional[str] = None,
                                plot_type: str = "bar",
                                figsize: Optional[tuple] = None,
                                colors: Optional[List[str]] = None,
                                patterns: Optional[List[str]] = None,
                                show_error_bars: bool = True,
                                error_bar_type: str = "ci") -> None:
        """Create bar chart comparison of aggregate metrics with error bars in subplots."""
        
        if plot_type not in ["bar", "line"]:
            raise ValueError("plot_type must be 'bar' or 'line'")
        
        # Calculate optimal grid layout
        n_metrics = len(metric_names)
        if n_metrics == 1:
            nrows, ncols = 1, 1
        elif n_metrics == 2:
            nrows, ncols = 1, 2
        elif n_metrics <= 4:
            nrows, ncols = 2, 2
        elif n_metrics <= 6:
            nrows, ncols = 2, 3
        elif n_metrics <= 9:
            nrows, ncols = 3, 3
        else:
            nrows = int(np.ceil(np.sqrt(n_metrics)))
            ncols = int(np.ceil(n_metrics / nrows))
        
        # Calculate figure size based on number of subplots
        subplot_width = 5
        subplot_height = 4
        fig_width = ncols * subplot_width
        fig_height = nrows * subplot_height
        
        if figsize is None:
            figsize = (fig_width, fig_height)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        
        # Ensure axes is always a flat array for easier indexing
        if n_metrics == 1:
            axes = [axes]
        elif nrows == 1 or ncols == 1:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        else:
            axes = axes.flatten()
        
        if colors is None:
            colors = ['#8B8B8B', '#B8C5E1', '#5A5A5A', '#7CB8D4', '#003F7F']
        
        if patterns is None:
            patterns = ['', '', '', '', '///']
        
        # Prepare data
        format_names = [data.name for data in comparison_data]
        metric_values, metric_errors = self._extract_aggregate_data(
            comparison_data, metric_names, error_bar_type
        )
        
        # Create subplot for each metric
        for i, metric_name in enumerate(metric_names):
            ax = axes[i]
            
            if plot_type == "bar":
                self._create_single_metric_bar_plot(
                    ax, metric_name, format_names, comparison_data,
                    metric_values[metric_name], metric_errors[metric_name],
                    colors, patterns, show_error_bars
                )
            else:
                self._create_single_metric_line_plot(
                    ax, metric_name, format_names,
                    metric_values[metric_name], metric_errors[metric_name],
                    colors, show_error_bars
                )
            
            # Set subplot title
            ax.set_title(metric_name.replace('_', ' ').title(), fontsize=12, fontweight='bold', pad=10)
        
        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        # Add overall title
        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
        # Add a single legend for all subplots
        if comparison_data:
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02),
                      frameon=True, fancybox=True, shadow=True,
                      fontsize=10, framealpha=0.9, ncol=min(len(format_names), 5))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)  # Make room for legend
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
            evaluator_type = self._detect_evaluator_type(comparison_data[0].result)
        
        # Create evaluator-specific plots
        evaluator_plotter = EvaluatorPlotterFactory.create_plotter(
            evaluator_type, figsize=self.figsize, colors=self.colors
        )
        
        if evaluator_plotter:
            evaluator_plotter.create_plots(comparison_data, output_dir)
        else:
            # Use standard evaluator-based plotting
            self._create_standard_evaluator_plots(evaluator_type, comparison_data, output_dir)
    
    def create_performance_comparison_chart(self,
                                          comparison_data: Dict[str, EvalResult],
                                          key_metric_names: List[Tuple[str, str]],
                                          output_path: Union[str, Path],
                                          title: Optional[str] = None,
                                          show_error_bars: bool = True,
                                          error_bar_type: str = "ci") -> None:
        """Create multi-subplot comparison chart with error bars."""
        
        plt.figure(figsize=(6 * len(key_metric_names), 8))
        
        # Enhanced colors for consistency
        colors = ['#8B8B8B', '#B8C5E1', '#5A5A5A', '#7CB8D4', '#003F7F']
        patterns = ['', '', '', '', '///']
        
        method_names = list(comparison_data.keys())
        x = np.arange(len(key_metric_names))
        width = 0.8 / len(method_names)
        
        ylim = 1
        
        # Create grouped bars
        for i, method_name in enumerate(method_names):
            eval_result = comparison_data[method_name]
            values = []
            errors = []
            
            # Extract values and errors for each metric
            for metric_name, plot_title in key_metric_names:
                value = eval_result.overall_metrics
                for key in metric_name.split('.'):
                    if key == "average_kl_divergence":
                        ylim = 3.0
                    if isinstance(value, dict) and key in value:
                        value = value[key]
                    else:
                        value = 0.0
                        break
                
                if isinstance(value, (list, tuple)):
                    value = np.mean(value) if value else 0.0
                
                values.append(float(value) if value is not None else 0.0)
                
                if (metric_name.startswith("avg_")) and (metric_name.replace("avg_", "std_") in eval_result.overall_metrics):
                    errors.append(eval_result.overall_metrics[metric_name.replace("avg_", "std_")])
                else:
                    errors.append(0.0)
            
            print(f"values: {values}")
            print(f"errors: {errors}")
            # Calculate offset for grouped bars
            offset = (i - len(method_names)/2 + 0.5) * width
            
            # Create bars for this method
            color = colors[i % len(colors)]
            pattern = patterns[i % len(patterns)] if patterns else ''
            
            if show_error_bars:
                plt.bar(x + offset, values, width,
                       label=method_name,
                       color=color,
                       alpha=0.8,
                       hatch=pattern,
                       edgecolor='white',
                       linewidth=0.5,
                       yerr=errors,
                       capsize=3,
                       error_kw={'elinewidth': 1, 'ecolor': 'black'})
            else:
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
        
        plt.ylim(0, ylim)

        # Add legend
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15),
                  frameon=True, fancybox=True, shadow=True,
                  fontsize=10, framealpha=0.9, ncol=min(len(method_names), 4))
        
        if title:
            plt.title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(output_path / "comparison_chart.png", dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
    
    def _detect_evaluator_type(self, result: EvalResult) -> str:
        """Detect evaluator type from result metrics."""
        if "average_similarity" in result.overall_metrics:
            return "diversity"
        elif "fluency" in result.overall_metrics:
            return "ttct"
        elif "average_creativity_index" in result.overall_metrics:
            return "creativity_index"
        elif "mean_token_length" in result.overall_metrics:
            return "length"
        elif "average_ngram_diversity" in result.overall_metrics:
            return "ngram"
        elif "response_distribution" in result.overall_metrics:
            return "response_count"
        elif "accuracy_given_attempted" in result.overall_metrics:
            return "factuality"
        else:
            return "generic"
    
    def _create_standard_evaluator_plots(self, evaluator_type: str, comparison_data: List[ComparisonData], output_dir: Path):
        """Create standard evaluator plots using evaluator class information."""
        try:
            from ..analysis.evals import get_evaluator
            evaluator_class = get_evaluator(evaluator_type)
            
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
        except Exception:
            # Fallback to generic plotting
            generic_plotter = EvaluatorPlotterFactory.create_generic_plotter(self)
            generic_plotter.create_plots(comparison_data, output_dir)
    
    def _extract_aggregate_data(self, comparison_data: List[ComparisonData], 
                              metric_names: List[str], error_bar_type: str):
        """Extract aggregate data and errors for plotting."""
        metric_values = {metric: [] for metric in metric_names}
        metric_errors = {metric: [] for metric in metric_names}
        
        for data in comparison_data:
            for metric in metric_names:
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
                
                if (metric.startswith("avg_")) and (metric.replace("avg_", "std_") in data.result.overall_metrics):
                    error_key = metric.replace('avg_', 'std_')
                    error = data.result.overall_metrics[error_key]
                    metric_errors[metric].append(error)
                else:
                    metric_errors[metric].append(0.0)
                        
        return metric_values, metric_errors
    
    def _create_single_metric_bar_plot(self, ax, metric_name, format_names, comparison_data,
                                     values, errors, colors, patterns, show_error_bars):
        """Create bar plot for a single metric in a subplot."""
        x = np.arange(len(format_names))
        width = 0.6
        
        bars = []
        for i, (format_name, data) in enumerate(zip(format_names, comparison_data)):
            color = data.color or colors[i % len(colors)]
            pattern = patterns[i % len(patterns)] if patterns else ''
            
            if show_error_bars:
                bar = ax.bar(x[i], values[i], width,
                           label=format_name,
                           color=color,
                           alpha=0.8,
                           hatch=pattern,
                           edgecolor='white',
                           linewidth=0.5,
                           yerr=errors[i],
                           capsize=3,
                           error_kw={'elinewidth': 1, 'ecolor': 'black'})
            else:
                bar = ax.bar(x[i], values[i], width,
                           label=format_name,
                           color=color,
                           alpha=0.8,
                           hatch=pattern,
                           edgecolor='white',
                           linewidth=0.5)
            bars.append(bar)
        
        ax.set_xlabel('Methods', fontsize=10, fontweight='bold')
        ax.set_ylabel('Score', fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(format_names, rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Set y-axis limits based on data range
        if show_error_bars and errors:
            max_val = max(v + e for v, e in zip(values, errors))
            min_val = min(max(0, v - e) for v, e in zip(values, errors))
        else:
            max_val = max(values) if values else 1
            min_val = min(values) if values else 0
        
        y_range = max_val - min_val
        ax.set_ylim(min_val - y_range * 0.05, max_val + y_range * 0.1)

    def _create_single_metric_line_plot(self, ax, metric_name, format_names,
                                      values, errors, colors, show_error_bars):
        """Create line plot for a single metric in a subplot."""
        line_styles = ['-', '--', '-.', ':', '-']
        markers = ['o', 's', '^', 'D', 'v']
        
        x = np.arange(len(format_names))
        
        for i, format_name in enumerate(format_names):
            color = colors[i % len(colors)]
            linestyle = line_styles[i % len(line_styles)]
            marker = markers[i % len(markers)]
            
            if show_error_bars:
                ax.errorbar(x[i], values[i], yerr=errors[i],
                          marker=marker, label=format_name,
                          color=color, linewidth=2.5, markersize=8,
                          linestyle=linestyle, markerfacecolor='white',
                          markeredgewidth=2, markeredgecolor=color,
                          capsize=5, elinewidth=1)
            else:
                ax.plot(x[i], values[i],
                       marker=marker, label=format_name,
                       color=color, linewidth=2.5, markersize=8,
                       linestyle=linestyle, markerfacecolor='white',
                       markeredgewidth=2, markeredgecolor=color)
        
        ax.set_xlabel('Methods', fontsize=10, fontweight='bold')
        ax.set_ylabel('Score', fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(format_names, rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Set y-axis limits based on data range
        if show_error_bars and errors:
            max_val = max(v + e for v, e in zip(values, errors))
            min_val = min(max(0, v - e) for v, e in zip(values, errors))
        else:
            max_val = max(values) if values else 1
            min_val = min(values) if values else 0
        
        y_range = max_val - min_val
        ax.set_ylim(min_val - y_range * 0.05, max_val + y_range * 0.1)

    # Backward compatibility methods
    def compare_instance_metrics(self, *args, **kwargs):
        """Backward compatibility method - now uses compare_distributions."""
        return self.compare_distributions(*args, **kwargs) 