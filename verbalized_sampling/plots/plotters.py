from typing import List
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class BasePlotter(ABC):
    """Base class for all plot types."""
    
    def __init__(self, colors: List[str]):
        self.colors = colors
    
    @abstractmethod
    def plot(self, all_data: List[List[float]], labels: List[str], colors: List[str], **kwargs):
        """Create the plot."""
        pass

class HistogramPlotter(BasePlotter):
    """Creates histogram plots."""
    
    def plot(self, all_data: List[List[float]], labels: List[str], colors: List[str], 
             bins: int = 100, alpha: float = 0.7, **kwargs):
        plt.hist(all_data, bins=bins, alpha=alpha, label=labels, color=colors)
        plt.ylabel('Frequency')

class ViolinPlotter(BasePlotter):
    """Creates violin plots."""
    
    def plot(self, all_data: List[List[float]], labels: List[str], colors: List[str], **kwargs):
        # Prepare data for seaborn
        df_data = []
        for i, (data_values, label) in enumerate(zip(all_data, labels)):
            for value in data_values:
                df_data.append({"Value": value, "Method": label})
        
        df = pd.DataFrame(df_data)
        palette = {label: color for label, color in zip(labels, colors)}
        
        # Create violin plot
        ax = sns.violinplot(
            data=df,
            x="Value", 
            y="Method",
            order=labels,
            palette=palette,
            scale="width",
            cut=0,
            alpha=0.6
        )
        
        # Add quartile lines and median points
        for i, label in enumerate(labels):
            vals = df[df["Method"] == label]["Value"].values
            if len(vals) > 0:
                q1, median, q3 = np.percentile(vals, [25, 50, 75])
                ax.hlines(y=i, xmin=q1, xmax=q3, color="black", linewidth=6, zorder=3)
                ax.scatter(median, i, color="white", edgecolor="black", s=50, zorder=4)

class KDEPlotter(BasePlotter):
    """Creates KDE plots."""
    
    def plot(self, all_data: List[List[float]], labels: List[str], colors: List[str], **kwargs):
        for data_values, label, color in zip(all_data, labels, colors):
            if len(data_values) > 1:  # Need at least 2 points for KDE
                sns.kdeplot(data_values, label=label, color=color)
        plt.ylabel('Density')

class BoxPlotter(BasePlotter):
    """Creates box plots."""
    
    def plot(self, all_data: List[List[float]], labels: List[str], colors: List[str], **kwargs):
        box_plot = plt.boxplot(all_data, labels=labels, patch_artist=True, vert=False)
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Make the median lines more bold
        for median in box_plot['medians']:
            median.set_linewidth(3)
            median.set_color('black')
        
        plt.xlabel('Value')

class SNSHistogramPlotter(BasePlotter):
    """Creates seaborn histogram plots."""
    
    def plot(self, all_data: List[List[float]], labels: List[str], colors: List[str], 
             bins: int = 100, alpha: float = 0.7, **kwargs):
        sns.histplot(all_data, alpha=alpha, color=colors, kde=True)
        plt.ylabel('Frequency') 