from typing import List, Optional
from .plotters import (
    BasePlotter, HistogramPlotter, ViolinPlotter, 
    KDEPlotter, BoxPlotter, SNSHistogramPlotter
)
from .evaluator_specific import (
    EvaluatorSpecificPlotter, ResponseCountPlotter, 
    FactualityPlotter, GenericPlotter
)

class PlotterFactory:
    """Factory for creating different plot types."""
    
    PLOTTERS = {
        "histogram": HistogramPlotter,
        "sns_histogram": SNSHistogramPlotter,
        "violin": ViolinPlotter,
        "kde": KDEPlotter,
        "box": BoxPlotter
    }
    
    @classmethod
    def create_plotter(cls, plot_type: str, colors: List[str]) -> BasePlotter:
        if plot_type not in cls.PLOTTERS:
            raise ValueError(f"Unknown plot_type: {plot_type}. Available: {list(cls.PLOTTERS.keys())}")
        return cls.PLOTTERS[plot_type](colors)
    
    @classmethod
    def get_available_plot_types(cls) -> List[str]:
        return list(cls.PLOTTERS.keys())

class EvaluatorPlotterFactory:
    """Factory for creating evaluator-specific plotters."""
    
    PLOTTERS = {
        "response_count": ResponseCountPlotter,
        "factuality": FactualityPlotter
    }
    
    @classmethod
    def create_plotter(cls, evaluator_type: str, **kwargs) -> Optional[EvaluatorSpecificPlotter]:
        if evaluator_type in cls.PLOTTERS:
            return cls.PLOTTERS[evaluator_type](**kwargs)
        return None
    
    @classmethod
    def create_generic_plotter(cls, comparison_plotter) -> GenericPlotter:
        return GenericPlotter(comparison_plotter)
    
    @classmethod
    def get_available_evaluator_types(cls) -> List[str]:
        return list(cls.PLOTTERS.keys()) 