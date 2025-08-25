"""
Configuration module for rlpref project.

This module centralizes configuration settings for the experiment.
"""
from typing import Dict, Any, Optional, List, Literal
import os

class ExperimentConfig:
    """
    Central configuration class for experiment settings.
    
    Attributes:
        model_name: Name of the model to use for experiments
        dataset_name: Name of the dataset to use for experiments
        num_samples: Number of samples to use from the dataset
        random_seed: Random seed for reproducibility
        use_4bit: Whether to use 4-bit quantization
        results_dir: Directory to save results
        skip_analysis: If True, skips analysis and plotting
        token_filter_threshold: Optional threshold for filtering tokens by log probability
        output_format: Format for saving plots ('pdf' or 'png', default 'pdf')
        figure_size: Default size for matplotlib figures
        jitter_seed: Seed for jittering points in scatter plots
        model_cache: Whether to keep model loaded in memory after experiments
    """
    def __init__(
        self,
        model_name: str = "distilgpt2",
        dataset_name: str = "HuggingFaceH4/summarize-from-feedback",
        num_samples: int = 100,
        random_seed: int = 42,
        use_4bit: bool = False,
        results_dir: Optional[str] = None,
        skip_analysis: bool = False,
        token_filter_threshold: Optional[float] = None,
        output_format: Literal["pdf", "png"] = "pdf",
        figure_size: tuple = (12, 7),
        jitter_seed: int = 42,
        model_cache: bool = True
    ):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.num_samples = num_samples
        self.random_seed = random_seed
        self.use_4bit = use_4bit
        self.results_dir = results_dir
        self.skip_analysis = skip_analysis
        self.token_filter_threshold = token_filter_threshold
        self.output_format = output_format
        self.figure_size = figure_size
        self.jitter_seed = jitter_seed
        self.model_cache = model_cache
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        return {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "num_samples": self.num_samples,
            "random_seed": self.random_seed,
            "use_4bit": self.use_4bit,
            "results_dir": self.results_dir,
            "skip_analysis": self.skip_analysis,
            "token_filter_threshold": self.token_filter_threshold,
            "output_format": self.output_format,
            "figure_size": self.figure_size,
            "jitter_seed": self.jitter_seed,
            "model_cache": self.model_cache
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create a configuration from a dictionary."""
        # Only include keys that are actual parameters of the class
        valid_params = {k: v for k, v in config_dict.items() 
                        if k in cls.__init__.__annotations__}
        return cls(**valid_params)
    
    def update(self, **kwargs) -> 'ExperimentConfig':
        """Create a new config with updated parameters."""
        config_dict = self.to_dict()
        config_dict.update(**kwargs)
        return self.from_dict(config_dict)
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        return "\n".join([f"{k}: {v}" for k, v in self.to_dict().items()])


# Default configuration
DEFAULT_CONFIG = ExperimentConfig()

# Create configuration from command line args
def config_from_args(args) -> ExperimentConfig:
    """
    Create configuration from command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        ExperimentConfig instance
    """
    config_dict = {
        "model_name": getattr(args, "model", DEFAULT_CONFIG.model_name),
        "dataset_name": getattr(args, "dataset", DEFAULT_CONFIG.dataset_name),
        "num_samples": getattr(args, "samples", DEFAULT_CONFIG.num_samples),
        "random_seed": getattr(args, "seed", DEFAULT_CONFIG.random_seed),
        "use_4bit": getattr(args, "use_4bit", DEFAULT_CONFIG.use_4bit),
        "results_dir": getattr(args, "output", DEFAULT_CONFIG.results_dir),
        "skip_analysis": getattr(args, "skip_analysis", DEFAULT_CONFIG.skip_analysis),
        "token_filter_threshold": getattr(args, "token_filter", DEFAULT_CONFIG.token_filter_threshold),
        "model_cache": not getattr(args, "unload_model", not DEFAULT_CONFIG.model_cache),
    }
    return ExperimentConfig.from_dict(config_dict)

# Environment-specific settings
PLOT_SETTINGS = {
    "style": "seaborn-v0_8-whitegrid",
    "figure.figsize": (12, 7),
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "Bitstream Vera Sans", "sans-serif"],
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.title_fontsize": 12,
    "figure.titlesize": 16,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1
}

# Apply matplotlib settings
def setup_matplotlib():
    """Configure matplotlib with project settings."""
    try:
        import matplotlib as mpl
        for key, value in PLOT_SETTINGS.items():
            mpl.rcParams[key] = value
    except ImportError:
        print("Warning: Could not import matplotlib to apply settings")
    except Exception as e:
        print(f"Warning: Could not apply matplotlib settings: {e}")

# Dataset paths and settings
DATASET_CONFIG = {
    "name": "openai/summarize_from_feedback",
    "variants": ["axis", "comparisons"],
    "cache_dir": os.path.expanduser("~/.cache/rlpref"),
}