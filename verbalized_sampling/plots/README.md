# Verbalized Sampling - Plots Module

This module provides a comprehensive plotting framework for comparing evaluation results across different sampling methods. The module is designed with modularity and extensibility in mind.

## Overview

The plots module is organized into several components:

- **Base classes**: Core data structures and utilities
- **Plotters**: Individual plot type implementations
- **Evaluator-specific plotters**: Specialized plotting for different evaluator types
- **Factories**: Creation patterns for easy extensibility
- **Comparison tools**: Main plotting orchestration
- **Convenience functions**: Easy-to-use high-level interfaces

## Module Structure
```
verbalized_sampling/plots/
├── init.py # Public API exports
├── base.py # Base classes and data structures
├── plotters.py # Individual plot type implementations
├── evaluator_specific.py # Evaluator-specific plotting logic
├── factory.py # Factory classes for creating plotters
├── comparison.py # Main comparison plotting class
├── convenience.py # High-level convenience functions
└── README.md # This file
```