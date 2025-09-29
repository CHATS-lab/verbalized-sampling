# OSS Conversion Plan for Verbalized Sampling

## Todo List

### Immediate Foundation
- [ ] Add LICENSE file to repository (MIT license already declared in pyproject.toml)
- [ ] Clean up repository structure - move/organize loose files and directories
- [ ] Set up pre-commit hooks for code formatting (black, isort, ruff, mypy)
- [ ] Add comprehensive test suite and improve pytest configuration

### Core Functionality
- [ ] Implement Persuasion Simulation task (mentioned as missing)
- [ ] Update and standardize LLM interfaces and evaluation code
- [ ] Create unified storage solution for generated responses (move from scattered directories)
- [ ] Set up external storage integration to reduce repository size
- [ ] Enhance CLI functionality and ensure all tasks are runnable via bash scripts

### Release Infrastructure
- [ ] Add GitHub Actions CI/CD pipeline (test, lint, build, publish)
- [ ] Create comprehensive documentation structure (docs/ with Sphinx)
- [ ] Prepare PyPI publishing workflow and verify package installation

### OSS Community Standards
- [ ] Add CONTRIBUTING.md and CODE_OF_CONDUCT.md for OSS guidelines
- [ ] Create example notebooks and tutorial documentation
- [ ] Review and update README.md for final OSS release
- [ ] Security audit - remove any hardcoded secrets or sensitive data
- [ ] Add version management and release automation

## Current Repository Analysis

### Strengths
-  Well-configured pyproject.toml with proper dependencies
-  Functional CLI interface using Typer
-  Comprehensive README with examples
-  MIT license declared
-  Multiple task implementations (creativity, math, safety)
-  Multiple sampling methods implemented
-  Evaluation framework with multiple metrics

### Areas Needing Attention
- L No LICENSE file (only declared in pyproject.toml)
- L Repository cluttered with experiment files and loose directories
- L No pre-commit hooks or CI/CD
- L Limited test coverage
- L Generated responses scattered across multiple directories
- L Missing OSS community standards files
- L No proper documentation structure (just README)

### Repository Structure Issues
The following files/directories should be organized:
- Loose experiment files: `book_diversity_tuning_comparison.pdf`, `joke_diversity_tuning_comparison.pdf`, etc.
- Generated data directories: `generated_data/`, `synthetic_*/`, `method_results_*/`, etc.
- Analysis scripts: `extract_joke_results.py`, `plot_max_diversity.py`, etc.
- Temporary files: `temp_combine_script.py`, `test.py`

### Missing Features
- Persuasion Simulation task (mentioned in original requirements)
- Unified storage solution for generated responses
- External storage integration to reduce repo size