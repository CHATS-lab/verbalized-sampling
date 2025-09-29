# OSS Conversion Plan for Verbalized Sampling

## Todo List

### Immediate Foundation
- [x] Add LICENSE file to repository (Apache-2.0 for CHATS-Lab)
- [x] Add Makefile for development workflow automation
- [x] Add setup.py for backward compatibility
- [x] Clean up repository structure - move/organize loose files and directories
- [x] Create new directory structure (scripts/, generated_data/, plots/, examples/, docs/)
- [x] Move all experiment files to organized locations
- [x] Update .gitignore for new structure
- [x] Remove temporary and duplicate files
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
- ✅ LICENSE file now implemented (Apache-2.0)
- ✅ Repository structure completely reorganized and clean
- ✅ Professional directory structure with organized file placement
- ❌ No pre-commit hooks or CI/CD
- ❌ Limited test coverage
- ❌ Missing OSS community standards files
- ❌ No proper documentation structure (just README)

### Repository Structure - COMPLETED ✅
**New Professional Structure:**
```
verbalized-sampling/
├── scripts/                    # All executable scripts
│   ├── analysis/              # Analysis and plotting code (includes latex/)
│   ├── data_processing/       # Data processing scripts + processing/
│   ├── experiments/           # Experiment runners and tests
│   └── tasks/                 # Existing task scripts
├── generated_data/            # All experiment results (organized)
├── plots/                     # Generated plots and figures
├── examples/                  # Usage examples (demo_safety_usage.py)
├── docs/                      # Ready for documentation
├── ablation_data/             # Ablation study results (kept)
├── analyse/                   # Analysis code (kept)
├── data/                      # Source datasets (kept)
└── verbalized_sampling/       # Main package (kept)
```

**Major Cleanup Accomplished:**
- ✅ 8 Python scripts → organized into scripts/ subdirectories
- ✅ 9 experiment directories → moved to generated_data/
- ✅ PNG files + latex_figures → organized in plots/
- ✅ processing/ → moved to scripts/data_processing/
- ✅ Removed temporary files (temp_combine_script.py, test.py, __init__.py)
- ✅ Updated .gitignore for new structure

### Missing Features
- Persuasion Simulation task (mentioned in original requirements)
- External storage integration to reduce repo size further