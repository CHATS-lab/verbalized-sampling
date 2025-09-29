print-%: ; @echo $* = $($*)
SHELL=/bin/bash
PROJECT_NAME     = verbalized-sampling
COPYRIGHT        = "CHATS-Lab. All Rights Reserved."
PROJECT_PATH     = verbalized_sampling
SOURCE_FOLDERS   = $(PROJECT_PATH) tests examples
LINT_PATHS       = ${PROJECT_PATH} tests examples

check_install = python3 -c "import $(1)" || pip3 install $(1) --upgrade
check_install_extra = python3 -c "import $(1)" || pip3 install $(2) --upgrade

.PHONY: help install install-dev test lint format type-check clean build publish

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

go-install:  ## Install Go (required for addlicense)
	# requires go >= 1.16
	command -v go || (sudo apt-get install -y golang && sudo ln -sf /usr/lib/go/bin/go /usr/bin/go)

addlicense-install: go-install  ## Install addlicense tool
	command -v addlicense || go install github.com/google/addlicense@latest

install:  ## Install package
	pip install -e .

install-dev:  ## Install package with development dependencies
	pip install -e ".[dev]"

test:  ## Run tests
	$(call check_install, pytest)
	pytest -s

test-verbose:  ## Run tests with verbose output
	$(call check_install, pytest)
	pytest -v

test-cov:  ## Run tests with coverage
	$(call check_install, pytest)
	$(call check_install_extra, pytest_cov, pytest-cov)
	pytest --cov=verbalized_sampling --cov-report=html --cov-report=term

lint:  ## Run linting
	$(call check_install, isort)
	$(call check_install, ruff)
	isort --check --diff --project=${PROJECT_PATH} ${LINT_PATHS}
	ruff check .

lint-fix:  ## Run linting with auto-fix
	$(call check_install, ruff)
	ruff check . --fix

format:  ## Format code
	$(call check_install, autoflake)
	autoflake --remove-all-unused-imports -i -r ${LINT_PATHS}
	$(call check_install, black)
	black ${LINT_PATHS}
	$(call check_install, isort)
	isort ${LINT_PATHS}

format-check:  ## Check code formatting
	$(call check_install, black)
	$(call check_install, isort)
	black --check ${LINT_PATHS}
	isort --check-only ${LINT_PATHS}

type-check:  ## Run type checking
	$(call check_install, mypy)
	mypy .

check-docstyle:  ## Check docstring style
	$(call check_install, pydocstyle)
	pydocstyle ${PROJECT_PATH} --convention=google

checks: lint check-docstyle type-check  ## Run all code quality checks

check-all: format-check lint type-check test  ## Run all checks including tests

clean:  ## Clean build artifacts
	@-rm -rf build/ dist/ .eggs/ site/ *.egg-info .pytest_cache .mypy_cache .ruff_cache
	@-find . -name '*.pyc' -type f -exec rm -rf {} +
	@-find . -name '__pycache__' -exec rm -rf {} +

build: clean  ## Build package
	$(call check_install_extra, build, build)
	python -m build

package: clean  ## Build package (alias for build)
	PRODUCTION_MODE=yes python -m build

publish-test:  ## Publish to test PyPI
	$(call check_install, twine)
	twine upload --repository testpypi dist/*

publish:  ## Publish to PyPI
	$(call check_install, twine)
	twine upload dist/*

setup-dev:  ## Setup development environment
	pip install -e ".[dev]"
	$(call check_install, pre-commit)
	pre-commit install

addlicense: addlicense-install  ## Add license headers to source files
	$(command -v addlicense || echo $(HOME)/go/bin/addlicense) -c $(COPYRIGHT) -ignore tests/coverage.xml -l apache $(SOURCE_FOLDERS)

check-license: addlicense-install  ## Check license headers in source files
	$(command -v addlicense || echo $(HOME)/go/bin/addlicense) -c $(COPYRIGHT) -ignore tests/coverage.xml -l apache -check $(SOURCE_FOLDERS)

# Development workflow
dev-setup: install-dev setup-dev  ## Complete development setup

dev-check: format lint type-check test  ## Run all development checks

# Release workflow
release-check: clean build test  ## Prepare for release

# Quick commands
quick-test:  ## Quick test run (no coverage)
	$(call check_install, pytest)
	pytest -x

quick-check: format-check lint  ## Quick format and lint check