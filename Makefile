.DEFAULT_GOAL := help

# Try to autodetect if python3 or python is the python executable used.
PYTHONPATH := $(shell which python 2>/dev/null || which python3 2>/dev/null)

SHELL=bash

# Get absolute path of venv directory
BASE_DIR := $(shell realpath .)
VENV=$(BASE_DIR)/.venv
ifeq ($(OS),Windows_NT)
    VENV_BIN := $(VENV)/Scripts
else
    VENV_BIN := $(VENV)/bin
endif

# Define the path to main Python pyproject.toml
PYPROJECT_PATH := python/quantum-pecos/pyproject.toml

# Virtual environment
# -------------------

.PHONY: venv
venv:  ## Build Python virtual environment
	$(PYTHONPATH) -m venv $(VENV)

# Requirements
# ------------
.PHONY: requirements
requirements: metadeps installreqs  ## Install/refresh Python project requirements including those needed for development

.PHONY: updatereqs
updatereqs:  ## Generate/update lockfiles for both packages
	@echo "Upgrading uv..."
	$(VENV_BIN)/python -m pip install --upgrade uv
	@echo "Generating lockfiles..."
	cd python/pecos-rslib && $(VENV_BIN)/uv lock
	cd python/quantum-pecos && $(VENV_BIN)/uv lock

.PHONY: installreqs
installreqs: ## Install Python project requirements from lockfiles
	@unset CONDA_PREFIX \
	&& $(VENV_BIN)/python -m pip install --upgrade uv
	@echo "Installing from lockfiles..."
	cd python/pecos-rslib && $(VENV_BIN)/uv sync
	cd python/quantum-pecos && $(VENV_BIN)/uv sync

.PHONY: metadeps
metadeps: ## Install uv and maturin for building
	$(VENV_BIN)/python -m pip install --upgrade uv maturin

# Function to extract version from pyproject.toml
define get_version
$(shell grep -m 1 'version\s*=' $(PYPROJECT_PATH) | sed 's/.*version\s*=\s*"\(.*\)".*/\1/')
endef

# Building development environments
# ---------------------------------
.PHONY: build
build: requirements ## Compile and install for development
	@unset CONDA_PREFIX && cd python/pecos-rslib/ && $(VENV_BIN)/maturin develop
	@unset CONDA_PREFIX && cd python/quantum-pecos && $(VENV_BIN)/uv pip install -e .[all]

.PHONY: build-basic
build-basic: requirements ## Compile and install for development but do not include install extras
	@unset CONDA_PREFIX && cd python/pecos-rslib/ && $(VENV_BIN)/maturin develop
	@unset CONDA_PREFIX && cd python/quantum-pecos && $(VENV_BIN)/uv pip install -e .

.PHONY: build-release
build-release: requirements ## Build a faster version of binaries
	@unset CONDA_PREFIX && cd python/pecos-rslib/ && $(VENV_BIN)/maturin develop --release
	@unset CONDA_PREFIX && cd python/quantum-pecos && $(VENV_BIN)/uv pip install -e .[all]

.PHONY: build-native
build-native: requirements ## Build a faster version of binaries with native CPU optimization
	@unset CONDA_PREFIX && cd python/pecos-rslib/ && RUSTFLAGS='-C target-cpu=native' \
	&& $(VENV_BIN)/maturin develop --release
	@unset CONDA_PREFIX && cd python/quantum-pecos && $(VENV_BIN)/uv pip install -e .[all]

# Documentation
# -------------

# .PHONY: docs
# docs:  ## Generate documentation
# 	#TODO: ...

# Linting / formatting
# --------------------

.PHONY: check
check:  ## Run cargo check with all features
	cargo check --workspace --all-targets --all-features

.PHONY: clippy
clippy:  ## Run cargo clippy with all features
	cargo clippy --workspace --all-targets --all-features -- -D warnings

.PHONY: fmt
fmt: ## Run autoformatting for cargo
	cargo fmt --all -- --check

.PHONY: lint  ## Run all quality checks / linting / reformatting
lint: fmt clippy
	$(VENV_BIN)/pre-commit run --all-files

# Testing
# -------

.PHONY: rstest
rstest:  ## Run Rust tests
	cargo test

.PHONY: pytest
pytest:  ## Run tests on the Python package (not including optional dependencies). ASSUMES: previous build command
	@cd python/tests/ && $(VENV_BIN)/pytest . -m "not optional_dependency"

.PHONY: pytest-dep
pytest-dep: ## Run tests on the Python package only for optional dependencies. ASSUMES: previous build command
	@cd python/ && $(VENV_BIN)/pytest tests -m optional_dependency

# .PHONY: pytest-doc
# pydoctest:  ## Run doctests with pytest. ASSUMES: A build command was ran previously. ASSUMES: previous build command
# 	# TODO: update and install docs requirements
# 	@cd python/ && $(VENV_BIN)/pytest docs --doctest-glob=*.rst --doctest-continue-on-failure

.PHONY: test
test: rstest pytest pytest-dep ## Run all tests. ASSUMES: previous build command

# Utility
# -------

.PHONY: clean
clean:  ## Clean up caches and build artifacts
	@rm -rf *.egg-info
	@rm -rf dist
	@rm -rf **/build/
	@rm -rf python/docs/_build
	@rm -rf **/.pytest_cache/
	@rm -rf **/.ipynb_checkpoints
	@rm -rf .ruff_cache/
	@rm -rf **/.hypothesis/
	@rm -rf **/junit/
	@cargo clean

# Help
# ----

.PHONY: help
help:  ## Show the help menu
	@echo "Available make commands:"
	@echo ""
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
