.DEFAULT_GOAL := help

# Try to autodetect if python3 or python is the python executable used.
PYTHONPATH := $(shell which python 2>/dev/null || which python3 2>/dev/null)

SHELL=bash

# Virtual environment
# -------------------

.PHONY: venv
venv:  ## Create virtual environment and install all dependencies
	@echo "Installing uv..."
	$(PYTHONPATH) -m pip install --upgrade uv
	@echo "Creating venv and installing dependencies..."
	uv sync

# Requirements
# ------------

.PHONY: updatereqs
updatereqs:  ## Generate/update lockfiles for both packages
	@echo "Ensuring uv is installed..."
	uv self update
	@echo "Generating lock files..."
	uv lock

.PHONY: installreqs
installreqs: ## Install Python project requirements to root .venv
	@echo "Installing requirements..."
	@unset CONDA_PREFIX && uv sync

# Building development environments
# ---------------------------------
.PHONY: build
build: installreqs ## Compile and install for development
	@unset CONDA_PREFIX && cd python/pecos-rslib/ && uv run maturin develop --uv
	@unset CONDA_PREFIX && cd python/quantum-pecos && uv pip install -e .[all]

.PHONY: build-basic
build-basic: installreqs ## Compile and install for development but do not include install extras
	@unset CONDA_PREFIX && cd python/pecos-rslib/ && uv run maturin develop --uv
	@unset CONDA_PREFIX && cd python/quantum-pecos && uv pip install -e .

.PHONY: build-release
build-release: installreqs ## Build a faster version of binaries
	@unset CONDA_PREFIX && cd python/pecos-rslib/ && uv run maturin develop --uv --release
	@unset CONDA_PREFIX && cd python/quantum-pecos && uv pip install -e .[all]

.PHONY: build-native
build-native: installreqs ## Build a faster version of binaries with native CPU optimization
	@unset CONDA_PREFIX && cd python/pecos-rslib/ && RUSTFLAGS='-C target-cpu=native' \
	&& uv run maturin develop --uv --release
	@unset CONDA_PREFIX && cd python/quantum-pecos && uv pip install -e .[all]

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
	uv run pre-commit run --all-files

# Testing
# -------

.PHONY: rstest
rstest:  ## Run Rust tests
	cargo test

.PHONY: pytest
pytest:  ## Run tests on the Python package (not including optional dependencies). ASSUMES: previous build command
	uv run pytest ./python/tests/ -m "not optional_dependency"

.PHONY: pytest-dep
pytest-dep: ## Run tests on the Python package only for optional dependencies. ASSUMES: previous build command
	uv run pytest ./python/tests/ -m optional_dependency

.PHONY: pytest-all
pytest-all:  ## Run all tests on the Python package ASSUMES: previous build command
	uv run pytest ./python/tests/

# .PHONY: pytest-doc
# pydoctest:  ## Run doctests with pytest. ASSUMES: A build command was ran previously. ASSUMES: previous build command
# 	# TODO: update and install docs requirements
# 	uv run pytest docs --doctest-glob=*.rst --doctest-continue-on-failure

.PHONY: test
test: rstest pytest-all ## Run all tests. ASSUMES: previous build command

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
