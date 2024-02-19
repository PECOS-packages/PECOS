.DEFAULT_GOAL := help

# Minimum Python version required for the project
MIN_PYTHON_VERSION := 3.10
# Automatically select the appropriate Python interpreter.
BASEPYTHON := $(shell which python3 2>/dev/null || which python 2>/dev/null)
# Specifies the shell to use for executing commands.
SHELL=/bin/bash
# Directory for the Python virtual environment.
VENV=.venv

# Adjust the virtual environment binaries path based on the operating system.
ifeq ($(OS),Windows_NT)
	VENV_BIN=$(VENV)/Scripts
else
	VENV_BIN=$(VENV)/bin
endif

.PHONY: venv
venv: ## Create a Python virtual environment in the .venv directory
	$(BASEPYTHON) -m venv $(VENV)

.PHONY: requirements
requirements: upgrade-pip  ## Install main, documentation, and development Python project requirements.
	$(VENV_BIN)/pip install -U build pip-tools pre-commit wheel Cython typos
	$(VENV_BIN)/pip install --upgrade -r python/requirements.txt
	$(VENV_BIN)/pip install --upgrade -r python/docs/requirements.txt

.PHONY: build
build: venv requirements  ## Compiles the Cython and Python packages for development.
	@unset CONDA_PREFIX && source $(VENV_BIN)/activate \
	&& pip install -e ./cython \
	&& maturin develop -m python/Cargo.toml

.PHONY: build-allex
build-allex: venv requirements  ## Compiles the Cython and Python packages for development with the "all" extra requirements.
	@unset CONDA_PREFIX && source $(VENV_BIN)/activate \
	&& pip install -e ./cython \
	&& maturin develop -E=all -m python/Cargo.toml

.PHONY: clippy
clippy:  ## Execute the Rust linter, clippy, across all project targets with all features enabled.
	cargo clippy --workspace --all-targets --all-features --locked -- -D warnings -D clippy::dbg_macro

.PHONY: clippy-default
clippy-default: ## Run clippy with default features for a quicker analysis.
	cargo clippy --all-targets --locked -- -D warnings -D clippy::dbg_macro

.PHONY: pre-commit
pre-commit: clippy clippy-default ## Run formatting and linting tools on the Python and Rust codebase.
	$(VENV_BIN)/pre-commit run --all-files
	cargo fmt --all
	$(VENV_BIN)/typos

.PHONY: clean
clean: ## Removes directories and files related to the build process, ensuring a clean state.
	# Remove the Python virtual environment and Rust target directory to clean the project workspace.
	@rm -rf .venv/
	@rm -rf target/
    # Remove the Cargo lock file and clean the Rust project to ensure a fresh start on the next build.
	@rm -f Cargo.lock
	@cargo clean
    # Call the clean target of the Makefile in the python/ and cython/ directories
	@$(MAKE) -s -C python/ $@
	@$(MAKE) -s -C cython/ $@

.PHONY: upgrade-pip
upgrade-pip: ## Update the pip version in the virtual environment.
	$(VENV_BIN)/python -m pip install --upgrade pip

help:  ## Show this help menu
	@echo "Usage: make [TARGET]..."
	@echo "Targets:"
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
