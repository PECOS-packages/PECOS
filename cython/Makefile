# A set of commands for development utilizing venv to develop, lint, test, document, and build the project.
# The goal is to concretely capture the development/build process for reproducibility of the development workflow.

.DEFAULT_GOAL := help

# Try to autodetect if python3 or python is the python executable used.
BASEPYTHON := $(shell which python3 2>/dev/null || which python 2>/dev/null)
VENV=../.venv

ifeq ($(OS),Windows_NT)
	VENV_BIN=$(VENV)/Scripts
else
	VENV_BIN=$(VENV)/bin
endif

# Requirements
# ------------

.PHONY: requirements
requirements: upgrade-pip  ## Install/refresh Python project requirements
	$(VENV_BIN)/pip install --upgrade -r requirements.txt
	$(VENV_BIN)/pip install --upgrade -r docs/requirements.txt

.PHONY: updatereqs
updatereqs: upgrade-pip  ## Autogenerate requirements.txt
	$(VENV_BIN)/pip install -U pip-tools
	-@rm requirements.txt
	$(VENV_BIN)/pip-compile --extra=tests --no-annotate --no-emit-index-url --output-file=requirements.txt --strip-extras pyproject.toml

.PHONY: metadeps
metadeps: upgrade-pip  ## Install extra dependencies used to develop/build this package
	$(VENV_BIN)/pip install -U build pip-tools pre-commit wheel pytest

# Installation
# ------------

.PHONY: upgrade-pip
install: upgrade-pip  ## Install PECOS
	$(VENV_BIN)/pip install .

.PHONY: install-all
install-all: upgrade-pip  ## Install PECOS with all optional dependencies
	$(VENV_BIN)/pip install .[all]

# Documentation
# -------------

.PHONY: docs
docs: install-all  ## Generate documentation
	$(VENV_BIN)/pip install -r ./docs/requirements.txt
	$(MAKE) -C docs SPHINXBUILD=../$(VENV_BIN)/sphinx-build clean html

# Linting / formatting
# --------------------

.PHONY: lint
lint: metadeps  ## Run all quality checks / linting / reformatting
	$(VENV_BIN)/pre-commit run --all-files

# Testing
# -------

.PHONY: tests
tests: install metadeps  ## Run tests on the Python package (not including optional dependencies)
	$(VENV_BIN)/pytest tests -m "not optional_dependency"

.PHONY: tests-dep
tests-dep: install-all metadeps ## Run tests on the Python package only for optional dependencies
	$(VENV_BIN)/pytest tests -m optional_dependency

.PHONY: doctests
doctests:  ## Run doctests with pytest
	$(VENV_BIN)/pytest ./docs --doctest-glob=*.rst --doctest-continue-on-failure

.PHONY: tests-all
tests-all: tests tests-dep doctests ## Run all tests

# Building / Developing
# ---------------------

.PHONY: clean
clean:  ## Clean up caches, build artifacts, and generated library files
	-rm -rf *.egg-info dist build docs/_build .pytest_cache/ .ruff_cache/ .venv/
	@find . \( -type d -name "__pycache__" \
	          -o -type f -name "*.pyc" \
	          -o -type f -name "*.so" \
	          -o -type f -name "*.dll" \) \
	          -print0 | xargs -0 rm -rf


.PHONY: venv
venv:  ## Build a new Python virtual environment from scratch
	-rm -rf .venv/
	$(BASEPYTHON) -m venv $(VENV)

.PHONY: dev-all
dev-all: dev-setup  ## Create a development environment from scratch with all optional dependencies and PECOS installed in editable mode
	$(VENV_BIN)/pip install -e .[all]

.PHONY: build
build: dev-setup  ## Clean, create new environment, and build PECOS for pypi
	$(VENV_BIN)/python -m build --sdist --wheel -n

.PHONY: build-full
build-full: dev-setup updatereqs install-all docs lint tests-all  ## Go through the full linting, testing, and building process
	$(VENV_BIN)/python -m build --sdist --wheel -n

# Help
# ----

.PHONY: help
help:  ## Show the help menu
	@echo "Available make commands:"
	@echo ""
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# Utility targets
# ---------------

.PHONY: upgrade-pip
upgrade-pip:
	$(VENV_BIN)/python -m pip install --upgrade pip

.PHONY: dev-setup
dev-setup: clean venv requirements metadeps
