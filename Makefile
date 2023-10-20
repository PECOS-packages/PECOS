# A set of commands for development utilizing venv to develop, lint, test, document, and build the project.
# The goal is to concretely capture the development/build process for reproducibility of the development workflow.

.DEFAULT_GOAL := help

.PHONY: requirements updatereqs metadeps install install-all docs lint tests tests-dep doctests tests-all clean venv dev-all build build-full help upgrade-pip dev-setup

# Try to autodetect if python3 or python is the python executable used.
BASEPYTHON := $(shell which python3 2>/dev/null || which python 2>/dev/null)
VENV=.venv

ifeq ($(OS),Windows_NT)
	VENV_BIN=$(VENV)/Scripts
else
	VENV_BIN=$(VENV)/bin
endif

# Requirements
# ------------

requirements: upgrade-pip  ## Install/refresh Python project requirements
	$(VENV_BIN)/pip install --upgrade -r requirements.txt
	$(VENV_BIN)/pip install --upgrade -r docs/requirements.txt

updatereqs: upgrade-pip  ## Autogenerate requirements.txt
	$(VENV_BIN)/pip install -U pip-tools
	-@rm requirements.txt
	$(VENV_BIN)/pip-compile --extra=tests --no-annotate --no-emit-index-url --output-file=requirements.txt --strip-extras pyproject.toml

metadeps: upgrade-pip  ## Install extra dependencies used to develop/build this package
	$(VENV_BIN)/pip install -U build pip-tools pre-commit wheel pytest

# Installation
# ------------

install: upgrade-pip  ## Install PECOS
	$(VENV_BIN)/pip install .

install-all: upgrade-pip  ## Install PECOS with all optional dependencies
	$(VENV_BIN)/pip install .[all]

# Documentation
# -------------

docs: install  ## Generate documentation
	$(VENV_BIN)/pip install -r ./docs/requirements.txt
	$(MAKE) -C docs SPHINXBUILD=../$(VENV_BIN)/sphinx-build clean html

# Linting / formatting
# --------------------

lint: metadeps  ## Run all quality checks / linting / reformatting
	$(VENV_BIN)/pre-commit run --all-files

# Testing
# -------

tests: venv install metadeps  ## Run tests on the Python package (not including optional dependencies)
	$(VENV_BIN)/pytest tests -m "not optional_dependency"

tests-dep: venv install-all metadeps ## Run tests on the Python package only for optional dependencies
	$(VENV_BIN)/pytest tests -m optional_dependency

doctests:  ## Run doctests with pytest
	$(VENV_BIN)/pytest ./docs --doctest-glob=*.rst --doctest-continue-on-failure

tests-all: tests tests-dep doctests ## Run all tests

# Building / Developing
# ---------------------

clean:  ## Clean up caches and build artifacts
	-rm -rf *.egg-info dist build docs/_build .pytest_cache/ .ruff_cache/

venv:  ## Build a new Python virtual environment from scratch
	-rm -rf .venv/
	$(BASEPYTHON) -m venv $(VENV)

dev-all: dev-setup  ## Create a development environment from scratch with all optional dependencies and PECOS installed in editable mode
	$(VENV_BIN)/pip install -e .[all]

build: dev-setup  ## Clean, create new environment, and build PECOS for pypi
	$(VENV_BIN)/python -m build --sdist --wheel -n

build-full: dev-setup updatereqs install-all docs lint tests-all  ## Go through the full linting, testing, and building process
	$(VENV_BIN)/python -m build --sdist --wheel -n

# Help
# ----

help:  ## Show the help menu
	@echo "Available make commands:"
	@echo ""
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# Utility targets
# ---------------

upgrade-pip:
	$(VENV_BIN)/python -m pip install --upgrade pip

dev-setup: clean venv requirements metadeps
