# A set of commands for development utilizing venv to develop, lint, test, document, and build the project.

.PHONY: requirements updatereqs metadeps install install-all docs lint tests doctest doctest2 clean venv dev-all build build-full

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

requirements:  ## Install/refresh Python project requirements
	$(VENV_BIN)/python -m pip install --upgrade pip
	$(VENV_BIN)/pip install --upgrade -r requirements.txt
	$(VENV_BIN)/pip install --upgrade -r docs/requirements.txt

updatereqs:  ## Autogenerate requirements
	$(VENV_BIN)/python -m pip install --upgrade pip
	$(VENV_BIN)/pip install -U pip-tools
	rm requirements.txt
	$(VENV_BIN)/pip-compile --extra=tests --no-annotate --no-emit-index-url --output-file=requirements.txt --strip-extras pyproject.toml

metadeps:  ## Install packages used to develop/build this package
	$(VENV_BIN)/python -m pip install --upgrade pip
	$(VENV_BIN)/pip install -U build pip-tools pre-commit wheel

# Installation
# ------------

install:  ## Install PECOS
	$(VENV_BIN)/python -m pip install --upgrade pip
	$(VENV_BIN)/pip install .

install-all:  ## Install PECOS with all optional dependencies
	$(VENV_BIN)/python -m pip install --upgrade pip
	$(VENV_BIN)/pip install .[all]

# Documentation
# -------------

docs: install  ## Generate documentation
	$(VENV_BIN)/python -m pip install --upgrade pip
	$(VENV_BIN)/pip install -r ./docs/requirements.txt
	cd docs && make clean && make html && cd -  # <<< Will run using the base env... change that make to use .venv?
# TODO: Maybe call sphinx-build directly...

# Linting / formatting
# --------------------

lint: metadeps  ## Run all quality checks / linting
	$(VENV_BIN)/pre-commit run --all-files

# Testing
# -------

tests: install  ## Run tests
	$(VENV_BIN)/pytest tests

doctest:  ## Run doctests
	$(VENV_BIN)/sphinx-build -b doctest ./docs ./docs/_build

doctest2:  ## Run doctests using pytest
	$(VENV_BIN)/pytest ./docs --doctest-glob=*.rst # --doctest-module

# Building / Developing
# ---------------------

clean:  ## Clean up caches and build artifacts
	rm -rf *.egg-info dist build docs/_build .pytest_cache/ .ruff_cache/

venv:  ## Build a new Python virtual environment from scratch
	rm -rf .venv/
	$(BASEPYTHON) -m venv $(VENV)

dev-all: clean venv requirements metadeps  ## Create a development environment from scratch
	$(VENV_BIN)/python -m pip install --upgrade pip
	$(VENV_BIN)/pip install -e .[all]

build: clean venv requirements metadeps  ## Clean, create new environment, and build PECOS for pypi
	$(VENV_BIN)/python -m build --sdist --wheel -n

build-full: clean venv requirements metadeps updatereqs install-all docs lint tests doctest  ## Go through the full linting, testing, and building process
	$(VENV_BIN)/python -m build --sdist --wheel -n
