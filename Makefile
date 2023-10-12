# A set of commands for development utilizing venv to develop, lint, test, document, and build the project.

.PHONY: venv update-pip requirements install install-all dev dev-all tests lint docs doctest doctest2 clean build metadeps updatereqs

# Try to autodetect if python3 or python is the python executable used.
BASEPYTHON := $(shell which python3 2>/dev/null || which python 2>/dev/null)
VENV=.venv

ifeq ($(OS),Windows_NT)
	VENV_BIN=$(VENV)/Scripts
else
	VENV_BIN=$(VENV)/bin
endif

PYTHON := $(VENV_BIN)/python
PIP := $(VENV_BIN)/pip

venv:  ## Set up a new Python virtual environment and install requirements
	rm -rf .venv/
	$(BASEPYTHON) -m venv $(VENV)
	$(MAKE) requirements

update-pip:  ## Update to latest pip
	$(PYTHON) -m pip install --upgrade pip

requirements: update-pip  ## Install/refresh Python project requirements
	$(PIP) install --upgrade -r requirements.txt
	$(PIP) install --upgrade -r docs/requirements.txt

install: update-pip  ## Install PECOS
	$(PIP) install .

install-all: update-pip  ## Install PECOS with all optional dependencies
	$(PIP) install .[all]

dev: update-pip  ## Install PECOS in editing mode for development
	$(PIP) install -e .

dev-all: update-pip  ## Install PECOS in editing mode for development with all optional dependencies
	$(PIP) install -e .[all]

uninstall:  ## Uninstall PECOS
	$(PIP) uninstall quantum-pecos

lint: metadeps  ## Run all quality checks / linting
	$(VENV_BIN)/pre-commit run --all-files

docs: install  ## Generate documentation
	$(PIP) install -r ./docs/requirements.txt
	cd docs && make clean && make html && cd -  # <<< Will run using the base env... change that make to use .venv?
# TODO: Maybe call sphinx-build directly...

tests: install  ## Run tests
	$(VENV_BIN)/pytest tests

doctest:  ## Run doctests
	$(VENV_BIN)/sphinx-build -b doctest ./docs ./docs/_build

doctest2:  ## Run doctests using pytest
	$(VENV_BIN)/pytest ./docs --doctest-glob=*.rst # --doctest-module

clean:  ## Clean up caches and build artifacts
	rm -rf *.egg-info dist build docs/_build .pytest_cache/ .ruff_cache/

metadeps:  ## Install packages used to develop/build this package
	$(PIP) install -U build pip-tools pre-commit wheel

updatereqs: update-pip  ## Autogenerate requirements
	$(PIP) install -U pip-tools
	rm requirements.txt
	$(VENV_BIN)/pip-compile --extra=tests --no-annotate --no-emit-index-url --output-file=requirements.txt --strip-extras pyproject.toml

build: clean venv metadeps  ## Build Python package for upload to pypi
	$(PYTHON) -m build --sdist --wheel -n
