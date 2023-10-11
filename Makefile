.PHONY: .venv update-pip requirements install install-all dev dev-all tests lint docs doctest doctest2 clean build metadeps updatereqs

# Try to autodetect if python3 or python is the python executable used.
PYTHON := $(shell which python3 2>/dev/null || which python 2>/dev/null)
PIP := $(shell which pip3 2>/dev/null || which pip 2>/dev/null)
PYTHONPATH=
SHELL=/bin/bash
VENV=.venv

all:
ifeq ($(PYTHON),)
	$(error "No python interpreter found")
endif
	@echo Using $(PYTHON)

ifeq ($(OS),Windows_NT)
	VENV_BIN=$(VENV)/Scripts
else
	VENV_BIN=$(VENV)/bin
endif

.venv:  ## Set up a new Python virtual environment and install requirements
	rm -rf .venv/
	$(PYTHON) -m venv $(VENV)
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

lint:  ## Run all quality checks / linting
	pre-commit run --all-files

docs: install  ## Generate documentation
	$(PIP) install -r ./docs/requirements.txt
	cd docs && make clean && make html && cd -

tests: install  ## Run tests
	pytest tests

doctest:  ## Run doctests
	sphinx-build -b doctest ./docs ./docs/_build

doctest2:  ## Run doctests using pytest
	pytest ./docs --doctest-glob=*.rst # --doctest-module

clean:  ## Clean up caches and build artifacts
	rm -rf *.egg-info dist build docs/_build .pytest_cache/ .ruff_cache/
	rm -r **/__pycache__

build: clean
	python -m build --sdist --wheel -n

metadeps:  ## Install packages used to develop/build this package
	$(PIP) install -U build pip-tools pre-commit wheel

updatereqs: update-pip  ## Autogenerate requirements
	$(PIP) install -U pip-tools
	rm requirements.txt
	pip-compile --extra=tests --no-annotate --no-emit-index-url --output-file=requirements.txt --strip-extras pyproject.toml
