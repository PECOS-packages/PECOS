.PHONY: update-pip .venv requirements install install-all dev dev-all tests lint docs doctest doctest2 clean build metadeps updatereqs

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

update-pip:
	$(PYTHON) -m pip install --upgrade pip

.venv:  ## Set up Python virtual environment (if it doesn't already exist) and install requirements
	[ -d $(VENV) ] || $(PYTHON) -m venv $(VENV)
	$(MAKE) requirements

requirements: update-pip  ## Install/refresh Python project requirements
	$(PIP) install --upgrade -r requirements.txt
	$(PIP) install --upgrade -r docs/requirements.txt

install: update-pip
	$(PIP) install .

install-all: update-pip
	$(PIP) install .[all]

dev: update-pip
	$(PIP) install -e .

dev-all: update-pip
	$(PIP) install -e .[all]

uninstall:
	$(PIP) uninstall quantum-pecos

lint:
	pre-commit run --all-files

docs: install
	$(PIP) install -r ./docs/requirements.txt
	cd docs && make clean && make html && cd -

tests: install
	pytest tests

doctest:
	sphinx-build -b doctest ./docs ./docs/_build

doctest2:
	pytest ./docs --doctest-glob=*.rst # --doctest-module

clean:
	rm -rf *.egg-info dist build docs/_build .venv/ .pytest_cache/ .ruff_cache/

build: clean
	python -m build --sdist --wheel -n

metadeps:
	$(PIP) install -U build pip-tools pre-commit wheel

updatereqs: update-pip
	$(PIP) install -U pip-tools
	rm requirements.txt
	pip-compile --extra=tests --no-annotate --no-emit-index-url --output-file=requirements.txt --strip-extras pyproject.toml
