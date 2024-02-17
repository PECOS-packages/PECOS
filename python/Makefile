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

.PHONY: venv
venv:  ## Setup a new Python virtual environment
	$(MAKE) -C .. $@

# Requirements
# ------------

.PHONY: requirements
requirements: upgrade-pip  ## Install Python project requirements
	$(Make) -C .. $@

.PHONY: updatereqs
updatereqs: upgrade-pip ## Autogenerate requirements.txt
	# $(VENV_BIN)/python -m pip install --upgrade pip
	$(VENV_BIN)/pip install -U pip-tools
	-@rm requirements.txt
	$(VENV_BIN)/pip-compile --extra=tests --no-annotate --no-emit-index-url --output-file=requirements.txt --strip-extras pyproject.toml

# Building / Installing
# ---------------------

.PHONY: build
build: clean venv
	$(Make) -C .. $@

# .PHONY: install-all
# install-all: upgrade-pip  ## Install PECOS with all optional dependencies
# 	$(VENV_BIN)/pip install ../cython
# 	$(VENV_BIN)/pip install .[all]
# .PHONY: build-full
# build-full: dev-setup updatereqs install-all docs lint tests-all  ## Go through the full linting, testing, and building process
# 	$(VENV_BIN)/python -m build --sdist --wheel -n

.PHONY: dev-setup
dev-setup: clean venv requirements metadeps

# Documentation
# -------------

.PHONY: docs
docs: build  ## Build documentation
	$(VENV_BIN)/pip install -r ./docs/requirements.txt
	$(MAKE) -C docs SPHINXBUILD=../$(VENV_BIN)/sphinx-build clean html

# Linting / formatting
# --------------------

.PHONY: pre-commit
pre-commit: ## Run all quality checks / linting / reformatting
	$(Make) -C .. $@

# Testing
# -------

.PHONY: test
tests: install metadeps  ## Run tests on the Python package (not including optional dependencies)
	$(VENV_BIN)/pytest tests -m "not optional_dependency"

.PHONY: test-dep
tests-dep: install-all metadeps ## Run tests on the Python package only for optional dependencies
	$(VENV_BIN)/pytest tests -m optional_dependency

.PHONY: doctest
doctests:  ## Run doctests
	$(VENV_BIN)/pytest ./docs --doctest-glob=*.rst --doctest-continue-on-failure

.PHONY: test-all
tests-all: tests tests-dep doctests ## Run all tests

# Utility
# -------

.PHONY: clean
clean:  ## Clean up caches, build artifacts, and generated library files
	-rm -rf *.egg-info dist build docs/_build .pytest_cache/ .ruff_cache/ .venv/
	@find . \( -type d -name "__pycache__" \
	          -o -type f -name "*.pyc" \
	          -o -type f -name "*.so" \
	          -o -type f -name "*.dll" \) \
	          -print0 | xargs -0 rm -rf

.PHONY: upgrade-pip
upgrade-pip:
	$(MAKE) -C .. upgrade-pip

# Help
# ----

.PHONY: help
help:  ## Show the help menu
	@echo "Available make commands:"
	@echo ""
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
