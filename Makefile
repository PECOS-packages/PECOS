.DEFAULT_GOAL := help

# Try to autodetect if python3 or python is the python executable used.
PYTHONPATH := $(shell which python 2>/dev/null || which python3 2>/dev/null)
SHELL=bash

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
	uv sync

.PHONY: buildrng
buildrng:
	@echo "Building and installing RNG library..."
	uv pip install nanobind
	cd clibs/pecos-rng && CC=gcc CXX=g++ uv pip install --python $(shell uv run which python) -e .

# Building development environments
# ---------------------------------
.PHONY: build
build: installreqs ## Compile and install for development
	@unset CONDA_PREFIX && cd python/pecos-rslib/ && uv run maturin develop --uv
	$(MAKE) buildrng
	@unset CONDA_PREFIX && cd python/quantum-pecos && uv pip install -e .[all]

.PHONY: build-basic
build-basic: installreqs ## Compile and install for development but do not include install extras
	@unset CONDA_PREFIX && cd python/pecos-rslib/ && uv run maturin develop --uv
	$(MAKE) buildrng
	@unset CONDA_PREFIX && cd python/quantum-pecos && uv pip install -e .

.PHONY: build-release
build-release: installreqs ## Build a faster version of binaries
	@unset CONDA_PREFIX && cd python/pecos-rslib/ && uv run maturin develop --uv --release
	$(MAKE) buildrng
	@unset CONDA_PREFIX && cd python/quantum-pecos && uv pip install -e .[all]

.PHONY: build-native
build-native: installreqs ## Build a faster version of binaries with native CPU optimization
	@unset CONDA_PREFIX && cd python/pecos-rslib/ && RUSTFLAGS='-C target-cpu=native' \
	&& uv run maturin develop --uv --release
	$(MAKE) buildrng
	@unset CONDA_PREFIX && cd python/quantum-pecos && uv pip install -e .[all]

# Documentation
# -------------

.PHONY: docs-build
docs-build:  ## Clean, install deps, and build documentation
	@uv run mkdocs build --clean

.PHONY: docs-serve
docs-serve:  ## Serve documentation (for  other ports add... -dev-addr=127.0.0.1:9000)
	@uv run mkdocs serve

.PHONY: docs-test
docs-test:  ## Test all code examples in documentation
	@uv run python scripts/docs/test_code_examples.py

.PHONY: docs-test-working
docs-test-working:  ## Test only working code examples in documentation
	@uv run python scripts/docs/test_working_examples.py

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
lint: check fmt clippy
	uv run pre-commit run --all-files

# Testing
# -------

.PHONY: qir-staticlib
qir-staticlib:  ## Build the QIR static library (needed for QIR compilation)
	cargo rustc -p pecos-qir --lib --crate-type=staticlib

.PHONY: qir-staticlib-if-needed
qir-staticlib-if-needed:  ## Build QIR static library only if it doesn't exist in persistent location
	@if [ ! -f ~/.cargo/pecos-qir/libpecos_qir.a ] && [ ! -f ~/.cargo/pecos-qir/pecos_qir.lib ]; then \
		echo "Building QIR static library..."; \
		$(MAKE) qir-staticlib; \
	fi

.PHONY: rstest
rstest: qir-staticlib-if-needed  ## Run Rust tests
	cargo test --workspace


.PHONY: pytest
pytest:  ## Run tests on the Python package (not including optional dependencies). ASSUMES: previous build command
	uv run pytest ./python/tests/ -m "not optional_dependency"
	uv run pytest ./python/pecos-rslib/tests/

.PHONY: pytest-dep
pytest-dep: ## Run tests on the Python package only for optional dependencies. ASSUMES: previous build command
	uv run pytest ./python/tests/ -m optional_dependency

.PHONY: pytest-all
pytest-all:  pytest ## Run all tests on the Python package ASSUMES: previous build command
	uv run pytest ./python/tests/ -m "optional_dependency"

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
ifeq ($(OS),Windows_NT)
	-@powershell -Command "exit 0" > NUL 2>&1 && $(MAKE) clean-windows-ps || $(MAKE) clean-windows-cmd
else
	$(MAKE) clean-unix
endif

.PHONY: clean-unix
clean-unix:
	@rm -rf *.egg-info
	@rm -rf dist
	@find . -type d -name "build" -exec rm -rf {} +
	@rm -rf python/docs/_build
	@rm -rf site
	@find . -type d -name ".pytest_cache" -exec rm -rf {} +
	@find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	@rm -rf .ruff_cache/
	@find . -type d -name ".hypothesis" -exec rm -rf {} +
	@find . -type d -name "junit" -exec rm -rf {} +
	@find python -name "*.so" -delete
	@find python -name "*.pyd" -delete
	@# Clean clibs build artifacts
	@find clibs -type d -name "build" -exec rm -rf {} +
	@find clibs -type d -name "dist" -exec rm -rf {} +
	@find clibs -type d -name "*.egg-info" -exec rm -rf {} +
	@find clibs -type d -name ".venv" -exec rm -rf {} +
	@find clibs -name "uv.lock" -delete
	@# Clean all target directories in crates (in case they were built independently)
	@find crates -type d -name "target" -exec rm -rf {} +
	@find python -type d -name "target" -exec rm -rf {} +
	@# Clean the root workspace target directory
	@cargo clean
	@# Clean the persistent QIR library directory
	@rm -rf ~/.cargo/pecos-qir/

.PHONY: clean-windows-ps
clean-windows-ps:
	@powershell -Command "if (Test-Path '*.egg-info') { Remove-Item -Recurse -Force *.egg-info }"
	@powershell -Command "if (Test-Path 'dist') { Remove-Item -Recurse -Force dist }"
	@powershell -Command "Get-ChildItem -Path . -Recurse -Directory -Filter 'build' | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue"
	@powershell -Command "if (Test-Path 'python\docs\_build') { Remove-Item -Recurse -Force python\docs\_build }"
	@powershell -Command "if (Test-Path 'site') { Remove-Item -Recurse -Force site }"
	@powershell -Command "Get-ChildItem -Path . -Recurse -Directory -Filter '.pytest_cache' | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue"
	@powershell -Command "Get-ChildItem -Path . -Recurse -Directory -Filter '.ipynb_checkpoints' | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue"
	@powershell -Command "if (Test-Path '.ruff_cache') { Remove-Item -Recurse -Force .ruff_cache }"
	@powershell -Command "Get-ChildItem -Path . -Recurse -Directory -Filter '.hypothesis' | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue"
	@powershell -Command "Get-ChildItem -Path . -Recurse -Directory -Filter 'junit' | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue"
	@powershell -Command "Get-ChildItem -Path python -Recurse -File -Include '*.so','*.pyd' | Remove-Item -Force -ErrorAction SilentlyContinue"
	@# Clean clibs build artifacts
	@powershell -Command "Get-ChildItem -Path clibs -Recurse -Directory -Filter 'build' | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue"
	@powershell -Command "Get-ChildItem -Path clibs -Recurse -Directory -Filter 'dist' | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue"
	@powershell -Command "Get-ChildItem -Path clibs -Recurse -Directory -Filter '*.egg-info' | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue"
	@powershell -Command "Get-ChildItem -Path clibs -Recurse -Directory -Filter '.venv' | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue"
	@powershell -Command "Get-ChildItem -Path clibs -Recurse -File -Filter 'uv.lock' | Remove-Item -Force -ErrorAction SilentlyContinue"
	@# Clean all target directories in crates
	@powershell -Command "Get-ChildItem -Path crates -Recurse -Directory -Filter 'target' | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue"
	@powershell -Command "Get-ChildItem -Path python -Recurse -Directory -Filter 'target' | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue"
	@cargo clean
	@# Clean the persistent QIR library directory
	@powershell -Command "if (Test-Path '$env:USERPROFILE\.cargo\pecos-qir') { Remove-Item -Recurse -Force $env:USERPROFILE\.cargo\pecos-qir }"

.PHONY: clean-windows-cmd
clean-windows-cmd:
	-@if exist *.egg-info rd /s /q *.egg-info
	-@if exist dist rd /s /q dist
	-@if exist python\docs\_build rd /s /q python\docs\_build
	-@if exist site rd /s /q site
	-@if exist .ruff_cache rd /s /q .ruff_cache
	-@for /f "delims=" %%d in ('dir /s /b /ad build 2^>nul') do @rd /s /q "%%d" 2>nul
	-@for /f "delims=" %%d in ('dir /s /b /ad .pytest_cache 2^>nul') do @rd /s /q "%%d" 2>nul
	-@for /f "delims=" %%d in ('dir /s /b /ad .ipynb_checkpoints 2^>nul') do @rd /s /q "%%d" 2>nul
	-@for /f "delims=" %%d in ('dir /s /b /ad .hypothesis 2^>nul') do @rd /s /q "%%d" 2>nul
	-@for /f "delims=" %%d in ('dir /s /b /ad junit 2^>nul') do @rd /s /q "%%d" 2>nul
	-@for /f "delims=" %%f in ('dir /s /b python\*.so python\*.pyd 2^>nul') do @del "%%f" 2>nul
	-@REM Clean clibs build artifacts
	-@for /f "delims=" %%d in ('dir /s /b /ad clibs\build 2^>nul') do @rd /s /q "%%d" 2>nul
	-@for /f "delims=" %%d in ('dir /s /b /ad clibs\dist 2^>nul') do @rd /s /q "%%d" 2>nul
	-@for /f "delims=" %%d in ('dir /s /b /ad clibs\*.egg-info 2^>nul') do @rd /s /q "%%d" 2>nul
	-@for /f "delims=" %%d in ('dir /s /b /ad clibs\.venv 2^>nul') do @rd /s /q "%%d" 2>nul
	-@for /f "delims=" %%f in ('dir /s /b clibs\uv.lock 2^>nul') do @del "%%f" 2>nul
	-@REM Clean all target directories in crates
	-@for /f "delims=" %%d in ('dir /s /b /ad crates\target 2^>nul') do @rd /s /q "%%d" 2>nul
	-@for /f "delims=" %%d in ('dir /s /b /ad python\target 2^>nul') do @rd /s /q "%%d" 2>nul
	-@cargo clean
	-@REM Clean the persistent QIR library directory
	-@if exist %USERPROFILE%\.cargo\pecos-qir rd /s /q %USERPROFILE%\.cargo\pecos-qir

.PHONY: pip-install-uv
pip-install-uv:  ## Install uv using pip and create a venv. (Recommended to instead follow: https://docs.astral.sh/uv/getting-started/installation/
	@echo "Installing uv..."
	$(PYTHONPATH) -m pip install --upgrade uv
	@echo "Creating venv and installing dependencies..."
	uv sync

.PHONY: dev
dev: clean build test  ## Run the typical sequence of commands to check everything is running correctly

.PHONY: devl  ## Run the commands to make sure everything runs + lint
devl: dev lint

# Help
# ----

.PHONY: help
help:  ## Show the help menu
	@echo "Available make commands:"
	@echo ""
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
