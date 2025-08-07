.DEFAULT_GOAL := help

# Try to autodetect if python3 or python is the python executable used.
PYTHON := $(shell which python 2>/dev/null || which python3 2>/dev/null)
SHELL=bash

# Requirements
# ------------

.PHONY: updatereqs
updatereqs:  ## Generate/update lockfiles for both packages
	@echo "Ensuring uv is installed..."
	uv self update
	@echo "Generating lock files..."
	uv lock --project .

.PHONY: installreqs
installreqs: ## Install Python project requirements to root .venv
	@echo "Installing requirements..."
	@if [ -n "$(UV_PYTHON)" ]; then \
		echo "Using pinned Python: $(UV_PYTHON)"; \
		uv sync --project . --python "$(UV_PYTHON)"; \
	else \
		uv sync --project .; \
	fi

# Building development environments
# ---------------------------------
.PHONY: build
build: installreqs ## Compile and install for development
	@unset CONDA_PREFIX && cd python/pecos-rslib/ && uv run maturin develop --uv
	@unset CONDA_PREFIX && uv pip install -e "./python/quantum-pecos[all]"

.PHONY: build-basic
build-basic: installreqs ## Compile and install for development but do not include install extras
	@unset CONDA_PREFIX && cd python/pecos-rslib/ && uv run maturin develop --uv
	@unset CONDA_PREFIX && uv pip install -e ./python/quantum-pecos

.PHONY: build-release
build-release: installreqs ## Build a faster version of binaries
	@unset CONDA_PREFIX && cd python/pecos-rslib/ && uv run maturin develop --uv --release
	@unset CONDA_PREFIX && uv pip install -e "./python/quantum-pecos[all]"

.PHONY: build-native
build-native: installreqs ## Build a faster version of binaries with native CPU optimization
	@unset CONDA_PREFIX && cd python/pecos-rslib/ && RUSTFLAGS='-C target-cpu=native' \
	&& uv run maturin develop --uv --release
	@unset CONDA_PREFIX && uv pip install -e "./python/quantum-pecos[all]"

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

.PHONY: rstest-all
rstest-all: qir-staticlib-if-needed  ## Run Rust tests with all features (includes WASM, decoders, etc.)
	cargo test --workspace --all-features

# Decoder-specific commands
# -------------------------

.PHONY: build-decoders
build-decoders: ## Build all decoder crates with all features
	cargo build --package pecos-decoders --all-features

.PHONY: build-decoder
build-decoder: ## Build specific decoder. Usage: make build-decoder DECODER=ldpc
	@if [ -z "$(DECODER)" ]; then \
		echo "Error: DECODER not specified. Usage: make build-decoder DECODER=ldpc"; \
		echo "Available decoders: ldpc"; \
		exit 1; \
	fi
	cargo build --package pecos-decoders --features $(DECODER)

.PHONY: test-decoders
test-decoders: ## Test all decoder crates
	cargo test --package pecos-decoders --all-features

.PHONY: test-decoder
test-decoder: ## Test specific decoder. Usage: make test-decoder DECODER=ldpc
	@if [ -z "$(DECODER)" ]; then \
		echo "Error: DECODER not specified. Usage: make test-decoder DECODER=ldpc"; \
		exit 1; \
	fi
	cargo test --package pecos-decoders --features $(DECODER)

.PHONY: decoder-info
decoder-info: ## Show available decoders and their features
	@echo "Available decoders in PECOS:"
	@echo "  • ldpc:           LDPC decoders (BP-OSD, MBP, etc.)"
	@echo ""
	@echo "To build specific decoder: make build-decoder DECODER=ldpc"
	@echo "To build all decoders:     make build-decoders"
	@echo "See DECODERS.md for detailed documentation."

.PHONY: decoder-cache-status
decoder-cache-status: ## Show decoder download cache status
	@CACHE_DIR="$${PECOS_CACHE_DIR:-$$HOME/.cache/pecos-decoders}"; \
	if [ -d "$$CACHE_DIR" ]; then \
		echo "Cache directory: $$CACHE_DIR"; \
		echo "Contents:"; \
		du -sh "$$CACHE_DIR"/* 2>/dev/null || echo "  (empty)"; \
	else \
		echo "No cache directory found at $$CACHE_DIR"; \
		echo "Cache will be created when building decoders"; \
	fi

.PHONY: decoder-cache-clean
decoder-cache-clean: ## Clean decoder download cache
	@CACHE_DIR="$${PECOS_CACHE_DIR:-$$HOME/.cache/pecos-decoders}"; \
	if [ -d "$$CACHE_DIR" ]; then \
		echo "Cleaning cache directory: $$CACHE_DIR"; \
		rm -rf "$$CACHE_DIR"; \
		echo "Cache cleaned"; \
	else \
		echo "No cache directory found"; \
	fi

.PHONY: pytest
pytest:  ## Run tests on the Python package (not including optional dependencies). ASSUMES: previous build command
	uv run pytest ./python/tests/ --doctest-modules -m "not optional_dependency"
	uv run pytest ./python/pecos-rslib/tests/

.PHONY: pytest-dep
pytest-dep: ## Run tests on the Python package only for optional dependencies. ASSUMES: previous build command
	uv run pytest ./python/tests/ --doctest-modules -m optional_dependency

.PHONY: pytest-all
pytest-all:  pytest ## Run all tests on the Python package ASSUMES: previous build command
	uv run pytest ./python/tests/ -m "optional_dependency"

# .PHONY: pytest-doc
# pydoctest:  ## Run doctests with pytest. ASSUMES: A build command was ran previously. ASSUMES: previous build command
# 	# TODO: update and install docs requirements
# 	uv run pytest docs --doctest-glob=*.rst --doctest-continue-on-failure

.PHONY: test
test: rstest-all pytest-all ## Run all tests. ASSUMES: previous build command

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
	-@REM Clean all target directories in crates
	-@for /f "delims=" %%d in ('dir /s /b /ad crates\target 2^>nul') do @rd /s /q "%%d" 2>nul
	-@for /f "delims=" %%d in ('dir /s /b /ad python\target 2^>nul') do @rd /s /q "%%d" 2>nul
	-@cargo clean
	-@REM Clean the persistent QIR library directory
	-@if exist %USERPROFILE%\.cargo\pecos-qir rd /s /q %USERPROFILE%\.cargo\pecos-qir

.PHONY: pip-install-uv
pip-install-uv:  ## Install uv using pip and create a venv. (Recommended to instead follow: https://docs.astral.sh/uv/getting-started/installation/
	@echo "Installing uv..."
	$(PYTHON) -m pip install --upgrade uv
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
