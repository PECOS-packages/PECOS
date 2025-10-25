.DEFAULT_GOAL := help

# Try to autodetect if python3 or python is the python executable used.
PYTHON := $(shell which python 2>/dev/null || which python3 2>/dev/null)
SHELL=bash

# LLVM Configuration
# LLVM is automatically detected by build.rs files using pecos-llvm-utils
# No manual configuration needed!

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

.PHONY: check-llvm
check-llvm: ## Check LLVM 14 installation status
	@cargo run -q --release --package pecos-llvm-utils --bin pecos-llvm -- check || true

# LLVM Detection Helper
# Auto-detect LLVM if not already set
SETUP_LLVM = \
	if [ -z "$$LLVM_SYS_140_PREFIX" ]; then \
		DETECTED_LLVM=$$(cargo run -q --release -p pecos-llvm-utils --bin pecos-llvm -- find 2>/dev/null); \
		if [ -n "$$DETECTED_LLVM" ]; then \
			export PECOS_LLVM="$$DETECTED_LLVM"; \
			export LLVM_SYS_140_PREFIX="$$DETECTED_LLVM"; \
			echo "Auto-detected LLVM at: $$LLVM_SYS_140_PREFIX"; \
		fi; \
	fi

# Helper to unset CONDA_PREFIX and add LLVM to PATH for runtime tools
ifdef OS
    # Windows (running in Git Bash/MSYS)
    UNSET_CONDA = set "CONDA_PREFIX=" &&
    ifdef LLVM_SYS_140_PREFIX
        ADD_LLVM_TO_PATH = export PATH="$(LLVM_SYS_140_PREFIX)/bin:$$PATH" &&
    else
        ADD_LLVM_TO_PATH =
    endif
else
    # Unix/Linux/macOS
    UNSET_CONDA = unset CONDA_PREFIX &&
    ifdef LLVM_SYS_140_PREFIX
        ADD_LLVM_TO_PATH = export PATH="$(LLVM_SYS_140_PREFIX)/bin:$$PATH" &&
    else
        ADD_LLVM_TO_PATH =
    endif
endif

.PHONY: build
build: installreqs ## Compile and install for development
	@$(SETUP_LLVM); $(UNSET_CONDA) cd python/pecos-rslib/ && uv run maturin develop --uv
	@$(UNSET_CONDA) uv pip install -e "./python/quantum-pecos[all]"
	@if command -v julia >/dev/null 2>&1; then \
		echo "Julia detected, building Julia FFI library..."; \
		$(SETUP_LLVM); cd julia/pecos-julia-ffi && cargo build; \
		echo "Julia FFI library built successfully"; \
	else \
		echo "Julia not detected, skipping Julia build"; \
	fi

.PHONY: build-basic
build-basic: installreqs ## Compile and install for development but do not include install extras
	@$(SETUP_LLVM); $(UNSET_CONDA) cd python/pecos-rslib/ && uv run maturin develop --uv
	@$(UNSET_CONDA) uv pip install -e ./python/quantum-pecos

.PHONY: build-release
build-release: installreqs ## Build a faster version of binaries
	@$(SETUP_LLVM); $(UNSET_CONDA) cd python/pecos-rslib/ && uv run maturin develop --uv --release
	@$(UNSET_CONDA) uv pip install -e "./python/quantum-pecos[all]"
	@if command -v julia >/dev/null 2>&1; then \
		echo "Julia detected, building Julia FFI library (release)..."; \
		cd julia/pecos-julia-ffi && cargo build --release; \
		echo "Julia FFI library built successfully"; \
	else \
		echo "Julia not detected, skipping Julia build"; \
	fi

.PHONY: build-native
build-native: installreqs ## Build a faster version of binaries with native CPU optimization
	@$(UNSET_CONDA) cd python/pecos-rslib/ && RUSTFLAGS='-C target-cpu=native' \
	uv run maturin develop --uv --release
	@$(UNSET_CONDA) uv pip install -e "./python/quantum-pecos[all]"

.PHONY: build-cuda
build-cuda: installreqs ## Compile and install for development with CUDA support
	@$(UNSET_CONDA) cd python/pecos-rslib/ && uv run maturin develop --uv
	@$(UNSET_CONDA) uv pip install -e "./python/quantum-pecos[all,cuda]"
	@if command -v julia >/dev/null 2>&1; then \
		echo "Julia detected, building Julia FFI library..."; \
		cd julia/pecos-julia-ffi && cargo build; \
		echo "Julia FFI library built successfully"; \
	else \
		echo "Julia not detected, skipping Julia build"; \
	fi

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

# Detect CUDA availability for GPU features
CUDA_AVAILABLE := $(shell command -v nvcc >/dev/null 2>&1 && echo "yes" || (test -n "$$CUDA_PATH" && echo "yes" || echo "no"))

# Get all features for pecos package except gpu (lazy evaluation - only computed when used)
PECOS_FEATURES_NO_GPU = $(shell cargo metadata --no-deps --format-version 1 2>/dev/null | jq -r '.packages[] | select(.name == "pecos") | .features | keys[] | select(. | IN("gpu") | not)' | tr '\n' ',' | sed 's/,$$//' 2>/dev/null)

# Get all features for pecos-quest package except gpu and cuda (lazy evaluation - only computed when used)
PECOS_QUEST_FEATURES_NO_GPU = $(shell cargo metadata --no-deps --format-version 1 2>/dev/null | jq -r '.packages[] | select(.name == "pecos-quest") | .features | keys[] | select(. | IN("gpu", "cuda") | not)' | tr '\n' ',' | sed 's/,$$//' 2>/dev/null)

# When CUDA is not available, we check all packages with all their features except GPU features
# This is done by checking packages separately
.PHONY: check
check:  ## Run cargo check (with GPU features only if CUDA available)
	@if [ "$(CUDA_AVAILABLE)" = "no" ]; then \
		echo "CUDA not detected - checking all features except GPU"; \
		echo "Checking workspace packages (excluding those with GPU features)..."; \
		cargo check --workspace --exclude pecos --exclude pecos-quest --all-targets --all-features; \
		echo "Checking pecos with all features except gpu..."; \
		cargo check -p pecos --all-targets --features "$(PECOS_FEATURES_NO_GPU)"; \
		echo "Checking pecos-quest with all features except gpu/cuda..."; \
		cargo check -p pecos-quest --all-targets --features "$(PECOS_QUEST_FEATURES_NO_GPU)"; \
	else \
		echo "CUDA detected - checking with all features"; \
		cargo check --workspace --all-targets --all-features; \
	fi

.PHONY: clippy
clippy:  ## Run cargo clippy (with GPU features only if CUDA available)
	@if [ "$(CUDA_AVAILABLE)" = "no" ]; then \
		echo "CUDA not detected - running clippy on all features except GPU"; \
		echo "Running clippy on workspace packages (excluding those with GPU features)..."; \
		cargo clippy --workspace --exclude pecos --exclude pecos-quest --all-targets --all-features -- -D warnings; \
		echo "Running clippy on pecos with all features except gpu..."; \
		cargo clippy -p pecos --all-targets --features "$(PECOS_FEATURES_NO_GPU)" -- -D warnings; \
		echo "Running clippy on pecos-quest with all features except gpu/cuda..."; \
		cargo clippy -p pecos-quest --all-targets --features "$(PECOS_QUEST_FEATURES_NO_GPU)" -- -D warnings; \
	else \
		echo "CUDA detected - running clippy with all features"; \
		cargo clippy --workspace --all-targets --all-features -- -D warnings; \
	fi

.PHONY: fmt
fmt: ## Check Rust formatting (without fixing)
	cargo fmt --all -- --check

.PHONY: fmt-fix
fmt-fix: ## Fix Rust formatting issues
	cargo fmt --all

.PHONY: lint
lint: fmt clippy  ## Run all quality checks / linting / reformatting (check only)
	uv run pre-commit run --all-files
	@if command -v julia >/dev/null 2>&1; then \
		echo "Julia detected, running Julia formatting check and linting..."; \
		$(MAKE) julia-format-check julia-lint; \
	else \
		echo "Julia not detected, skipping Julia linting"; \
	fi

.PHONY: normalize-line-endings
normalize-line-endings:  ## Normalize line endings according to .gitattributes
	@echo "Normalizing line endings according to .gitattributes..."
	@echo "This will refresh all tracked files to apply .gitattributes rules"
	@git rm --cached -r . >/dev/null 2>&1 || true
	@git reset --hard >/dev/null 2>&1
	@echo "Line endings normalized. Check 'git status' for any changes."

.PHONY: lint-fix
lint-fix:  ## Fix all auto-fixable linting issues (Rust, Python, Julia)
	@echo "Fixing Rust formatting..."
	cargo fmt --all
	@if [ "$(CUDA_AVAILABLE)" = "no" ]; then \
		echo "CUDA not detected - running clippy fix on all features except GPU"; \
		echo "Fixing workspace packages (excluding those with GPU features)..."; \
		cargo clippy --fix --workspace --exclude pecos --exclude pecos-quest --all-targets --all-features --allow-staged --allow-dirty; \
		echo "Fixing pecos with all features except gpu..."; \
		cargo clippy --fix -p pecos --all-targets --features "$(PECOS_FEATURES_NO_GPU)" --allow-staged --allow-dirty; \
		echo "Fixing pecos-quest with all features except gpu/cuda..."; \
		cargo clippy --fix -p pecos-quest --all-targets --features "$(PECOS_QUEST_FEATURES_NO_GPU)" --allow-staged --allow-dirty; \
	else \
		echo "CUDA detected - running clippy fix with all features"; \
		cargo clippy --fix --workspace --all-targets --all-features --allow-staged --allow-dirty; \
	fi
	@echo ""
	@echo "Running pre-commit fixes..."
	uv run pre-commit run --all-files || true
	@echo ""
	@if command -v julia >/dev/null 2>&1; then \
		echo "Fixing Julia formatting..."; \
		$(MAKE) julia-format; \
		echo ""; \
		echo "Note: Some Julia linting issues from Aqua.jl may require manual fixes."; \
	else \
		echo "Julia not detected, skipping Julia formatting"; \
	fi
	@echo ""
	@echo "Linting fixes applied! Run 'make lint' to check for remaining issues."

# Testing
# -------

.PHONY: rstest
rstest:  ## Run Rust tests
	@$(ADD_LLVM_TO_PATH) cargo test --workspace --release

.PHONY: rstest-all
rstest-all:  ## Run Rust tests with all features except GPU
	@$(ADD_LLVM_TO_PATH) cargo test --workspace --exclude pecos-quest --exclude pecos-decoders
	@$(ADD_LLVM_TO_PATH) cargo test -p pecos-quest
	@$(ADD_LLVM_TO_PATH) cargo test -p pecos-decoders --all-features

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
	@$(ADD_LLVM_TO_PATH) cargo test --package pecos-decoders --all-features

.PHONY: test-decoder
test-decoder: ## Test specific decoder. Usage: make test-decoder DECODER=ldpc
	@if [ -z "$(DECODER)" ]; then \
		echo "Error: DECODER not specified. Usage: make test-decoder DECODER=ldpc"; \
		exit 1; \
	fi
	@$(ADD_LLVM_TO_PATH) cargo test --package pecos-decoders --features $(DECODER)

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
	@$(ADD_LLVM_TO_PATH) uv run pytest ./python/quantum-pecos/tests/ --doctest-modules -m "not optional_dependency"
	@$(ADD_LLVM_TO_PATH) uv run pytest ./python/pecos-rslib/tests/
	@$(ADD_LLVM_TO_PATH) uv run pytest ./python/slr-tests/ -m "not optional_dependency"

.PHONY: pytest-dep
pytest-dep: ## Run tests on the Python package only for optional dependencies. ASSUMES: previous build command
	@$(ADD_LLVM_TO_PATH) uv run pytest ./python/quantum-pecos/tests/ --doctest-modules -m optional_dependency

.PHONY: pytest-all
pytest-all: ## Run all tests on the Python package ASSUMES: previous build command
	@$(ADD_LLVM_TO_PATH) uv run pytest ./python/quantum-pecos/tests/ -m ""
	@$(ADD_LLVM_TO_PATH) uv run pytest ./python/pecos-rslib/tests/

# .PHONY: pytest-doc
# pydoctest:  ## Run doctests with pytest. ASSUMES: A build command was ran previously. ASSUMES: previous build command
# 	# TODO: update and install docs requirements
# 	uv run pytest docs --doctest-glob=*.rst --doctest-continue-on-failure

.PHONY: test
test: rstest-all pytest-all ## Run all tests. ASSUMES: previous build command
	@if command -v julia >/dev/null 2>&1; then \
		echo "Julia detected, running Julia tests..."; \
		$(MAKE) julia-test; \
	else \
		echo "Julia not detected, skipping Julia tests"; \
	fi

.PHONY: test-all
test-all: rstest-all pytest-all ## Run all tests including Julia (warns if Julia not installed)
	@if command -v julia >/dev/null 2>&1; then \
		echo "Julia detected, running Julia tests..."; \
		$(MAKE) julia-test; \
	else \
		echo ""; \
		echo "WARNING: Julia is not installed. Skipping Julia tests."; \
		echo "   To run Julia tests, please install Julia from https://julialang.org/downloads/"; \
		echo ""; \
	fi

# Julia bindings
# --------------

.PHONY: julia-build
julia-build: ## Build Julia FFI library
	@echo "Building Julia FFI library..."
	cd julia/pecos-julia-ffi && cargo build --release
	@echo "Julia library built at: target/release/libpecos_julia.{so,dylib,dll}"

.PHONY: julia-build-debug
julia-build-debug: ## Build Julia FFI library in debug mode
	@echo "Building Julia FFI library (debug)..."
	cd julia/pecos-julia-ffi && cargo build
	@echo "Julia library built at: target/debug/libpecos_julia.{so,dylib,dll}"

.PHONY: julia-test
julia-test: julia-build ## Run Julia tests (requires Julia installed)
	@echo "Running Julia tests..."
	@if command -v julia >/dev/null 2>&1; then \
		cd julia/PECOS.jl && julia --project=. -e 'using Pkg; Pkg.instantiate(); include("test/runtests.jl")'; \
	else \
		echo "Julia not found. Please install Julia to run tests."; \
		exit 1; \
	fi

.PHONY: julia-examples
julia-examples: julia-build-debug ## Run Julia examples (requires Julia installed)
	@echo "Running Julia examples..."
	@if command -v julia >/dev/null 2>&1; then \
		cd julia/PECOS.jl && julia --project=. examples/demo.jl; \
		cd julia/PECOS.jl && julia --project=. examples/basic_usage.jl; \
	else \
		echo "Julia not found. Please install Julia to run examples."; \
		exit 1; \
	fi

.PHONY: julia-clean
julia-clean: ## Clean Julia build artifacts
	@echo "Cleaning Julia artifacts..."
	@rm -rf julia/PECOS.jl/Manifest.toml
	@rm -rf julia/PECOS.jl/dev/PECOS_julia_jll/Manifest.toml
	@rm -rf julia/PECOS.jl/dev/PECOS_julia_jll/src/Manifest.toml
	@find julia -name "*.jl.*.cov" -delete
	@find julia -name "*.jl.cov" -delete
	@find julia -name "*.jl.mem" -delete

.PHONY: julia-info
julia-info: ## Show Julia package information
	@echo "Julia Package Information:"
	@echo "========================="
	@echo "Package name: PECOS.jl"
	@echo "Location: julia/PECOS.jl"
	@echo "FFI library: julia/pecos-julia-ffi"
	@echo ""
	@echo "To install for development:"
	@echo "  1. Build FFI library: make julia-build"
	@echo "  2. In Julia REPL: ] add julia/PECOS.jl"
	@echo ""
	@echo "To run tests: make julia-test"
	@echo "To run examples: make julia-examples"

.PHONY: julia-format
julia-format: ## Format Julia code using JuliaFormatter
	@echo "Formatting Julia code..."
	@if command -v julia >/dev/null 2>&1; then \
		cd julia/PECOS.jl && julia -e 'using Pkg; if !haskey(Pkg.project().dependencies, "JuliaFormatter"); Pkg.add("JuliaFormatter"); end; using JuliaFormatter; format("."; verbose=true)'; \
	else \
		echo "Julia not found. Please install Julia to format code."; \
		exit 1; \
	fi

.PHONY: julia-format-check
julia-format-check: ## Check Julia code formatting without modifying files
	@echo "Checking Julia code formatting..."
	@if command -v julia >/dev/null 2>&1; then \
		cd julia/PECOS.jl && julia -e 'using Pkg; if !haskey(Pkg.project().dependencies, "JuliaFormatter"); Pkg.add("JuliaFormatter"); end; using JuliaFormatter; if !format("."; verbose=false, overwrite=false); println("Formatting issues found. Run `make julia-format` to fix."); exit(1); else println("All Julia code is properly formatted."); end'; \
	else \
		echo "Julia not found. Please install Julia to check formatting."; \
		exit 1; \
	fi

.PHONY: julia-lint
julia-lint: julia-build ## Run Aqua.jl quality checks on Julia code
	@echo "Running Julia code quality checks with Aqua.jl..."
	@if command -v julia >/dev/null 2>&1; then \
		cd julia/PECOS.jl && julia --project=. test/aqua_tests.jl; \
	else \
		echo "Julia not found. Please install Julia to run linting."; \
		exit 1; \
	fi

# Utility
# -------

.PHONY: clean
clean:  ## Clean up caches and build artifacts
ifeq ($(OS),Windows_NT)
	# Check if Unix commands are available (from Git Bash, MSYS2, etc. in PATH)
	@if command -v rm >/dev/null 2>&1 && command -v /usr/bin/find >/dev/null 2>&1; then \
		$(MAKE) clean-unix; \
	else \
		powershell -Command "exit 0" > NUL 2>&1 && $(MAKE) clean-windows-ps || $(MAKE) clean-windows-cmd; \
	fi
else
	$(MAKE) clean-unix
endif

.PHONY: clean-unix
clean-unix:
	@rm -rf *.egg-info
	@rm -rf dist
	@/usr/bin/find . -type d -name "build" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf python/docs/_build
	@rm -rf site
	@/usr/bin/find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@/usr/bin/find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .ruff_cache/
	@/usr/bin/find . -type d -name ".hypothesis" -exec rm -rf {} + 2>/dev/null || true
	@/usr/bin/find . -type d -name "junit" -exec rm -rf {} + 2>/dev/null || true
	@/usr/bin/find python -name "*.so" -delete 2>/dev/null || true
	@/usr/bin/find python -name "*.pyd" -delete 2>/dev/null || true
	@# Clean all target directories in crates (in case they were built independently)
	@/usr/bin/find crates -type d -name "target" -exec rm -rf {} + 2>/dev/null || true
	@/usr/bin/find python -type d -name "target" -exec rm -rf {} + 2>/dev/null || true
	@# Clean Julia artifacts
	@rm -rf julia/PECOS.jl/Manifest.toml
	@rm -rf julia/PECOS.jl/dev/PECOS_julia_jll/Manifest.toml
	@rm -rf julia/PECOS.jl/dev/PECOS_julia_jll/src/Manifest.toml
	@/usr/bin/find julia -name "*.jl.*.cov" -delete 2>/dev/null || true
	@/usr/bin/find julia -name "*.jl.cov" -delete 2>/dev/null || true
	@/usr/bin/find julia -name "*.jl.mem" -delete 2>/dev/null || true
	@# Clean the root workspace target directory
	@cargo clean

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

.PHONY: pip-install-uv
pip-install-uv:  ## Install uv using pip and create a venv. (Recommended to instead follow: https://docs.astral.sh/uv/getting-started/installation/
	@echo "Installing uv..."
	$(PYTHON) -m pip install --upgrade uv
	@echo "Creating venv and installing dependencies..."
	uv sync

.PHONY: dev
dev: clean build test  ## Run the typical sequence of commands to check everything is running correctly

.PHONY: devl
devl: dev lint  ## Run the commands to make sure everything runs + lint

.PHONY: devc
devc: clean build-cuda test  ## Run dev sequence with CUDA support (requires CUDA Toolkit 13)

.PHONY: devcl
devcl: devc lint  ## Run dev sequence with CUDA support + lint (requires CUDA Toolkit 13)

# Help
# ----

.PHONY: help
help:  ## Show the help menu
	@echo "Available make commands:"
	@echo ""
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Note: Julia support is automatically detected."
	@echo "  - 'make build' will also build Julia FFI if Julia is installed"
	@echo "  - 'make test' will also run Julia tests if Julia is installed"
	@echo "  - 'make lint' checks code quality; 'make lint-fix' fixes issues"
	@echo "  - Use 'make julia-info' for more Julia-specific information"
	@echo ""
	@echo "CUDA GPU Simulator Support:"
	@echo "  - 'make build-cuda' builds with CUDA GPU simulator support"
	@echo "  - 'make devc' runs full dev cycle with CUDA support"
	@echo "  - 'make devcl' runs dev + linting with CUDA support"
	@echo "  - Requires: CUDA Toolkit 13 (see docs/user-guide/cuda-setup.md)"
