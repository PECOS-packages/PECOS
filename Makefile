.DEFAULT_GOAL := help

# =============================================================================
# DEPRECATED: This Makefile is deprecated in favor of the Justfile.
# Please use `just` instead of `make` for all development commands.
# Install just: cargo install just (or see https://github.com/casey/just)
# Run `just --list` to see available commands.
# =============================================================================

# Try to autodetect if python3 or python is the python executable used.
PYTHON := $(shell which python 2>/dev/null || which python3 2>/dev/null)
SHELL=bash

# FFI crates that should be excluded from workspace-wide cargo commands
# These are built separately by maturin (Python), Julia, and Go tooling
FFI_CRATES := pecos-rslib pecos-julia-ffi pecos-go-ffi

# Generate --exclude flags for cargo commands
CARGO_EXCLUDE_FFI := $(foreach crate,$(FFI_CRATES),--exclude $(crate))

# LLVM Configuration
# LLVM 14 is required for QIR/LLVM IR features (pecos-llvm, pecos-engines with llvm feature)
# Run 'make install-llvm' to download and install LLVM 14 to ~/.pecos/llvm/
# Run 'make check-llvm' to verify installation status

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

# LLVM Setup
# ----------

.PHONY: install-llvm
install-llvm: ## Install LLVM 14 to ~/.pecos/llvm/ (required for QIR features)
	@echo "Installing LLVM 14..."
	@cargo run --release --package pecos-dev -- llvm install

.PHONY: check-llvm
check-llvm: ## Check LLVM 14 installation status
	@cargo run -q --release --package pecos-dev -- llvm check || true

# CUDA Setup
# ----------
# CUDA Toolkit is required for GPU support (pecos-quest, selene-quest with GPU)
# Run 'make install-cuda' to download and install CUDA to ~/.pecos/cuda/
# Run 'make check-cuda' to verify installation status
# Note: This installs compile-time dependencies only - no GPU hardware needed

.PHONY: install-cuda
install-cuda: ## Install CUDA Toolkit to ~/.pecos/cuda/ (for GPU support, no GPU needed)
	@echo "Installing CUDA Toolkit..."
	@cargo run --release --package pecos-dev -- cuda install

.PHONY: check-cuda
check-cuda: ## Check CUDA installation status (local or system)
	@cargo run -q --release --package pecos-dev -- cuda check || true

.PHONY: validate-cuda
validate-cuda: ## Validate CUDA installation integrity
	@cargo run -q --release --package pecos-dev -- cuda validate

# Helper to unset CONDA_PREFIX (prevents conda interference with builds)
# Note: LLVM_SYS_140_PREFIX is set via .cargo/config.toml (run `pecos-dev llvm configure`)
ifdef OS
    # Windows (running in Git Bash/MSYS)
    UNSET_CONDA = set "CONDA_PREFIX=" &&
else
    # Unix/Linux/macOS
    UNSET_CONDA = unset CONDA_PREFIX &&
endif

# Build profile configuration
# Usage: make build PROFILE=debug|release|native (default: debug)
# Build scripts detect the profile via Cargo's PROFILE env var.
PROFILE ?= debug

# Profile-specific Cargo/Maturin settings
# - debug: uses default cargo (debug) profile - fast compile, no optimization
# - release: uses --release flag - full optimization
# - native: uses --profile native (custom profile inheriting from release) + CPU-specific opts
#
# For native profile, we also pass -C target-cpu=native to Rust via RUSTFLAGS.
# Build scripts detect PROFILE=native and add -march=native for C++ code.
ifeq ($(PROFILE),native)
    MATURIN_RELEASE_FLAG := --release
    CARGO_PROFILE_FLAG := --profile native
    RUSTFLAGS_EXTRA := -C target-cpu=native
    PROFILE_DESC := native (release + CPU optimizations)
else ifeq ($(PROFILE),release)
    MATURIN_RELEASE_FLAG := --release
    CARGO_PROFILE_FLAG := --release
    RUSTFLAGS_EXTRA :=
    PROFILE_DESC := release (optimized)
else
    # debug profile (default)
    MATURIN_RELEASE_FLAG :=
    CARGO_PROFILE_FLAG :=
    RUSTFLAGS_EXTRA :=
    PROFILE_DESC := debug (fast compile, unoptimized)
endif

# Helper to build FFI crates with the correct profile
# Uses pecos-dev julia/go build for cross-platform tool detection and building
define BUILD_FFI_CRATES
	@cargo run -q -p pecos-dev -- julia build --profile $(PROFILE) $(if $(RUSTFLAGS_EXTRA),--rustflags "$(RUSTFLAGS_EXTRA)") 2>/dev/null || true
	@cargo run -q -p pecos-dev -- go build --profile $(PROFILE) $(if $(RUSTFLAGS_EXTRA),--rustflags "$(RUSTFLAGS_EXTRA)") 2>/dev/null || true
endef

.PHONY: build
build: installreqs build-selene ## Build PECOS (use PROFILE=debug|release|native, default: debug)
	@cargo run -p pecos-dev -- python build --profile $(PROFILE)
	$(BUILD_FFI_CRATES)

.PHONY: build-selene
build-selene: ## Build and install Selene plugins for development
	@echo "Building Selene plugins..."
	@# Build Rust libraries (with GPU support if CUDA available)
	@if cargo run -q -p pecos-dev -- cuda check -q >/dev/null 2>&1; then \
		echo "CUDA detected, building with GPU support..."; \
		cargo build --release -p pecos-selene-quest --features cuda; \
	else \
		echo "CUDA not detected, building CPU-only..."; \
		cargo build --release -p pecos-selene-quest; \
	fi
	@cargo build --release -p pecos-selene-qulacs -p pecos-selene-sparsestab -p pecos-selene-statevec
	@# Copy libraries to Python package directories (cross-platform via pecos-dev)
	@echo "Copying libraries to Python packages..."
	@cargo run -p pecos-dev -- selene install
	@# Install Python packages in editable mode
	@echo "Installing Selene plugins in editable mode..."
	@$(UNSET_CONDA) uv pip install -e ./python/selene-plugins/pecos-selene-quest
	@$(UNSET_CONDA) uv pip install -e ./python/selene-plugins/pecos-selene-qulacs
	@$(UNSET_CONDA) uv pip install -e ./python/selene-plugins/pecos-selene-sparsestab
	@$(UNSET_CONDA) uv pip install -e ./python/selene-plugins/pecos-selene-statevec
	@echo "Selene plugins built and installed successfully"

.PHONY: build-cuda
build-cuda: installreqs ## Build PECOS with CUDA support (use PROFILE=debug|release|native, default: debug)
	@cargo run -p pecos-dev -- python build --profile $(PROFILE) --cuda
	$(BUILD_FFI_CRATES)

# Convenience aliases for common build profiles
.PHONY: build-debug
build-debug: ## Alias for: make build PROFILE=debug
	@$(MAKE) build PROFILE=debug

.PHONY: build-release
build-release: ## Alias for: make build PROFILE=release
	@$(MAKE) build PROFILE=release

.PHONY: build-native
build-native: ## Alias for: make build PROFILE=native
	@$(MAKE) build PROFILE=native

.PHONY: build-cuda-debug
build-cuda-debug: ## Alias for: make build-cuda PROFILE=debug
	@$(MAKE) build-cuda PROFILE=debug

.PHONY: build-cuda-release
build-cuda-release: ## Alias for: make build-cuda PROFILE=release
	@$(MAKE) build-cuda PROFILE=release

.PHONY: build-cuda-native
build-cuda-native: ## Alias for: make build-cuda PROFILE=native
	@$(MAKE) build-cuda PROFILE=native

# Documentation
# -------------

.PHONY: docs-build
docs-build:  ## Clean, install deps, and build documentation
	@uv run mkdocs build --clean

.PHONY: docs
docs:  ## Serve documentation and open in browser (PORT=9000 to change port)
	@uv run mkdocs serve -a 127.0.0.1:$(or $(PORT),8000) 2>&1 | while IFS= read -r line; do \
		echo "$$line"; \
		case "$$line" in *"Serving on"*) xdg-open http://127.0.0.1:$(or $(PORT),8000)/PECOS/ 2>/dev/null ;; esac; \
	done

.PHONY: docs-test
docs-test:  ## Test all code examples in documentation
	@uv run python scripts/docs/test_code_examples.py

.PHONY: docs-test-working
docs-test-working:  ## Test only working code examples in documentation
	@uv run python scripts/docs/test_working_examples.py

# Linting / formatting
# --------------------

# Rust check, clippy, fmt - use pecos-dev for CUDA-aware handling
.PHONY: check
check:  ## Run cargo check (with GPU features only if CUDA available)
	@cargo run -p pecos-dev -- rust check --include-ffi

.PHONY: clippy
clippy:  ## Run cargo clippy (with GPU features only if CUDA available)
	@echo "==> Running clippy via pecos-dev..."
	cargo run -p pecos-dev -- rust clippy --include-ffi

.PHONY: fmt
fmt: ## Check Rust formatting (without fixing)
	@echo "==> Running fmt check via pecos-dev..."
	cargo run -p pecos-dev -- rust fmt --check

.PHONY: fmt-fix
fmt-fix: ## Fix Rust formatting issues
	@cargo run -p pecos-dev -- rust fmt

.PHONY: lint
lint: fmt clippy  ## Run all quality checks / linting / reformatting (check only)
	@echo "==> Running pre-commit..."
	uv run pre-commit run --all-files
	@if cargo run -q -p pecos-dev -- julia check -q >/dev/null 2>&1; then \
		echo "Julia detected, running Julia formatting check and linting..."; \
		cargo run -q -p pecos-dev -- julia fmt --check; \
		cargo run -q -p pecos-dev -- julia lint; \
	else \
		echo "Julia not detected, skipping Julia linting"; \
	fi
	@if cargo run -q -p pecos-dev -- go check -q >/dev/null 2>&1; then \
		echo "Go detected, running Go formatting check and linting..."; \
		cargo run -q -p pecos-dev -- go fmt --check; \
		cargo run -q -p pecos-dev -- go lint; \
	else \
		echo "Go not detected, skipping Go linting"; \
	fi

.PHONY: normalize-line-endings
normalize-line-endings:  ## Normalize line endings according to .gitattributes
	@echo "Normalizing line endings according to .gitattributes..."
	@echo "This will refresh all tracked files to apply .gitattributes rules"
	@git rm --cached -r . >/dev/null 2>&1 || true
	@git reset --hard >/dev/null 2>&1
	@echo "Line endings normalized. Check 'git status' for any changes."

.PHONY: lint-fix
lint-fix:  ## Fix all auto-fixable linting issues (Rust, Python, Julia, Go)
	@echo "Fixing Rust formatting and clippy issues..."
	@cargo run -p pecos-dev -- rust fmt
	@cargo run -p pecos-dev -- rust clippy --fix --include-ffi
	@echo ""
	@echo "Running pre-commit fixes..."
	uv run pre-commit run --all-files || true
	@echo ""
	@if cargo run -q -p pecos-dev -- julia check -q >/dev/null 2>&1; then \
		echo "Fixing Julia formatting..."; \
		cargo run -q -p pecos-dev -- julia fmt; \
		echo ""; \
		echo "Note: Some Julia linting issues from Aqua.jl may require manual fixes."; \
	else \
		echo "Julia not detected, skipping Julia formatting"; \
	fi
	@if cargo run -q -p pecos-dev -- go check -q >/dev/null 2>&1; then \
		echo "Fixing Go formatting..."; \
		cargo run -q -p pecos-dev -- go fmt; \
	else \
		echo "Go not detected, skipping Go formatting"; \
	fi
	@echo ""
	@echo "Linting fixes applied! Run 'make lint' to check for remaining issues."

# Testing
# -------

.PHONY: rstest
rstest:  ## Run Rust tests (with GPU features only if CUDA available)
	@cargo run -q -p pecos-dev -- rust test --release

.PHONY: rstest-all
rstest-all:  ## Run Rust tests with all features (including GPU if CUDA available)
	@cargo run -q -p pecos-dev -- rust test

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
	@cargo test --package pecos-decoders --all-features

.PHONY: test-decoder
test-decoder: ## Test specific decoder. Usage: make test-decoder DECODER=ldpc
	@if [ -z "$(DECODER)" ]; then \
		echo "Error: DECODER not specified. Usage: make test-decoder DECODER=ldpc"; \
		exit 1; \
	fi
	@cargo test --package pecos-decoders --features $(DECODER)

.PHONY: decoder-info
decoder-info: ## Show available decoders and their features
	@echo "Available decoders in PECOS:"
	@echo "  • ldpc:           LDPC decoders (BP-OSD, MBP, etc.)"
	@echo ""
	@echo "To build specific decoder: make build-decoder DECODER=ldpc"
	@echo "To build all decoders:     make build-decoders"
	@echo "See DECODERS.md for detailed documentation."

.PHONY: decoder-cache-status
decoder-cache-status: ## Show decoder download cache status (now managed by pecos-dev)
	@cargo run -q -p pecos-dev -- list -v

.PHONY: decoder-cache-clean
decoder-cache-clean: clean-cache  ## Clean decoder download cache (same as clean-cache)
	@echo "Decoder cache cleaned (part of ~/.pecos/cache/)"

.PHONY: pytest
pytest:  ## Run tests on the Python package (excluding numpy and optional deps). ASSUMES: previous build command
	@cargo run -q -p pecos-dev -- python test

.PHONY: pytest-numpy
pytest-numpy:  ## Run NumPy/SciPy compatibility tests. ASSUMES: previous build command
	@cargo run -q -p pecos-dev -- python test --numpy

.PHONY: pytest-perf
pytest-perf: build-release ## Run performance tests on pecos-rslib with release build
	@echo "Running pecos-rslib performance tests with release build..."
	@uv run --group numpy-compat pytest ./python/pecos-rslib/tests/ -m "performance" -v

.PHONY: pytest-dep
pytest-dep: ## Run tests on the Python package only for optional dependencies. ASSUMES: previous build command
	@cargo run -q -p pecos-dev -- python test -m optional_dependency

.PHONY: pytest-selene
pytest-selene: ## Run tests for Selene plugins. ASSUMES: previous build command
	@cargo run -q -p pecos-dev -- python test --selene

.PHONY: pytest-all
pytest-all: pytest pytest-numpy pytest-selene ## Run all tests (core + numpy compat + selene) on the Python package. ASSUMES: previous build command
	@echo "All Python tests completed (core + NumPy/SciPy compatibility + Selene plugins)"

# .PHONY: pytest-doc
# pydoctest:  ## Run doctests with pytest. ASSUMES: A build command was ran previously. ASSUMES: previous build command
# 	# TODO: update and install docs requirements
# 	uv run pytest docs --doctest-glob=*.rst --doctest-continue-on-failure

.PHONY: test
test: rstest-all pytest-all ## Run all tests. ASSUMES: previous build command
	@if cargo run -q -p pecos-dev -- julia check -q >/dev/null 2>&1; then \
		echo "Julia detected, running Julia tests..."; \
		cargo run -q -p pecos-dev -- julia test; \
	else \
		echo "Julia not detected, skipping Julia tests"; \
	fi
	@if cargo run -q -p pecos-dev -- go check -q >/dev/null 2>&1; then \
		echo "Go detected, running Go tests..."; \
		cargo run -q -p pecos-dev -- go test; \
	else \
		echo "Go not detected, skipping Go tests"; \
	fi

.PHONY: test-all
test-all: rstest-all pytest-all ## Run all tests including Julia and Go (warns if not installed)
	@if cargo run -q -p pecos-dev -- julia check -q >/dev/null 2>&1; then \
		echo "Julia detected, running Julia tests..."; \
		cargo run -q -p pecos-dev -- julia test; \
	else \
		echo ""; \
		echo "WARNING: Julia is not installed. Skipping Julia tests."; \
		echo "   To run Julia tests, please install Julia from https://julialang.org/downloads/"; \
		echo ""; \
	fi
	@if cargo run -q -p pecos-dev -- go check -q >/dev/null 2>&1; then \
		echo "Go detected, running Go tests..."; \
		cargo run -q -p pecos-dev -- go test; \
	else \
		echo ""; \
		echo "WARNING: Go is not installed. Skipping Go tests."; \
		echo "   To run Go tests, please install Go from https://go.dev/dl/"; \
		echo ""; \
	fi

# Julia bindings
# --------------

.PHONY: julia-build
julia-build: ## Build Julia FFI library
	@cargo run -q -p pecos-dev -- julia build

.PHONY: julia-build-debug
julia-build-debug: ## Build Julia FFI library in debug mode
	@cargo run -q -p pecos-dev -- julia build --profile debug

.PHONY: julia-test
julia-test: ## Run Julia tests (requires Julia installed)
	@cargo run -q -p pecos-dev -- julia test

.PHONY: julia-examples
julia-examples: julia-build-debug ## Run Julia examples (requires Julia installed)
	@echo "Running Julia examples..."
	@if cargo run -q -p pecos-dev -- julia check -q >/dev/null 2>&1; then \
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
	@echo "  1. Build FFI library: pecos-dev julia build"
	@echo "  2. In Julia REPL: ] add julia/PECOS.jl"
	@echo ""
	@echo "To run tests: pecos-dev julia test"
	@echo "To run examples: make julia-examples"

.PHONY: julia-format
julia-format: ## Format Julia code using JuliaFormatter
	@cargo run -q -p pecos-dev -- julia fmt

.PHONY: julia-format-check
julia-format-check: ## Check Julia code formatting without modifying files
	@cargo run -q -p pecos-dev -- julia fmt --check

.PHONY: julia-lint
julia-lint: ## Run Aqua.jl quality checks on Julia code
	@cargo run -q -p pecos-dev -- julia lint

# Go bindings
# -----------

.PHONY: go-build
go-build: ## Build Go FFI library
	@cargo run -q -p pecos-dev -- go build

.PHONY: go-build-debug
go-build-debug: ## Build Go FFI library in debug mode
	@cargo run -q -p pecos-dev -- go build --profile debug

.PHONY: go-test
go-test: ## Run Go tests (requires Go installed)
	@cargo run -q -p pecos-dev -- go test

.PHONY: go-clean
go-clean: ## Clean Go build artifacts
	@echo "Cleaning Go artifacts..."
	@rm -rf go/pecos/go.sum
	@if cargo run -q -p pecos-dev -- go check -q >/dev/null 2>&1; then \
		cd go/pecos && go clean -cache 2>/dev/null || true; \
	fi

.PHONY: go-info
go-info: ## Show Go package information
	@echo "Go Package Information:"
	@echo "======================="
	@echo "Package name: github.com/PECOS-packages/PECOS/go/pecos"
	@echo "Location: go/pecos"
	@echo "FFI library: go/pecos-go-ffi"
	@echo ""
	@echo "To build and test:"
	@echo "  1. Build FFI library: pecos-dev go build"
	@echo "  2. Run tests: pecos-dev go test"
	@echo ""
	@echo "To use in your Go project:"
	@echo "  1. Set LD_LIBRARY_PATH to include target/release"
	@echo "  2. Import: github.com/PECOS-packages/PECOS/go/pecos"

.PHONY: go-fmt
go-fmt: ## Format Go code using gofmt
	@cargo run -q -p pecos-dev -- go fmt

.PHONY: go-fmt-check
go-fmt-check: ## Check Go code formatting without modifying files
	@cargo run -q -p pecos-dev -- go fmt --check

.PHONY: go-lint
go-lint: ## Run Go linting with go vet
	@cargo run -q -p pecos-dev -- go lint

# Cleaning
# --------
# Cross-platform cleaning via Python script (works on Windows, macOS, Linux)
# Uses uv to run scripts/clean.py which handles all platforms via pathlib/shutil

.PHONY: clean
clean:  ## Clean build artifacts (cross-platform, no compilation needed)
	@uv run python scripts/clean.py

.PHONY: clean-selene
clean-selene:  ## Clean Selene plugin build artifacts
	@uv run python scripts/clean.py --selene

.PHONY: clean-cache
clean-cache:  ## Clean ~/.pecos/cache/ and ~/.pecos/tmp/ (downloaded archives)
	@uv run python scripts/clean.py --cache

.PHONY: clean-deps
clean-deps:  ## Clean ~/.pecos/deps/ (extracted C++ dependencies)
	@uv run python scripts/clean.py --deps

.PHONY: clean-llvm
clean-llvm:  ## Clean ~/.pecos/llvm/ (LLVM installation - large, slow to reinstall)
	@uv run python scripts/clean.py --llvm

.PHONY: clean-cuda
clean-cuda:  ## Clean ~/.pecos/cuda/ (CUDA installation - large, slow to reinstall)
	@uv run python scripts/clean.py --cuda

.PHONY: clean-pecos-home
clean-pecos-home:  ## Clean ~/.pecos/ except LLVM and CUDA
	@uv run python scripts/clean.py --cache --deps

.PHONY: clean-all
clean-all:  ## Clean project artifacts + ~/.pecos/ (except LLVM/CUDA)
	@uv run python scripts/clean.py --cache --deps

.PHONY: clean-everything
clean-everything:  ## Nuclear option: clean everything including LLVM and CUDA
	@uv run python scripts/clean.py --all

.PHONY: pip-install-uv
pip-install-uv:  ## Install uv using pip and create a venv. (Recommended to instead follow: https://docs.astral.sh/uv/getting-started/installation/
	@echo "Installing uv..."
	$(PYTHON) -m pip install --upgrade uv
	@echo "Creating venv and installing dependencies..."
	uv sync

.PHONY: pre-check
pre-check:  ## Verify LLVM configuration before building
	@cargo run -q -p pecos-dev -- llvm check

.PHONY: dev
dev: pre-check clean build-debug test  ## Run the typical sequence of commands to check everything is running correctly

.PHONY: devl
devl: dev lint  ## Run the commands to make sure everything runs + lint

.PHONY: devc
devc: pre-check clean build-cuda test  ## Run dev sequence with CUDA support (requires CUDA Toolkit)

.PHONY: devcl
devcl: devc lint  ## Run dev sequence with CUDA support + lint (requires CUDA Toolkit)

# Help
# ----

.PHONY: help
help:  ## Show the help menu
	@echo "Available make commands:"
	@echo ""
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Note: Julia and Go support is automatically detected."
	@echo "  - 'make build-debug' will also build Julia/Go FFI if they are installed"
	@echo "  - 'make test' will also run Julia/Go tests if they are installed"
	@echo "  - 'make lint' checks code quality; 'make lint-fix' fixes issues"
	@echo "  - Use 'make julia-info' or 'make go-info' for more information"
	@echo ""
	@echo "CUDA GPU Simulator Support:"
	@echo "  - 'make install-cuda' downloads CUDA Toolkit to ~/.pecos/cuda/"
	@echo "  - 'make check-cuda' shows CUDA installation status"
	@echo "  - 'make build-cuda' builds with CUDA GPU simulator support"
	@echo "  - 'make devc' runs full dev cycle with CUDA support"
	@echo "  - No GPU hardware needed - CUDA is for compile-time only"
