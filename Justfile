# PECOS Development Justfile
# Cross-platform command runner (Windows, macOS, Linux)
# Install: cargo install just
# Usage: just <recipe> or just --list

# Default recipe: show help
default:
    @just --list

# =============================================================================
# Settings
# =============================================================================

# Use bash by default (Windows users should use Git Bash, WSL, or PowerShell recipes)
set shell := ["bash", "-cu"]
set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

# =============================================================================
# Requirements
# =============================================================================

# Generate/update lockfiles
updatereqs:
    @echo "Ensuring uv is installed..."
    uv self update
    @echo "Generating lock files..."
    uv lock --project .

# Install Python project requirements to root .venv
installreqs:
    @echo "Installing requirements..."
    uv sync --project .

# Install requirements with specific Python version
installreqs-python version:
    @echo "Installing requirements with Python {{version}}..."
    uv sync --project . --python "{{version}}"

# =============================================================================
# LLVM Setup
# =============================================================================

# Install LLVM 14 to ~/.pecos/llvm/ (required for QIR features)
install-llvm:
    @echo "Installing LLVM 14..."
    cargo run --release -p pecos --features cli -- llvm install

# Check LLVM 14 installation status
check-llvm:
    -cargo run --release -p pecos --features cli -- llvm check

# Configure LLVM paths in .cargo/config.toml
configure-llvm:
    cargo run --release -p pecos --features cli -- llvm configure

# =============================================================================
# CUDA Setup
# =============================================================================

# Install CUDA Toolkit to ~/.pecos/cuda/ (for GPU support, no GPU needed)
install-cuda:
    @echo "Installing CUDA Toolkit..."
    cargo run -p pecos --features cli -- cuda install

# Check CUDA installation status (local or system)
check-cuda:
    -cargo run -p pecos --features cli -- cuda check

# Validate CUDA installation integrity
validate-cuda:
    cargo run -p pecos --features cli -- cuda validate

# =============================================================================
# Building
# =============================================================================

# Build PECOS (profile: debug, release, native)
build profile="debug": installreqs build-selene
    cargo run -p pecos --features cli -- python build --profile {{profile}}
    # Build FFI crates if tools available
    cargo run -p pecos --features cli -- julia build --profile {{profile}} 2>/dev/null || true
    cargo run -p pecos --features cli -- go build --profile {{profile}} 2>/dev/null || true

# Build and install Selene plugins for development
build-selene:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Building Selene plugins..."

    # Build Rust libraries (with GPU support if CUDA available)
    if cargo run -p pecos --features cli -- cuda check -q >/dev/null 2>&1; then
        echo "CUDA detected, building with GPU support..."
        cargo build --release -p pecos-selene-quest --features cuda
    else
        echo "CUDA not detected, building CPU-only..."
        cargo build --release -p pecos-selene-quest
    fi

    cargo build --release -p pecos-selene-qulacs -p pecos-selene-sparsestab -p pecos-selene-statevec

    # Copy libraries to Python package directories
    echo "Copying libraries to Python packages..."
    cargo run -p pecos --features cli -- selene install

    # Install Python packages in editable mode
    echo "Installing Selene plugins in editable mode..."
    unset CONDA_PREFIX 2>/dev/null || true
    uv pip install -e ./python/selene-plugins/pecos-selene-quest
    uv pip install -e ./python/selene-plugins/pecos-selene-qulacs
    uv pip install -e ./python/selene-plugins/pecos-selene-sparsestab
    uv pip install -e ./python/selene-plugins/pecos-selene-statevec
    echo "Selene plugins built and installed successfully"

# Build PECOS with CUDA support
build-cuda profile="debug": installreqs
    cargo run -p pecos --features cli -- python build --profile {{profile}} --cuda
    # Build FFI crates if tools available
    cargo run -p pecos --features cli -- julia build --profile {{profile}} 2>/dev/null || true
    cargo run -p pecos --features cli -- go build --profile {{profile}} 2>/dev/null || true

# Convenience aliases
build-debug: (build "debug")
build-release: (build "release")
build-native: (build "native")
build-cuda-debug: (build-cuda "debug")
build-cuda-release: (build-cuda "release")
build-cuda-native: (build-cuda "native")

# =============================================================================
# Documentation
# =============================================================================

# Build documentation
docs-build:
    uv run mkdocs build --clean

# Serve documentation and open in browser
docs port="8000":
    cargo run -p pecos --features cli -- docs --port {{port}}

# Test all code examples in documentation
docs-test:
    uv run python scripts/docs/test_code_examples.py

# Test only working code examples in documentation
docs-test-working:
    uv run python scripts/docs/test_working_examples.py

# =============================================================================
# Linting / Formatting
# =============================================================================

# Run cargo check (with GPU features only if CUDA available)
check:
    cargo run -p pecos --features cli -- rust check --include-ffi

# Run cargo clippy (with GPU features only if CUDA available)
clippy:
    @echo "==> Running clippy via pecos..."
    cargo run -p pecos --features cli -- rust clippy --include-ffi

# Check Rust formatting (without fixing)
fmt:
    @echo "==> Running fmt check via pecos..."
    cargo run -p pecos --features cli -- rust fmt --check

# Fix Rust formatting issues
fmt-fix:
    cargo run -p pecos --features cli -- rust fmt

# Run all quality checks / linting (check only)
lint: fmt clippy
    #!/usr/bin/env bash
    set -euo pipefail
    echo "==> Running pre-commit..."
    uv run pre-commit run --all-files

    if cargo run -p pecos --features cli -- julia check -q >/dev/null 2>&1; then
        echo "Julia detected, running Julia formatting check and linting..."
        cargo run -p pecos --features cli -- julia fmt --check
        cargo run -p pecos --features cli -- julia lint
    else
        echo "Julia not detected, skipping Julia linting"
    fi

    if cargo run -p pecos --features cli -- go check -q >/dev/null 2>&1; then
        echo "Go detected, running Go formatting check and linting..."
        cargo run -p pecos --features cli -- go fmt --check
        cargo run -p pecos --features cli -- go lint
    else
        echo "Go not detected, skipping Go linting"
    fi

# Fix all auto-fixable linting issues (Rust, Python, Julia, Go)
lint-fix:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Fixing Rust formatting and clippy issues..."
    cargo run -p pecos --features cli -- rust fmt
    cargo run -p pecos --features cli -- rust clippy --fix --include-ffi
    echo ""
    echo "Running pre-commit fixes..."
    uv run pre-commit run --all-files || true
    echo ""

    if cargo run -p pecos --features cli -- julia check -q >/dev/null 2>&1; then
        echo "Fixing Julia formatting..."
        cargo run -p pecos --features cli -- julia fmt
        echo ""
        echo "Note: Some Julia linting issues from Aqua.jl may require manual fixes."
    else
        echo "Julia not detected, skipping Julia formatting"
    fi

    if cargo run -p pecos --features cli -- go check -q >/dev/null 2>&1; then
        echo "Fixing Go formatting..."
        cargo run -p pecos --features cli -- go fmt
    else
        echo "Go not detected, skipping Go formatting"
    fi
    echo ""
    echo "Linting fixes applied! Run 'just lint' to check for remaining issues."

# Normalize line endings according to .gitattributes
normalize-line-endings:
    @echo "Normalizing line endings according to .gitattributes..."
    @echo "This will refresh all tracked files to apply .gitattributes rules"
    -git rm --cached -r .
    git reset --hard
    @echo "Line endings normalized. Check 'git status' for any changes."

# =============================================================================
# Testing
# =============================================================================

# Run Rust tests (with GPU features only if CUDA available)
rstest:
    cargo run -p pecos --features cli -- rust test --release

# Run Rust tests with all features
rstest-all:
    cargo run -p pecos --features cli -- rust test

# Run Python tests (excluding numpy and optional deps)
pytest:
    cargo run -p pecos --features cli -- python test

# Run NumPy/SciPy compatibility tests
pytest-numpy:
    cargo run -p pecos --features cli -- python test --numpy

# Run performance tests with release build
pytest-perf: build-release
    @echo "Running pecos-rslib performance tests with release build..."
    uv run --group numpy-compat pytest ./python/pecos-rslib/tests/ -m "performance" -v

# Run tests for optional dependencies
pytest-dep:
    cargo run -p pecos --features cli -- python test -m optional_dependency

# Run Selene plugin tests
pytest-selene:
    cargo run -p pecos --features cli -- python test --selene

# Run all Python tests (core + numpy compat + selene)
pytest-all: pytest pytest-numpy pytest-selene
    @echo "All Python tests completed (core + NumPy/SciPy compatibility + Selene plugins)"

# Run all tests (Rust + Python + Julia + Go if available)
test: rstest-all pytest-all
    #!/usr/bin/env bash
    set -euo pipefail
    if cargo run -p pecos --features cli -- julia check -q >/dev/null 2>&1; then
        echo "Julia detected, running Julia tests..."
        cargo run -p pecos --features cli -- julia test
    else
        echo "Julia not detected, skipping Julia tests"
    fi

    if cargo run -p pecos --features cli -- go check -q >/dev/null 2>&1; then
        echo "Go detected, running Go tests..."
        cargo run -p pecos --features cli -- go test
    else
        echo "Go not detected, skipping Go tests"
    fi

# Run all tests with warnings for missing tools
test-all: rstest-all pytest-all
    #!/usr/bin/env bash
    set -euo pipefail
    if cargo run -p pecos --features cli -- julia check -q >/dev/null 2>&1; then
        echo "Julia detected, running Julia tests..."
        cargo run -p pecos --features cli -- julia test
    else
        echo ""
        echo "WARNING: Julia is not installed. Skipping Julia tests."
        echo "   To run Julia tests, please install Julia from https://julialang.org/downloads/"
        echo ""
    fi

    if cargo run -p pecos --features cli -- go check -q >/dev/null 2>&1; then
        echo "Go detected, running Go tests..."
        cargo run -p pecos --features cli -- go test
    else
        echo ""
        echo "WARNING: Go is not installed. Skipping Go tests."
        echo "   To run Go tests, please install Go from https://go.dev/dl/"
        echo ""
    fi

# =============================================================================
# Decoders
# =============================================================================

# Build all decoder crates with all features
build-decoders:
    cargo build --package pecos-decoders --all-features

# Build specific decoder (e.g., just build-decoder ldpc)
build-decoder decoder:
    cargo build --package pecos-decoders --features {{decoder}}

# Test all decoder crates
test-decoders:
    cargo test --package pecos-decoders --all-features

# Test specific decoder
test-decoder decoder:
    cargo test --package pecos-decoders --features {{decoder}}

# Show available decoders and their features
decoder-info:
    @echo "Available decoders in PECOS:"
    @echo "  - ldpc: LDPC decoders (BP-OSD, MBP, etc.)"
    @echo ""
    @echo "To build specific decoder: just build-decoder ldpc"
    @echo "To build all decoders:     just build-decoders"
    @echo "See DECODERS.md for detailed documentation."

# Show decoder download cache status
decoder-cache-status:
    cargo run -p pecos --features cli -- list -v

# Clean decoder download cache (same as clean-cache)
decoder-cache-clean: clean-cache
    @echo "Decoder cache cleaned (part of ~/.pecos/cache/)"

# =============================================================================
# Julia Bindings
# =============================================================================

# Build Julia FFI library
julia-build profile="release":
    cargo run -p pecos --features cli -- julia build --profile {{profile}}

# Build Julia FFI library in debug mode
julia-build-debug:
    cargo run -p pecos --features cli -- julia build --profile debug

# Run Julia tests (requires Julia installed)
julia-test:
    cargo run -p pecos --features cli -- julia test

# Run Julia examples
julia-examples: julia-build-debug
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Running Julia examples..."
    if cargo run -p pecos --features cli -- julia check -q >/dev/null 2>&1; then
        cd julia/PECOS.jl && julia --project=. examples/demo.jl
        cd julia/PECOS.jl && julia --project=. examples/basic_usage.jl
    else
        echo "Julia not found. Please install Julia to run examples."
        exit 1
    fi

# Show Julia package information
julia-info:
    @echo "Julia Package Information:"
    @echo "========================="
    @echo "Package name: PECOS.jl"
    @echo "Location: julia/PECOS.jl"
    @echo "FFI library: julia/pecos-julia-ffi"
    @echo ""
    @echo "To install for development:"
    @echo "  1. Build FFI library: pecos julia build"
    @echo "  2. In Julia REPL: ] add julia/PECOS.jl"
    @echo ""
    @echo "To run tests: pecos julia test"
    @echo "To run examples: just julia-examples"

# Format Julia code
julia-format:
    cargo run -p pecos --features cli -- julia fmt

# Check Julia code formatting
julia-format-check:
    cargo run -p pecos --features cli -- julia fmt --check

# Run Aqua.jl quality checks on Julia code
julia-lint:
    cargo run -p pecos --features cli -- julia lint

# Clean Julia build artifacts
julia-clean:
    @echo "Cleaning Julia artifacts..."
    rm -f julia/PECOS.jl/Manifest.toml || true
    rm -f julia/PECOS.jl/dev/PECOS_julia_jll/Manifest.toml || true
    find julia -name "*.jl.*.cov" -delete 2>/dev/null || true
    find julia -name "*.jl.cov" -delete 2>/dev/null || true
    find julia -name "*.jl.mem" -delete 2>/dev/null || true

# =============================================================================
# Go Bindings
# =============================================================================

# Build Go FFI library
go-build profile="release":
    cargo run -p pecos --features cli -- go build --profile {{profile}}

# Build Go FFI library in debug mode
go-build-debug:
    cargo run -p pecos --features cli -- go build --profile debug

# Run Go tests (requires Go installed)
go-test:
    cargo run -p pecos --features cli -- go test

# Show Go package information
go-info:
    @echo "Go Package Information:"
    @echo "======================="
    @echo "Package name: github.com/PECOS-packages/PECOS/go/pecos"
    @echo "Location: go/pecos"
    @echo "FFI library: go/pecos-go-ffi"
    @echo ""
    @echo "To build and test:"
    @echo "  1. Build FFI library: pecos go build"
    @echo "  2. Run tests: pecos go test"
    @echo ""
    @echo "To use in your Go project:"
    @echo "  1. Set LD_LIBRARY_PATH to include target/release"
    @echo "  2. Import: github.com/PECOS-packages/PECOS/go/pecos"

# Format Go code
go-fmt:
    cargo run -p pecos --features cli -- go fmt

# Check Go code formatting
go-fmt-check:
    cargo run -p pecos --features cli -- go fmt --check

# Run Go linting with go vet
go-lint:
    cargo run -p pecos --features cli -- go lint

# Clean Go build artifacts
go-clean:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Cleaning Go artifacts..."
    rm -f go/pecos/go.sum || true
    if cargo run -p pecos --features cli -- go check -q >/dev/null 2>&1; then
        cd go/pecos && go clean -cache 2>/dev/null || true
    fi

# =============================================================================
# Cleaning (Cross-platform via Python script)
# =============================================================================

# Clean build artifacts (cross-platform)
clean:
    uv run python scripts/clean.py

# Clean Selene plugin build artifacts
clean-selene:
    uv run python scripts/clean.py --selene

# Clean ~/.pecos/cache/ and ~/.pecos/tmp/
clean-cache:
    uv run python scripts/clean.py --cache

# Clean ~/.pecos/deps/ (extracted C++ dependencies)
clean-deps:
    uv run python scripts/clean.py --deps

# Clean ~/.pecos/llvm/ (WARNING: slow to reinstall)
clean-llvm:
    uv run python scripts/clean.py --llvm

# Clean ~/.pecos/cuda/ (WARNING: slow to reinstall)
clean-cuda:
    uv run python scripts/clean.py --cuda

# Clean ~/.pecos/ except LLVM and CUDA
clean-pecos-home:
    uv run python scripts/clean.py --cache --deps

# Clean project artifacts + ~/.pecos/ (except LLVM/CUDA)
clean-all:
    uv run python scripts/clean.py --cache --deps

# Nuclear option: clean everything including LLVM and CUDA
clean-everything:
    uv run python scripts/clean.py --all

# Preview what would be cleaned (dry run)
clean-dry-run:
    uv run python scripts/clean.py --dry-run

# =============================================================================
# Development Workflows
# =============================================================================

# Verify LLVM configuration before building
pre-check:
    cargo run --release -p pecos --features cli -- llvm check

# Dev cycle: incremental build + test (fast, for normal development)
dev cuda="false": pre-check (build-dev cuda) test

# Dev cycle with CUDA support
devc: (dev "true")

# Full dev cycle: clean build + test + lint (pre-merge)
dev-full cuda="false": pre-check clean (build-dev cuda) test lint

# Full dev cycle with CUDA support
devc-full: (dev-full "true")

# Internal: build for dev cycle with optional CUDA
[private]
build-dev cuda="false": installreqs build-selene
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ "{{cuda}}" == "true" ]]; then
        cargo run -p pecos --features cli -- python build --profile debug --cuda
    else
        cargo run -p pecos --features cli -- python build --profile debug
    fi
    # Build FFI crates if tools available
    cargo run -p pecos --features cli -- julia build --profile debug 2>/dev/null || true
    cargo run -p pecos --features cli -- go build --profile debug 2>/dev/null || true

# Install uv using pip (prefer: https://docs.astral.sh/uv/getting-started/installation/)
pip-install-uv:
    @echo "Installing uv..."
    python -m pip install --upgrade uv
    @echo "Creating venv and installing dependencies..."
    uv sync

# Show system information
sys-info:
    cargo run -p pecos --features cli -- sys-info

# List installed and cached dependencies
list-deps:
    cargo run -p pecos --features cli -- list -v
