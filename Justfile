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

# PECOS CLI - must be installed (run 'just install-cli' first)
pecos := "pecos"

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
# Automatically includes cuda group if CUDA packages were previously installed
installreqs:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Installing requirements..."
    if python -c "import cupy" >/dev/null 2>&1; then
        echo "(including CUDA packages)"
        uv sync --project . --all-packages --group cuda
    else
        uv sync --project . --all-packages
    fi

# Install requirements with specific Python version
installreqs-python version:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Installing requirements with Python {{version}}..."
    if python -c "import cupy" >/dev/null 2>&1; then
        echo "(including CUDA packages)"
        uv sync --project . --all-packages --python "{{version}}" --group cuda
    else
        uv sync --project . --all-packages --python "{{version}}"
    fi

# =============================================================================
# PECOS CLI
# =============================================================================

# Check if PECOS CLI is installed, fail with helpful message if not
[private]
check-cli:
    #!/usr/bin/env bash
    if ! command -v pecos >/dev/null 2>&1; then
        echo ""
        echo "Error: PECOS CLI not found."
        echo ""
        echo "Install it with:"
        echo "  just install-cli"
        echo ""
        echo "Or manually:"
        echo "  cargo install --path crates/pecos --features cli"
        echo ""
        exit 1
    fi

    # Check if the installed CLI might be stale
    stale=false
    reasons=()

    # Check 1: version mismatch
    expected_version=$(grep '^version = ' Cargo.toml | head -1 | sed 's/version = "\(.*\)"/\1/')
    installed_version=$(pecos --version 2>/dev/null | awk '{print $2}')
    if [[ "$installed_version" != "$expected_version" ]]; then
        stale=true
        reasons+=("Version mismatch (installed: ${installed_version:-unknown}, expected: $expected_version)")
    fi

    # Check 2: missing expected subcommands
    if ! pecos rust --help >/dev/null 2>&1; then
        stale=true
        reasons+=("Missing 'rust' subcommand")
    fi

    if [[ "$stale" == "true" ]]; then
        echo ""
        echo "Warning: PECOS CLI may be outdated."
        for reason in "${reasons[@]}"; do
            echo "  - $reason"
        done
        echo ""
        echo "  Update with: just reinstall-cli"
        echo ""
    fi

    # Informational: suggest CUDA Python packages if toolkit available but cupy isn't
    if pecos cuda check -q >/dev/null 2>&1; then
        if ! python -c "import cupy" >/dev/null 2>&1; then
            echo ""
            echo "Note: CUDA toolkit detected but Python CUDA packages not installed."
            echo "      To enable GPU-accelerated simulations:"
            echo "        pecos cuda setup-python"
            echo "      Or manually:"
            echo "        uv sync --group cuda"
            echo ""
        fi
    fi

# Install PECOS CLI (required for most recipes)
install-cli:
    @echo "Installing PECOS CLI..."
    cargo install --path crates/pecos --features cli
    @echo ""
    @echo "Done! You can now run: just dev"

# Reinstall PECOS CLI (run after changing CLI code)
reinstall-cli:
    @echo "Reinstalling PECOS CLI..."
    cargo install --path crates/pecos --features cli --force
    @echo ""
    @echo "Done!"

# =============================================================================
# LLVM Setup
# =============================================================================

# Install LLVM 14 to ~/.pecos/llvm/ (required for QIR features)
install-llvm:
    @echo "Installing LLVM 14..."
    {{pecos}} install llvm

# Check LLVM 14 installation status
check-llvm:
    -{{pecos}} llvm check

# Configure LLVM paths in .cargo/config.toml
configure-llvm:
    {{pecos}} llvm configure

# =============================================================================
# CUDA Setup
# =============================================================================

# Install CUDA Toolkit to ~/.pecos/cuda/ (for GPU support, no GPU needed)
install-cuda:
    @echo "Installing CUDA Toolkit..."
    {{pecos}} install cuda

# Check CUDA installation status (local or system)
check-cuda:
    -{{pecos}} cuda check

# Validate CUDA installation integrity
validate-cuda:
    {{pecos}} cuda validate

# Install CUDA Python packages (cupy, cuquantum, pytket-cutensornet)
# Requires CUDA toolkit to be installed first (just install-cuda or system CUDA)
install-cuda-python:
    {{pecos}} cuda setup-python

# Full CUDA setup: toolkit + Python packages
setup-cuda: install-cuda install-cuda-python
    @echo "Full CUDA setup complete (toolkit + Python packages)"

# =============================================================================
# Building
# =============================================================================

# Build PECOS (profile: debug, release, native)
build profile="debug": check-cli installreqs build-selene
    {{pecos}} python build --profile {{profile}}
    # Build FFI crates if tools available (- prefix ignores errors)
    -{{pecos}} julia build --profile {{profile}}
    -{{pecos}} go build --profile {{profile}}

# Build and install Selene plugins for development
build-selene:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Building Selene plugins..."

    # Discover and build all selene plugins
    CARGO_ARGS=""
    for DIR in python/selene-plugins/pecos-selene-*/; do
        CARGO_ARGS="$CARGO_ARGS -p $(basename "$DIR")"
    done
    if [ -n "$CARGO_ARGS" ]; then
        cargo build --release $CARGO_ARGS
    fi

    # Copy libraries to Python package directories
    echo "Copying libraries to Python packages..."
    {{pecos}} selene install --profile release

    # Selene plugins are workspace members, so uv sync --all-packages handles editable installs
    echo "Selene plugins built and installed successfully"

# Build PECOS with CUDA support
build-cuda profile="debug": installreqs
    {{pecos}} python build --profile {{profile}} --cuda
    # Build FFI crates if tools available (- prefix ignores errors)
    -{{pecos}} julia build --profile {{profile}}
    -{{pecos}} go build --profile {{profile}}

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
docs port="8000": check-cli
    {{pecos}} docs --port {{port}}

# Test Python code examples in documentation (excludes slow tests and Rust tests)
docs-test:
    uv run python scripts/docs/generate_doc_tests.py
    uv run pytest python/quantum-pecos/tests/docs/generated -v -k "not rust" -m "not slow"

# Test all Python code examples including slow tests (transversal CNOT - takes >2 hours)
docs-test-slow:
    uv run python scripts/docs/generate_doc_tests.py
    uv run pytest python/quantum-pecos/tests/docs/generated -v -k "not rust"

# Generate doc tests without running them
docs-test-generate:
    uv run python scripts/docs/generate_doc_tests.py

# Run doc tests with pytest options (e.g., just docs-test-run "-k bell_state")
docs-test-run *args:
    uv run pytest python/quantum-pecos/tests/docs/generated {{args}}

# Legacy: test code examples with old script
docs-test-legacy:
    uv run python scripts/docs/test_code_examples.py

# =============================================================================
# Linting / Formatting
# =============================================================================

# Run cargo check (with GPU features only if CUDA available)
check: check-cli
    {{pecos}} rust check --include-ffi

# Run cargo clippy (with GPU features only if CUDA available)
clippy: check-cli
    @echo "==> Running clippy via pecos..."
    {{pecos}} rust clippy --include-ffi

# Check Rust formatting (without fixing)
fmt: check-cli
    @echo "==> Running fmt check via pecos..."
    {{pecos}} rust fmt --check

# Fix Rust formatting issues
fmt-fix: check-cli
    {{pecos}} rust fmt

# Run all quality checks / linting (check only)
lint: fmt clippy
    #!/usr/bin/env bash
    set -euo pipefail
    echo "==> Running pre-commit..."
    uv run pre-commit run --all-files

    if {{pecos}} julia check -q >/dev/null 2>&1; then
        echo "Julia detected, running Julia formatting check and linting..."
        {{pecos}} julia fmt --check
        {{pecos}} julia lint
    else
        echo "Julia not detected, skipping Julia linting"
    fi

    if {{pecos}} go check -q >/dev/null 2>&1; then
        echo "Go detected, running Go formatting check and linting..."
        {{pecos}} go fmt --check
        {{pecos}} go lint
    else
        echo "Go not detected, skipping Go linting"
    fi

# Fix all auto-fixable linting issues (Rust, Python, Julia, Go)
lint-fix:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Fixing Rust formatting and clippy issues..."
    {{pecos}} rust fmt
    {{pecos}} rust clippy --fix --include-ffi
    echo ""
    echo "Running pre-commit fixes..."
    uv run pre-commit run --all-files || true
    echo ""

    if {{pecos}} julia check -q >/dev/null 2>&1; then
        echo "Fixing Julia formatting..."
        {{pecos}} julia fmt
        echo ""
        echo "Note: Some Julia linting issues from Aqua.jl may require manual fixes."
    else
        echo "Julia not detected, skipping Julia formatting"
    fi

    if {{pecos}} go check -q >/dev/null 2>&1; then
        echo "Fixing Go formatting..."
        {{pecos}} go fmt
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

# Run Rust tests.
# Includes pecos-gpu-sims when the GPU probe succeeds.
rstest: check-cli
    {{pecos}} rust test --release

# Run Rust tests in the default cargo profile.
# Includes pecos-gpu-sims when the GPU probe succeeds.
rstest-all: check-cli
    {{pecos}} rust test

# Run Python tests (excluding numpy and optional deps)
pytest: check-cli
    {{pecos}} python test

# Run NumPy/SciPy compatibility tests
pytest-numpy: check-cli
    {{pecos}} python test --numpy

# Run performance tests with release build
pytest-perf: build-release
    @echo "Running pecos-rslib performance tests with release build..."
    uv run --group numpy-compat pytest ./python/pecos-rslib/tests/ -m "performance" -v

# Run tests for optional dependencies
pytest-dep: check-cli
    {{pecos}} python test -m optional_dependency

# Run Selene plugin tests
pytest-selene: check-cli
    {{pecos}} python test --selene

# Run all Python tests (core + numpy compat + selene)
pytest-all: pytest pytest-numpy pytest-selene
    @echo "All Python tests completed (core + NumPy/SciPy compatibility + Selene plugins)"

# Run all tests (Rust + Python + Julia + Go if available)
test: rstest-all pytest-all
    #!/usr/bin/env bash
    set -euo pipefail
    if {{pecos}} julia check -q >/dev/null 2>&1; then
        echo "Julia detected, running Julia tests..."
        {{pecos}} julia test
    else
        echo "Julia not detected, skipping Julia tests"
    fi

    if {{pecos}} go check -q >/dev/null 2>&1; then
        echo "Go detected, running Go tests..."
        {{pecos}} go test
    else
        echo "Go not detected, skipping Go tests"
    fi

# Run all tests with warnings for missing tools
test-all: rstest-all pytest-all
    #!/usr/bin/env bash
    set -euo pipefail
    if {{pecos}} julia check -q >/dev/null 2>&1; then
        echo "Julia detected, running Julia tests..."
        {{pecos}} julia test
    else
        echo ""
        echo "WARNING: Julia is not installed. Skipping Julia tests."
        echo "   To run Julia tests, please install Julia from https://julialang.org/downloads/"
        echo ""
    fi

    if {{pecos}} go check -q >/dev/null 2>&1; then
        echo "Go detected, running Go tests..."
        {{pecos}} go test
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
    {{pecos}} list -v

# Clean decoder download cache (same as clean-cache)
decoder-cache-clean: clean-cache
    @echo "Decoder cache cleaned (part of ~/.pecos/cache/)"

# =============================================================================
# Julia Bindings
# =============================================================================

# Build Julia FFI library
julia-build profile="release":
    {{pecos}} julia build --profile {{profile}}

# Build Julia FFI library in debug mode
julia-build-debug:
    {{pecos}} julia build --profile debug

# Run Julia tests (requires Julia installed)
julia-test:
    {{pecos}} julia test

# Run Julia examples
julia-examples: julia-build-debug
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Running Julia examples..."
    if {{pecos}} julia check -q >/dev/null 2>&1; then
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
    {{pecos}} julia fmt

# Check Julia code formatting
julia-format-check:
    {{pecos}} julia fmt --check

# Run Aqua.jl quality checks on Julia code
julia-lint:
    {{pecos}} julia lint

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
    {{pecos}} go build --profile {{profile}}

# Build Go FFI library in debug mode
go-build-debug:
    {{pecos}} go build --profile debug

# Run Go tests (requires Go installed)
go-test:
    {{pecos}} go test

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
    {{pecos}} go fmt

# Check Go code formatting
go-fmt-check:
    {{pecos}} go fmt --check

# Run Go linting with go vet
go-lint:
    {{pecos}} go lint

# Clean Go build artifacts
go-clean:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Cleaning Go artifacts..."
    rm -f go/pecos/go.sum || true
    if {{pecos}} go check -q >/dev/null 2>&1; then
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
pre-check: check-cli
    {{pecos}} llvm check

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
        {{pecos}} python build --profile debug --cuda
    else
        {{pecos}} python build --profile debug
    fi
    # Build FFI crates if tools available
    {{pecos}} julia build --profile debug 2>/dev/null || true
    {{pecos}} go build --profile debug 2>/dev/null || true

# Install uv using pip (prefer: https://docs.astral.sh/uv/getting-started/installation/)
pip-install-uv:
    @echo "Installing uv..."
    python -m pip install --upgrade uv
    @echo "Creating venv and installing dependencies..."
    uv sync

# Show system information
sys-info:
    {{pecos}} sys-info

# List installed and cached dependencies
list-deps:
    {{pecos}} list -v
