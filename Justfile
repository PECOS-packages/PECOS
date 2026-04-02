# PECOS Development Justfile
# Cross-platform command runner (Windows, macOS, Linux)
# Install: cargo install just
# Usage: just <recipe> or just --list

# Default recipe: show quick-start guide + recipe list
default:
    @echo "PECOS Development"
    @echo "================="
    @echo ""
    @echo "Quick start:"
    @echo "  just install-cli    # First time: install the pecos CLI"
    @echo "  just build          # Build PECOS (auto-installs dependencies)"
    @echo "  just test           # Run all tests"
    @echo "  just dev            # Build + test (daily workflow)"
    @echo "  just lint           # Check formatting and linting"
    @echo ""
    @echo "All commands:"
    @just --list --list-heading ''

# =============================================================================
# Settings
# =============================================================================

# Use bash by default (Windows users should use Git Bash, WSL, or PowerShell recipes)
set shell := ["bash", "-cu"]
set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

# PECOS CLI - must be installed (run 'just install-cli' first)
pecos := "pecos"

# =============================================================================
# Getting Started
# =============================================================================

# Install PECOS CLI (required for most recipes)
[group('setup')]
install-cli:
    @echo "Installing PECOS CLI..."
    cargo install --path crates/pecos --features cli
    @echo ""
    @echo "Done! You can now run: just build"

# Reinstall PECOS CLI (run after changing CLI code)
[group('setup')]
reinstall-cli:
    cargo install --path crates/pecos --features cli --force

# Set up build environment (detect and install missing dependencies)
[group('setup')]
setup: check-cli
    {{pecos}} setup

# Set up build environment, accepting all prompts (for CI)
[group('setup')]
setup-ci: check-cli
    {{pecos}} setup --yes

# Show system information
[group('setup')]
sys-info: check-cli
    {{pecos}} sys-info

# List installed and cached dependencies
[group('setup')]
list-deps: check-cli
    {{pecos}} list -v

# =============================================================================
# Building
# =============================================================================

# Build PECOS (profile: debug, release, native)
[group('build')]
build profile="debug": check-cli setup-quiet installreqs build-selene
    {{pecos}} python build --profile {{profile}}
    -{{pecos}} julia build --profile {{profile}}
    -{{pecos}} go build --profile {{profile}}

# Build PECOS without dependency setup prompts
[group('build')]
build-lite profile="debug": check-cli installreqs build-selene
    {{pecos}} python build --profile {{profile}}
    -{{pecos}} julia build --profile {{profile}}
    -{{pecos}} go build --profile {{profile}}

# Build PECOS with CUDA support
[group('build')]
build-cuda profile="debug": check-cli setup-quiet installreqs
    {{pecos}} python build --profile {{profile}} --cuda
    -{{pecos}} julia build --profile {{profile}}
    -{{pecos}} go build --profile {{profile}}

# =============================================================================
# Testing
# =============================================================================

# Run Python tests (core)
[group('test')]
pytest:
    uv run pytest python/pecos-rslib/tests -m "not performance and not numpy"
    uv run pytest python/quantum-pecos/tests -m "not optional_dependency and not numpy"

# Run Rust tests (CUDA-aware, release mode)
[group('test')]
rstest: check-cli
    {{pecos}} rust test --release

# Run all tests (Rust + Python + Julia + Go if available)
[group('test')]
test: rstest-all pytest-all
    #!/usr/bin/env bash
    set -euo pipefail
    if command -v julia >/dev/null 2>&1; then
        echo "Julia detected, running Julia tests..."
        {{pecos}} julia test
    else
        echo "Julia not detected, skipping Julia tests"
    fi
    if command -v go >/dev/null 2>&1; then
        echo "Go detected, running Go tests..."
        {{pecos}} go test
    else
        echo "Go not detected, skipping Go tests"
    fi

# Run all Python tests (core + numpy compat + selene)
[group('test')]
pytest-all: pytest pytest-numpy pytest-selene
    @echo "All Python tests completed"

# =============================================================================
# Linting / Formatting
# =============================================================================

# Run all quality checks (fmt + clippy + pre-commit + Julia/Go if available)
[group('lint')]
lint: fmt clippy
    #!/usr/bin/env bash
    set -euo pipefail
    echo "==> Running pre-commit..."
    uv run pre-commit run --all-files

    if command -v julia >/dev/null 2>&1; then
        echo "Julia detected, running Julia formatting check and linting..."
        {{pecos}} julia fmt --check
        {{pecos}} julia lint
    else
        echo "Julia not detected, skipping Julia linting"
    fi

    if command -v go >/dev/null 2>&1; then
        echo "Go detected, running Go formatting check and linting..."
        test -z "$(gofmt -l go/pecos)" || (gofmt -l go/pecos && exit 1)
        cd go/pecos && go vet ./...
    else
        echo "Go not detected, skipping Go linting"
    fi

# Fix all auto-fixable linting issues
[group('lint')]
lint-fix:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Fixing Rust formatting and clippy issues..."
    cargo fmt --all
    cargo clippy --workspace --all-targets --fix --allow-staged --allow-dirty -- -D warnings
    echo ""
    echo "Running pre-commit fixes..."
    uv run pre-commit run --all-files || true
    echo ""
    if command -v julia >/dev/null 2>&1; then
        echo "Fixing Julia formatting..."
        {{pecos}} julia fmt
    else
        echo "Julia not detected, skipping Julia formatting"
    fi
    if command -v go >/dev/null 2>&1; then
        echo "Fixing Go formatting..."
        gofmt -w go/pecos
    else
        echo "Go not detected, skipping Go formatting"
    fi
    echo ""
    echo "Linting fixes applied! Run 'just lint' to check for remaining issues."

# Run cargo check
[group('lint')]
check:
    cargo check --workspace --all-targets

# Run cargo clippy
[group('lint')]
clippy:
    cargo clippy --workspace --all-targets -- -D warnings

# Check Rust formatting
[group('lint')]
fmt:
    cargo fmt --all -- --check

# Fix Rust formatting
[group('lint')]
fmt-fix:
    cargo fmt --all

# =============================================================================
# Dev Workflows
# =============================================================================

# Dev cycle: build + test (fast, for normal development)
[group('dev')]
dev cuda="false": pre-check (build-dev cuda) test

# Full dev cycle: clean + build + test + lint (pre-merge)
[group('dev')]
dev-full cuda="false": pre-check clean (build-dev cuda) test lint

# Dev cycle with CUDA support
[group('dev')]
devc: (dev "true")

# Full dev cycle with CUDA support
[group('dev')]
devc-full: (dev-full "true")

# Clean build artifacts
[group('dev')]
[group('clean')]
clean:
    uv run python scripts/clean.py

# =============================================================================
# Documentation
# =============================================================================

# Serve documentation and open in browser
[group('docs')]
docs port="8000": check-cli
    {{pecos}} docs --port {{port}}

# Build documentation
[group('docs')]
docs-build:
    uv run mkdocs build --clean

# Test Python code examples in documentation
[group('docs')]
docs-test:
    uv run python scripts/docs/generate_doc_tests.py
    uv run pytest python/quantum-pecos/tests/docs/generated -v -k "not rust" -m "not slow"

# =============================================================================
# Deps Management (prefer `just setup` or `pecos install <target>`)
# =============================================================================

# Install LLVM 14
[group('deps')]
install-llvm: check-cli
    {{pecos}} install llvm

# Install CUDA Toolkit
[group('deps')]
install-cuda: check-cli
    {{pecos}} install cuda

# Configure LLVM paths in .cargo/config.toml
[group('deps')]
configure-llvm: check-cli
    {{pecos}} llvm configure

# Check LLVM 14 installation status
[group('deps')]
check-llvm: check-cli
    -{{pecos}} llvm check

# Check CUDA installation status
[group('deps')]
check-cuda: check-cli
    -{{pecos}} cuda check

# =============================================================================
# Julia Bindings
# =============================================================================

# Build Julia FFI library
[group('julia')]
julia-build profile="release": check-cli
    {{pecos}} julia build --profile {{profile}}

# Run Julia tests
[group('julia')]
julia-test: check-cli
    {{pecos}} julia test

# Format Julia code
[group('julia')]
julia-format: check-cli
    {{pecos}} julia fmt

# Check Julia code formatting
[group('julia')]
julia-format-check: check-cli
    {{pecos}} julia fmt --check

# Run Aqua.jl quality checks
[group('julia')]
julia-lint: check-cli
    {{pecos}} julia lint

# =============================================================================
# Go Bindings
# =============================================================================

# Build Go FFI library
[group('go')]
go-build profile="release": check-cli
    {{pecos}} go build --profile {{profile}}

# Run Go tests
[group('go')]
go-test: check-cli
    {{pecos}} go test

# Format Go code
[group('go')]
go-fmt:
    gofmt -w go/pecos

# Check Go code formatting
[group('go')]
go-fmt-check:
    @test -z "$(gofmt -l go/pecos)" || (gofmt -l go/pecos && exit 1)

# Run Go linting with go vet
[group('go')]
go-lint:
    cd go/pecos && go vet ./...

# =============================================================================
# Decoders
# =============================================================================

# Build all decoder crates
[group('decoders')]
build-decoders:
    cargo build --package pecos-decoders --all-features

# Test all decoder crates
[group('decoders')]
test-decoders:
    cargo test --package pecos-decoders --all-features

# Build specific decoder (e.g., just build-decoder ldpc)
[group('decoders')]
build-decoder decoder:
    cargo build --package pecos-decoders --features {{decoder}}

# Test specific decoder
[group('decoders')]
test-decoder decoder:
    cargo test --package pecos-decoders --features {{decoder}}

# =============================================================================
# Additional Testing
# =============================================================================

# Run Rust tests in default cargo profile
[private]
rstest-all: check-cli
    {{pecos}} rust test

# Run NumPy/SciPy compatibility tests
[group('test')]
pytest-numpy:
    uv run --group numpy-compat pytest python/pecos-rslib/tests -m "numpy and not performance"

# Run performance tests with release build
[group('test')]
pytest-perf: build-release
    uv run --group numpy-compat pytest python/pecos-rslib/tests -m "performance" -v

# Run tests for optional dependencies
[group('test')]
pytest-dep:
    uv run pytest python/pecos-rslib/tests -m "optional_dependency"
    uv run pytest python/quantum-pecos/tests -m "optional_dependency"

# Run Selene plugin tests
[group('test')]
pytest-selene:
    uv run pytest python/selene-plugins

# Run all tests with warnings for missing tools
[group('test')]
test-all: rstest-all pytest-all
    #!/usr/bin/env bash
    set -euo pipefail
    if command -v julia >/dev/null 2>&1; then
        echo "Julia detected, running Julia tests..."
        {{pecos}} julia test
    else
        echo ""
        echo "WARNING: Julia is not installed. Skipping Julia tests."
        echo "   To run Julia tests, please install Julia from https://julialang.org/downloads/"
        echo ""
    fi
    if command -v go >/dev/null 2>&1; then
        echo "Go detected, running Go tests..."
        {{pecos}} go test
    else
        echo ""
        echo "WARNING: Go is not installed. Skipping Go tests."
        echo "   To run Go tests, please install Go from https://go.dev/dl/"
        echo ""
    fi

# =============================================================================
# Cleaning
# =============================================================================

# Clean Selene plugin build artifacts
[group('clean')]
clean-selene:
    uv run python scripts/clean.py --selene

# Clean ~/.pecos/cache/ and ~/.pecos/tmp/
[group('clean')]
clean-cache:
    uv run python scripts/clean.py --cache

# Clean ~/.pecos/deps/ (extracted C++ dependencies)
[group('clean')]
clean-deps:
    uv run python scripts/clean.py --deps

# Clean everything including LLVM and CUDA
[group('clean')]
clean-everything:
    uv run python scripts/clean.py --all

# Preview what would be cleaned
[group('clean')]
clean-dry-run:
    uv run python scripts/clean.py --dry-run

# =============================================================================
# Private / Internal Recipes
# =============================================================================

[private]
check-cli:
    #!/usr/bin/env bash
    if ! command -v pecos >/dev/null 2>&1; then
        echo "Error: PECOS CLI not found. Install with: just install-cli"
        exit 1
    fi
    expected=$(grep '^version = ' Cargo.toml | head -1 | sed 's/version = "\(.*\)"/\1/')
    installed=$(pecos --version 2>/dev/null | awk '{print $2}')
    if [[ "$installed" != "$expected" ]]; then
        echo "Warning: PECOS CLI outdated (installed: ${installed:-unknown}, expected: $expected)"
        echo "  Update with: just reinstall-cli"
    fi

[private]
setup-quiet: check-cli
    {{pecos}} setup --quiet

[private]
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

[private]
build-selene:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Building Selene plugins..."
    CARGO_ARGS=""
    for DIR in python/selene-plugins/pecos-selene-*/; do
        CARGO_ARGS="$CARGO_ARGS -p $(basename "$DIR")"
    done
    if [ -n "$CARGO_ARGS" ]; then
        cargo build --release $CARGO_ARGS
    fi
    echo "Copying libraries to Python packages..."
    {{pecos}} selene install --profile release
    echo "Selene plugins built and installed successfully"

[private]
pre-check: check-cli setup-quiet

[private]
build-dev cuda="false": installreqs build-selene
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ "{{cuda}}" == "true" ]]; then
        {{pecos}} python build --profile debug --cuda
    else
        {{pecos}} python build --profile debug
    fi
    {{pecos}} julia build --profile debug 2>/dev/null || true
    {{pecos}} go build --profile debug 2>/dev/null || true

# Convenience aliases
[private]
build-debug: (build "debug")
[private]
build-release: (build "release")
[private]
build-native: (build "native")
[private]
build-lite-debug: (build-lite "debug")
[private]
build-lite-release: (build-lite "release")
[private]
build-cuda-debug: (build-cuda "debug")
[private]
build-cuda-release: (build-cuda "release")
[private]
build-cuda-native: (build-cuda "native")

# Remaining utility recipes

# Generate/update lockfiles
[group('setup')]
updatereqs:
    uv self update
    uv lock --project .

# Install requirements with specific Python version
[private]
installreqs-python version:
    #!/usr/bin/env bash
    set -euo pipefail
    if python -c "import cupy" >/dev/null 2>&1; then
        uv sync --project . --all-packages --python "{{version}}" --group cuda
    else
        uv sync --project . --all-packages --python "{{version}}"
    fi

# Install uv using pip
[group('setup')]
pip-install-uv:
    python -m pip install --upgrade uv
    uv sync

# Normalize line endings according to .gitattributes
[private]
normalize-line-endings:
    -git rm --cached -r .
    git reset --hard

# Install CUDA Python packages (requires CUDA toolkit)
[private]
install-cuda-python:
    {{pecos}} cuda setup-python

# Full CUDA setup: toolkit + Python packages
[private]
install-cuda-full: install-cuda install-cuda-python

# Validate CUDA installation integrity
[private]
validate-cuda:
    {{pecos}} cuda validate

# Docs extras
[private]
docs-test-slow:
    uv run python scripts/docs/generate_doc_tests.py
    uv run pytest python/quantum-pecos/tests/docs/generated -v -k "not rust"

[private]
docs-test-generate:
    uv run python scripts/docs/generate_doc_tests.py

[private]
docs-test-run *args:
    uv run pytest python/quantum-pecos/tests/docs/generated {{args}}

[private]
docs-test-legacy:
    uv run python scripts/docs/test_code_examples.py

# Julia/Go extras
[private]
julia-build-debug:
    {{pecos}} julia build --profile debug

[private]
julia-examples: julia-build-debug
    #!/usr/bin/env bash
    set -euo pipefail
    if command -v julia >/dev/null 2>&1; then
        cd julia/PECOS.jl && julia --project=. examples/demo.jl
        cd julia/PECOS.jl && julia --project=. examples/basic_usage.jl
    else
        echo "Julia not found."; exit 1
    fi

[private]
julia-info:
    @echo "Julia: julia/PECOS.jl | FFI: julia/pecos-julia-ffi"

[private]
julia-clean:
    rm -f julia/PECOS.jl/Manifest.toml julia/PECOS.jl/dev/PECOS_julia_jll/Manifest.toml || true

[private]
go-build-debug:
    {{pecos}} go build --profile debug

[private]
go-info:
    @echo "Go: go/pecos | FFI: go/pecos-go-ffi"

[private]
go-clean:
    rm -f go/pecos/go.sum || true

[private]
decoder-info:
    @echo "Decoders: ldpc (BP-OSD, MBP). See DECODERS.md"

[private]
decoder-cache-status:
    {{pecos}} list -v

[private]
decoder-cache-clean: clean-cache

[private]
clean-llvm:
    uv run python scripts/clean.py --llvm

[private]
clean-cuda:
    uv run python scripts/clean.py --cuda

[private]
clean-pecos-home:
    uv run python scripts/clean.py --cache --deps

[private]
clean-all:
    uv run python scripts/clean.py --cache --deps
