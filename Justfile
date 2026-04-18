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
    @echo "  just install-cli    # Optional: install pecos CLI for direct use"
    @echo "  just setup          # First time: detect and install dependencies"
    @echo "  just build          # Build PECOS (runs setup if needed)"
    @echo "  just test           # Run all tests"
    @echo "  just dev            # Build + test (daily workflow)"
    @echo "  just lint           # Check formatting and linting"
    @echo "  just doctor         # Diagnose environment problems"
    @echo ""
    @echo "All commands:"
    @just --list --list-heading ''

# =============================================================================
# Settings
# =============================================================================

# Requires bash (Windows: use Git Bash from https://git-scm.com or WSL)
set shell := ["bash", "-cu"]

# PECOS CLI - must be installed (run 'just install-cli' first)
pecos := "cargo run -p pecos-cli --"

# =============================================================================
# Getting Started
# =============================================================================

# Install or update the PECOS CLI
[group('setup')]
install-cli:
    @echo "Installing PECOS CLI..."
    cargo install --path crates/pecos-cli --force
    @echo ""
    @echo "Done! You can now run: just build"

# Set up build environment (detect and install missing dependencies)
[group('setup')]
setup:
    {{pecos}} setup

# Set up build environment, accepting all prompts (for CI)
[group('setup')]
setup-ci:
    {{pecos}} setup --yes

# Check development environment for common problems
[group('setup')]
doctor:
    #!/usr/bin/env bash
    set -euo pipefail
    PROBLEMS=0
    ok()   { echo "  [OK] $1: $2"; }
    fail() { echo "  [!!] $1: $2"; PROBLEMS=$((PROBLEMS + 1)); }

    echo "LLVM 14:"
    LLVM_DIR=""
    for d in "$HOME/.pecos/deps/llvm-14" "$HOME/.pecos/deps/llvm"; do
        [ -d "$d/bin" ] && LLVM_DIR="$d" && break
    done
    if [ -n "$LLVM_DIR" ]; then
        VERSION=$("$LLVM_DIR/bin/llvm-config" --version 2>/dev/null || echo "unknown")
        ok "installed" "$VERSION at $LLVM_DIR"
    else
        fail "installed" "not found (run: pecos setup)"
    fi
    if [ -f .cargo/config.toml ] && grep -q "LLVM_SYS_140_PREFIX" .cargo/config.toml 2>/dev/null; then
        ok ".cargo/config.toml" "LLVM_SYS_140_PREFIX configured"
    else
        fail ".cargo/config.toml" "LLVM_SYS_140_PREFIX not set (run: pecos llvm configure)"
    fi
    echo ""

    echo "Python:"
    if command -v uv >/dev/null 2>&1; then
        ok "uv" "$(uv --version)"
    else
        fail "uv" "not found (see: https://docs.astral.sh/uv/)"
    fi
    PECOS_VER=$(uv run python -c "import pecos; print(pecos.__version__)" 2>/dev/null) \
        && ok "import pecos" "v$PECOS_VER" \
        || fail "import pecos" "failed (run: just build)"
    RSLIB_VER=$(uv run python -c "import pecos_rslib; print(pecos_rslib.__version__)" 2>/dev/null) \
        && ok "pecos_rslib" "v$RSLIB_VER" \
        || fail "pecos_rslib" "native library failed to load (run: just build)"
    echo ""

    echo "CUDA (optional):"
    NVCC=$(command -v nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)
    if [ -x "$NVCC" ]; then
        CUDA_VER=$("$NVCC" --version 2>/dev/null | grep release | sed 's/.*release //' | sed 's/,.*//')
        ok "CUDA" "$CUDA_VER"
    else
        echo "  [--] CUDA: not found (optional)"
    fi
    echo ""

    if [ "$PROBLEMS" -eq 0 ]; then
        echo "No problems found."
    else
        echo "$PROBLEMS problem(s) found. See above for fixes."
    fi

# Show system information
[group('setup')]
sys-info:
    {{pecos}} sys-info

# List installed and cached dependencies
[group('setup')]
list-deps:
    {{pecos}} list -v

# =============================================================================
# Building
# =============================================================================

# Build PECOS (profile: debug, release, native)
[group('build')]
build profile="debug": setup-quiet sync-deps build-selene
    #!/usr/bin/env bash
    set -euo pipefail
    {{pecos}} python build --profile {{profile}}
    command -v julia >/dev/null 2>&1 && just julia-build {{profile}} || true
    command -v go >/dev/null 2>&1 && just go-build {{profile}} || true

# Build PECOS without dependency setup or sync (profile: debug, release, native)
[group('build')]
build-lite profile="debug": build-selene
    {{pecos}} python build --profile {{profile}}

# Build PECOS with CUDA Python extras (profile: debug, release, native)
[group('build')]
build-cuda profile="debug": setup-quiet
    {{pecos}} python build --profile {{profile}} --cuda

# =============================================================================
# Testing
# =============================================================================

# Run Python tests (or: just pytest <custom args>)
[group('test')]
pytest *args:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -n "{{args}}" ]; then
        uv run pytest {{args}}
    else
        uv run pytest python/pecos-rslib/tests -m "not performance"
        uv run --group numpy-compat pytest python/pecos-rslib/tests -m "numpy and not performance"
        uv run pytest python/quantum-pecos/tests -m "not optional_dependency and not slow"
        uv run pytest python/selene-plugins
    fi

# Run Rust tests (CUDA-aware; mode: debug or release)
[group('test')]
rstest mode="release":
    #!/usr/bin/env bash
    set -euo pipefail
    if [ "{{mode}}" = "release" ]; then
        {{pecos}} rust test --release
    else
        {{pecos}} rust test
    fi

# Run all tests (Rust + Python + Julia + Go if available)
[group('test')]
test mode="release": (rstest mode) pytest
    #!/usr/bin/env bash
    set -euo pipefail
    if command -v julia >/dev/null 2>&1; then
        echo "Julia detected, running Julia tests..."
        just julia-test
    else
        echo "Julia not detected, skipping Julia tests"
    fi
    if command -v go >/dev/null 2>&1; then
        echo "Go detected, running Go tests..."
        just go-test
    else
        echo "Go not detected, skipping Go tests"
    fi

# =============================================================================
# Linting / Formatting
# =============================================================================

# Fix formatting and linting issues (or: just lint check)
[group('lint')]
lint mode="fix":
    #!/usr/bin/env bash
    set -euo pipefail
    # Detect CUDA: only use --all-features when CUDA toolkit is available
    if command -v nvcc >/dev/null 2>&1 || [ -n "${CUDA_PATH:-}" ] || [ -d /usr/local/cuda ]; then
        CLIPPY_FEATURES="--all-features"
        echo "(CUDA detected -- linting with all features)"
    else
        CLIPPY_FEATURES=""
        echo "(No CUDA -- linting with default features only)"
    fi

    if [ "{{mode}}" = "check" ]; then
        echo "==> Checking Rust formatting..."
        cargo fmt --all -- --check
        echo "==> Running clippy..."
        cargo clippy --workspace --all-targets $CLIPPY_FEATURES -- -D warnings
        echo "==> Running pre-commit..."
        uv run pre-commit run --all-files
        if command -v julia >/dev/null 2>&1; then
            echo "==> Checking Julia formatting..."
            just julia-fmt-check
            just julia-lint
        fi
        if command -v go >/dev/null 2>&1; then
            echo "==> Checking Go formatting..."
            just go-fmt-check
            just go-lint
        fi
    else
        echo "==> Fixing Rust formatting and clippy..."
        cargo fmt --all
        cargo clippy --workspace --all-targets $CLIPPY_FEATURES --fix --allow-staged --allow-dirty -- -D warnings
        echo "==> Running pre-commit..."
        uv run pre-commit run --all-files || true
        if command -v julia >/dev/null 2>&1; then
            echo "==> Fixing Julia formatting..."
            just julia-fmt
        fi
        if command -v go >/dev/null 2>&1; then
            echo "==> Fixing Go formatting..."
            just go-fmt
        fi
    fi

# Run cargo check
[group('lint')]
check:
    cargo check --workspace --all-targets

# Run cargo clippy (CUDA-aware: uses --all-features only when CUDA is available)
[group('lint')]
clippy:
    #!/usr/bin/env bash
    set -euo pipefail
    if command -v nvcc >/dev/null 2>&1 || [ -n "${CUDA_PATH:-}" ] || [ -d /usr/local/cuda ]; then
        echo "(CUDA detected -- clippy with all features)"
        cargo clippy --workspace --all-targets --all-features -- -D warnings
    else
        echo "(No CUDA -- clippy with default features)"
        cargo clippy --workspace --all-targets -- -D warnings
    fi

# Check Rust formatting
[group('lint')]
fmt:
    cargo fmt --all -- --check

# Run benchmarks (profile: release/native; features: optional; pattern: filter)
[group('test')]
bench profile="release" features="" pattern="":
    #!/usr/bin/env bash
    set -euo pipefail
    ARGS="bench -p benchmarks --bench benchmarks"
    if [ "{{profile}}" = "native" ]; then
        ARGS="$ARGS --profile=native"
        export RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=native"
    elif [ "{{profile}}" != "release" ]; then
        echo "Unknown profile: {{profile}}. Use release or native."; exit 1
    fi
    if [ -n "{{features}}" ]; then ARGS="$ARGS --features={{features}}"; fi
    if [ -n "{{pattern}}" ]; then ARGS="$ARGS -- {{pattern}}"; fi
    cargo $ARGS

# =============================================================================
# Dev Workflows
# =============================================================================

# Dev cycle: build + test (lang: all, rust, python, julia, go)
[group('dev')]
dev lang="all":
    #!/usr/bin/env bash
    set -euo pipefail
    case "{{lang}}" in
        all)
            just build
            just test debug
            ;;
        rust)
            just rstest debug
            ;;
        python)
            just build
            just pytest
            ;;
        julia)
            just julia-build
            just julia-test
            ;;
        go)
            just go-build
            just go-test
            ;;
        *)
            echo "Unknown language: {{lang}}. Use: all, rust, python, julia, go"
            exit 1
            ;;
    esac

# Clean build + test + lint check (run before opening a PR)
[group('dev')]
check-all: clean (build "release") (test "release") (lint "check")

# Clean build artifacts (or: just clean cache/deps/all/dry-run)
[group('clean')]
clean *target:
    uv run python scripts/clean.py {{ if target == "cache" { "--cache" } else if target == "deps" { "--deps" } else if target == "all" { "--all" } else if target == "dry-run" { "--dry-run" } else { "" } }}

# =============================================================================
# Documentation
# =============================================================================

# Serve documentation locally (port: default 8000)
[group('docs')]
docs port="8000":
    uv run mkdocs serve -a "127.0.0.1:{{port}}"

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
install-llvm:
    {{pecos}} install llvm

# Install CUDA Toolkit
[group('deps')]
install-cuda:
    {{pecos}} install cuda

# Configure LLVM paths in .cargo/config.toml
[group('deps')]
configure-llvm:
    {{pecos}} llvm configure

# Check LLVM 14 installation status
[group('deps')]
check-llvm:
    -{{pecos}} llvm check

# Check CUDA installation status
[group('deps')]
check-cuda:
    -{{pecos}} cuda check

# =============================================================================
# Julia Bindings
# =============================================================================

# Build Julia FFI library (profile: debug, release, native; rustflags: optional)
[group('julia')]
julia-build profile="release" rustflags="":
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -n "{{rustflags}}" ]; then
        export RUSTFLAGS="${RUSTFLAGS:-} {{rustflags}}"
    fi
    case "{{profile}}" in
        native)  cargo build --profile native -p pecos-julia-ffi ;;
        release) cargo build --release -p pecos-julia-ffi ;;
        dev|debug) cargo build -p pecos-julia-ffi ;;
        *) echo "Unknown profile: {{profile}}"; exit 1 ;;
    esac

# Run Julia tests
[group('julia')]
julia-test: (julia-build "release")
    cd julia/PECOS.jl && julia --project=. -e 'using Pkg; Pkg.instantiate(); include("test/runtests.jl")'

# Format Julia code
[group('julia')]
julia-fmt:
    #!/usr/bin/env bash
    set -euo pipefail
    julia -e 'using Pkg; Pkg.activate(); haskey(Pkg.project().dependencies, "JuliaFormatter") || Pkg.add("JuliaFormatter")'
    cd julia/PECOS.jl && julia -e 'using JuliaFormatter; format("."; verbose=true)'

# Check Julia code formatting
[group('julia')]
julia-fmt-check:
    #!/usr/bin/env bash
    set -euo pipefail
    julia -e 'using Pkg; Pkg.activate(); haskey(Pkg.project().dependencies, "JuliaFormatter") || Pkg.add("JuliaFormatter")'
    cd julia/PECOS.jl && julia -e 'using JuliaFormatter; format("."; verbose=false, overwrite=false) || (println("Run just julia-fmt to fix."); exit(1))'

# Run Aqua.jl quality checks
[group('julia')]
julia-lint: (julia-build "release")
    cd julia/PECOS.jl && julia --project=. test/aqua_tests.jl

# =============================================================================
# Go Bindings
# =============================================================================

# Build Go FFI library (profile: debug, release, native; rustflags: optional)
[group('go')]
go-build profile="release" rustflags="":
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -n "{{rustflags}}" ]; then
        export RUSTFLAGS="${RUSTFLAGS:-} {{rustflags}}"
    fi
    case "{{profile}}" in
        native)  cargo build --profile native -p pecos-go-ffi ;;
        release) cargo build --release -p pecos-go-ffi ;;
        dev|debug) cargo build -p pecos-go-ffi ;;
        *) echo "Unknown profile: {{profile}}"; exit 1 ;;
    esac

# Run Go tests
[group('go')]
go-test: (go-build "release")
    #!/usr/bin/env bash
    set -euo pipefail
    LIB_DIR="$(pwd)/target/release"
    export LD_LIBRARY_PATH="$LIB_DIR:${LD_LIBRARY_PATH:-}"
    export DYLD_LIBRARY_PATH="$LIB_DIR:${DYLD_LIBRARY_PATH:-}"
    cd go/pecos && go test -v

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
go-lint: (go-build "release")
    #!/usr/bin/env bash
    set -euo pipefail
    LIB_DIR="$(pwd)/target/release"
    export LD_LIBRARY_PATH="$LIB_DIR:${LD_LIBRARY_PATH:-}"
    export DYLD_LIBRARY_PATH="$LIB_DIR:${DYLD_LIBRARY_PATH:-}"
    cd go/pecos && go vet ./...

# =============================================================================
# Additional Testing
# =============================================================================

# Run performance tests with release build
[group('test')]
pytest-perf: build-release
    uv run --group numpy-compat pytest python/pecos-rslib/tests -m "performance" -v

# Run tests for optional dependencies
[group('test')]
pytest-dep:
    uv run pytest python/pecos-rslib/tests -m "optional_dependency"
    uv run pytest python/quantum-pecos/tests -m "optional_dependency"

# Run the slower integration lane (excluded from the default fast lane)
[group('test')]
pytest-slow:
    uv run pytest python/quantum-pecos/tests -m "slow and not optional_dependency"




# =============================================================================
# Private / Internal Recipes
# =============================================================================

[private]
setup-quiet:
    {{pecos}} setup --quiet

# Sync Python deps (fast if already installed, skips maturin rebuilds)
[private]
sync-deps:
    #!/usr/bin/env bash
    set -euo pipefail
    # Quick check: ensure the packages used by the default dev/test lane are importable.
    # This catches newly added workspace members that an older .venv may be missing.
    if uv run --frozen python -c "import importlib.util, sys; required = ('pecos', 'pecos_rslib', 'pecos_selene_stab_vec', 'pecos_selene_stabilizer', 'pecos_selene_statevec', 'pecos_selene_stab_mps', 'pecos_selene_mast'); missing = [name for name in required if importlib.util.find_spec(name) is None]; sys.exit(1 if missing else 0)" 2>/dev/null; then
        exit 0
    fi
    echo "Python deps incomplete, running uv sync..."
    uv sync --project . --all-packages

[private]
build-selene:
    #!/usr/bin/env bash
    set -euo pipefail
    PLUGIN_DIRS=()
    for DIR in python/selene-plugins/pecos-selene-*/; do
        [ -d "$DIR" ] || continue
        [ -f "$DIR/Cargo.toml" ] || continue
        [ -f "$DIR/pyproject.toml" ] || continue
        PLUGIN_DIRS+=("$DIR")
    done
    # Check if any selene source changed since last install
    NEEDS_BUILD=false
    for DIR in "${PLUGIN_DIRS[@]}"; do
        PKG=$(basename "$DIR")
        DEST="$DIR/python/${PKG//-/_}/_dist/lib/"
        SO=$(find "$DEST" -name "*.so" 2>/dev/null | head -1 || true)
        if [ -z "$SO" ]; then
            NEEDS_BUILD=true
            break
        fi
        # Check if any Rust source is newer than the installed .so
        NEWER=$(find "crates/" "$DIR" -name "*.rs" -newer "$SO" 2>/dev/null | head -1 || true)
        if [ -n "$NEWER" ]; then
            NEEDS_BUILD=true
            break
        fi
    done
    if [ "$NEEDS_BUILD" = false ]; then
        echo "Selene plugins: up to date"
        exit 0
    fi
    echo "Building Selene plugins..."
    CARGO_ARGS=""
    for DIR in "${PLUGIN_DIRS[@]}"; do
        CARGO_ARGS="$CARGO_ARGS -p $(basename "$DIR")"
    done
    if [ -n "$CARGO_ARGS" ]; then
        cargo build --release $CARGO_ARGS
    fi
    echo "Copying libraries to Python packages..."
    {{pecos}} selene install --profile release
    echo "Selene plugins built and installed successfully"


# Convenience aliases
[private]
build-debug: (build "debug")
[private]
build-release: (build "release")

# Regenerate all lockfiles from scratch
[group('setup')]
updatelocks:
    rm -f uv.lock Cargo.lock
    uv lock --project .
    cargo generate-lockfile

# Install CUDA Python packages (requires CUDA toolkit)
[private]
install-cuda-python:
    {{pecos}} cuda setup-python

# Validate CUDA installation integrity
[private]
validate-cuda:
    {{pecos}} cuda validate

# Run Julia examples
[private]
julia-examples: (julia-build "debug")
    #!/usr/bin/env bash
    set -euo pipefail
    if command -v julia >/dev/null 2>&1; then
        cd julia/PECOS.jl && julia --project=. examples/demo.jl
        cd julia/PECOS.jl && julia --project=. examples/basic_usage.jl
    else
        echo "Julia not found."; exit 1
    fi
