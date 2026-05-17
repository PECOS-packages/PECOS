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
install-cli: _msvc-bootstrap
    @echo "Installing PECOS CLI..."
    cargo install --path crates/pecos-cli --force
    @echo ""
    @echo "Done! You can now run: just build"

# Set up build environment (detect and install missing dependencies)
[group('setup')]
setup: _msvc-bootstrap
    {{pecos}} setup

# Set up build environment, accepting all prompts (for CI)
[group('setup')]
setup-ci: _msvc-bootstrap
    {{pecos}} setup --yes

# Ensure CI has a runtime-valid LLVM and export PECOS build env files
[group('setup')]
ci-env: _msvc-bootstrap
    #!/usr/bin/env bash
    set -euo pipefail
    {{pecos}} llvm ensure --managed --no-configure
    {{pecos}} env --github-actions

# Check development environment for common problems
[group('setup')]
doctor: _msvc-bootstrap
    #!/usr/bin/env bash
    set -euo pipefail
    PROBLEMS=0
    ok()   { echo "  [OK] $1: $2"; }
    fail() { echo "  [!!] $1: $2"; PROBLEMS=$((PROBLEMS + 1)); }

    echo "LLVM 14:"
    if LLVM_DIR=$({{pecos}} llvm find 2>/dev/null); then
        VERSION=$("$LLVM_DIR/bin/llvm-config" --version 2>/dev/null || {{pecos}} llvm version 2>/dev/null | head -1 || echo "unknown")
        ok "installed" "$VERSION at $LLVM_DIR"
    else
        fail "installed" "not found (run: just setup)"
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

    echo "Optional decoders:"
    CMAKE_BIN=""
    # Prefer PECOS-managed install (mirrors find_cmake() in pecos-build).
    for d in "$HOME"/.pecos/deps/cmake-*; do
        [ -d "$d" ] || continue
        # macOS layout nests cmake inside CMake.app/Contents/bin/.
        for candidate in "$d/CMake.app/Contents/bin/cmake" "$d/bin/cmake" "$d/bin/cmake.exe"; do
            if [ -x "$candidate" ]; then
                CMAKE_BIN="$candidate"
                break 2
            fi
        done
    done
    if [ -z "$CMAKE_BIN" ] && command -v cmake >/dev/null 2>&1; then
        CMAKE_BIN=$(command -v cmake)
    fi
    if [ -n "$CMAKE_BIN" ]; then
        CMAKE_VER=$("$CMAKE_BIN" --version 2>/dev/null | head -1 | awk '{print $3}')
        ok "cmake" "$CMAKE_VER (MWPF decoder available) at $CMAKE_BIN"
    else
        echo "  [--] cmake: not found — MWPF decoder disabled"
        echo "       Install via 'pecos setup' / 'pecos install cmake', or see:"
        echo "       https://github.com/PECOS-packages/PECOS/blob/dev/docs/user-guide/cmake-setup.md"
    fi
    echo ""

    if [ "$PROBLEMS" -eq 0 ]; then
        echo "No problems found."
    else
        echo "$PROBLEMS problem(s) found. See above for fixes."
    fi

# Show system information
[group('setup')]
sys-info: _msvc-bootstrap
    {{pecos}} sys-info

# List installed and cached dependencies
[group('setup')]
list-deps: _msvc-bootstrap
    {{pecos}} list -v

# =============================================================================
# Building
# =============================================================================

# Build PECOS (profile: dev/debug, release, native)
[group('build')]
build profile="debug": _msvc-bootstrap (validate-profile "build" profile) setup-quiet sync-deps (build-selene profile)
    #!/usr/bin/env bash
    set -euo pipefail
    PROFILE="{{profile}}"
    {{pecos}} python build --profile "$PROFILE"
    if command -v julia >/dev/null 2>&1; then
        just julia-build "$PROFILE"
    fi
    if command -v go >/dev/null 2>&1; then
        just go-build "$PROFILE"
    fi

# Build PECOS without dependency setup or sync (profile: dev/debug, release, native)
[group('build')]
build-lite profile="debug": _msvc-bootstrap (validate-profile "build-lite" profile) (build-selene profile)
    #!/usr/bin/env bash
    set -euo pipefail
    PROFILE="{{profile}}"
    {{pecos}} python build --profile "$PROFILE"

# Build PECOS with CUDA Python extras (profile: dev/debug, release, native)
[group('build')]
build-cuda profile="debug": _msvc-bootstrap (validate-profile "build-cuda" profile) setup-quiet
    #!/usr/bin/env bash
    set -euo pipefail
    PROFILE="{{profile}}"
    {{pecos}} python build --profile "$PROFILE" --cuda

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

# Run Rust tests (CUDA-aware; mode: dev/debug, release, native)
[group('test')]
rstest mode="release": _msvc-bootstrap (validate-test-mode "rstest" mode)
    #!/usr/bin/env bash
    set -euo pipefail
    MODE="{{mode}}"
    {{pecos}} rust test --profile "$MODE"

# Run all tests (Rust + Python + Julia + Go if available; mode: dev/debug, release, native)
[group('test')]
test mode="release": (validate-test-mode "test" mode) (rstest mode) pytest
    #!/usr/bin/env bash
    set -euo pipefail
    MODE="{{mode}}"
    if command -v julia >/dev/null 2>&1; then
        echo "Julia detected, running Julia tests..."
        just julia-test "$MODE"
    else
        echo "Julia not detected, skipping Julia tests"
    fi
    if command -v go >/dev/null 2>&1; then
        echo "Go detected, running Go tests..."
        just go-test "$MODE"
    else
        echo "Go not detected, skipping Go tests"
    fi

# =============================================================================
# Linting / Formatting
# =============================================================================

# Fix formatting and linting issues (or: just lint check)
[group('lint')]
lint mode="fix": _msvc-bootstrap (validate-lint-mode mode) python-workspace-check
    #!/usr/bin/env bash
    set -euo pipefail
    MODE="{{mode}}"
    # Detect CUDA: only use --all-features when CUDA toolkit is available
    if command -v nvcc >/dev/null 2>&1 || [ -n "${CUDA_PATH:-}" ] || [ -d /usr/local/cuda ]; then
        CLIPPY_FEATURES="--all-features"
        echo "(CUDA detected -- linting with all features)"
    else
        CLIPPY_FEATURES=""
        echo "(No CUDA -- linting with default features only)"
    fi

    if [ "$MODE" = "check" ]; then
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
check: _msvc-bootstrap
    cargo check --workspace --all-targets

# Check Python workspace metadata
[group('lint')]
python-workspace-check:
    @uv run python scripts/check_python_workspace.py

# Run cargo clippy (CUDA-aware: uses --all-features only when CUDA is available)
[group('lint')]
clippy: _msvc-bootstrap
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
bench profile="release" features="" pattern="": _msvc-bootstrap (validate-bench-profile "bench" profile)
    #!/usr/bin/env bash
    set -euo pipefail
    PROFILE="{{profile}}"
    FEATURES="{{features}}"
    PATTERN="{{pattern}}"
    case "$FEATURES" in
        features=*)
            VALUE="${FEATURES#features=}"
            echo "Invalid features argument: $FEATURES"
            echo "Just recipe parameters are positional. Use: just bench $PROFILE $VALUE"
            exit 2
            ;;
    esac
    case "$PATTERN" in
        pattern=*)
            VALUE="${PATTERN#pattern=}"
            echo "Invalid pattern argument: $PATTERN"
            echo "Just recipe parameters are positional. Use: just bench $PROFILE '$FEATURES' '$VALUE'"
            exit 2
            ;;
    esac
    ARGS=(bench -p benchmarks --bench benchmarks)
    if [ "$PROFILE" = "native" ]; then
        ARGS+=(--profile=native)
        export RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=native"
    fi
    if [ -n "$FEATURES" ]; then ARGS+=(--features "$FEATURES"); fi
    if [ -n "$PATTERN" ]; then ARGS+=(-- "$PATTERN"); fi
    cargo "${ARGS[@]}"

# =============================================================================
# Dev Workflows
# =============================================================================

# Dev cycle: build + test (lang: all, rust, python, julia, go)
[group('dev')]
dev lang="all": (validate-dev-lang lang)
    #!/usr/bin/env bash
    set -euo pipefail
    DEV_LANG="{{lang}}"
    case "$DEV_LANG" in
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
            echo "Unknown language: $DEV_LANG. Use: all, rust, python, julia, go"
            exit 1
            ;;
    esac

# Clean build + test + lint check (run before opening a PR)
[group('dev')]
check-all: clean (build "release") (test "release") (lint "check")

# Clean build artifacts (or: just clean cache/deps/selene/all/dry-run; multiple OK, e.g. just clean selene deps)
[group('clean')]
clean *target:
    #!/usr/bin/env bash
    set -euo pipefail
    TARGETS="{{target}}"
    ARGS=()
    if [ -n "$TARGETS" ]; then
        for TARGET in $TARGETS; do
            case "$TARGET" in
                cache|deps|selene|all|dry-run) ARGS+=("--$TARGET") ;;
                target=*)
                    VALUE="${TARGET#target=}"
                    echo "Invalid clean target argument: $TARGET"
                    echo "Just variadic arguments are positional. Use: just clean $VALUE"
                    exit 2
                    ;;
                *)
                    echo "Unknown clean target: $TARGET"
                    echo "Supported targets: cache, deps, selene, all, dry-run"
                    exit 2
                    ;;
            esac
        done
    fi
    # macOS bash 3.2: ${arr[@]+"${arr[@]}"} expands to nothing when arr is empty/unset
    # under `set -u` (which otherwise trips on empty @-expansion).
    uv run python scripts/clean.py ${ARGS[@]+"${ARGS[@]}"}

# =============================================================================
# Documentation
# =============================================================================

# Serve documentation locally (port: default 8000)
[group('docs')]
docs port="8000": (validate-port port)
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
install-llvm: _msvc-bootstrap
    {{pecos}} install llvm

# Install CUDA Toolkit
[group('deps')]
install-cuda: _msvc-bootstrap
    {{pecos}} install cuda

# Configure LLVM paths in .cargo/config.toml
[group('deps')]
configure-llvm: _msvc-bootstrap
    {{pecos}} llvm configure

# Check LLVM 14 installation status
[group('deps')]
check-llvm: _msvc-bootstrap
    -{{pecos}} llvm check

# Check CUDA installation status
[group('deps')]
check-cuda: _msvc-bootstrap
    -{{pecos}} cuda check

# =============================================================================
# Julia Bindings
# =============================================================================

# Build Julia FFI library (profile: dev/debug, release, native; rustflags: optional)
[group('julia')]
julia-build profile="release" rustflags="": _msvc-bootstrap (validate-profile "julia-build" profile)
    #!/usr/bin/env bash
    set -euo pipefail
    PROFILE="{{profile}}"
    RUSTFLAGS_ARG="{{rustflags}}"
    case "$RUSTFLAGS_ARG" in
        rustflags=*)
            VALUE="${RUSTFLAGS_ARG#rustflags=}"
            echo "Invalid rustflags argument: $RUSTFLAGS_ARG"
            echo "Just recipe parameters are positional. Use: just julia-build $PROFILE '$VALUE'"
            exit 2
            ;;
    esac
    if [ -n "$RUSTFLAGS_ARG" ]; then
        export RUSTFLAGS="${RUSTFLAGS:-} $RUSTFLAGS_ARG"
    fi
    # The native profile inherits release; -C target-cpu=native is injected here
    # rather than via profile.native.rustflags (which is still unstable in cargo).
    if [ "$PROFILE" = "native" ]; then
        export RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=native"
    fi
    case "$PROFILE" in
        native)  cargo build --profile native -p pecos-julia-ffi ;;
        release) cargo build --release -p pecos-julia-ffi ;;
        dev|debug) cargo build -p pecos-julia-ffi ;;
        *) echo "Unknown profile: $PROFILE"; exit 1 ;;
    esac

# Run Julia tests (profile: dev/debug, release, native)
[group('julia')]
julia-test profile="release": (validate-profile "julia-test" profile) (julia-build profile)
    #!/usr/bin/env bash
    set -euo pipefail
    PROFILE="{{profile}}"
    case "$PROFILE" in
        native) LIB_DIR="$(pwd)/target/native" ;;
        release) LIB_DIR="$(pwd)/target/release" ;;
        dev|debug) LIB_DIR="$(pwd)/target/debug" ;;
    esac
    export PECOS_JULIA_LIB_DIR="$LIB_DIR"
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

# Build Go FFI library (profile: dev/debug, release, native; rustflags: optional)
[group('go')]
go-build profile="release" rustflags="": _msvc-bootstrap (validate-profile "go-build" profile)
    #!/usr/bin/env bash
    set -euo pipefail
    PROFILE="{{profile}}"
    RUSTFLAGS_ARG="{{rustflags}}"
    case "$RUSTFLAGS_ARG" in
        rustflags=*)
            VALUE="${RUSTFLAGS_ARG#rustflags=}"
            echo "Invalid rustflags argument: $RUSTFLAGS_ARG"
            echo "Just recipe parameters are positional. Use: just go-build $PROFILE '$VALUE'"
            exit 2
            ;;
    esac
    if [ -n "$RUSTFLAGS_ARG" ]; then
        export RUSTFLAGS="${RUSTFLAGS:-} $RUSTFLAGS_ARG"
    fi
    # See julia-build for why -C target-cpu=native is injected here.
    if [ "$PROFILE" = "native" ]; then
        export RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=native"
    fi
    case "$PROFILE" in
        native)  cargo build --profile native -p pecos-go-ffi ;;
        release) cargo build --release -p pecos-go-ffi ;;
        dev|debug) cargo build -p pecos-go-ffi ;;
        *) echo "Unknown profile: $PROFILE"; exit 1 ;;
    esac

# Run Go tests (profile: dev/debug, release, native)
[group('go')]
go-test profile="release": (validate-profile "go-test" profile) (go-build profile)
    #!/usr/bin/env bash
    set -euo pipefail
    PROFILE="{{profile}}"
    case "$PROFILE" in
        native) LIB_DIR="$(pwd)/target/native" ;;
        release) LIB_DIR="$(pwd)/target/release" ;;
        dev|debug) LIB_DIR="$(pwd)/target/debug" ;;
    esac
    export CGO_LDFLAGS="-L$LIB_DIR ${CGO_LDFLAGS:-}"
    export LIBRARY_PATH="$LIB_DIR:${LIBRARY_PATH:-}"
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

# Run Go linting with go vet (profile: dev/debug, release, native)
[group('go')]
go-lint profile="release": (validate-profile "go-lint" profile) (go-build profile)
    #!/usr/bin/env bash
    set -euo pipefail
    PROFILE="{{profile}}"
    case "$PROFILE" in
        native) LIB_DIR="$(pwd)/target/native" ;;
        release) LIB_DIR="$(pwd)/target/release" ;;
        dev|debug) LIB_DIR="$(pwd)/target/debug" ;;
    esac
    export CGO_LDFLAGS="-L$LIB_DIR ${CGO_LDFLAGS:-}"
    export LIBRARY_PATH="$LIB_DIR:${LIBRARY_PATH:-}"
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
validate-profile recipe profile:
    #!/usr/bin/env bash
    set -euo pipefail
    RECIPE="{{recipe}}"
    PROFILE="{{profile}}"
    case "$PROFILE" in
        dev|debug|release|native) ;;
        profile=*)
            VALUE="${PROFILE#profile=}"
            echo "Invalid profile argument: $PROFILE"
            echo "Just recipe parameters are positional. Use: just $RECIPE $VALUE"
            exit 2
            ;;
        *)
            echo "Unknown profile: $PROFILE"
            echo "Supported profiles: dev, debug, release, native"
            exit 2
            ;;
    esac

[private]
validate-test-mode recipe mode:
    #!/usr/bin/env bash
    set -euo pipefail
    RECIPE="{{recipe}}"
    MODE="{{mode}}"
    case "$MODE" in
        dev|debug|release|native) ;;
        mode=*)
            VALUE="${MODE#mode=}"
            echo "Invalid mode argument: $MODE"
            echo "Just recipe parameters are positional. Use: just $RECIPE $VALUE"
            exit 2
            ;;
        *)
            echo "Unknown test mode: $MODE"
            echo "Supported modes: dev, debug, release, native"
            exit 2
            ;;
    esac

[private]
validate-lint-mode mode:
    #!/usr/bin/env bash
    set -euo pipefail
    MODE="{{mode}}"
    case "$MODE" in
        fix|check) ;;
        mode=*)
            VALUE="${MODE#mode=}"
            echo "Invalid mode argument: $MODE"
            echo "Just recipe parameters are positional. Use: just lint $VALUE"
            exit 2
            ;;
        *)
            echo "Unknown lint mode: $MODE"
            echo "Supported modes: fix, check"
            exit 2
            ;;
    esac

[private]
validate-bench-profile recipe profile:
    #!/usr/bin/env bash
    set -euo pipefail
    RECIPE="{{recipe}}"
    PROFILE="{{profile}}"
    case "$PROFILE" in
        release|native) ;;
        profile=*)
            VALUE="${PROFILE#profile=}"
            echo "Invalid benchmark profile argument: $PROFILE"
            echo "Just recipe parameters are positional. Use: just $RECIPE $VALUE"
            exit 2
            ;;
        *)
            echo "Unknown benchmark profile: $PROFILE"
            echo "Supported benchmark profiles: release, native"
            exit 2
            ;;
    esac

[private]
validate-dev-lang lang:
    #!/usr/bin/env bash
    set -euo pipefail
    DEV_LANG="{{lang}}"
    case "$DEV_LANG" in
        all|rust|python|julia|go) ;;
        lang=*)
            VALUE="${DEV_LANG#lang=}"
            echo "Invalid language argument: $DEV_LANG"
            echo "Just recipe parameters are positional. Use: just dev $VALUE"
            exit 2
            ;;
        *)
            echo "Unknown language: $DEV_LANG"
            echo "Supported languages: all, rust, python, julia, go"
            exit 2
            ;;
    esac

[private]
validate-port port:
    #!/usr/bin/env bash
    set -euo pipefail
    PORT="{{port}}"
    case "$PORT" in
        port=*)
            VALUE="${PORT#port=}"
            echo "Invalid port argument: $PORT"
            echo "Just recipe parameters are positional. Use: just docs $VALUE"
            exit 2
            ;;
        *) ;;
    esac
    if ! [[ "$PORT" =~ ^[0-9]+$ ]] || [ "$PORT" -lt 1 ] || [ "$PORT" -gt 65535 ]; then
        echo "Invalid docs port: $PORT"
        echo "Port must be an integer from 1 to 65535"
        exit 2
    fi

[private]
setup-quiet:
    #!/usr/bin/env bash
    set -euo pipefail
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
    SYNC_ARGS=(--project . --all-packages)
    # Include CUDA Python packages (cupy, cuquantum, pytket-cutensornet) when
    # the toolkit is installed AND an NVIDIA GPU is present. Pure Rust users
    # and machines without a GPU skip this -- mirrors `pecos python build`.
    if {{pecos}} cuda check -q 2>/dev/null && nvidia-smi -L 2>/dev/null | grep -q "^GPU "; then
        echo "CUDA toolkit + NVIDIA GPU detected -- including CUDA Python packages"
        SYNC_ARGS+=(--group cuda)
    fi
    uv sync "${SYNC_ARGS[@]}"

# Windows MSVC bootstrap: write the correct linker + LIB/INCLUDE into
# .cargo/config.toml (read by cargo *after* it spawns, so it bypasses
# git-bash's link.exe shadowing and LIB mangling). Scoped TOML merge -- it
# only owns [target.x86_64-pc-windows-msvc] and the MSVC [env] keys, leaving
# the LLVM/cuQuantum keys the Rust writers own untouched. Prereq of every
# cargo entrypoint so a fresh checkout / a VS update is picked up. The unix
# variant is a no-op so the dependency is portable. Requires PowerShell 7
# (pwsh) on Windows -- the de facto repo requirement; asserted by the script.
[private]
[windows]
_msvc-bootstrap:
    pwsh -NoProfile -File scripts/win-msvc-bootstrap.ps1

[private]
[unix]
_msvc-bootstrap:
    @true

[private]
build-selene profile="release":
    #!/usr/bin/env bash
    set -euo pipefail
    PROFILE="{{profile}}"
    case "$PROFILE" in
        native)    CARGO_PROFILE_FLAGS=(--profile native); TARGET_DIR="target/native" ;;
        release)   CARGO_PROFILE_FLAGS=(--release);        TARGET_DIR="target/release" ;;
        dev|debug) CARGO_PROFILE_FLAGS=();                 TARGET_DIR="target/debug" ;;
        *) echo "build-selene: unknown profile $PROFILE" >&2; exit 2 ;;
    esac
    # See julia-build for why -C target-cpu=native is injected here.
    if [ "$PROFILE" = "native" ]; then
        export RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=native"
    fi
    case "$(uname -s)" in
        Darwin)               LIB_PREFIX="lib"; LIB_EXT="dylib" ;;
        MINGW*|MSYS*|CYGWIN*) LIB_PREFIX="";    LIB_EXT="dll" ;;
        *)                    LIB_PREFIX="lib"; LIB_EXT="so" ;;
    esac
    PLUGIN_DIRS=()
    for DIR in python/selene-plugins/pecos-selene-*/; do
        [ -d "$DIR" ] || continue
        [ -f "$DIR/Cargo.toml" ] || continue
        [ -f "$DIR/pyproject.toml" ] || continue
        PLUGIN_DIRS+=("$DIR")
    done
    # Skip cargo if the cargo output for this profile already exists and no Rust
    # source is newer. We compare against target/<profile>/ (cargo's output) rather
    # than _dist/lib/ (the installed copy) so switching profile correctly triggers
    # a rebuild even when sources are unchanged.
    # macOS bash 3.2: ${arr[@]+"${arr[@]}"} expands to nothing when arr is
    # empty/unset under `set -u` (which otherwise trips on empty @-expansion).
    NEEDS_BUILD=false
    for DIR in ${PLUGIN_DIRS[@]+"${PLUGIN_DIRS[@]}"}; do
        PKG=$(basename "$DIR")
        LIB="$TARGET_DIR/${LIB_PREFIX}${PKG//-/_}.${LIB_EXT}"
        if [ ! -f "$LIB" ]; then
            NEEDS_BUILD=true
            break
        fi
        NEWER=$(find "crates/" "$DIR" -name "*.rs" -newer "$LIB" 2>/dev/null | head -1 || true)
        if [ -n "$NEWER" ]; then
            NEEDS_BUILD=true
            break
        fi
    done
    if [ "$NEEDS_BUILD" = true ]; then
        echo "Building Selene plugins ($PROFILE)..."
        CARGO_PKG_ARGS=()
        for DIR in ${PLUGIN_DIRS[@]+"${PLUGIN_DIRS[@]}"}; do
            CARGO_PKG_ARGS+=(-p "$(basename "$DIR")")
        done
        if [ ${#CARGO_PKG_ARGS[@]} -gt 0 ]; then
            cargo build ${CARGO_PROFILE_FLAGS[@]+"${CARGO_PROFILE_FLAGS[@]}"} "${CARGO_PKG_ARGS[@]}"
        fi
    else
        echo "Selene plugins: cargo output up to date ($PROFILE)"
    fi
    echo "Installing Selene plugin libraries ($PROFILE)..."
    {{pecos}} selene install --profile "$PROFILE"
    echo "Selene plugins ready ($PROFILE)"


# Convenience aliases
[private]
build-debug: (build "debug")
[private]
build-release: (build "release")
[private]
build-native: (build "native")

# Regenerate all lockfiles from scratch
[group('setup')]
updatelocks: _msvc-bootstrap
    rm -f uv.lock Cargo.lock
    uv lock --project .
    cargo generate-lockfile

# Install CUDA Python packages (requires CUDA toolkit)
[private]
install-cuda-python: _msvc-bootstrap
    {{pecos}} cuda setup-python

# Validate CUDA installation integrity
[private]
validate-cuda: _msvc-bootstrap
    {{pecos}} cuda validate

# Run Julia examples
[private]
julia-examples: (julia-build "debug")
    #!/usr/bin/env bash
    set -euo pipefail
    if command -v julia >/dev/null 2>&1; then
        export PECOS_JULIA_LIB_DIR="$(pwd)/target/debug"
        cd julia/PECOS.jl && julia --project=. examples/demo.jl
        cd julia/PECOS.jl && julia --project=. examples/basic_usage.jl
    else
        echo "Julia not found."; exit 1
    fi
