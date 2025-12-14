# Developer Tools CLI

The `pecos-dev` CLI provides tools for PECOS development, including build utilities, LLVM setup, and dependency management.

## Installation

```bash
cargo install pecos-dev
```

Or build from source:

```bash
cargo build -p pecos-dev --release
```

## Commands Overview

```
$ pecos-dev --help
PECOS developer tools - build, test, and manage PECOS development

Commands:
  rust      Rust/Cargo commands (CUDA-aware) [aliases: rs]
  python    Python build and test commands [aliases: py]
  cuda      CUDA availability and info
  julia     Julia build and test commands [aliases: jl]
  go        Go build and test commands
  llvm      LLVM 14 management
  selene    Selene plugin management
  clean     Clean build artifacts and caches
  features  Query package features
  deps      Dependency manifest management (pecos.toml)
  info      Show PECOS home directory info and status
  list      List installed and cached dependencies
```

## Rust Commands (`rust` / `rs`)

CUDA-aware cargo commands that automatically handle GPU feature detection.

```bash
# Run cargo check (automatically excludes GPU features if CUDA unavailable)
pecos-dev rust check
pecos-dev rs check          # short alias

# Run cargo clippy
pecos-dev rust clippy
pecos-dev rust clippy --fix # auto-fix issues

# Run cargo test
pecos-dev rust test
pecos-dev rust test --release

# Run cargo fmt
pecos-dev rust fmt
pecos-dev rust fmt --check  # check only, don't modify

# Include FFI crates in checks
pecos-dev rust check --include-ffi
pecos-dev rust clippy --include-ffi
```

When CUDA is not available, these commands automatically:

- Exclude `pecos` and `pecos-quest` from workspace commands
- Run them separately with GPU features filtered out

## Python Commands (`python` / `py`)

Build and test commands for Python packages.

```bash
# Check if Python/uv is available
pecos-dev python check
pecos-dev py check          # short alias

# Build pecos-rslib and quantum-pecos
pecos-dev python build
pecos-dev python build --profile release
pecos-dev python build --cuda  # with CUDA support

# Run Python tests
pecos-dev python test
pecos-dev python test -v        # verbose
pecos-dev python test --selene  # Selene plugin tests only
pecos-dev python test --numpy   # NumPy/SciPy compat tests
```

## CUDA Commands (`cuda`)

```bash
# Check if CUDA/nvcc is available
pecos-dev cuda check

# Quiet mode (exit code only, for scripting)
pecos-dev cuda check -q
```

## Julia Commands (`julia` / `jl`)

```bash
# Check if Julia is available
pecos-dev julia check
pecos-dev jl check          # short alias

# Build Julia FFI library
pecos-dev julia build
pecos-dev julia build --profile release
pecos-dev julia build --profile debug

# Run Julia tests
pecos-dev julia test

# Format Julia code
pecos-dev julia fmt
pecos-dev julia fmt --check  # check only

# Run Julia linting (Aqua.jl)
pecos-dev julia lint
```

## Go Commands (`go`)

```bash
# Check if Go is available
pecos-dev go check

# Build Go FFI library
pecos-dev go build
pecos-dev go build --profile release
pecos-dev go build --profile debug

# Run Go tests
pecos-dev go test

# Format Go code
pecos-dev go fmt
pecos-dev go fmt --check  # check only

# Run Go linting (go vet)
pecos-dev go lint
```

## LLVM Management (`llvm`)

The `llvm` subcommand helps set up LLVM 14, which is required for QIS program support.

### Check LLVM Status

```bash
pecos-dev llvm check
```

Verifies if LLVM 14 is available and properly configured.

### Install LLVM

```bash
# Automated installation (downloads pre-built binaries)
pecos-dev llvm install

# Force reinstall
pecos-dev llvm install --force
```

This downloads and installs LLVM 14 to the PECOS home directory (`~/.pecos/llvm/` by default).

### Configure Cargo

```bash
pecos-dev llvm configure
```

Updates `.cargo/config.toml` with the correct `LLVM_SYS_140_PREFIX` environment variable.

### Find LLVM Path

```bash
# Find LLVM installation
pecos-dev llvm find

# Export for shell evaluation
pecos-dev llvm find --export

# Find a specific tool
pecos-dev llvm tool clang
pecos-dev llvm tool llvm-config
```

### Validate Installation

```bash
pecos-dev llvm validate
```

Checks that all required LLVM components are present and functional.

### Show Version

```bash
pecos-dev llvm version
```

## Selene Plugin Management (`selene`)

Manage Selene simulator plugins.

```bash
# List plugins and their status
pecos-dev selene list

# Install plugins (copy built libraries to Python packages)
pecos-dev selene install
pecos-dev selene install --profile release
pecos-dev selene install --plugin pecos-selene-quest

# Clean plugin artifacts (quiet by default)
pecos-dev selene clean
pecos-dev selene clean --venv     # also clean from .venv/
pecos-dev selene clean -v         # verbose output
```

## Clean Commands (`clean`)

Clean various build artifacts and caches. By default, output is quiet. Use `-v` for verbose output.

```bash
# Clean build artifacts (Python, Rust, Julia)
pecos-dev clean build
pecos-dev clean build -v           # verbose output
pecos-dev clean build -vv          # more verbose
pecos-dev clean build --dry-run    # preview what would be deleted
pecos-dev clean build --skip-cargo # don't run cargo clean

# Clean ~/.pecos/cache/ and tmp/
pecos-dev clean cache
pecos-dev clean cache -v           # verbose output

# Clean ~/.pecos/deps/
pecos-dev clean deps

# Clean ~/.pecos/llvm/
pecos-dev clean llvm

# Clean everything (deps + cache + tmp)
pecos-dev clean all
pecos-dev clean all --include-llvm  # also remove LLVM
```

## Feature Queries (`features`)

Query package features for build configuration.

```bash
# List all features for a package
pecos-dev features list --package pecos

# Exclude certain features
pecos-dev features list --package pecos --exclude cuda

# Output as JSON
pecos-dev features list --package pecos-quest --json
```

## Dependency Management (`deps`)

Manage the `pecos.toml` manifest which tracks external dependencies.

### Show Status

```bash
pecos-dev deps status
```

### Initialize Manifest

```bash
pecos-dev deps init
```

### Sync Manifests

```bash
pecos-dev deps sync
pecos-dev deps sync --dry-run
```

### Verify Dependencies

```bash
pecos-dev deps verify
```

## Cache Management

### Show Info

```bash
pecos-dev info
```

Displays PECOS home directory location and status.

### List Cached Items

```bash
pecos-dev list
pecos-dev list --verbose
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PECOS_HOME` | PECOS cache and data directory | `~/.pecos` |
| `LLVM_SYS_140_PREFIX` | LLVM 14 installation path | auto-detected |
| `RUST_LOG` | Log level for build output (`info` shows download progress) | `warn` |

## Typical Workflows

### Setting Up LLVM for the First Time

```bash
# 1. Check if LLVM is already available
pecos-dev llvm check

# 2. If not, install it
pecos-dev llvm install

# 3. Configure Cargo to find it
pecos-dev llvm configure

# 4. Verify the installation
pecos-dev llvm validate

# 5. Now you can build with LLVM support
cargo build -p pecos --features llvm
```

### Running Lints Before Committing

```bash
# Check code compiles
pecos-dev rs check

# Run clippy
pecos-dev rs clippy

# Check formatting
pecos-dev rs fmt --check
```

### Building FFI Libraries

```bash
# Build Julia and Go FFI libraries
pecos-dev julia build --profile release
pecos-dev go build --profile release
```

### Cleaning Up

```bash
# See what's cached
pecos-dev info
pecos-dev list

# Clean build artifacts
pecos-dev clean build

# Clean everything
pecos-dev clean all
```

## See Also

- [LLVM Setup](../user-guide/llvm-setup.md) - Detailed LLVM installation guide
- [Development Guide](DEVELOPMENT.md) - Contributing to PECOS
