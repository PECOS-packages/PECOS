# PECOS Development Tools

PECOS provides development tools through the `pecos` CLI and the `Justfile`.

## pecos CLI Commands

The `pecos` CLI includes commands for building, testing, and managing dependencies:

```bash
# Show all available commands
pecos --help

# Rust commands (CUDA-aware)
pecos rust check              # Run cargo check
pecos rust clippy             # Run cargo clippy
pecos rust test               # Run cargo test
pecos rust fmt                # Run cargo fmt

# Python commands
pecos python build            # Build pecos-rslib with maturin
pecos python test             # Run pytest

# Dependency installation (apt-like)
pecos install llvm            # Install LLVM 14 to ~/.pecos/llvm/
pecos install cuda            # Install CUDA Toolkit to ~/.pecos/cuda/
pecos install cuquantum       # Install cuQuantum SDK to ~/.pecos/cuquantum/
pecos install --all           # Install all optional dependencies
pecos uninstall llvm          # Uninstall LLVM
pecos upgrade llvm            # Upgrade (force reinstall) LLVM

# LLVM inspection
pecos llvm check              # Check LLVM installation status
pecos llvm configure          # Configure .cargo/config.toml

# CUDA inspection
pecos cuda check              # Check CUDA availability

# Julia commands
pecos julia build             # Build Julia FFI library
pecos julia test              # Run Julia tests

# Go commands
pecos go build                # Build Go FFI library
pecos go test                 # Run Go tests

# Selene plugin management
pecos selene install          # Install Selene plugins
pecos selene list             # List plugin status

# Dependency management
pecos deps list               # List available dependencies
pecos deps sync               # Sync dependency manifests

# System info
pecos sys-info                # Show toolchain and environment info
```

When running from the repository:
```bash
cargo run -p pecos --features cli -- install llvm
cargo run -p pecos --features cli -- rust clippy
```

## Build and Test Commands (Justfile)

Most development tasks are managed through the Justfile. Make sure you have `just` installed:

```bash
# Install just
cargo install just
```

### Quick Reference

```bash
# Setup
just llvm-install              # Install LLVM 14
just setup                     # Build all components

# Building
just build                     # Build all (release)
just build-debug               # Build all (debug)
just python-build              # Build Python package

# Testing
just test                      # Run all tests
just rust-test                 # Rust tests only
just python-test               # Python tests only

# Code quality
just lint                      # Run all linters
just fmt                       # Format all code
just fmt-check                 # Check formatting

# Cleaning
just clean                     # Clean build artifacts
```

### Available Commands

Run `just --list` to see all available commands:

```bash
$ just --list
Available recipes:
    build profile='release'    # Build all components
    build-debug                # Build in debug mode
    clean                      # Clean build artifacts
    fmt                        # Format all code
    fmt-check                  # Check formatting
    lint                       # Run linters
    llvm-check                 # Check LLVM installation
    llvm-configure             # Configure LLVM paths
    llvm-install               # Install LLVM 14
    python-build               # Build Python package
    python-test                # Run Python tests
    rust-check                 # Check Rust code
    rust-clippy                # Run clippy
    rust-fmt                   # Format Rust code
    rust-test                  # Run Rust tests
    setup                      # Initial setup
    test                       # Run all tests
    ...
```

## LLVM Management Details

### Install LLVM

```bash
# Automated installation (downloads pre-built binaries)
pecos install llvm

# Force reinstall
pecos install llvm --force

# Skip automatic configuration
pecos install llvm --no-configure
```

This downloads and installs LLVM 14 to `~/.pecos/llvm/`.

### Check LLVM Status

```bash
pecos llvm check

# Quiet mode (exit code only, for scripting)
pecos llvm check --quiet
```

### Configure Cargo

```bash
pecos llvm configure
```

Updates `.cargo/config.toml` with the correct `LLVM_SYS_140_PREFIX` environment variable.

### Find LLVM Path

```bash
# Find LLVM installation
pecos llvm find

# Export for shell evaluation
pecos llvm find --export
```

## Dependency Management Details

### List Dependencies

```bash
pecos deps list
```

Shows all available external dependencies defined in `pecos.toml`.

### Sync Manifests

```bash
pecos deps sync
```

Syncs crate-level `pecos.toml` manifests from the workspace-level manifest.

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
pecos llvm check

# 2. If not, install it
pecos install llvm

# 3. Now you can build with LLVM support
cargo build -p pecos --features llvm
```

Or using Justfile:
```bash
just llvm-install
just build
```

### Running Lints Before Committing

```bash
just lint
just fmt-check
```

### Building and Testing Python

```bash
just python-build
just python-test
```

### Cleaning Up

```bash
# Clean build artifacts
just clean
```

## See Also

- [LLVM Setup](../user-guide/llvm-setup.md) - Detailed LLVM installation guide
- [Development Guide](DEVELOPMENT.md) - Contributing to PECOS
