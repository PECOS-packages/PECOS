# PECOS Development Tools

PECOS provides development tools through the `pecos` CLI and the `Justfile`.

## pecos CLI Commands

The `pecos` CLI handles dependency management, CUDA-aware builds, and system inspection.
Daily dev workflows (fmt, test, lint, bench) live in the Justfile.

```bash
# Show all available commands
pecos --help

# Rust commands (CUDA-aware)
pecos rust check              # Run cargo check (auto-excludes CUDA if unavailable)
pecos rust clippy             # Run cargo clippy (CUDA-aware)
pecos rust test               # Run cargo test (CUDA-aware)

# Python build (maturin + quantum-pecos)
pecos python build            # Build pecos-rslib with maturin

# Dependency installation
pecos install llvm            # Install LLVM 14 to ~/.pecos/deps/llvm-14/
pecos install cuda            # Install CUDA Toolkit to ~/.pecos/deps/cuda/
pecos install cuquantum       # Install cuQuantum SDK to ~/.pecos/deps/cuquantum/
pecos install --all           # Install all optional dependencies
pecos uninstall llvm          # Uninstall LLVM
pecos upgrade llvm            # Upgrade (force reinstall) LLVM

# Inspection
pecos llvm check              # Check LLVM installation status
pecos llvm configure          # Configure .cargo/config.toml
pecos cuda check              # Check CUDA availability
pecos sys-info                # Show toolchain and environment info

# Selene plugin management
pecos selene install          # Install Selene plugins
pecos selene list             # List plugin status

# Dependency manifests
pecos deps list               # List available dependencies
pecos deps sync               # Sync dependency manifests
```

When running from the repository:
```bash
cargo run -p pecos-cli -- install llvm
cargo run -p pecos-cli -- rust clippy
```

## Build and Test Commands (Justfile)

Most development tasks are managed through the Justfile. Make sure you have `just` installed:

```bash
# Install just
cargo install just
```

### Quick Reference

```bash
# Setup & diagnosis
just install-cli               # Install the pecos CLI
just setup                     # Detect and install missing dependencies
just doctor                    # Check dev environment for common problems

# Building
just build                     # Build all (debug, default)
just build release             # Build all (release)

# Testing
just test                      # Run all tests (Rust + Python + Julia + Go)
just rstest                    # Rust tests only (CUDA-aware, via CLI)
just pytest                    # Python tests only

# Code quality
just lint                      # Run all checks (fmt + clippy + pre-commit + Julia/Go)
just lint-fix                  # Auto-fix all fixable issues
just fmt                       # Check Rust formatting
just fmt-fix                   # Fix Rust formatting
just clippy                    # Run clippy

# Cleaning
just clean                     # Clean build artifacts
```

Run `just --list` to see all available commands.

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

This downloads and installs LLVM 14 to `~/.pecos/deps/llvm-14/`.

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
just install-llvm
just build
```

### Daily Development

```bash
just build                # Build everything
just test                 # Run all tests
just lint                 # Check formatting and linting
just doctor               # Diagnose environment issues
```

## See Also

- [LLVM Setup](../user-guide/llvm-setup.md) - Detailed LLVM installation guide
- [Development Guide](DEVELOPMENT.md) - Contributing to PECOS
