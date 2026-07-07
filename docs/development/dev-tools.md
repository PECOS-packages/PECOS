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
pecos rust test               # Run cargo test (CUDA-aware; default profile=dev)
pecos rust test --profile release   # Same but with release optimisations
pecos rust test --profile native    # Release + -C target-cpu=native + --march=native for C++

# Python build (maturin + quantum-pecos)
pecos python build            # Build pecos-rslib with maturin (default profile=dev)
pecos python build --profile release  # Release build
pecos python build --profile native   # Release + native-CPU codegen (Rust and C++)

# Dependency installation
pecos install llvm            # Install managed LLVM 21.1 where supported
pecos install cuda            # Install CUDA Toolkit to ~/.pecos/deps/cuda/
pecos install cuquantum       # Install cuQuantum SDK to ~/.pecos/deps/cuquantum/
pecos install --all           # Install all optional dependencies
pecos uninstall llvm          # Uninstall LLVM
pecos upgrade llvm            # Upgrade (force reinstall) LLVM

# Inspection
pecos llvm check              # Check LLVM installation status
pecos llvm configure          # Configure .cargo/config.toml using detected LLVM
pecos llvm configure /path/to/llvm  # Configure an explicit user-managed LLVM
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

# Dependency and security policy
just security-check            # Dependency integrity + cargo-deny checks
just cargo-deny                # Rust dependency policy checks

# Cleaning
just clean                     # Clean build artifacts
```

Run `just --list` to see all available commands.

## LLVM Management Details

### Install LLVM

```bash
# Automated installation where PECOS can provide shared LLVM
pecos install llvm

# Accept the managed-install prompt
pecos install llvm --yes

# Force reinstall
pecos install llvm --force

# Skip automatic configuration
pecos install llvm --no-configure
```

On Debian/Ubuntu-compatible Linux systems this downloads apt.llvm.org shared
LLVM packages into `~/.pecos/deps/llvm-21.1/` without `sudo`. The managed
install is the preferred developer path where it is available, but it is a
large toolchain install. `pecos install llvm` prints what it is about to install
and asks for confirmation before downloading.

On macOS, install Homebrew LLVM 21 (`brew install llvm@21`) and run
`pecos llvm configure`. On native Windows MSVC, use
`scripts\ci\install-llvm-21-windows.ps1` to install the conda-forge LLVM 21.1
toolchain, then configure `~\.pecos\deps\llvm-21.1\Library`.

`pecos rust test` requires shared LLVM for the workspace HUGR test lane. LLVM
21.1 static test links can use multiple GB of RAM each, so PECOS fails early
instead of letting `just dev` spawn enough concurrent linkers to overwhelm a
normal development machine.

### Check LLVM Status

```bash
pecos llvm check

# Quiet mode (exit code only, for scripting)
pecos llvm check --quiet
```

### Configure Cargo

```bash
pecos llvm configure

# Or explicitly use a system/Homebrew/apt LLVM instead of the managed install
pecos llvm configure /path/to/llvm
```

Updates `.cargo/config.toml` with the correct `LLVM_SYS_211_PREFIX` environment variable.
Explicit paths are canonicalized, so configuring a symlink records the resolved
LLVM directory. Re-run `pecos llvm configure /path/to/llvm` after repointing the
symlink.

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

## Dependency and Security Policy

Run these recipes when changing dependencies, lockfiles, CI workflows, action references, cache behavior, or security policy:

```bash
just security-check            # Full local dependency/security policy check
just dependency-integrity-check # Lock discipline, CI posture, action pins, cache posture
just cargo-deny                # Both Rust cargo-deny checks covered by CI
just cargo-deny-workspace      # Root Rust workspace only
just cargo-deny-native-bench   # Standalone native benchmark crate only
```

`cargo-deny` checks the resolved Rust dependency graph against `deny.toml`. In this repo it checks:

- `advisories`: known Rust security advisories
- `bans`: disallowed crates or dependency patterns
- `sources`: approved registries and git sources

Install the same `cargo-deny` version used by CI before running these locally:

```bash
cargo install --locked --version 0.19.6 cargo-deny
```

Use `just check-all` before a broad PR; it runs the build, tests, lint gate, and `just security-check`.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PECOS_HOME` | PECOS cache and data directory | `~/.pecos` |
| `LLVM_SYS_211_PREFIX` | LLVM 21.1 installation path | auto-detected |
| `RUST_LOG` | Log level for build output (`info` shows download progress) | `warn` |

## Typical Workflows

### Setting Up LLVM for the First Time

```bash
# 1. Check if LLVM is already available
pecos llvm check

# 2. If not, install it where managed shared LLVM is supported
pecos install llvm

# 3. Now you can build with LLVM support
cargo build -p pecos --features llvm
```

On macOS use `brew install llvm@21 && pecos llvm configure`. On native Windows
MSVC, use `scripts\ci\install-llvm-21-windows.ps1` and configure
`~\.pecos\deps\llvm-21.1\Library`.

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
