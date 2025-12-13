# Developer Tools CLI

The `pecos-dev` CLI provides tools for PECOS development, including LLVM setup, dependency management, and build utilities.

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
PECOS developer tools - LLVM setup, dependency management, and build utilities

Commands:
  info   Show PECOS home directory info and status
  list   List installed and cached dependencies
  clean  Clean cached dependencies
  llvm   LLVM management commands
  deps   Dependency manifest management (pecos.toml)
```

## LLVM Management

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

# Specify installation directory
pecos-dev llvm install --prefix ~/.local/llvm14
```

This downloads and installs LLVM 14 to the PECOS home directory (`~/.pecos/llvm/` by default).

### Configure Cargo

```bash
pecos-dev llvm configure
```

Updates `.cargo/config.toml` with the correct `LLVM_SYS_140_PREFIX` environment variable so Rust crates can find LLVM.

### Find LLVM Path

```bash
# Find LLVM installation
pecos-dev llvm find

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

## Dependency Management

The `deps` subcommand manages the `pecos.toml` manifest which tracks external dependencies (C++ libraries, git repositories, etc.).

### Show Status

```bash
pecos-dev deps status
```

Shows the current manifest, listing all dependencies and which crates use them:

```
Manifest Status
===============

pecos.toml: /path/to/PECOS/pecos.toml
  Version: 1
  LLVM: version 14 (required: true)

  Crates (5):
    pecos: none [LLVM]
    pecos-engines: none [LLVM]
    pecos-ldpc-decoders: stim, pymatching, ldpc, tesseract, chromobius, boost
    pecos-quest: quest
    pecos-qulacs: qulacs, eigen, boost

  Dependencies (9):
    boost: 1.83.0 - C++ Boost libraries
    eigen: 3.4.0 - C++ linear algebra library
    quest: v4.1.0 - QuEST quantum simulator
    ...
```

### Initialize Manifest

```bash
pecos-dev deps init
```

Creates a new `pecos.toml` manifest in the current directory.

### Sync Manifests

```bash
pecos-dev deps sync
```

Synchronizes crate-level `pecos.toml` files from the workspace manifest.

### Verify Dependencies

```bash
pecos-dev deps verify
```

Downloads dependencies and verifies their checksums match the manifest.

## Cache Management

### Show Info

```bash
pecos-dev info
```

Displays PECOS home directory location and cached items:

```
PECOS Home Directory
====================

Location: /home/user/.pecos

Directories:
  Cache:  /home/user/.pecos/cache
  Deps:   /home/user/.pecos/deps
  LLVM:   /home/user/.pecos/llvm
  Temp:   /home/user/.pecos/tmp

Environment:
  PECOS_HOME: (not set, using default)
```

### List Cached Items

```bash
pecos-dev list
```

Shows all installed and cached dependencies.

### Clean Cache

```bash
# Clean all cached items
pecos-dev clean

# Clean only temporary files
pecos-dev clean --temp

# Clean only downloaded dependencies
pecos-dev clean --deps
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PECOS_HOME` | PECOS cache and data directory | `~/.pecos` |
| `LLVM_SYS_140_PREFIX` | LLVM 14 installation path | auto-detected |

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

### Building with External Dependencies

```bash
# 1. Check what dependencies are needed
pecos-dev deps status

# 2. Verify dependencies are available
pecos-dev deps verify

# 3. Build the crate
cargo build -p pecos-quest
```

### Cleaning Up

```bash
# See what's cached
pecos-dev info
pecos-dev list

# Clean everything
pecos-dev clean
```

## Integration with pecos CLI

The `pecos` CLI forwards certain commands to `pecos-dev` if it's installed:

```bash
# These commands are forwarded to pecos-dev
pecos llvm check      # -> pecos-dev llvm check
pecos deps status     # -> pecos-dev deps status
pecos info            # Shows pecos info (different from pecos-dev info)
```

If `pecos-dev` is not installed, these commands show a helpful message:

```
Command 'llvm' requires pecos-dev.

Install with:
  cargo install pecos-dev
```

## pecos.toml Format

The `pecos.toml` manifest tracks external dependencies:

```toml
version = 1

[llvm]
version = "14"
required = true
required_by = ["pecos-engines", "pecos"]

[dependencies.quest]
version = "v4.1.0"
repository = "https://github.com/QuEST-Kit/QuEST"
description = "QuEST quantum simulator"
sha256 = "abc123..."

[dependencies.boost]
version = "1.83.0"
url = "https://archives.boost.io/release/1.83.0/source/boost_1_83_0.tar.gz"
description = "C++ Boost libraries"
sha256 = "def456..."

[crates.pecos-quest]
dependencies = ["quest"]

[crates.pecos-qulacs]
dependencies = ["qulacs", "eigen", "boost"]
```

## See Also

- [LLVM Setup](../user-guide/llvm-setup.md) - Detailed LLVM installation guide
- [QIS Architecture](QIS_ARCHITECTURE.md) - QIS system design
- [Development Guide](DEVELOPMENT.md) - Contributing to PECOS
