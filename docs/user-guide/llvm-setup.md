# LLVM Setup Guide

!!! note "Python Users"
    **Python users can skip this guide entirely.** Pre-built Python wheels already include LLVM support, so no additional setup is required.

This guide is for **Rust users building PECOS from source** who need LLVM support for QIS (Quantum Instruction Set) with LLVM IR/QIR execution.

## When is LLVM Needed?

LLVM is **optional** and only required when building PECOS Rust crates with the `llvm` feature flag enabled.

```bash
# Build without LLVM (default)
cargo build

# Build with LLVM support
cargo build --features llvm
```

If you don't need QIS LLVM IR/QIR execution features, you can skip LLVM installation entirely.

## Installation Options

### Option 1: Automatic Installation (Recommended)

Use the `pecos-llvm` CLI tool to automatically download and install LLVM 14.0.6:

```bash
# Install LLVM 14.0.6 to ~/.pecos/deps/llvm-14/ (~400MB, ~5 minutes)
cargo run -p pecos-cli -- install llvm

# Build PECOS with LLVM support
cargo build --features llvm
```

The `install` command automatically:

- Downloads the correct LLVM binary for your platform
- Extracts it to `~/.pecos/deps/llvm-14/`
- Configures PECOS by updating `.cargo/config.toml`

This is the **recommended approach** for all platforms, especially Windows where system package managers may not provide LLVM 14 development files.

### Option 2: System Package Manager

Install LLVM 14 using your system's package manager, then configure PECOS:

=== "macOS"
    ```bash
    brew install llvm@14
    cargo run -p pecos-cli -- llvm configure
    cargo build --features llvm
    ```

    Works on both Intel and Apple Silicon Macs.

=== "Linux (Debian/Ubuntu)"
    ```bash
    sudo apt update
    sudo apt install llvm-14 llvm-14-dev
    cargo run -p pecos-cli -- llvm configure
    cargo build --features llvm
    ```

=== "Linux (Fedora/RHEL)"
    ```bash
    sudo dnf install llvm14 llvm14-devel
    cargo run -p pecos-cli -- llvm configure
    cargo build --features llvm
    ```

=== "Linux (Arch)"
    ```bash
    yay -S llvm14  # May need to build from AUR
    cargo run -p pecos-cli -- llvm configure
    cargo build --features llvm
    ```

=== "Windows"
    !!! warning "Windows LLVM Requirement"
        The official LLVM Windows installer (`LLVM-*.exe`) is **toolchain-only** and lacks required development files (`llvm-config.exe` and headers).

    **Recommended:** Use Option 1 (automatic installation) above.

    **Alternative:** Download a full development package from:

    - [bitgate/llvm-windows-full-builds](https://github.com/bitgate/llvm-windows-full-builds) (recommended)
    - [vovkos/llvm-package-windows](https://github.com/vovkos/llvm-package-windows)

    Extract to `C:\LLVM`, then:

    ```cmd
    set LLVM_SYS_140_PREFIX=C:\LLVM
    cargo run -p pecos-cli -- llvm configure
    cargo build --features llvm
    ```

## Verifying Installation

After installing LLVM, you can verify the installation using these commands:

```bash
# Check if LLVM 14 is detected
cargo run -p pecos-cli -- llvm check

# Show LLVM version and path
cargo run -p pecos-cli -- llvm version

# Find LLVM installation path
cargo run -p pecos-cli -- llvm find
```

## pecos-llvm CLI Reference

The `pecos llvm` CLI tool provides several useful commands:

### `install`

Download and install LLVM 14.0.6 to `~/.pecos/deps/llvm-14/`:

```bash
cargo run -p pecos-cli -- install llvm

# Reinstall even if already present
cargo run -p pecos-cli -- install llvm --force

# Skip automatic configuration after install
cargo run -p pecos-cli -- install llvm --no-configure
```

### `configure`

Auto-configure PECOS to use detected LLVM installation:

```bash
cargo run -p pecos-cli -- llvm configure
```

This updates `.cargo/config.toml` with the LLVM path.

### `check`

Verify LLVM 14 is available:

```bash
cargo run -p pecos-cli -- llvm check

# Suppress output messages
cargo run -p pecos-cli -- llvm check --quiet
```

Exit code: 0 if found, 1 if not found.

### `version`

Show LLVM version information:

```bash
cargo run -p pecos-cli -- llvm version
```

### `find`

Locate LLVM installation:

```bash
# Print LLVM path
cargo run -p pecos-cli -- llvm find

# Print export command for shell evaluation
cargo run -p pecos-cli -- llvm find --export
```

### `validate`

Verify LLVM installation integrity:

```bash
cargo run -p pecos-cli -- llvm validate /path/to/llvm
```

Checks for critical files, libraries, headers, and runtime functionality.

### `tool`

Find specific LLVM tools:

```bash
cargo run -p pecos-cli -- llvm tool llvm-as
cargo run -p pecos-cli -- llvm tool clang
cargo run -p pecos-cli -- llvm tool llvm-link
```

## Technical Details

### Version Requirement

PECOS specifically requires **LLVM version 14.x** (14.0.x). Other versions are not compatible with the current implementation.

### Configuration File

The `configure` command updates `.cargo/config.toml` in the project root with:

```toml
[env]
LLVM_SYS_140_PREFIX = { value = "/path/to/llvm", force = true }
```

**Important notes:**

- This file is auto-generated and machine-specific
- It's in `.gitignore` and should not be committed
- The `force = true` setting ensures the configured LLVM path takes priority over environment variables

### Detection Priority

The `pecos-llvm` tool searches for LLVM 14 in this order:

1. **Home directory:**
   - Windows: `~/.pecos/deps/llvm-14`
   - Unix: `~/.pecos/deps/llvm-14`

2. **Project-local:** `<repo-root>/llvm/`

3. **System installations:**
   - **macOS:** Homebrew locations (`/opt/homebrew/opt/llvm@14`, `/usr/local/opt/llvm@14`)
   - **Linux:** Via `llvm-config-14` command and common paths
   - **Windows:** Common paths (`C:\Program Files\LLVM`, `C:\LLVM`, etc.)

### Platform-Specific Notes

**macOS:**

- Supports both Intel and Apple Silicon architectures
- Automatically detects Homebrew installations
- Downloads appropriate binary for each platform

**Linux:**

- Detects system LLVM via `llvm-config-14` command
- Supports x86_64 and aarch64 architectures

**Windows:**

- Uses `.7z` archives for distribution
- Pure Rust extraction (no external tools required)
- Official LLVM Windows installer lacks development files - use `pecos-llvm install` or community packages

### Security

All downloaded LLVM packages are verified with SHA256 checksums to ensure integrity.

## Troubleshooting

### LLVM not found after installation

Run the `configure` command to update `.cargo/config.toml`:

```bash
cargo run -p pecos-cli -- llvm configure
```

### Build fails with LLVM errors

Verify LLVM is correctly installed and detected:

```bash
cargo run -p pecos-cli -- llvm check
cargo run -p pecos-cli -- llvm version
```

### Wrong LLVM version detected

PECOS requires LLVM 14.x. If you have multiple LLVM versions installed, the tool will prioritize LLVM 14. Use the `find` command to see which installation is detected:

```bash
cargo run -p pecos-cli -- llvm find
```

### Manual configuration

If automatic configuration doesn't work, you can manually set the environment variable:

```bash
# Unix/macOS
export LLVM_SYS_140_PREFIX=/path/to/llvm

# Windows
set LLVM_SYS_140_PREFIX=C:\path\to\llvm
```

Or add to `.cargo/config.toml`:

```toml
[env]
LLVM_SYS_140_PREFIX = { value = "/path/to/llvm", force = true }
```

## PECOS Home Directory

LLVM is installed to `~/.pecos/deps/llvm-14/`, which is part of the PECOS home directory structure:

```
~/.pecos/
├── deps/
│   ├── llvm/       # LLVM-14 installation
│   ├── cuda/       # CUDA Toolkit
│   └── cuquantum/  # cuQuantum SDK
└── cache/          # Build artifacts
```

You can override the PECOS home location using the `PECOS_HOME` environment variable or in `.cargo/config.toml`:

```toml
[env]
PECOS_HOME = { value = "/custom/path", force = true }
```

For more details, see the [Development Guide](../development/DEVELOPMENT.md#pecos-home-directory).

## See Also

- [Getting Started Guide](getting-started.md) - Main installation guide
- [Development Guide](../development/DEVELOPMENT.md) - Developer setup and PECOS home directory
