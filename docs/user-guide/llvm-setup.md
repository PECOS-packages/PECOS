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

### Option 1: PECOS-Managed Installation (Recommended Where Available)

Use the `pecos` CLI (`pecos install llvm`, or `cargo run -p pecos-cli -- install llvm` in a source checkout) to automatically download and install LLVM 21.1.8 where PECOS can provide a verified shared LLVM package:

```bash
# Install LLVM 21.1.8 to ~/.pecos/deps/llvm-21.1/
cargo run -p pecos-cli -- install llvm

# Build PECOS with LLVM support
cargo build --features llvm
```

The `install` command automatically:

- Downloads a shared LLVM toolchain on supported platforms
- Extracts it to `~/.pecos/deps/llvm-21.1/`
- Configures PECOS by updating `.cargo/config.toml`

This is the **recommended approach** where PECOS can provide a verified shared
LLVM package. On Debian/Ubuntu-compatible Linux distributions, PECOS downloads
the apt.llvm.org LLVM 21 packages into `~/.pecos/deps/llvm-21.1/` without using
`sudo`. On macOS, use Homebrew for LLVM 21. On Windows MSVC, use the
conda-forge helper in the Windows section below; it installs a full LLVM
development environment under `~/.pecos/deps/llvm-21.1/` and configures
`~/.pecos/deps/llvm-21.1/Library` as the LLVM prefix.

This is a developer toolchain install: the CLI prints the install size/behavior
and asks for confirmation before downloading. Use `--yes` to accept the prompt
in scripts. Depending on platform and archive layout, the extracted toolchain
can occupy several GB.

### Option 2: System Package Manager

Install LLVM 21.1 using your system's package manager, then configure PECOS:

=== "macOS"
    ```bash
    brew install llvm@21
    cargo run -p pecos-cli -- llvm configure
    cargo build --features llvm
    ```

    Works on both Intel and Apple Silicon Macs.

=== "Linux (Debian/Ubuntu)"
    ```bash
    sudo apt update
    sudo apt install llvm-21 llvm-21-dev
    cargo run -p pecos-cli -- llvm configure
    cargo build --features llvm
    ```

    If your distribution repositories do not provide LLVM 21, use the LLVM
    project's Debian/Ubuntu repository at <https://apt.llvm.org/>.

=== "Linux (Fedora/RHEL)"
    ```bash
    sudo dnf install llvm21 llvm21-devel
    cargo run -p pecos-cli -- llvm configure
    cargo build --features llvm
    ```

=== "Linux (Arch)"
    ```bash
    yay -S llvm21  # May need to build from AUR
    cargo run -p pecos-cli -- llvm configure
    cargo build --features llvm
    ```

=== "Windows"
    !!! warning "Windows LLVM Requirement"
        The official LLVM Windows installer (`LLVM-*.exe`) is **toolchain-only** and lacks required development files (`llvm-config.exe`, headers, and `libclang.dll`). Use a full LLVM development package built for the MSVC dynamic runtime.

    **Recommended for full development tests:** Use the PECOS conda-forge helper:

    ```powershell
    .\scripts\ci\install-llvm-21-windows.ps1 -InstallDir "$env:USERPROFILE\.pecos\deps\llvm-21.1"
    cargo run -p pecos-cli -- llvm configure "$env:USERPROFILE\.pecos\deps\llvm-21.1\Library"
    cargo build --features llvm
    ```

    **Alternative:** Configure another full LLVM 21.1 development package that includes `llvm-config.exe`, headers, static MSVC libraries built against the dynamic runtime, and `libclang.dll`.

    - [bitgate/llvm-windows-full-builds](https://github.com/bitgate/llvm-windows-full-builds) (recommended)
    - [vovkos/llvm-package-windows](https://github.com/vovkos/llvm-package-windows)

    Extract to `C:\LLVM`, then:

    ```cmd
    set LLVM_SYS_211_PREFIX=C:\LLVM
    cargo run -p pecos-cli -- llvm configure C:\LLVM
    cargo build --features llvm
    ```

## Verifying Installation

After installing LLVM, you can verify the installation using these commands:

```bash
# Check if LLVM 21.1 is detected
cargo run -p pecos-cli -- llvm check

# Show LLVM version and path
cargo run -p pecos-cli -- llvm version

# Find LLVM installation path
cargo run -p pecos-cli -- llvm find
```

`llvm check` also reports LLVM's link mode. PECOS Rust builds prefer
`libLLVM-21.so` when a shared LLVM 21.1 installation is available; static LLVM
is only suitable for targeted builds.

For `pecos rust test` and `just dev`, PECOS requires shared LLVM. On a Linux
x86_64 developer machine, one static LLVM test link measured about 4 GB peak
RSS, while the same target linked against shared LLVM measured about 0.8 GB.
Failing early on static LLVM is intentional: full workspace tests can spawn
many LLVM-linking test binaries at once.

## `pecos llvm` CLI Reference

The `pecos llvm` subcommand provides several useful commands:

### `install`

Download and install LLVM 21.1.8 to `~/.pecos/deps/llvm-21.1/` on supported
platforms:

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

# Or explicitly configure a user-managed LLVM installation
cargo run -p pecos-cli -- llvm configure /path/to/llvm
```

This updates `.cargo/config.toml` with the LLVM path.

Explicit paths are canonicalized before being written. If `/path/to/llvm` is a
symlink, PECOS writes the resolved LLVM directory into `.cargo/config.toml`; run
`pecos llvm configure /path/to/llvm` again after repointing that symlink.

### `check`

Verify LLVM 21.1 is available:

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

PECOS specifically requires **LLVM version 21.1.x** (21.1.x). Other versions are not compatible with the current implementation.

### Configuration File

The `configure` command updates `.cargo/config.toml` in the project root with:

```toml
[env]
LLVM_SYS_211_PREFIX = { value = "/path/to/llvm", force = true }
```

**Important notes:**

- This file is auto-generated and machine-specific
- It's in `.gitignore` and should not be committed
- The `force = true` setting ensures the configured LLVM path takes priority over environment variables

### Shared vs Static LLVM

PECOS enables inkwell's `llvm21-1-prefer-dynamic` feature. That means Rust
builds use `libLLVM-21.so` / `libLLVM.dylib` when `llvm-config --link-shared`
can provide it. The managed installer rejects static LLVM because the normal
development test path links many LLVM-using test binaries.

System package manager installs usually provide shared LLVM. On Debian/Ubuntu
compatible Linux distributions, the managed installer uses the apt.llvm.org
LLVM 21 packages locally under `~/.pecos/deps/llvm-21.1/`, without installing
system packages.

When LLVM is shared, PECOS CLI commands add LLVM's `libdir` to the runtime
library path for child Cargo commands. That lets locally configured shared LLVM
installs work without editing your shell startup files.

### Detection Priority

Build commands that need to match Cargo's behavior first honor
`.cargo/config.toml` if it sets `LLVM_SYS_211_PREFIX`, then fall back to the
normal detector. The normal `pecos llvm` detector searches for LLVM 21.1 in
this order:

1. **Home directory:**
   - Windows: `~/.pecos/deps/llvm-21.1`
   - Unix: `~/.pecos/deps/llvm-21.1`

2. **Legacy home directory:** `~/.pecos/llvm`

3. **Project-local:** `<repo-root>/llvm/`

4. **System installations:**
   - **macOS:** Homebrew locations (`/opt/homebrew/opt/llvm@21`, `/usr/local/opt/llvm@21`)
   - **Linux:** Via `llvm-config-21` command and common paths
   - **Windows:** Common paths (`C:\Program Files\LLVM`, `C:\LLVM`, etc.)

### Platform-Specific Notes

**macOS:**

- Use Homebrew LLVM 21: `brew install llvm@21`
- Automatically detects Homebrew installations

**Linux:**

- Detects system LLVM via `llvm-config-21` command
- Managed install uses apt.llvm.org on Debian/Ubuntu-compatible x86_64 and
  aarch64 systems
- Other Linux distributions should install shared LLVM 21 through their package
  manager and run `pecos llvm configure /path/to/llvm`

**Windows:**

- The official LLVM installer is not sufficient for PECOS development builds
- Use `scripts\ci\install-llvm-21-windows.ps1` for the conda-forge LLVM 21.1 toolchain
- Configure `~\.pecos\deps\llvm-21.1\Library`, not the conda environment root

### Security

Linux managed packages are checked against apt metadata hashes. Windows helper
packages are installed by micromamba from conda-forge, using conda package
metadata and checksums.

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

PECOS requires LLVM 21.1.x. If you have multiple LLVM versions installed, the tool will prioritize LLVM 21.1. Use the `find` command to see which installation is detected:

```bash
cargo run -p pecos-cli -- llvm find
```

### Manual configuration

If automatic configuration doesn't work, you can manually set the environment variable:

```bash
# Unix/macOS
export LLVM_SYS_211_PREFIX=/path/to/llvm

# Windows
set LLVM_SYS_211_PREFIX=C:\path\to\llvm
```

Or add to `.cargo/config.toml`:

```toml
[env]
LLVM_SYS_211_PREFIX = { value = "/path/to/llvm", force = true }
```

## PECOS Home Directory

LLVM is installed to `~/.pecos/deps/llvm-21.1/`, which is part of the PECOS home directory structure:

```
~/.pecos/
├── deps/
│   ├── llvm-21.1/  # LLVM 21.1 installation
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
