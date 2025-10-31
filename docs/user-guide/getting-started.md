# Getting Started

This guide will help you get up and running with PECOS quickly, whether you're using the Python package, the Rust crates, or both.

## Installation

=== "Python"

    To install the main Python package for general usage:

    ```bash
    pip install quantum-pecos
    ```

    This will install both `quantum-pecos` and its dependency `pecos-rslib`.

    For optional dependencies that should work on all systems:

    ```bash
    pip install quantum-pecos[all]
    ```

    !!! note "Import Name"
        The `quantum-pecos` package is imported as `import pecos` and not `import quantum_pecos`.

    To install pre-releases (the latest development code) from PyPI:

    ```bash
    pip install quantum-pecos==X.Y.Z.devN  # Replace with actual version number
    ```

=== "Rust"

    To use PECOS in your Rust project, add the following to your `Cargo.toml`:

    ```toml
    [dependencies]
    pecos-core = "0.1.x"  # Replace with the latest version
    # Add other PECOS crates as needed:
    # pecos-engines = "0.1.x"
    # pecos-qsim = "0.1.x"
    ```

## Optional Dependencies

### LLVM for QIS Support

LLVM version 14 is required for LLVM IR execution with QIS (Quantum Instruction Set) support.

**Setup Steps:**

**Option 1 - Use pecos-llvm installer (recommended for all platforms):**

```bash
# Install LLVM 14.0.6 to ~/.pecos/llvm/ (~400MB, ~5 minutes)
cargo run -p pecos-llvm-utils --bin pecos-llvm -- install

# Build PECOS
cargo build
```

The installer automatically configures PECOS after installation.

**Option 2 - Manual installation:**

1. **Install LLVM 14** for your platform:

   === "macOS"
       ```bash
       brew install llvm@14
       ```
       Works on both Intel and Apple Silicon Macs.

   === "Linux (Debian/Ubuntu)"
       ```bash
       sudo apt update
       sudo apt install llvm-14 llvm-14-dev
       ```

   === "Linux (Fedora/RHEL)"
       ```bash
       sudo dnf install llvm14 llvm14-devel
       ```

   === "Linux (Arch)"
       ```bash
       yay -S llvm14  # May need to build from AUR
       ```

   === "Windows"
       !!! warning "Windows LLVM Requirement"
           The official LLVM Windows installer (`LLVM-*.exe`) is **toolchain-only** and lacks required development files (`llvm-config.exe` and headers). You need a **full development package**.

       **Recommended: Use pecos-llvm installer** (see Option 1 above)

       **For system-wide installation:**

       Download a full development package from community sources:

       - [bitgate/llvm-windows-full-builds](https://github.com/bitgate/llvm-windows-full-builds) (recommended)
       - [vovkos/llvm-package-windows](https://github.com/vovkos/llvm-package-windows)

       Extract to a location like `C:\LLVM` or `C:\Program Files\LLVM-14`, then set:
       ```cmd
       set LLVM_SYS_140_PREFIX=C:\LLVM
       ```

2. **Configure PECOS** to detect your LLVM installation:
   ```bash
   cargo run -p pecos-llvm-utils --bin pecos-llvm -- configure
   ```

3. **Build PECOS**:
   ```bash
   cargo build
   ```

**Check LLVM Status:**

```bash
cargo run -p pecos-llvm-utils --bin pecos-llvm -- check
cargo run -p pecos-llvm-utils --bin pecos-llvm -- version
```

!!! warning
    PECOS's LLVM IR implementation is currently only compatible with LLVM version 14.x.

!!! note
    The `.cargo/config.toml` file is auto-generated and machine-specific. It's in `.gitignore` and should not be committed.

### Simulators with Special Requirements

Some simulators from `pecos.simulators` require external packages:

- **QuEST**: Installed with the Python package `pyquest` via `pip install .[all]`. For 32-bit float point precision, follow the installation instructions [here](https://github.com/rrmeister/pyQuEST/tree/develop).

- **CuStateVec** and **MPS** (GPU simulators): Require NVIDIA GPU, CUDA Toolkit 13/12, and additional Python packages. See the comprehensive [CUDA Setup Guide](cuda-setup.md) for detailed installation instructions.

    Quick install (after installing CUDA Toolkit):
    ```bash
    uv pip install quantum-pecos[cuda]
    ```

## Verification

Verify your installation:

=== "Python"
    ```python
    import pecos

    print(pecos.__version__)
    ```

=== "Rust"
    Create a simple Rust program and run:

    ```rust
    // This example assumes you have added pecos-core to your Cargo.toml
    // use pecos_core;

    fn main() {
        println!("PECOS Rust crates would be loaded here!");
        // Once loaded, you can use PECOS functionality
    }
    ```
