# Getting Started

This guide will help you get up and running with PECOS quickly, whether you're using the Python package, the Rust crates, or both.

## Installation

=== ":fontawesome-brands-python: Python"

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

=== ":fontawesome-brands-rust: Rust"

    To use PECOS in your Rust project, add the following to your `Cargo.toml`:

    ```toml
    [dependencies]
    pecos = "0.1.x"  # Replace with the latest version
    ```

    The `pecos` crate is a metacrate that re-exports functionality from all PECOS crates.

    For specific functionality, you can alternatively depend on individual crates:

    ```toml
    [dependencies]
    pecos-core = "0.1.x"
    pecos-engines = "0.1.x"
    pecos-qsim = "0.1.x"
    # etc.
    ```

## Optional Dependencies

### LLVM for QIS Support (Rust Only)

!!! note "Python Users"
    **Python users can skip this section.** Pre-built Python wheels already include LLVM support, so no additional setup is required.

For **Rust users building from source**, LLVM version 14 is optional and only needed for QIS (Quantum Instruction Set) with LLVM IR/QIR execution support.

**Quick Setup (Recommended):**

```bash
# Install LLVM 14.0.6 to ~/.pecos/llvm/ (~400MB, ~5 minutes)
cargo run -p pecos-llvm-utils --bin pecos-llvm -- install

# Build PECOS with LLVM support
cargo build --features llvm
```

The `pecos-llvm install` command automatically downloads, installs, and configures LLVM for your platform.

!!! info "Detailed Setup Instructions"
    For complete LLVM installation options, system package manager instructions, troubleshooting, and CLI reference, see the [LLVM Setup Guide](llvm-setup.md).

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

=== ":fontawesome-brands-python: Python"
    ```python
    import pecos

    print(pecos.__version__)
    ```

=== ":fontawesome-brands-rust: Rust"
    Create a simple Rust program and run:

    ```rust
    // This example assumes you have added pecos-core to your Cargo.toml
    // use pecos_core;

    fn main() {
        println!("PECOS Rust crates would be loaded here!");
        // Once loaded, you can use PECOS functionality
    }
    ```
