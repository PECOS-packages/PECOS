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

### LLVM for QIR Support

LLVM version 14 is required for QIR (Quantum Intermediate Representation) support:

=== "Linux"
    ```bash
    sudo apt install llvm-14
    ```

=== "macOS"
    ```bash
    brew install llvm@14
    ```

=== "Windows"
    Download LLVM 14.x installer from [LLVM releases](https://releases.llvm.org/download.html#14.0.0)

!!! warning
    PECOS's QIR implementation is currently only compatible with LLVM version 14.x.

If LLVM 14 is not installed, PECOS will still function normally but QIR-related features will be disabled.

### Simulators with Special Requirements

Some simulators from `pecos.simulators` require external packages:

- **QuEST**: Installed with the Python package `pyquest` via `pip install .[all]`. For 32-bit float point precision, follow the installation instructions [here](https://github.com/rrmeister/pyQuEST/tree/develop).

- **CuStateVec**: Requires a Linux machine with an NVIDIA GPU. Installation via conda is recommended, as discussed [here](https://docs.nvidia.com/cuda/cuquantum/latest/getting_started/getting_started.html#installing-cuquantum).

- **MPS**: Uses `pytket-cutensornet` and can be installed via `pip install .[cuda]`. These simulators use NVIDIA GPUs and cuQuantum. Follow the instructions for `CuStateVec` above to install cuQuantum.

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
