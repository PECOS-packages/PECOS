# Installation

This guide provides comprehensive instructions for installing PECOS in both Python and Rust environments. PECOS offers multiple installation methods to suit different workflows and requirements.

## System Requirements

PECOS is compatible with the following platforms:

- **Operating Systems**:
  - Linux (Ubuntu, Debian, CentOS, etc.)
  - macOS (Intel and Apple Silicon)
  - Windows 10/11

- **Python**: 3.10 or newer
- **Rust**: Latest stable version (for Rust crates or building from source)

## Quick Installation

### Python Package

The simplest way to install PECOS for Python is via pip:

```bash
pip install quantum-pecos
```

This installs both `quantum-pecos` and its dependency `pecos-rslib`.

!!! note "Import Name"
    The `quantum-pecos` package is imported as `import pecos` and not `import quantum_pecos`.

### Rust Crates

To use PECOS in your Rust project, add the following to your `Cargo.toml`:

```toml
[dependencies]
pecos = "0.1.1"  # Replace with the latest version
```

## Detailed Installation Options

### Python Installation Options

#### Installing from PyPI

For standard installation:

```bash
pip install quantum-pecos
```

For all optional dependencies (recommended for full functionality):

```bash
pip install quantum-pecos[all]
```

PECOS provides several installation extras:

- `[all]`: Installs all optional dependencies
- `[cuda]`: Installs dependencies for GPU-accelerated simulators
- `[dev]`: Installs development dependencies (testing, linting, etc.)

#### Installing Development Versions

To install pre-release development versions from PyPI:

```bash
pip install quantum-pecos==X.Y.Z.devN  # Replace with actual version number
```

#### Installing in a Virtual Environment

Using a virtual environment is recommended to avoid conflicts with other packages:

=== "Using uv (recommended)"
    ```bash
    uv venv pecos-env
    source pecos-env/bin/activate  # On Windows: pecos-env\Scripts\activate
    uv pip install quantum-pecos
    ```

=== "Using venv"
    ```bash
    python -m venv pecos-env
    source pecos-env/bin/activate  # On Windows: pecos-env\Scripts\activate
    pip install quantum-pecos
    ```

=== "Using conda"
    ```bash
    conda create -n pecos-env python=3.10
    conda activate pecos-env
    pip install quantum-pecos
    ```

### Rust Installation Options

#### Using cargo-add

The easiest way to add PECOS to an existing Rust project:

```bash
cargo add pecos
```

To add a specific version:

```bash
cargo add pecos@0.1.1
```

For specific PECOS components:

```bash
cargo add pecos-core  # Core functionality
cargo add pecos-qsim  # Quantum simulators
cargo add pecos-engines  # Execution engines
cargo add pecos-qec  # Quantum error correction
```

#### Manual Cargo.toml Configuration

Add the following to your `Cargo.toml` file:

```toml
[dependencies]
# Main PECOS package (includes all components)
pecos = "0.1.1"  # Replace with latest version

# Or individual components as needed
pecos-core = "0.1.1"
pecos-qsim = "0.1.1"
pecos-engines = "0.1.1"
pecos-qec = "0.1.1"
```

## Installing from Source

Building from source provides the latest features and allows for customization.

### Prerequisites

Before building PECOS, ensure you have the following tools installed:

- **[Rust](https://www.rust-lang.org/tools/install)** (latest stable version)
- **[Python](https://www.python.org/downloads/)** (3.10 or newer)
- **[Git](https://git-scm.com/downloads)** for version control
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** for Python dependency management (recommended)
- **[LLVM 14.x](https://releases.llvm.org/download.html#14.0.0)** (optional, for QIR support)

### Platform-Specific Prerequisites

=== "Linux"
    ```bash
    # Install Rust
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    source $HOME/.cargo/env
    
    # Install Python dependencies
    sudo apt update
    sudo apt install python3 python3-pip python3-venv

    # Install uv
    curl -L --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/astral-sh/uv/main/install.sh | sh
    source ~/.zshrc  # or ~/.bashrc depending on your shell
    
    # Install LLVM 14 (optional, for QIR support)
    sudo apt install llvm-14
    ```

=== "macOS"
    ```bash
    # Install Rust
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    source $HOME/.cargo/env
    
    # Install Python and other dependencies with Homebrew
    brew install python
    
    # Install uv
    curl -L --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/astral-sh/uv/main/install.sh | sh
    source ~/.zshrc  # or ~/.bashrc depending on your shell
    
    # Install LLVM 14 (optional, for QIR support)
    brew install llvm@14
    echo 'export PATH="/usr/local/opt/llvm@14/bin:$PATH"' >> ~/.zshrc  # or ~/.bashrc
    source ~/.zshrc  # or ~/.bashrc
    ```

=== "Windows"
    ```powershell
    # Install Rust
    Invoke-WebRequest -Uri https://win.rustup.rs/x86_64 -OutFile rustup-init.exe
    .\rustup-init.exe
    # Follow the prompts to complete installation
    
    # Install Python from the official website or Microsoft Store
    # https://www.python.org/downloads/
    
    # Install uv
    Invoke-WebRequest -Uri https://github.com/astral-sh/uv/releases/latest/download/uv-installer.ps1 -OutFile uv-installer.ps1
    powershell -ExecutionPolicy Bypass -File .\uv-installer.ps1
    # Restart your terminal after installation
    
    # Install LLVM 14 (optional, for QIR support)
    # Download LLVM 14.x installer from https://releases.llvm.org/download.html#14.0.0
    # and follow the installation wizard
    # Add LLVM's bin directory (e.g., C:\Program Files\LLVM\bin) to your PATH environment variable
    ```

### Building from Source Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/PECOS-packages/PECOS.git
   cd PECOS
   ```

2. **Set Up the Development Environment**:
   ```bash
   uv sync
   ```
   Activate the virtual environment:

    === "Linux/macOS"
        ```bash
        source .venv/bin/activate
        ```
    === "Windows"
        ```powershell
        .\.venv\Scripts\activate
        ```

3. **Build PECOS**:
   
    === "Standard Development Build"
        ```bash
        make build
        ```
        This builds both Rust components and Python packages in development mode with all extras.
    
    === "Basic Build (No Extras)"
        ```bash 
        make build-basic
        ```
        This builds the core functionality without optional dependencies.
    
    === "Optimized Release Build"
        ```bash
        make build-release
        ```
        This builds with release optimizations for better performance.
    
    === "CPU-Optimized Build"
        ```bash
        make build-native
        ```
        This builds with native CPU optimizations for maximum performance on your specific hardware.

    On Windows, you might need to run these commands using `make` from Git Bash or use the equivalent commands directly:
    
    ```bash
    # Equivalent to make build on Windows
    uv sync
    cd python/pecos-rslib/ && uv run maturin develop --uv
    cd ../../python/quantum-pecos && uv pip install -e .[all]
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
    echo 'export PATH="/usr/local/opt/llvm@14/bin:$PATH"' >> ~/.zshrc  # or ~/.bashrc
    source ~/.zshrc  # or ~/.bashrc
    ```

=== "Windows"
    Download LLVM 14.x installer from [LLVM releases](https://releases.llvm.org/download.html#14.0.0)
    and add LLVM's bin directory (e.g., `C:\Program Files\LLVM\bin`) to your PATH environment variable.

!!! warning
    Only LLVM version 14.x is compatible. LLVM 15 or later versions will not work with PECOS's QIR implementation.

If LLVM 14 is not installed, PECOS will still function normally but QIR-related features will be disabled.

### GPU-Accelerated Simulators

For simulators that support GPU acceleration:

- **CuStateVec**: Requires a Linux machine with an NVIDIA GPU. Installation via conda is recommended:
  ```bash
  conda install -c nvidia cuda-quantum
  ```
  For more details, see the [NVIDIA cuQuantum documentation](https://docs.nvidia.com/cuda/cuquantum/latest/getting_started/getting_started.html#installing-cuquantum).

- **MPS Simulators**: Uses `pytket-cutensornet` and can be installed via:
  ```bash
  pip install quantum-pecos[cuda]
  ```
  These simulators also require NVIDIA GPUs and cuQuantum.

## Installation Verification

### Python

Verify your Python installation with:

```python
import pecos
print(pecos.__version__)

# Check if Rust extensions are available
from pecos import rslib
print("Rust extensions loaded successfully!")

# Test a basic quantum circuit
from pecos.circuits import QuantumCircuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
print("Bell state circuit created successfully!")
```

### Rust

Create a simple Rust program to verify your installation:

```rust
use pecos::prelude::*;

fn main() {
    println!("PECOS version: {}", pecos::version());
    
    // Create a simple circuit
    let mut circuit = pecos::circuits::Circuit::new();
    let q0 = circuit.allocate_qubit();
    let q1 = circuit.allocate_qubit();
    
    circuit.h(q0);
    circuit.cx(q0, q1);
    
    println!("Bell state circuit created successfully!");
}
```

## Common Installation Issues

### General Troubleshooting

- **Missing Dependencies**: Ensure all prerequisites are installed for your platform.
- **Version Conflicts**: Use a virtual environment to avoid conflicts with other packages.
- **Path Issues**: Make sure all tools (Rust, Python, LLVM) are in your system PATH.

### Platform-Specific Issues

#### Linux

- **Missing LLVM**: If you encounter errors related to missing LLVM tools, ensure LLVM 14 is installed and in your PATH.
- **Linker errors**: You might need additional development packages:
  ```bash
  sudo apt install build-essential libssl-dev
  ```

#### macOS

- **LLVM Path Issues**: If LLVM tools aren't found, explicitly set the LLVM path:
  ```bash
  export LLVM_CONFIG=/usr/local/opt/llvm@14/bin/llvm-config
  export PATH="/usr/local/opt/llvm@14/bin:$PATH"
  ```
- **XCode Command Line Tools**: Ensure XCode command line tools are installed:
  ```bash
  xcode-select --install
  ```

#### Windows

- **Path Length Issues**: Windows has path length limitations. Clone the repository to a directory with a short path.
- **Visual Studio Build Tools**: Ensure you have Visual Studio Build Tools installed for C++ development.
- **LLVM Path**: Add LLVM's bin directory to your PATH environment variable.
- **PowerShell Execution Policy**: You may need to adjust your execution policy:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

### Python-Specific Issues

- **Cannot import pecos**: Make sure you're using `import pecos` not `import quantum_pecos`.
- **Missing Rust extensions**: If you get errors about missing Rust extensions, try reinstalling with:
  ```bash
  pip uninstall -y quantum-pecos pecos-rslib
  pip install quantum-pecos --no-cache-dir
  ```
- **Binary incompatibility**: If you see "ImportError: incompatible library version", you may need to compile from source:
  ```bash
  pip install --no-binary=pecos-rslib quantum-pecos
  ```

### Rust-Specific Issues

- **Cargo build errors**: Update Rust with `rustup update` and try again.
- **Feature flags**: If you need specific features, ensure they're enabled in your `Cargo.toml`:
  ```toml
  pecos = { version = "0.1.1", features = ["your-feature"] }
  ```

## Next Steps

Now that you have PECOS installed, you can:

- Explore the [Quick Start Guide](quick-start.md) for a brief introduction
- Learn about [Core Concepts](concepts/index.md) in quantum error correction
- Check out the [Python API](https://quantum-pecos.readthedocs.io/en/latest/) or [Rust API](https://docs.rs/pecos/latest/pecos/) reference
- Try the examples in each language:
  - Python examples: `/python/quantum-pecos/examples/`
  - Rust examples: `/crates/*/examples/`
