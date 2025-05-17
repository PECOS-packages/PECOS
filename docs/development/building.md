# Building PECOS from Source

This guide covers how to build PECOS from source code on different platforms. Building from source gives you the most flexibility for development, customization, and accessing the latest features.

## Prerequisites

Before building PECOS, ensure you have the following tools installed:

- **[Rust](https://www.rust-lang.org/tools/install)** (latest stable version)
- **[Python](https://www.python.org/downloads/)** (3.10 or newer)
- **[Git](https://git-scm.com/downloads)** for version control
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** for Python dependency management
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

!!! warning
    PECOS requires LLVM version 14.x specifically for QIR support. Later versions (15+) are not compatible with PECOS's QIR implementation. If LLVM 14 is not installed, PECOS will still function normally but QIR-related features will be disabled.

## Building PECOS from Source

### 1. Clone the Repository

```bash
git clone https://github.com/PECOS-packages/PECOS.git
cd PECOS
```

### 2. Set Up the Development Environment

PECOS uses `uv` for Python dependency management. This creates a virtual environment in `.venv/` and installs all required dependencies:

```bash
uv sync
```

You may want to activate the virtual environment explicitly:

=== "Linux/macOS"
    ```bash
    source .venv/bin/activate
    ```

=== "Windows"
    ```powershell
    .\.venv\Scripts\activate
    ```

### 3. Build PECOS

PECOS provides several build options through its Makefile:

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

### 4. Verify the Build

After building, verify that everything was installed correctly:

```python
# Python
import pecos
print(pecos.__version__)

# Try importing Rust extensions
from pecos import rslib
print("Rust extension loaded successfully!")
```

## Building Individual Components

If you only need specific components, you can build them separately.

### Building Just the Rust Crates

```bash
cargo build
```

For a release build with optimizations:

```bash
cargo build --release
```

To build a specific crate:

```bash
cargo build -p pecos-core
```

### Building Just the Python Bindings

```bash
cd python/pecos-rslib/
uv run maturin develop --uv
```

For a release build:

```bash
cd python/pecos-rslib/
uv run maturin develop --uv --release
```

### Building the Python Package

```bash
cd python/quantum-pecos
uv pip install -e .
```

With all extras:

```bash
cd python/quantum-pecos
uv pip install -e .[all]
```

## Platform-Specific Build Issues

### Linux

- **Missing LLVM**: If you encounter errors related to missing LLVM tools, ensure LLVM 14 is installed and in your PATH.
- **Linker errors**: You might need additional development packages:
  ```bash
  sudo apt install build-essential libssl-dev
  ```

### macOS

- **LLVM Path Issues**: If LLVM tools aren't found, explicitly set the LLVM path:
  ```bash
  export LLVM_CONFIG=/usr/local/opt/llvm@14/bin/llvm-config
  export PATH="/usr/local/opt/llvm@14/bin:$PATH"
  ```
- **XCode Command Line Tools**: Ensure XCode command line tools are installed:
  ```bash
  xcode-select --install
  ```

### Windows

- **Path Length Issues**: Windows has path length limitations. Clone the repository to a directory with a short path.
- **Visual Studio Build Tools**: Ensure you have Visual Studio Build Tools installed for C++ development.
- **LLVM Path**: Add LLVM's bin directory to your PATH environment variable.
- **PowerShell Execution Policy**: You may need to adjust your execution policy:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

## Troubleshooting

### QIR Runtime Library Issues

If you encounter issues with the QIR runtime library:

1. Delete the pre-built library to force a rebuild:

   === "Linux/macOS"
       ```bash
       rm -f target/debug/libqir_runtime.a target/release/libqir_runtime.a
       ```
   
   === "Windows"
       ```powershell
       Remove-Item -Force target\debug\qir_runtime.lib, target\release\qir_runtime.lib
       ```

2. Rebuild the library explicitly:
   ```bash
   cargo clean -p pecos-engines
   cargo build -p pecos-engines
   ```

3. Check for build errors with verbose output:
   ```bash
   CARGO_LOG=debug cargo build -p pecos-engines
   ```

### Python Extension Build Failures

If Maturin fails to build the Python extensions:

1. Make sure Rust and Python versions are compatible.
2. Check for missing development headers or libraries.
3. Try clearing the build cache:
   ```bash
   cd python/pecos-rslib
   rm -rf target/
   uv run maturin develop --uv
   ```

### Cleaning and Rebuilding

If you encounter persistent issues, cleaning the build and starting over can help:

```bash
make clean
make build
```

This removes all build artifacts and rebuilds the project from scratch.

## Running Tests

After building, you can run tests to verify everything works:

```bash
make test
```

This runs both Rust and Python tests.

To run just Rust tests:

```bash
make rstest
```

To run just Python tests:

```bash
make pytest
```

## Next Steps

Now that you've built PECOS from source, you can:

- Explore the [User Guide](../user-guide/index.md) to learn core concepts
- Check out the [Python API](https://quantum-pecos.readthedocs.io/en/latest/) or [Rust API](https://docs.rs/pecos/latest/pecos/) reference
- Try the examples in each language:
  - Python examples: `/python/quantum-pecos/examples/`
  - Rust examples: `/crates/*/examples/`
- See the [Development Guide](development.md) for information on contributing to PECOS