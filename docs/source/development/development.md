# Development Guide

This guide covers how to set up your development environment for contributing to PECOS or for advanced customization.

## Prerequisites

Before starting development work on PECOS, ensure you have the following tools installed:

- **Rust** (latest stable): Install via [rustup](https://rustup.rs/)
- **Python** (3.10+): Install via your preferred method
- **Git**: For version control
- **LLVM 14** (optional): For QIR support

## Setting Up the Development Environment

### 1. Clone the Repository

```bash
git clone https://github.com/PECOS-packages/PECOS.git
cd PECOS
```

### 2. Python Development Setup

Create a virtual environment and install the package in development mode:

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

### 3. Rust Development Setup

Rust dependencies are handled automatically by Cargo. Build the Rust crates with:

```bash
cargo build
```

For development, you might want to enable specific features:

```bash
cargo build --features="dev,testing"
```

## Repository Structure

PECOS consists of multiple interconnected components:

- `/python/`: Contains Python packages
  - `/python/quantum-pecos/`: Main Python package (imports as `pecos`)
  - `/python/pecos-rslib/`: Python package with Rust extensions
- `/crates/`: Contains Rust crates
  - `/crates/pecos/`: Main Rust crate
  - `/crates/pecos-core/`: Core Rust functionalities
  - `/crates/pecos-qsims/`: Quantum simulators
  - `/crates/pecos-qec/`: QEC code
  - `/crates/pecos-python/`: Rust code for Python extensions
  - `/crates/benchmarks/`: Performance benchmarks

## Development Workflow

### Running Tests

#### Python Tests

```bash
# Run all Python tests
pytest

# Run a specific test
pytest python/tests/path/to/test_file.py
```

#### Rust Tests

```bash
# Run all Rust tests
cargo test

# Run tests for a specific crate
cargo test -p pecos-core
```

### Code Formatting and Linting

#### Python

PECOS uses `ruff` for Python code linting and formatting:

```bash
# Run linting
ruff check

# Apply autoformat
ruff format
```

#### Rust

For Rust code, use rustfmt and clippy:

```bash
# Format code
cargo fmt

# Lint code
cargo clippy
```

### Documentation

#### Building Documentation

To build and preview the documentation locally:

##### Python API Documentation

```bash
cd python/docs
make html
python -m http.server -d _build/html
```

##### Rust API Documentation

```bash
cargo doc --open
```

##### MkDocs User Guide

```bash
# Install MkDocs and required plugins
pip install mkdocs mkdocs-material mkdocstrings

# Serve documentation locally
cd docs
mkdocs serve
```

### Pre-commit Hooks

PECOS uses pre-commit hooks to ensure code quality. Install them with:

```bash
pip install pre-commit
pre-commit install
```

## Contribution Guidelines

When contributing to PECOS:

1. **Create a branch**: Create a branch for your feature or bugfix
2. **Write tests**: Add tests for new functionality
3. **Update documentation**: Update relevant documentation
4. **Follow coding standards**: Adhere to the project's coding style
5. **Submit a PR**: Create a pull request to merge your changes

## Release Process

PECOS follows semantic versioning. For making releases:

1. Update version numbers in relevant files
2. Update CHANGELOG.md with release notes
3. Create a release tag
4. Build and publish packages

See the [DEVELOPMENT.md](https://github.com/PECOS-packages/PECOS/blob/master/DEVELOPMENT.md) file in the repository for more detailed information.