# PECOS QIR

This crate provides QIR (Quantum Intermediate Representation) execution capabilities for the PECOS framework.

## Overview

The PECOS QIR crate enables execution of quantum programs written in the Quantum Intermediate Representation (QIR), a common interface between different quantum programming languages and target quantum computation platforms.

This crate contains all QIR-related functionality, which was migrated from the `pecos-engines` crate to improve maintainability, allow better testing, and enable focused development of QIR capabilities.

## Requirements

- LLVM version 14.x with the 'llc' tool is required for QIR support
  - Linux: `sudo apt install llvm-14 llvm-14-dev`
  - macOS: `brew install llvm@14`
  - Windows: Download LLVM 14.x installer from [LLVM releases](https://releases.llvm.org/download.html#14.0.0)

**Note**: Only LLVM version 14.x is compatible. LLVM 15 or later versions will not work with PECOS's QIR implementation.

## Usage

### From Rust

```rust
use pecos_qir::QirEngine;
use std::path::PathBuf;

fn main() {
    // Create a QIR engine for a specific QIR file
    let qir_path = PathBuf::from("path/to/your/qir_file.ll");
    let mut engine = QirEngine::new(qir_path);

    // Pre-compile the QIR program for better performance
    engine.pre_compile().expect("Failed to pre-compile QIR program");

    // Run the QIR program (for a complete workflow, see examples)
    // ...
}
```

### From CLI

PECOS includes a command-line interface that supports executing QIR programs:

```sh
# Run a QIR program
pecos run path/to/qir_file.ll

# Run with specific number of shots
pecos run path/to/qir_file.ll -s 100

# Run with noise model
pecos run path/to/qir_file.ll -p 0.01
```

## Architecture

The QIR crate includes several components:

- **QirEngine**: The main entry point for executing QIR programs
- **QirCompiler**: Handles compilation of QIR programs to native code
- **QirLibrary**: Manages loading and interaction with compiled QIR libraries
- **Platform-specific modules**: Handle differences between Linux, macOS, and Windows

## Contributing

Contributions to improve the QIR implementation are welcome! Please follow the contribution guidelines in the main PECOS repository.

## License

This crate is licensed under the Apache-2.0 License, as is the rest of the PECOS project.
