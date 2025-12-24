# pecos-programs

Zero-dependency program types for PECOS quantum simulation.

This crate provides pure data types for quantum programs that can be used across different PECOS engine crates without creating dependencies between them.

## Supported Program Types

- **QASM**: OpenQASM 2.0 quantum circuit descriptions
- **LLVM**: LLVM IR (both text and bitcode formats)
- **HUGR**: Hierarchical Unified Graph Representation
- **WASM**: WebAssembly binary format
- **WAT**: WebAssembly Text format
- **PHIR-JSON**: PECOS High-level Intermediate Representation in JSON

## Usage

```rust
use pecos_programs::{QasmProgram, LlvmProgram, Program};

// Create a QASM program
let qasm = QasmProgram::from_string("OPENQASM 2.0; qreg q[2];");

// Load from file
let llvm = LlvmProgram::from_file("circuit.ll")?;

// Use the enum for runtime dispatch
let program: Program = qasm.into();
```

This crate has zero dependencies to ensure it can be used as a common interface between different parts of the PECOS ecosystem.
