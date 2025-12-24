# pecos-hugr

HUGR (Hierarchical Unified Graph Representation) compiler for PECOS.

This crate provides compilation of HUGR quantum programs to LLVM IR for execution in the PECOS quantum simulation framework.

## Features

- Compile HUGR files to LLVM IR
- Support for quantum gates and operations
- Integration with the tket2 quantum compiler toolkit
- Automatic handling of extension types and operations

## Usage

```rust
use pecos_hugr::compile_hugr_to_llvm;

// Compile a HUGR file to LLVM IR
let llvm_ir_path = compile_hugr_to_llvm("quantum_circuit.hugr", None)?;
```

For more information, see the [PECOS documentation](https://github.com/PECOS-packages/PECOS).
