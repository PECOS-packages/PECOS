# pecos-hugr-qis

HUGR to QIS compiler for PECOS.

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

## Acknowledgements

This crate builds on [tket2](https://github.com/Quantinuum/tket2), the quantum compiler toolkit developed by Quantinuum, and uses [HUGR](https://github.com/Quantinuum/hugr) (Hierarchical Unified Graph Representation) as its input format.

**Paper:**
- Koch, M., Borgna, A., Sivarajah, S., Lawrence, A., Edgington, A., Wilson, D., Roy, C., Mondada, L., Heidemann, L., & Duncan, R. (2025). "HUGR: A Quantum-Classical Intermediate Representation." [arXiv:2510.11420](https://arxiv.org/abs/2510.11420)

For more information, see the [PECOS documentation](https://github.com/PECOS-packages/PECOS).
