# pecos-llvm

LLVM IR generation using inkwell.

## Purpose

Provides Rust types for generating LLVM IR, designed to be compatible with Python's llvmlite API patterns. Used for compiling quantum programs to LLVM IR.

## Key Types

- `LLContext` - LLVM context wrapper
- `LLModule` - LLVM module for IR generation
- `LLFunction` - Function builder
- `LLIRBuilder` - Instruction builder
- `LLType`, `LLValue`, `LLConstant` - Type system wrappers

## Relationship to pecos-build

- **pecos-build**: Manages LLVM 14 *installation* (downloading, finding)
- **pecos-llvm**: *Uses* LLVM 14 (via inkwell) for IR generation

## Requirements

Requires LLVM 14. Install with:
```bash
cargo run -p pecos -- llvm install
```

## Acknowledgements

This crate uses [inkwell](https://github.com/TheDan64/inkwell), a safe Rust wrapper for the [LLVM](https://llvm.org/) compiler infrastructure.
