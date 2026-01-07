# pecos-phir

MLIR-inspired quantum program intermediate representation.

## Purpose

PHIR (PECOS High-level IR) provides an MLIR-inspired SSA representation for quantum programs. It supports parsing, optimization, and execution through multiple backends.

## Key Features

- Hierarchical structure: Operations contain Regions contain Blocks contain Operations
- Dialect system: builtin, HUGR, and QIS dialects
- Progressive lowering: parsing ops -> high-level ops -> low-level ops -> execution
- Multiple execution strategies: interpreter, MLIR lowering to LLVM

## Key Types

- `Module` - Top-level container
- `Operation` - SSA operations across dialects
- `PhirEngine` - Execution engine
- `Pipeline` - Compilation pipeline

## Relationship to pecos-phir-json

- **pecos-phir**: MLIR-inspired IR with execution pipeline (this crate)
- **pecos-phir-json**: JSON-based format for PHIR programs
