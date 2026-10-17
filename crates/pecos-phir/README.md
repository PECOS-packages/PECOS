# pecos-phir

MLIR-inspired quantum program intermediate representation.

## Purpose

PHIR (PECOS High-level IR) provides an MLIR-inspired SSA representation for quantum programs. It supports parsing, optimization, and execution through multiple backends.

## Key Features

- Hierarchical structure: Operations contain Regions contain Blocks contain Operations
- Dialect system: builtin and QIS dialects
- Progressive lowering: parsing ops -> high-level ops -> low-level ops -> execution
- Multiple execution strategies: interpreter, MLIR lowering to LLVM

## Key Types

- `Module` - Top-level container
- `Operation` - SSA operations across dialects
- `PhirEngine` - Execution engine

## Relationship to pecos-phir-json

- **pecos-phir**: MLIR-inspired IR with execution pipeline (this crate)
- **pecos-phir-json**: JSON-based format for PHIR programs

## Execution and MLIR lowering

The processor validates operand and result counts before execution, including
measurement, reset, preparation and deallocation. Broadcasts must be split into
individual instructions; the PHIR-JSON converter performs this split. Preparations
consume one existing qubit, as reset does. Measurements may discard their result.

`mlir_lowering::phir_to_mlir` currently lowers function bodies containing quantum
allocation, H, CX, measurement and Return. It rejects unsupported operations,
terminators and function control-flow structures with an error naming the rejected
operation or structure. It also rejects incorrect operand and result counts instead
of consuming a prefix. This API no longer emits TODO comments for omitted operations.

The PHIR-JSON converter represents every PHIR v0.1 quantum gate and alias,
including CZ, RZZ, identity, square roots of Y, face gates, XX/YY rotations and
combined XX/YY/ZZ rotations. The processor emits their existing exact native
byte-message gates. Conversion retains gate angles and target order.
