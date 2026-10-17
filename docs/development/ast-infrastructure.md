# AST Infrastructure Summary

This document summarizes the AST (Abstract Syntax Tree) infrastructure work for the PECOS quantum computing framework.

## Overview

The SLR-AST module provides a unified intermediate representation for quantum programs, enabling:
- Conversion from SLR programs
- Validation and analysis passes
- Optimization passes
- Code generation to multiple targets (QASM, Stim, Guppy, QIR, QuantumCircuit)

## Completed Work

### Validation Module (`src/pecos/slr/ast/validation/`)

Three validation passes:
- **TypeChecker** - Gate parameter types, arity checking
- **BoundsChecker** - Qubit/bit index bounds validation
- **AllocationValidator** - Allocator consistency, hierarchy validation

### Analysis Module (`src/pecos/slr/ast/analysis/`)

- Resource counting (gates, qubits, measurements)
- T-count analysis
- Circuit depth analysis
- Connectivity analysis
- Parallelism detection

### Optimization Module (`src/pecos/slr/ast/optimizations/`)

- Gate cancellation (adjacent inverse gates)
- Rotation merging (combine consecutive rotations)
- Identity removal
- Optimization pipeline

### Serialization & Comparison (`src/pecos/slr/ast/`)

- `serialize.py` - AST ↔ JSON serialization
- `compare.py` - Structural AST comparison and diff
- `pretty_print.py` - Human-readable AST output

### Code Generation (`src/pecos/slr/ast/codegen/`)

Targets:
- QASM (OpenQASM 2.0)
- Stim (Clifford simulator)
- Guppy (Python quantum DSL)
- QIR (Quantum Intermediate Representation)
- QuantumCircuit (internal tick-based format)

## Test Organization

Paths below are relative to `python/quantum-pecos/`.

```
tests/pecos/slr/ast_tests/       # Core SLR AST tests
├── test_ast_serialize.py        # Serialization tests
├── test_ast_compare.py          # Comparison tests
├── test_ast_roundtrip.py        # Code generation round-trips
├── optimizations/               # Optimization pass tests
├── analysis/                    # Analysis pass tests
└── validation/                  # Validation pass tests

tests/guppy/                     # Guppy compilation and execution tests
tests/selene/                    # Selene integration tests
```

## Key Files

| File | Purpose |
|------|---------|
| `src/pecos/slr/ast/nodes.py` | AST node definitions (GateKind, Statement types) |
| `src/pecos/slr/ast/converter.py` | SLR → AST conversion |
| `src/pecos/slr/ast/visitor.py` | BaseVisitor pattern for traversal |
| `src/pecos/slr/ast/validation/__init__.py` | Validation pipeline |
| `src/pecos/slr/ast/codegen/__init__.py` | Code generation exports |

## Running Tests

From the repository root:

```bash
uv run --frozen --no-sync pytest python/quantum-pecos/tests/pecos/slr/ast_tests -q
uv run --frozen --no-sync pytest python/quantum-pecos/tests/guppy python/quantum-pecos/tests/selene -q
```

## Architecture Notes

SLR programs convert to the AST through `pecos.slr.ast.converter`.
Guppy programs compile through `pecos.compilation_pipeline.compile_guppy_to_hugr`
and `pecos_rslib.hugr_lowering.compile_hugr_to_qis` for execution with `pecos.sim`.
For an executed circuit's DAG, use
`trace_program_to_tick_circuit(...).to_dag_circuit()`; this represents the traced
execution, including the branch taken, rather than the source control flow.

### AST Node Hierarchy

```
Program
├── declarations: tuple[Declaration, ...]
│   ├── AllocatorDecl (qubit allocators)
│   └── RegisterDecl (classical registers)
└── body: tuple[Statement, ...]
    ├── GateOp (quantum gates)
    ├── MeasureOp (measurements)
    ├── PrepareOp (qubit preparation)
    ├── IfStmt (conditionals)
    ├── WhileStmt (loops)
    ├── ForStmt (bounded loops)
    ├── RepeatStmt (repeat blocks)
    └── ParallelBlock (parallel operations)
```

## Contact

See the main PECOS documentation for more information.
