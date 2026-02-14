# AST Infrastructure Summary

This document summarizes the AST (Abstract Syntax Tree) infrastructure work for the PECOS quantum computing framework.

## Overview

The SLR-AST module provides a unified intermediate representation for quantum programs, enabling:
- Conversion from multiple input formats (SLR, HUGR/Guppy)
- Validation and analysis passes
- Optimization passes
- Code generation to multiple targets (QASM, Stim, Guppy, QIR, QuantumCircuit)

## Completed Work

### HUGR to AST Converter (`src/pecos/circuit_converters/hugr_to_ast.py`)

Converts compiled Guppy programs (HUGR format) to SLR-AST for analysis and code generation.

**Supported features:**
- Straight-line quantum circuits
- Conditionals (if/else based on measurement results)
- Nested conditionals
- While loops with classical conditions
- Gates: H, X, Y, Z, S, Sdg, T, Tdg, SX, SXdg, CX, CY, CZ, CH, RX, RY, RZ, RZZ

**Key functions:**
```python
from pecos.circuit_converters.hugr_to_ast import guppy_to_ast, hugr_to_ast

@guppy
def my_circuit() -> bool:
    q = qubit()
    h(q)
    return measure(q)

ast = guppy_to_ast(my_circuit)
```

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

```
tests/selene/                    # Selene/HUGR pipeline tests (111 tests)
├── test_hugr_to_ast.py          # HUGR → AST conversion
├── test_hugr_pipeline.py        # End-to-end Guppy → AST → codegen
├── test_hugr_roundtrip.py       # Serialization round-trips
├── test_hugr_to_dag.py          # HUGR → DAG conversion
├── test_hugr_compilation.py     # HUGR compilation tests
├── test_hugr_structure.py       # HUGR format tests
├── test_selene_*.py             # Selene integration tests

tests/pecos/slr/ast/             # Core AST tests
├── test_ast_serialize.py        # Serialization tests
├── test_ast_compare.py          # Comparison tests
├── test_ast_roundtrip.py        # Code generation round-trips
├── optimizations/               # Optimization pass tests
├── analysis/                    # Analysis pass tests
└── validation/                  # Validation pass tests

tests/guppy/                     # Guppy language feature tests
```

## Key Files

| File | Purpose |
|------|---------|
| `src/pecos/slr/ast/nodes.py` | AST node definitions (GateKind, Statement types) |
| `src/pecos/slr/ast/converter.py` | SLR → AST conversion |
| `src/pecos/slr/ast/visitor.py` | BaseVisitor pattern for traversal |
| `src/pecos/circuit_converters/hugr_to_ast.py` | HUGR → AST conversion |
| `src/pecos/slr/ast/validation/__init__.py` | Validation pipeline |
| `src/pecos/slr/ast/codegen/__init__.py` | Code generation exports |

## Usage Examples

### Full Pipeline: Guppy → AST → QASM

```python
from guppylang import guppy
from guppylang.std.quantum import h, cx, measure, qubit
from pecos.circuit_converters.hugr_to_ast import guppy_to_ast
from pecos.slr.ast.validation import validate
from pecos.slr.ast.codegen import generate

@guppy
def bell() -> tuple[bool, bool]:
    q0 = qubit()
    q1 = qubit()
    h(q0)
    cx(q0, q1)
    return measure(q0), measure(q1)

# Convert to AST
ast = guppy_to_ast(bell)

# Validate
result = validate(ast)
assert result.valid

# Generate QASM
qasm = generate(ast, "qasm")
print(qasm)
```

### Serialization Round-trip

```python
from pecos.slr.ast.serialize import ast_to_json, json_to_ast
from pecos.slr.ast.compare import ast_equal

json_str = ast_to_json(ast)
restored = json_to_ast(json_str)
assert ast_equal(ast, restored)
```

## Potential Future Work

### Near-term

1. **Additional gate support** - Add more gates as needed (e.g., multi-controlled gates)
2. **Better condition tracking** - Currently nested conditionals use simplified condition variables
3. **For loop support** - Add ForStmt generation from HUGR bounded loops

### Medium-term

4. **Optimization integration with codegen** - Add `validate_before_generate` option
5. **CodegenResult with metadata** - Include analysis results (T-count, depth) with generated code
6. **More analysis passes** - Gate scheduling, qubit routing hints

### Longer-term

7. **Bidirectional conversion** - AST → HUGR for round-trip editing
8. **Circuit visualization** - AST → diagram output
9. **Equivalence checking** - Verify optimizations preserve semantics

## Running Tests

```bash
# Run all Selene/HUGR tests
uv run pytest tests/selene/ -v

# Run all AST tests
uv run pytest tests/pecos/slr/ast/ -v

# Run specific test file
uv run pytest tests/selene/test_hugr_to_ast.py -v
```

## Architecture Notes

### HUGR → AST Conversion

The converter handles HUGR's CFG-based structure:

1. **CFG Analysis** - Identifies blocks, edges, conditionals, loops
2. **Loop Detection** - Finds back-edges to identify while loops
3. **Conditional Detection** - Identifies branch patterns for if/else
4. **Nested Structure** - Recursively processes branches for nesting
5. **Qubit Tracking** - Maps qubit wires across CFG blocks

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
