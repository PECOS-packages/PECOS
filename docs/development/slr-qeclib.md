# SLR and QECLib Developer Guide

This guide covers PECOS's Simple Logical Representation (SLR) and the QEC library (`qeclib`), which provide low-level programmatic control over quantum circuit construction.

## When to Use SLR vs Guppy

| Approach | Level | Best For |
|----------|-------|----------|
| **Guppy** (`pecos.guppy`) | High-level | General users, QEC experiments, quick prototyping |
| **SLR** (`pecos.slr`) | Low-level | Developers needing direct control, custom backends, advanced optimization |

**Use Guppy when:**

- You want the simplest path to running QEC experiments
- You need Guppy's type safety and control flow
- You're building on `pecos.qec` geometry

**Use SLR when:**

- You need fine-grained control over circuit construction
- You're building custom compilation pipelines
- You need direct QASM/QIR output without Guppy
- You're developing new simulator backends
- You need programmatic circuit manipulation

## SLR Overview

SLR (Simple Logical Representation) is a programmatic way to construct quantum programs using Python. It provides:

- **Hierarchical structure** - Programs are built from nested blocks
- **Register management** - Explicit qubit and classical register handling
- **Control flow** - If/else, loops, and parallel blocks
- **Multiple backends** - Convert to Guppy, QASM, QIR, or execute directly

### Core Concepts

```python
from pecos.slr import (
    Main,  # Top-level program container
    Block,  # Group of operations
    QReg,  # Quantum register
    CReg,  # Classical register (use CReg[0] for single bits)
    Qubit,  # Single qubit reference
    If,  # Conditional block (use .Then() method)
    For,  # For loop with variable (use .Do() method)
    While,  # While loop (use .Do() method)
    Repeat,  # Simple repetition (use .block() method)
    Parallel,  # Parallel execution block
    SlrConverter,  # Convert to QASM/QIR/Guppy
)
from pecos.slr import qeclib  # QEC operations
```

## Basic SLR Programs

### Hello World: Bell State

```python
from pecos.slr import Main, QReg, CReg, SlrConverter
from pecos.slr.qeclib import qubit as qb

# Define the program
prog = Main(
    q := QReg("q", 2),  # 2-qubit register
    c := CReg("c", 2),  # 2-bit classical register
    qb.H(q[0]),  # Hadamard on qubit 0
    qb.CX(q[0], q[1]),  # CNOT
    qb.Measure(q) > c,  # Measure all qubits into c
)

# Convert to QASM
qasm = SlrConverter(prog).qasm()
print(qasm)
```

Output:
```
OPENQASM 2.0;
include "hqslib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q -> c;
```

### Using Blocks

Group operations into logical blocks:

```python
from pecos.slr import Main, Block, QReg, CReg
from pecos.slr.qeclib import qubit as qb

prog = Main(
    q := QReg("q", 4),
    c := CReg("c", 4),
    # Initialization block
    Block(
        qb.Prep(q[0], "Z"),
        qb.Prep(q[1], "Z"),
        qb.Prep(q[2], "X"),
        qb.Prep(q[3], "X"),
    ),
    # Entanglement block
    Block(
        qb.H(q[0]),
        qb.CX(q[0], q[1]),
        qb.H(q[2]),
        qb.CX(q[2], q[3]),
    ),
    # Measurement
    qb.Measure(q) > c,
)
```

### Parallel Execution

Express operations that can run simultaneously:

```python
from pecos.slr import Main, Parallel, QReg, CReg, SlrConverter
from pecos.slr.qeclib import qubit as qb

prog = Main(
    q := QReg("q", 4),
    c := CReg("c", 4),
    # These Hadamards can run in parallel
    Parallel(
        qb.H(q[0]),
        qb.H(q[1]),
        qb.H(q[2]),
        qb.H(q[3]),
    ),
    qb.Measure(q) > c,
)

# SlrConverter optimizes parallel blocks by default
qasm = SlrConverter(prog).qasm()
```

### Control Flow

#### Conditionals

```python
from pecos.slr import Main, If, QReg, CReg
from pecos.slr.qeclib import qubit as qb

prog = Main(
    q := QReg("q", 2),
    c := CReg("c", 2),
    qb.H(q[0]),
    qb.Measure(q[0]) > c[0],
    # Conditional X gate (use .Then() method)
    If(c[0] == 1).Then(
        qb.X(q[1]),
    ),
    qb.Measure(q[1]) > c[1],
)
```

#### Loops

For simple repetition, use `Repeat`:

```python
from pecos.slr import Main, Repeat, QReg, CReg
from pecos.slr.qeclib import qubit as qb

prog = Main(
    q := QReg("q", 1),
    c := CReg("c", 1),
    # Apply H gate 3 times
    Repeat(3).block(
        qb.H(q[0]),
    ),
    qb.Measure(q) > c,
)
```

For iteration with a loop variable, use `For` with `LoopVar`:

```python
from pecos.slr import Main, For, LoopVar, QReg, CReg
from pecos.slr.qeclib import qubit as qb

# Create a loop variable for symbolic indexing
i = LoopVar("i")

prog = Main(
    q := QReg("q", 4),
    c := CReg("c", 4),
    # Apply H to each qubit using loop variable
    For(i, range(4)).Do(
        qb.H(q[i]),  # q[i] uses symbolic indexing
    ),
    qb.Measure(q) > c,
)
```

For while loops:

```python
from pecos.slr import Main, While, QReg, CReg
from pecos.slr.qeclib import qubit as qb

prog = Main(
    q := QReg("q", 1),
    c := CReg("c", 1),
    # While loop (use .Do() method)
    While(c[0] == 0).Do(
        qb.H(q[0]),
        qb.Measure(q[0]) > c[0],
    ),
)
```

## QECLib: Quantum Operations

The `qeclib` module provides quantum operations organized by category:

### Qubit Operations (`pecos.slr.qeclib.qubit`)

```python
from pecos.slr import Main, QReg, CReg
from pecos.slr.qeclib import qubit as qb

prog = Main(
    q := QReg("q", 2),
    c := CReg("c", 2),
    # Single-qubit Paulis
    qb.X(q[0]),
    qb.Y(q[0]),
    qb.Z(q[0]),
    # Hadamard
    qb.H(q[0]),
    # Phase gates
    qb.SZ(q[0]),  # S gate (sqrt Z)
    qb.SZdg(q[0]),  # S dagger
    qb.T(q[0]),  # T gate
    qb.Tdg(q[0]),  # T dagger
    # Rotations (angle in radians)
    qb.RX(q[0], 0.5),
    qb.RY(q[0], 0.5),
    qb.RZ(q[0], 0.5),
    # Two-qubit gates
    qb.CX(q[0], q[1]),  # CNOT
    qb.CY(q[0], q[1]),
    qb.CZ(q[0], q[1]),
    # Measurements and preparations
    qb.Measure(q[0]) > c[0],
    qb.Prep(q[0], "Z"),  # Prepare |0>
    qb.Prep(q[0], "X"),  # Prepare |+>
)
```

### Surface Code Operations (`pecos.slr.qeclib.surface`)

```python
from pecos.slr.qeclib import surface

# Surface code specific operations
# (Implementation varies by layout)
```

### Color Code Operations (`pecos.slr.qeclib.color488`)

```python
from pecos.slr.qeclib import color488

# 4.8.8 color code operations
```

### Steane Code Operations (`pecos.slr.qeclib.steane`)

```python
from pecos.slr.qeclib import steane

# Steane [[7,1,3]] code operations
```

## SlrConverter: Output Generation

SlrConverter can output to multiple formats and also convert from other formats to SLR.

### Output Formats

```python
from pecos.slr import Main, QReg, CReg, SlrConverter
from pecos.slr.qeclib import qubit as qb

# Create a simple program
prog = Main(
    q := QReg("q", 2),
    c := CReg("c", 2),
    qb.H(q[0]),
    qb.CX(q[0], q[1]),
    qb.Measure(q) > c,
)

converter = SlrConverter(prog)

# Guppy source code
guppy_source = converter.guppy()

# HUGR (compiled Guppy)
hugr_module = converter.hugr()

# OpenQASM 2.0
qasm = converter.qasm()
qasm = converter.qasm(skip_headers=True)  # Without OPENQASM header

# QIR (LLVM IR text)
qir = converter.qir()

# PECOS QuantumCircuit
quantum_circuit = converter.quantum_circuit()
```

Additional output formats:

```python
from pecos.slr import Main, QReg, CReg, SlrConverter
from pecos.slr.qeclib import qubit as qb

prog = Main(
    q := QReg("q", 2),
    c := CReg("c", 2),
    qb.H(q[0]),
    qb.Measure(q) > c,
)
converter = SlrConverter(prog)

# Stim circuit
stim_circuit = converter.stim()

# QIR bytecode (requires llvmlite)
qir_bc = converter.qir_bc()
```

### Parallel Optimization

```python
from pecos.slr import Main, QReg, CReg, SlrConverter
from pecos.slr.qeclib import qubit as qb

prog = Main(
    q := QReg("q", 2),
    c := CReg("c", 2),
    qb.H(q[0]),
    qb.Measure(q) > c,
)

# Optimization enabled by default
converter = SlrConverter(prog)

# Disable parallel optimization
converter = SlrConverter(prog, optimize_parallel=False)
```

### Converting FROM Other Formats

```python
from pecos.slr import Main, QReg, CReg, SlrConverter
from pecos.slr.qeclib import qubit as qb

# Create a program and convert to various formats
prog = Main(
    q := QReg("q", 2),
    c := CReg("c", 2),
    qb.H(q[0]),
    qb.Measure(q) > c,
)
converter = SlrConverter(prog)

# Get Stim and QuantumCircuit representations
stim_circuit = converter.stim()
qc = converter.quantum_circuit()

# Convert back from Stim circuit
slr_block = SlrConverter.from_stim(stim_circuit)

# Convert back from PECOS QuantumCircuit
slr_block = SlrConverter.from_quantum_circuit(qc)
```

### Running SLR Programs

```python
from pecos import sim, Qasm
from pecos.slr import Main, QReg, CReg, SlrConverter
from pecos.slr.qeclib import qubit as qb

# Create a program
prog = Main(
    q := QReg("q", 2),
    c := CReg("c", 2),
    qb.H(q[0]),
    qb.CX(q[0], q[1]),
    qb.Measure(q) > c,
)

# Option 1: Convert to QASM and simulate
qasm = SlrConverter(prog).qasm()
results = sim(Qasm(qasm)).seed(42).run(100)

# Option 2: Compile to HUGR and run
hugr = SlrConverter(prog).hugr()
# Use with PECOS HUGR engine
```

## Building QEC Circuits with SLR

### Example: Simple Syndrome Extraction

```python
from pecos.slr import Main, Block, QReg, CReg, Parallel, SlrConverter
from pecos.slr.qeclib import qubit as qb


def surface_code_syndrome(d: int):
    """Build a simple syndrome extraction circuit."""
    num_data = d * d
    num_ancilla = 2

    prog = Main(
        data := QReg("data", num_data),
        ancilla := QReg("anc", num_ancilla),
        syn := CReg("syn", num_ancilla),
        # Initialize data qubits
        Block(*[qb.Prep(data[i], "Z") for i in range(num_data)]),
        # X stabilizer measurement (simplified)
        Block(
            qb.Prep(ancilla[0], "X"),  # Prepare |+>
            qb.H(ancilla[0]),
            qb.CX(ancilla[0], data[0]),
            qb.CX(ancilla[0], data[1]),
            qb.H(ancilla[0]),
            qb.Measure(ancilla[0]) > syn[0],
        ),
        # Z stabilizer measurement (simplified)
        Block(
            qb.Prep(ancilla[1], "Z"),  # Prepare |0>
            qb.CX(data[0], ancilla[1]),
            qb.CX(data[3], ancilla[1]),
            qb.Measure(ancilla[1]) > syn[1],
        ),
        # Final measurement
        qb.Measure(data) > CReg("final", num_data),
    )

    return prog


# Generate QASM for d=3 surface code
prog = surface_code_syndrome(3)
qasm = SlrConverter(prog).qasm()
```

### Example: Parameterized Circuits

```python
from pecos.slr import Main, Block, QReg, CReg, SlrConverter
from pecos.slr.qeclib import qubit as qb


def ghz_state(n: int):
    """Create an n-qubit GHZ state."""
    prog = Main(
        q := QReg("q", n),
        c := CReg("c", n),
        # Hadamard on first qubit
        qb.H(q[0]),
        # CNOT chain
        Block(*[qb.CX(q[i], q[i + 1]) for i in range(n - 1)]),
        # Measure all
        qb.Measure(q) > c,
    )
    return prog


# Create 5-qubit GHZ state
prog = ghz_state(5)
qasm = SlrConverter(prog).qasm()
```

## Advanced Topics

### Custom Gate Definitions

For operations not in qeclib, you can extend the framework:

```python
from pecos.slr.cops import COp


# Custom operation (see pecos.slr.cops for details)
class MyGate(COp):
    def __init__(self, qubit):
        super().__init__("my_gate", [qubit])
```

### Working with the AST

SLR programs have an internal AST you can inspect:

```python
from pecos.slr import Main, QReg
from pecos.slr.qeclib import qubit as qb

prog = Main(
    q := QReg("q", 2),
    qb.H(q[0]),
)

# The program structure is accessible
print(prog.ops)  # List of operations
```

### Integration with pecos.qec

Combine SLR with `pecos.qec` geometry:

```python
from pecos.qec.surface import SurfacePatch
from pecos.slr import Main, Block, QReg, CReg
from pecos.slr.qeclib import qubit as qb

# Get geometry from pecos.qec
patch = SurfacePatch.create(distance=3)

# Use geometry to build SLR program
num_data = patch.num_data
x_stabs = patch.x_stabilizers

prog = Main(
    data := QReg("data", num_data),
    # ... build circuit using geometry info
)
```

## API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `Main` | Top-level program container |
| `Block` | Group of sequential operations |
| `Parallel` | Parallel execution block |
| `QReg` | Quantum register |
| `CReg` | Classical register (index with `[i]` for bits) |
| `Qubit` | Single qubit reference |
| `If` | Conditional block (use `.Then()` method) |
| `For` | For loop with variable (use `.Do()` method) |
| `LoopVar` | Loop variable for symbolic indexing (e.g., `q[i]`) |
| `While` | While loop (use `.Do()` method) |
| `Repeat` | Simple repetition (use `.block()` method) |
| `Return` | Return statement |
| `Barrier` | Barrier operation |
| `Comment` | Code comment |

### SlrConverter Methods

**Output methods:**

| Method | Description |
|--------|-------------|
| `guppy()` | Convert to Guppy source code |
| `hugr()` | Compile to HUGR via Guppy |
| `qasm()` | Convert to OpenQASM 2.0 |
| `qir()` | Convert to QIR (LLVM IR text) |
| `qir_bc()` | Convert to QIR bytecode |
| `stim()` | Convert to Stim circuit |
| `quantum_circuit()` | Convert to PECOS QuantumCircuit |

**Input methods (class methods):**

| Method | Description |
|--------|-------------|
| `from_stim(circuit)` | Convert Stim circuit to SLR |
| `from_quantum_circuit(qc)` | Convert QuantumCircuit to SLR |

### qeclib Submodules

| Module | Description |
|--------|-------------|
| `qeclib.qubit` | Basic qubit operations (H, X, CX, etc.) |
| `qeclib.surface` | Surface code operations |
| `qeclib.color488` | 4.8.8 color code operations |
| `qeclib.steane` | Steane [[7,1,3]] code operations |

## See Also

- [Parallel Execution](../user-guide/parallel-execution.md) - User guide for parallel blocks
- [Parallel Blocks and Optimization](parallel-blocks-and-optimization.md) - Deep dive into parallel optimization
- [QEC Geometry](../user-guide/qec-geometry.md) - Pure QEC geometry module
- [QEC with Guppy](../user-guide/qec-guppy.md) - High-level Guppy approach
