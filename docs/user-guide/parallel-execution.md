# Parallel Execution in PECOS

This guide explains how to express parallel quantum operations in PECOS using the `Parallel` block construct.

## Introduction

Modern quantum hardware often supports executing multiple quantum gates simultaneously on different qubits. PECOS provides the `Parallel` block to express this parallelism explicitly in your quantum programs.

By default, PECOS automatically optimizes operations within `Parallel` blocks to maximize parallel execution while preserving the semantics of your quantum circuit.

## Basic Usage

### Simple Parallel Operations

Use `Parallel` to indicate that operations can execute simultaneously:

```python
from pecos.slr import Main, Parallel, QReg, CReg
from pecos.slr.qeclib import qubit as qb

# Create a program with parallel Hadamard gates
prog = Main(
    q := QReg("q", 4),
    c := CReg("m", 4),
    Parallel(
        qb.H(q[0]),
        qb.H(q[1]),
        qb.H(q[2]),
        qb.H(q[3]),
    ),
    qb.Measure(q) > c,
)
```

### Grouping Related Operations

You can use nested blocks to group related operations:

```python
from pecos.slr import Main, Parallel, Block, QReg, CReg
from pecos.slr.qeclib import qubit as qb

# Three Bell pairs prepared in parallel
prog = Main(
    q := QReg("q", 6),
    c := CReg("m", 6),
    Parallel(
        Block(  # Bell pair 1
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
        ),
        Block(  # Bell pair 2
            qb.H(q[2]),
            qb.CX(q[2], q[3]),
        ),
        Block(  # Bell pair 3
            qb.H(q[4]),
            qb.CX(q[4], q[5]),
        ),
    ),
    qb.Measure(q) > c,
)
```

## Automatic Optimization

PECOS automatically optimizes `Parallel` blocks to maximize parallelism by default:

```python
from pecos.slr import Main, Parallel, QReg, CReg, SlrConverter
from pecos.slr.qeclib import qubit as qb

# Define a program to demonstrate optimization
prog = Main(
    q := QReg("q", 2),
    c := CReg("m", 2),
    Parallel(qb.H(q[0]), qb.H(q[1])),
)

# Optimization is enabled by default
qasm = SlrConverter(prog).qasm()

# To disable optimization:
qasm_unoptimized = SlrConverter(prog, optimize_parallel=False).qasm()
```

The optimizer will:
- Analyze dependencies between operations
- Group operations by gate type
- Reorder operations to maximize parallel execution

### Example: Bell State Preparation

Without optimization:
```
h q[0];
cx q[0], q[1];
h q[2];
cx q[2], q[3];
h q[4];
cx q[4], q[5];
```

With optimization:
```
h q[0];
h q[2];
h q[4];
cx q[0], q[1];
cx q[2], q[3];
cx q[4], q[5];
```

All Hadamard gates execute first in parallel, followed by all CNOT gates in parallel.

## Important Considerations

### Dependencies

Operations that share qubits cannot be parallelized:

```python
from pecos.slr import Parallel, QReg
from pecos.slr.qeclib import qubit as qb

q = QReg("q", 2)

# These operations must execute sequentially
Parallel(
    qb.H(q[0]),
    qb.CX(q[0], q[1]),  # Depends on H(q[0])
)
```

### Control Flow

Parallel blocks containing conditional operations are not optimized:

```python
from pecos.slr import Parallel, QReg, CReg, If
from pecos.slr.qeclib import qubit as qb

q = QReg("q", 2)
c = CReg("c", 2)

Parallel(
    qb.H(q[0]),
    If(c[0] == 1).Then(qb.X(q[1])),  # Control flow prevents optimization
)
```

## Tips

1. **Use Parallel for truly independent operations**: Only group operations that act on different qubits
2. **Consider hardware limitations**: Real devices have constraints on which gates can run in parallel
3. **Verify the output**: Check the generated QASM to ensure the optimization meets your needs

## Complete Example

Here's a complete example showing parallel quantum phase estimation:

```python
from pecos.slr import Main, Parallel, Block, QReg, CReg, SlrConverter
from pecos.slr.qeclib import qubit as qb
import numpy as np

# Parallel Quantum Phase Estimation
prog = Main(
    q := QReg("q", 4),
    c := CReg("m", 4),
    # Initialize ancillas in parallel
    Parallel(
        qb.H(q[0]),
        qb.H(q[1]),
        qb.H(q[2]),
    ),
    # Apply controlled rotations
    qb.CRZ[np.pi](q[0], q[3]),
    qb.CRZ[np.pi / 2](q[1], q[3]),
    qb.CRZ[np.pi / 4](q[2], q[3]),
    # Inverse QFT on ancillas
    qb.H(q[0]),
    qb.CRZ[-np.pi / 2](q[0], q[1]),
    qb.H(q[1]),
    qb.CRZ[-np.pi / 4](q[0], q[2]),
    qb.CRZ[-np.pi / 2](q[1], q[2]),
    qb.H(q[2]),
    # Measure ancillas
    Parallel(
        qb.Measure(q[0]) > c[0],
        qb.Measure(q[1]) > c[1],
        qb.Measure(q[2]) > c[2],
    ),
)

# Generate optimized QASM (optimization is on by default)
qasm = SlrConverter(prog).qasm()
```

## See Also

- [Development Guide: Parallel Blocks and Optimization](../development/parallel-blocks-and-optimization.md) - Technical details and extending the optimizer
- [Getting Started](getting-started.md) - Introduction to PECOS
- [QASM Simulation](qasm-simulation.md) - Running quantum simulations
