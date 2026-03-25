# Parallel Blocks and Optimization

This guide explains the `Parallel` block construct in PECOS's SLR (Simple Logical Representation) and the optimization transformations available for parallel quantum operations.

## Overview

The `Parallel` block is a semantic construct that indicates operations within it can be executed simultaneously on quantum hardware. While standard quantum circuit representations execute gates sequentially, real quantum hardware often supports parallel gate execution on disjoint qubits.

When using `SlrConverter`, parallel optimization is enabled by default. This means operations within `Parallel` blocks will be automatically reordered to maximize parallelism while respecting quantum gate dependencies. Programs without `Parallel` blocks are unaffected.

## Basic Usage

### Creating Parallel Blocks

```python
from pecos.slr import Main, Parallel, QReg
from pecos.slr.qeclib import qubit as qb

prog = Main(
    q := QReg("q", 4),
    Parallel(
        qb.H(q[0]),
        qb.H(q[1]),
        qb.X(q[2]),
        qb.Y(q[3]),
    ),
)
```

### Nested Structures

Parallel blocks can contain other blocks for logical grouping:

```python
from pecos.slr import Main, Parallel, Block, QReg
from pecos.slr.qeclib import qubit as qb

prog = Main(
    q := QReg("q", 6),
    Parallel(
        Block(  # Bell pair 1
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
        ),
        Block(  # Bell pair 2
            qb.H(q[2]),
            qb.CX(q[2], q[3]),
        ),
    ),
)
```

## Parallel Optimization

The `ParallelOptimizer` transformation pass analyzes operations within `Parallel` blocks and reorders them to maximize parallelism while respecting quantum gate dependencies.

### How It Works

1. **Dependency Analysis**: The optimizer tracks which qubits each operation acts on
2. **Operation Grouping**: Operations on disjoint qubits are grouped by gate type
3. **Reordering**: Groups are arranged to maximize parallel execution opportunities

### Example Transformation

**Before optimization:**

```python
from pecos.slr import Main, Parallel, Block, QReg
from pecos.slr.qeclib import qubit as qb

# Input SLR program with nested Parallel/Block structure
prog = Main(
    q := QReg("q", 6),
    Parallel(
        Block(qb.H(q[0]), qb.CX(q[0], q[1])),
        Block(qb.H(q[2]), qb.CX(q[2], q[3])),
        Block(qb.H(q[4]), qb.CX(q[4], q[5])),
    ),
)
```

**After optimization:**

The optimizer reorders operations to maximize parallelism by grouping compatible gates:

```hidden-python
from pecos.slr import Main, Parallel, Block, QReg, SlrConverter
from pecos.slr.qeclib import qubit as qb

prog = Main(
    q := QReg("q", 6),
    Parallel(
        Block(qb.H(q[0]), qb.CX(q[0], q[1])),
        Block(qb.H(q[2]), qb.CX(q[2], q[3])),
        Block(qb.H(q[4]), qb.CX(q[4], q[5])),
    ),
)
```

```python
from pecos.slr import SlrConverter

# Convert to QASM (optimizer runs automatically)
qasm = SlrConverter(prog).qasm()
print(qasm)
# All H gates are grouped in one parallel block,
# all CX gates in the next:
#   // parallel begin
#   h q[0]; h q[2]; h q[4];
#   // parallel end
#   // parallel begin
#   cx q[0], q[1]; cx q[2], q[3]; cx q[4], q[5];
#   // parallel end
```

### Using the Optimizer

#### With SlrConverter

The simplest way to use the optimizer is through `SlrConverter`:

```python
from pecos.slr import Main, Parallel, Block, QReg, SlrConverter
from pecos.slr.qeclib import qubit as qb

prog = Main(
    q := QReg("q", 6),
    Parallel(
        Block(qb.H(q[0]), qb.CX(q[0], q[1])),
        Block(qb.H(q[2]), qb.CX(q[2], q[3])),
    ),
)

# With optimization (default)
qasm = SlrConverter(prog).qasm()

# Without optimization
qasm_unoptimized = SlrConverter(prog, optimize_parallel=False).qasm()
```

#### Direct Usage

For more control, use the optimizer directly:

```python
from pecos.slr import Main, Parallel, Block, QReg
from pecos.slr.qeclib import qubit as qb
from pecos.slr.transforms import ParallelOptimizer

prog = Main(
    q := QReg("q", 4),
    Parallel(
        Block(qb.H(q[0]), qb.CX(q[0], q[1])),
        Block(qb.H(q[2]), qb.CX(q[2], q[3])),
    ),
)

optimizer = ParallelOptimizer()
optimized_prog = optimizer.transform(prog)
```

### QASM Output Comparison

Given the Bell state example above:

**Without optimization:**
```qasm
h q[0];
cx q[0], q[1];
h q[2];
cx q[2], q[3];
h q[4];
cx q[4], q[5];
```

**With optimization:**
```qasm
h q[0];
h q[2];
h q[4];
cx q[0], q[1];
cx q[2], q[3];
cx q[4], q[5];
```

## Limitations and Conservative Behavior

The optimizer is conservative to ensure correctness:

### Control Flow

Parallel blocks containing control flow (`If`, `Repeat`) are not optimized:

```python
from pecos.slr import Parallel, QReg, CReg, If
from pecos.slr.qeclib import qubit as qb

q = QReg("q", 4)
c = CReg("c", 4)

Parallel(
    qb.H(q[0]),
    If(c[0] == 1).Then(qb.X(q[1])),  # Control flow prevents optimization
    qb.H(q[2]),
)
```

### Dependencies

Operations with qubit dependencies maintain their order:

```python
from pecos.slr import Parallel, QReg
from pecos.slr.qeclib import qubit as qb

q = QReg("q", 4)

Parallel(
    qb.H(q[0]),
    qb.CX(q[0], q[1]),  # Depends on H(q[0])
    qb.X(q[1]),  # Depends on CX
)
# These operations cannot be reordered
```

## Implementation Details

### Transformation Process

1. **Bottom-up traversal**: Inner blocks are transformed first
2. **Conservative checking**: Blocks with control flow are skipped
3. **Dependency graph**: Built based on qubit usage
4. **Topological sorting**: Ensures dependency preservation
5. **Type-based grouping**: Operations grouped by gate type

### Code Structure

- `pecos/slr/misc.py` - Contains the `Parallel` class definition
- `pecos/slr/transforms/parallel_optimizer.py` - Optimization implementation
- `pecos/slr/gen_codes/gen_qasm.py` - QASM generation (treats Parallel as Block)
- `pecos/slr/gen_codes/gen_qir.py` - QIR generation (treats Parallel as Block)

## Future Enhancements

Potential improvements for the Parallel block system:

1. **Barrier semantics**: Use `Barrier` statements as optimization boundaries
2. **Classical operation handling**: Special treatment for measurements and classical ops
3. **Hardware-aware optimization**: Consider specific hardware connectivity
4. **Scheduling hints**: Allow users to specify scheduling preferences
5. **Performance metrics**: Report estimated parallelism improvements

## Testing

Comprehensive tests are available in:
- `python/tests/pecos/unit/test_parallel_optimizer.py` - Core functionality tests
- `python/tests/pecos/unit/test_parallel_optimizer_verification.py` - Transformation verification
- `python/tests/pecos/regression/test_qasm/random_cases/test_control_flow.py` - QASM generation tests

## Best Practices

1. **Use Parallel blocks for independent operations**: Only wrap operations that can truly execute in parallel
2. **Group related operations**: Use nested blocks for logical grouping (e.g., Bell pairs)
3. **Optimization is on by default**: Use `optimize_parallel=False` to disable when needed
4. **Verify transformations**: Check generated QASM/QIR to ensure desired optimization
5. **Consider hardware constraints**: Real devices have limited parallelism capabilities

## Example: Quantum Fourier Transform

Here's a more complex example showing parallel phase gates:

```python
import numpy as np
from pecos.slr import Main, Parallel, QReg
from pecos.slr.qeclib import qubit as qb


def qft_layer(q, n, k):
    """Generate parallel controlled rotations for QFT layer k"""
    operations = []
    for j in range(k + 1, n):
        angle = np.pi / (2 ** (j - k))
        operations.append(qb.CRZ[angle](q[j], q[k]))
    return Parallel(*operations) if len(operations) > 1 else operations[0]


# QFT with parallel phase gates
prog = Main(
    q := QReg("q", 4),
    qb.H(q[0]),
    qft_layer(q, 4, 0),
    qb.H(q[1]),
    qft_layer(q, 4, 1),
    qb.H(q[2]),
    qft_layer(q, 4, 2),
    qb.H(q[3]),
)
```

This structure makes the inherent parallelism in QFT explicit and allows the optimizer to group operations effectively.
