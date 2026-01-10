# QEC Geometry

This guide covers PECOS's pure QEC geometry module (`pecos.qec`), which provides data structures and utilities for quantum error correction codes without any simulation dependencies.

## What You'll Learn

- Working with surface code geometry
- Working with color code geometry
- Analyzing QEC simulation results
- Using the generic stabilizer framework
- Magic state distillation protocol geometry

## Overview

The `pecos.qec` module provides:

- **Pure geometry** - No SLR or simulation dependencies
- **Dataclass-based** - Immutable, well-typed data structures
- **Code-agnostic utilities** - Work across different QEC code families

```python
from pecos.qec import (
    # Surface codes
    SurfacePatch,
    SurfacePatchBuilder,
    # Color codes
    ColorCode488,
    # Analysis
    logical_fidelity,
    syndrome_to_detection_events,
    # Protocols
    create_msd_protocol,
)
```

## Surface Codes

Surface codes are the most widely studied topological QEC codes. PECOS supports both rotated (default) and standard layouts.

### Creating a Surface Code Patch

=== "Factory Method"

    ```python
    from pecos.qec.surface import SurfacePatch

    # Symmetric distance-5 patch (rotated layout, default)
    patch = SurfacePatch.create(distance=5)

    # Standard (non-rotated) layout
    patch = SurfacePatch.create(distance=5, rotated=False)

    # Asymmetric patch
    patch = SurfacePatch.create(dx=3, dz=5)
    ```

=== "Builder Pattern"

    ```python
    from pecos.qec.surface import SurfacePatchBuilder, PatchOrientation

    patch = (
        SurfacePatchBuilder()
        .with_distance(5)
        .with_orientation(PatchOrientation.Z_TOP_BOTTOM)
        .build()
    )

    # Non-rotated (standard) layout
    patch = (
        SurfacePatchBuilder()
        .with_distance(5)
        .standard()  # Use standard layout instead of rotated
        .build()
    )
    ```

### Accessing Geometry

Once you have a patch, you can access its geometry:

```python
from pecos.qec.surface import SurfacePatch

patch = SurfacePatch.create(distance=3)

# Basic properties
print(f"Distance: {patch.distance}")  # 3
print(f"Data qubits: {patch.num_data}")  # 9
print(f"Total qubits: {patch.num_qubits}")  # 11 (9 data + 2 ancilla)

# Stabilizers
for stab in patch.x_stabilizers:
    print(
        f"X stab {stab.index}: qubits={stab.data_qubits}, boundary={stab.is_boundary}"
    )

for stab in patch.z_stabilizers:
    print(f"Z stab {stab.index}: qubits={stab.data_qubits}, weight={stab.weight}")

# Logical operators
geom = patch.geometry
print(f"Logical X qubits: {geom.logical_x.data_qubits}")
print(f"Logical Z qubits: {geom.logical_z.data_qubits}")

# Parity check matrix
H_x = patch.get_parity_matrix("X")
H_z = patch.get_parity_matrix("Z")
```

### Rotated vs Standard Layout

PECOS supports two surface code layouts:

| Layout | Qubits | Description |
|--------|--------|-------------|
| **Rotated** (default) | $d^2$ | More common, fewer physical qubits |
| **Standard** | $d^2$ | Traditional square lattice |

```python
# Rotated layout (default) - more efficient
rotated = SurfacePatch.create(distance=5)

# Standard layout - traditional
standard = SurfacePatch.create(distance=5, rotated=False)

print(f"Rotated: {rotated.num_data} data qubits")  # 25
print(f"Standard: {standard.num_data} data qubits")  # 25
```

### Low-Level Layout Functions

For direct access to stabilizer supports without creating a patch:

```python
from pecos.qec.surface import (
    compute_x_stabilizer_supports,
    compute_z_stabilizer_supports,
    parity_matrix_x,
    parity_matrix_z,
)

# Get stabilizer supports for distance-3
x_stabs = compute_x_stabilizer_supports(d=3)
z_stabs = compute_z_stabilizer_supports(d=3)

for stab in x_stabs:
    print(f"X[{stab.index}]: {stab.data_qubits}, boundary={stab.is_boundary}")

# Get parity check matrices directly
H_x = parity_matrix_x(d=3)
H_z = parity_matrix_z(d=3)
```

## Color Codes

The 4.8.8 triangular color code is a topological code with transversal Clifford gates. Stabilizers are colored red, green, and blue.

### Creating a Color Code

```python
from pecos.qec.color import ColorCode488, ColorCode488Builder

# Factory method
code = ColorCode488.create(distance=3)

# Builder pattern
code = ColorCode488Builder().with_distance(5).build()
```

### Accessing Geometry

```python
from pecos.qec.color import ColorCode488

code = ColorCode488.create(distance=3)

# Basic properties
print(f"Distance: {code.distance}")
print(f"Data qubits: {code.num_data}")
print(f"Stabilizers: {code.num_stabilizers}")

# All stabilizers
for stab in code.stabilizers:
    print(
        f"Stab {stab.index}: color={stab.color}, qubits={stab.qubits}, weight={stab.weight}"
    )

# Filter by color
red_stabs = code.get_red_stabilizers()
green_stabs = code.get_green_stabilizers()
blue_stabs = code.get_blue_stabilizers()

# Logical operators (same support for self-dual code)
logical_x = code.get_logical_x()
logical_z = code.get_logical_z()

# Parity check matrix
H = code.get_parity_matrix()
```

### Color Code Properties

The 4.8.8 color code has special properties:

- **Self-dual**: Logical X and Z have the same qubit support
- **Transversal Clifford**: All Clifford gates can be applied transversally
- **Three colors**: Stabilizers are red, green, or blue

```python
code = ColorCode488.create(distance=3)

# Count stabilizers by color
colors = {"red": 0, "green": 0, "blue": 0}
for stab in code.stabilizers:
    colors[stab.color] += 1

print(f"Stabilizers by color: {colors}")
```

## Result Analysis

The `pecos.qec.analysis` module provides utilities for analyzing QEC simulation results.

### Extracting Logical Values

```python
from pecos.qec import logical_x_from_data, logical_z_from_data, logical_from_data

# Simulated measurement data (d^2 = 9 values for d=3)
data = [0, 1, 0, 1, 0, 1, 0, 1, 0]

# Extract logical values
d = 3
logical_x = logical_x_from_data(d, data)  # XOR of left column
logical_z = logical_z_from_data(d, data)  # XOR of top row

# Or get both at once
lx, lz = logical_from_data(d, data)
```

### Computing Fidelity

```python
from pecos.qec import logical_fidelity, logical_error_rate

# Multiple measurement outcomes from simulation
outcomes = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],  # Shot 1
    [0, 0, 0, 0, 0, 0, 0, 0, 0],  # Shot 2
    [1, 0, 0, 0, 0, 0, 0, 0, 0],  # Shot 3 (error)
    # ... more shots
]

d = 3
basis = 1  # 0=X, 1=Z
expected = 0  # Expected logical value

# Compute fidelity with error bars
fidelity, error = logical_fidelity(outcomes, d, basis, expected)
print(f"Fidelity: {fidelity:.4f} +/- {error:.4f}")

# Or compute error rate directly
error_rate, error_bar = logical_error_rate(outcomes, d, basis, expected)
print(f"Error rate: {error_rate:.4f} +/- {error_bar:.4f}")
```

### Processing Syndromes

```python
from pecos.qec import syndrome_difference, syndrome_to_detection_events

# Syndrome measurements from multiple rounds
syndromes = [
    [0, 1, 0, 0],  # Round 0
    [0, 1, 1, 0],  # Round 1
    [1, 0, 1, 0],  # Round 2
]

# Compute differences (for MWPM decoding)
diffs = syndrome_difference(syndromes)
# diffs[0] = syndromes[0] (compared to all-zeros)
# diffs[i] = syndromes[i] XOR syndromes[i-1]

# Convert to detection events (stabilizer, round) pairs
events = syndrome_to_detection_events(syndromes)
for stab_idx, round_idx in events:
    print(f"Detection at stabilizer {stab_idx}, round {round_idx}")
```

### Lower Bound Fidelity

When measuring in two bases, compute a statistical lower bound:

```python
from pecos.qec import lower_bound_fidelity

f_x = 0.95  # X-basis fidelity
f_z = 0.93  # Z-basis fidelity

bound = lower_bound_fidelity(f_x, f_z)
print(f"Lower bound on true fidelity: {bound:.4f}")
```

## Generic Stabilizer Framework

The `pecos.qec.generic` module provides code-agnostic abstractions:

```python
from pecos.qec.generic import StabilizerCheck, PauliType, CheckSchedule

# Create stabilizer checks
x_check = StabilizerCheck.x_check(index=0, qubits=(0, 1, 2, 3))
z_check = StabilizerCheck.z_check(index=1, qubits=(1, 2, 4, 5), is_boundary=True)

print(f"X check weight: {x_check.weight}")
print(f"Z check is boundary: {z_check.is_boundary}")
```

## Magic State Distillation

The `pecos.qec.protocols` module provides geometry for MSD protocols:

```python
from pecos.qec.protocols import create_msd_protocol, MSDProtocol

# Create MSD protocol with default parameters
protocol = create_msd_protocol(inner_rounds=2, outer_rounds=1)

# Inner code (distance-2, 4 qubits)
inner = protocol.inner
print(f"Inner code: {inner.num_data} data qubits")
print(f"Inner X stabilizers: {inner.x_stabilizers}")
print(f"Inner Z stabilizer: {inner.z_stabilizer}")

# Outer code (distance-3, 9 qubits)
outer = protocol.outer
print(f"Outer code: {outer.num_data} data qubits")
print(f"Expansion qubits: {outer.expansion_qubits}")

# Get preparation states for expansion
prep_states = protocol.get_expansion_prep_states()
# {2: '0', 5: '0', 6: '+', 7: '+', 8: '+'}

# Get initial states for inner code (T state distillation)
init_states = protocol.get_inner_init_states()
# {0: 'T+', 1: '0', 3: '+', 4: '+'}
```

### MSD Qubit Layout

The MSD protocol uses a 3x3 grid:

```
0  1  2
3  4  5
6  7  8
```

- **Inner code**: Qubits {0, 1, 3, 4} (top-left 2x2)
- **Expansion**: Qubits {2, 5, 6, 7, 8} (added for outer code)

## API Reference

### Surface Code Classes

| Class | Description |
|-------|-------------|
| `SurfacePatch` | Configurable surface code patch |
| `SurfacePatchBuilder` | Builder for creating patches |
| `PatchGeometry` | Underlying geometry data |
| `Stabilizer` | Individual stabilizer (index, qubits, type) |
| `LogicalOperator` | Logical X or Z operator |
| `PatchOrientation` | Boundary orientation enum |

### Color Code Classes

| Class | Description |
|-------|-------------|
| `ColorCode488` | 4.8.8 triangular color code |
| `ColorCode488Builder` | Builder for creating codes |
| `ColorCode488Geometry` | Underlying geometry data |
| `ColorCodeStabilizer` | Stabilizer with color attribute |

### Analysis Functions

| Function | Description |
|----------|-------------|
| `logical_x_from_data(d, data)` | Extract logical X from measurements |
| `logical_z_from_data(d, data)` | Extract logical Z from measurements |
| `logical_from_data(d, data)` | Extract both logical values |
| `logical_fidelity(outcomes, d, basis, expected)` | Compute fidelity with error bars |
| `logical_error_rate(outcomes, d, basis, expected)` | Compute error rate |
| `syndrome_difference(syndromes)` | Compute syndrome changes |
| `syndrome_to_detection_events(syndromes)` | Convert to (stab, round) pairs |
| `lower_bound_fidelity(f1, f2)` | Statistical lower bound |

### Protocol Classes

| Class | Description |
|-------|-------------|
| `MSDProtocol` | Magic state distillation geometry |
| `InnerCodeGeometry` | Distance-2 inner code |
| `OuterCodeGeometry` | Distance-3 outer code |

## Using Geometry with Circuit Generators

The `pecos.qec` geometry can be used with different circuit generation approaches:

| Approach | Module | Description |
|----------|--------|-------------|
| **Guppy** (recommended) | `pecos.guppy` | High-level, generates Guppy programs |
| **SLR** (advanced) | `pecos.slr.qeclib` | Low-level, generates Guppy/QASM/QIR |

```python
# Using geometry with Guppy (recommended)
from pecos.guppy import make_surface_code

prog = make_surface_code(distance=3, num_rounds=2, basis="Z")

# Using geometry with SLR (for developers needing more control)
from pecos.qec.surface import SurfacePatch
from pecos.slr import Main, QReg, SlrConverter

patch = SurfacePatch.create(distance=3)
# Build SLR program using patch.x_stabilizers, patch.z_stabilizers, etc.
```

## Next Steps

- **[QEC with Guppy](qec-guppy.md)** - Generate executable QEC circuits from geometry
- **[SLR and QECLib](../development/slr-qeclib.md)** - Low-level circuit construction (developers)
- **[Decoders](decoders.md)** - Decode syndromes to recover logical information
- **[HUGR & Guppy Simulation](hugr-simulation.md)** - Run Guppy programs on PECOS
