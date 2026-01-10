# QEC with Guppy

This guide covers PECOS's Guppy QEC code generation module (`pecos.guppy`), which generates executable Guppy quantum programs directly from QEC geometry.

## What You'll Learn

- Generating surface code memory experiments
- Generating color code memory experiments
- Transversal CNOT between code blocks
- Customizing generated code
- Running generated programs on PECOS

## Overview

The `pecos.guppy` module provides **direct Guppy code generation** for QEC circuits, bypassing intermediate representations for faster compilation:

```python
from pecos.guppy import (
    # Surface codes
    make_surface_code,
    get_num_qubits,
    # Color codes
    make_color_code,
    get_num_qubits_color,
    # Transversal operations
    make_css_transversal_cnot,
    get_transversal_num_qubits,
)
```

## Surface Code Memory Experiments

A memory experiment initializes a logical state, performs syndrome extraction rounds, and measures the final state.

### Quick Start

```python
from pecos import sim, state_vector
from pecos.guppy import make_surface_code, get_num_qubits

# Create a distance-3 Z-basis memory experiment with 3 rounds
prog = make_surface_code(distance=3, num_rounds=3, basis="Z")

# Get qubit count (d^2 data + 2 ancilla)
num_qubits = get_num_qubits(3)  # 11 qubits

# Run simulation
results = sim(prog).qubits(num_qubits).quantum(state_vector()).seed(42).run(100)

print(results.to_dict())
```

### X-Basis vs Z-Basis

```python
from pecos.guppy import make_surface_code

# Z-basis: Initialize |0_L>, measure in Z basis
z_prog = make_surface_code(distance=3, num_rounds=2, basis="Z")

# X-basis: Initialize |+_L>, measure in X basis
x_prog = make_surface_code(distance=3, num_rounds=2, basis="X")
```

### Understanding the Output

The generated program produces these result keys:

| Key | Description |
|-----|-------------|
| `init_synx` / `init_synz` | Initial syndrome (from initialization) |
| `synx` | X syndrome per round |
| `synz` | Z syndrome per round |
| `final` | Final data qubit measurements |

<!--skip: named result keys not yet captured (uses measurement_N instead)-->
```python
results = sim(prog).qubits(11).quantum(state_vector()).run(10)
data = results.to_dict()

# Access syndrome history
init_syn = data.get("init_synx", [])
synx_rounds = data.get("synx", [])
synz_rounds = data.get("synz", [])
final_meas = data.get("final", [])
```

## Color Code Memory Experiments

The 4.8.8 triangular color code supports transversal Clifford gates.

### Quick Start

<!--skip: named result keys not yet captured (uses measurement_N instead)-->
```python
from pecos import sim, state_vector
from pecos.guppy import make_color_code, get_num_qubits_color

# Create a distance-3 color code memory experiment
prog = make_color_code(distance=3, num_rounds=2, basis="Z")

# Get qubit count
num_qubits = get_num_qubits_color(3)

# Run simulation
results = sim(prog).qubits(num_qubits).quantum(state_vector()).seed(42).run(100)
```

### Comparing Surface and Color Codes

```python
from pecos.guppy import (
    make_surface_code,
    get_num_qubits,
    make_color_code,
    get_num_qubits_color,
)

d = 3

# Surface code
surface_prog = make_surface_code(distance=d, num_rounds=2, basis="Z")
surface_qubits = get_num_qubits(d)

# Color code
color_prog = make_color_code(distance=d, num_rounds=2, basis="Z")
color_qubits = get_num_qubits_color(d)

print(f"Surface code d={d}: {surface_qubits} qubits")
print(f"Color code d={d}: {color_qubits} qubits")
```

## Transversal CNOT

Transversal CNOT applies `CX(ctrl[i], tgt[i])` for all data qubits between two code blocks. This preserves the CSS structure.

### Generic CSS Transversal CNOT

<!--skip: named result keys not yet captured (uses measurement_N instead)-->
```python
from pecos import sim, state_vector
from pecos.guppy import make_css_transversal_cnot, get_transversal_num_qubits

# Create transversal CNOT for color codes
prog = make_css_transversal_cnot(
    code_type="color",  # or "surface"
    distance=3,
    num_rounds=1,
)

# Get total qubit count (2 patches + 4 ancillas)
num_qubits = get_transversal_num_qubits("color", 3)

# Run simulation
results = sim(prog).qubits(num_qubits).quantum(state_vector()).seed(42).run(100)
```

### Transversal CNOT with Logical X

Test the logical CNOT by preparing `|1_L>|0_L>` and verifying it becomes `|1_L>|1_L>`:

<!--skip: named result keys not yet captured (uses measurement_N instead)-->
```python
from pecos.guppy import make_css_transversal_cnot_with_x

# |1_L>|0_L> -> |1_L>|1_L>
prog = make_css_transversal_cnot_with_x(
    code_type="color",
    distance=3,
    num_rounds=1,
)

results = sim(prog).qubits(num_qubits).quantum(state_vector()).run(100)

# Check that both patches measure to logical 1
data = results.to_dict()
final_ctrl = data.get("final_ctrl", [])
final_tgt = data.get("final_tgt", [])
```

### Convenience Functions

For common cases, use the convenience functions:

```python
from pecos.guppy import (
    # Color code transversal CNOT
    make_color_transversal_cnot,
    make_color_transversal_cnot_with_x,
    make_color_transversal_cnot_d3,  # d=3 shortcut
    # Surface code transversal CNOT
    make_surface_transversal_cnot,
    make_surface_transversal_cnot_with_x,
)

# Quick d=3 color code transversal CNOT
prog = make_color_transversal_cnot_d3(num_rounds=1)

# Surface code transversal CNOT
prog = make_surface_transversal_cnot(distance=5, num_rounds=2)
```

## Generated Code Structure

The `pecos.guppy` module generates Guppy source code with these components:

### Struct Definitions

<!--skip: illustrative generated code, not executable-->
```python
@guppy.struct
class SurfaceCode_3x3:
    """Surface code patch with dx=3, dz=3 (9 data qubits)."""

    data: array[qubit, 9]


@guppy.struct
class Syndrome_3x3:
    """Syndrome for dx=3, dz=3 patch."""

    synx: array[bool, 4]
    synz: array[bool, 4]
```

### Stabilizer Measurements

<!--skip: illustrative generated code, not executable-->
```python
@guppy
def measure_x_stab_0(ax: qubit, data: array[qubit, 9]) -> bool:
    """Measure X stabilizer 0 (boundary): [0, 1]."""
    h(ax)
    cx(ax, data[0])
    cx(ax, data[1])
    h(ax)
    return measure_and_reset(ax)


@guppy
def measure_z_stab_0(az: qubit, data: array[qubit, 9]) -> bool:
    """Measure Z stabilizer 0 (boundary): [0, 3]."""
    cx(data[0], az)
    cx(data[3], az)
    return measure_and_reset(az)
```

### Syndrome Extraction

<!--skip: illustrative generated code, not executable-->
```python
@guppy
def syndrome_extraction(
    surf: SurfaceCode_3x3,
    ax: qubit,
    az: qubit,
) -> Syndrome_3x3:
    """Extract full syndrome."""
    # Z stabilizers
    sz0 = measure_z_stab_0(az, surf.data)
    sz1 = measure_z_stab_1(az, surf.data)
    # ...

    # X stabilizers
    sx0 = measure_x_stab_0(ax, surf.data)
    sx1 = measure_x_stab_1(ax, surf.data)
    # ...

    return Syndrome_3x3(array(sx0, sx1, ...), array(sz0, sz1, ...))
```

## Viewing Generated Source

To see the generated Guppy source code:

```python
from pecos.guppy import generate_surface_code_module, generate_color_code_module

# Surface code source
source = generate_surface_code_module(d=3)
print(source)

# Color code source
source = generate_color_code_module(d=3)
print(source)
```

This is useful for:

- Understanding the generated circuit structure
- Debugging issues
- Customizing the generated code

## Advanced: Working with Modules

For more control, access the generated module directly:

```python
from pecos.guppy import get_surface_code_module

# Get the loaded module
module = get_surface_code_module(d=3)

# Access individual functions
make_memory_z = module["make_memory_z"]
make_memory_x = module["make_memory_x"]

# Create custom experiments
prog = make_memory_z(num_rounds=5)

# Access metadata
print(f"Distance: {module['distance']}")
print(f"Data qubits: {module['num_data']}")
print(f"Stabilizers: {module['num_stab']}")
```

## Adding Noise

Add noise to QEC simulations:

<!--skip: named result keys not yet captured (uses measurement_N instead)-->
```python
from pecos import sim, state_vector, depolarizing_noise
from pecos.guppy import make_surface_code, get_num_qubits

prog = make_surface_code(distance=3, num_rounds=3, basis="Z")
num_qubits = get_num_qubits(3)

# Add depolarizing noise
results = (
    sim(prog)
    .qubits(num_qubits)
    .quantum(state_vector())
    .noise(depolarizing_noise(p=0.001))
    .seed(42)
    .run(1000)
)
```

## Complete Example: Threshold Estimation

Here's a complete example estimating the logical error rate:

<!--skip: named result keys not yet captured (uses measurement_N instead)-->
```python
from pecos import sim, state_vector, depolarizing_noise
from pecos.guppy import make_surface_code, get_num_qubits
from pecos.qec import logical_z_from_data


def estimate_logical_error_rate(distance: int, p: float, shots: int = 1000) -> float:
    """Estimate logical error rate for a surface code."""
    prog = make_surface_code(distance=distance, num_rounds=distance, basis="Z")
    num_qubits = get_num_qubits(distance)

    results = (
        sim(prog)
        .qubits(num_qubits)
        .quantum(state_vector())
        .noise(depolarizing_noise(p=p))
        .seed(42)
        .run(shots)
    )

    data = results.to_dict()
    final = data.get("final", [])

    # Count logical errors (expected logical Z = 0 for |0_L>)
    errors = 0
    for shot_data in final:
        logical = logical_z_from_data(distance, shot_data)
        if logical != 0:
            errors += 1

    return errors / shots


# Compare different distances
for d in [3, 5, 7]:
    error_rate = estimate_logical_error_rate(d, p=0.001)
    print(f"d={d}: logical error rate = {error_rate:.4f}")
```

## API Reference

### Surface Code Functions

| Function | Description |
|----------|-------------|
| `make_surface_code(distance, num_rounds, basis)` | Create memory experiment |
| `get_num_qubits(d)` | Get total qubit count (d^2 + 2) |
| `generate_surface_code_module(d)` | Get generated source code |
| `get_surface_code_module(d)` | Get loaded module dict |

### Color Code Functions

| Function | Description |
|----------|-------------|
| `make_color_code(distance, num_rounds, basis)` | Create memory experiment |
| `get_num_qubits_color(d)` | Get total qubit count |
| `generate_color_code_module(d)` | Get generated source code |
| `get_color_code_module(d)` | Get loaded module dict |

### Transversal CNOT Functions

| Function | Description |
|----------|-------------|
| `make_css_transversal_cnot(code_type, distance, num_rounds)` | Generic transversal CNOT |
| `make_css_transversal_cnot_with_x(...)` | With logical X on control |
| `get_transversal_num_qubits(code_type, distance)` | Get total qubit count |
| `make_color_transversal_cnot(distance, num_rounds)` | Color code shortcut |
| `make_surface_transversal_cnot(distance, num_rounds)` | Surface code shortcut |

## Alternative: SLR for Low-Level Control

For developers who need more direct control over circuit construction, PECOS also provides the SLR (Simple Logical Representation) framework. SLR is a lower-level programmatic approach that:

- Gives fine-grained control over circuit structure
- Outputs to Guppy, QASM, or QIR
- Supports custom compilation pipelines
- Enables programmatic circuit manipulation

```python
from pecos.slr import Main, Block, QReg, CReg, SlrConverter
from pecos.slr.qeclib import qubit as qb

# Build circuit programmatically
prog = Main(
    q := QReg("q", 9),
    c := CReg("c", 9),
    Block(
        qb.H(q[0]),
        qb.CX(q[0], q[1]),
        # ... more operations
    ),
    qb.Measure(q) > c,
)

# Convert to QASM
qasm = SlrConverter(prog).qasm()
```

See the [SLR and QECLib Developer Guide](../development/slr-qeclib.md) for details.

## Next Steps

- **[QEC Geometry](qec-geometry.md)** - Understand the underlying geometry
- **[Decoders](decoders.md)** - Decode syndromes to recover logical information
- **[Noise Model Builders](noise-model-builders.md)** - Custom noise configurations
- **[HUGR & Guppy Simulation](hugr-simulation.md)** - More Guppy features
