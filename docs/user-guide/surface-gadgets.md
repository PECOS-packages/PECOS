# Surface Code Gadgets

Each surface-code gadget has one physical definition and two renderings:
`TickCircuit` for circuit analysis and Guppy source for compiled execution.
Preparation, syndrome extraction, readout, and gates share the same step lists.

## The patch

Start with geometry, then allocate registers. The default allocation places data
first, followed by dedicated X-check and Z-check ancillas in stabilizer-index
order. A rotated distance-3 patch has 9 data qubits and 4 ancillas per family.

```python
from pecos.qec.surface import SurfacePatch, gadgets

patch = SurfacePatch.create(distance=3)
allocation = gadgets.default_allocation(patch)
assert (patch.dx, patch.dz, patch.rotated) == (3, 3, True)
assert allocation.data_qubits == list(range(9))
assert allocation.x_ancilla_qubits == list(range(9, 13))
assert allocation.z_ancilla_qubits == list(range(13, 17))
rectangle = SurfacePatch.create(dx=3, dz=5, rotated=True)
assert (rectangle.dx, rectangle.dz) == (3, 5)
```

Single-patch gadgets and their `TickCircuit` and Guppy renderers support
rotated square and rotated rectangular patches, and non-rotated square patches.
Transversal H requires a square patch. CX requires identical static geometry,
disjoint allocations, and the same current X/Z orientation on both patches;
the builder checks orientation, while the standalone gadget does not.
The logical builder requires even-weight checks. The protocol module accepts
only odd square rotated patches with distance at least 3.

Y product preparation renders at any distance, but carries logical Y after
projection only when both `dx` and `dz` are odd. On an even square patch,
the all-X product is a stabilizer, so the projected all-Y state has no logical-Y
content. `add_sz_via_teleportation` requires both `dx` and `dz` of the ancilla
to be odd. The [example](#preparation-in-y) shows odd- and even-distance product preparation.

```hidden-python
from pathlib import Path
import re

from pecos.qec.surface import SurfacePatch, LogicalCircuitBuilder, gadgets
from pecos.qec.surface.circuit_builder import TickCircuitRenderer, QubitAllocation
from pecos.guppy_gen.gadget_render import render_gadget_function, render_surface_gadget_module

patch = SurfacePatch.create(distance=3)
allocation = gadgets.default_allocation(patch)


def render_tick(gadget):
    return TickCircuitRenderer().render(list(gadget.steps), allocation, patch, 0, gadget.basis or "Z")


def guard_excerpt(section, lines):
    roots = [Path.cwd(), *Path.cwd().parents]
    path = next(root / "docs/user-guide/surface-gadgets.md" for root in roots
                if (root / "docs/user-guide/surface-gadgets.md").is_file())
    body = path.read_text().split("### " + section + "\n", 1)[1].split("\n##", 1)[0]
    fence = "`" * 3
    excerpts = re.findall(fence + r"text\n(.*?)" + fence, body, re.DOTALL)
    assert excerpts
    rendered = {line.strip() for line in lines}
    for excerpt in excerpts:
        for line in excerpt.splitlines():
            if line.strip() != "...":
                assert line.strip() in rendered, line
```

## The gadget library

`Gadget` is a frozen dataclass with `kind`, `name`, `steps`, `allocations`,
`dimensions`, `basis`, and `x_z_swapped`. Its steps are `SurfaceCircuitStep`
operations.
All single-patch functions below take `patch, allocation` first unless shown
otherwise. `round_order=None` uses the default schedule.

| Function and remaining parameters | `GadgetKind` | Effect | Guppy sideband tags |
|---|---|---|---|
| `prep_gadget(..., basis=)` | `PREP` | Product preparation in Z, X, or Y; projection follows separately | None |
| `init_syndrome_gadget(..., basis=, round_order=None, x_z_swapped=False)` | `INIT_SYNDROME` | Project the complementary family after Z or X preparation | `<label>:init:meas:<ordinal>` |
| `syndrome_round_gadget(..., round_index=, round_order=None, x_z_swapped=False)` | `SYNDROME_ROUND` | Measure both check families | `<label>:meas:<ordinal>` |
| `measure_out_gadget(..., basis=)` | `MEASURE_OUT` | Destructive Z or X data measurement | None; returns an array |
| `logical_pauli_gadget(..., pauli=)` | `LOGICAL_PAULI` | Apply the geometry's logical X or Z string | None |
| `transversal_layer_gadget(..., gate=)` | `TRANSVERSAL` | H exchanges X/Z orientation; SZ and SZDG are physical S layers | None |
| `transversal_cx_gadget(ctrl_patch, ctrl_allocation, tgt_patch, tgt_allocation)` | `TWO_PATCH` | CX between corresponding data indices | None |
| `memory_gadgets(patch, num_rounds, basis, allocation=None, round_order=None)` | List of gadget kinds | Z/X prep, initial projection, full rounds, readout | Constituent gadget tags plus wrapper outputs: scalar `final:meas:<index>` and aggregate `init_synx`/`init_synz`, `synx`/`synz`, and `final` |

During generation the builder toggles its tracked orientation after a
transversal H; standalone callers track orientation themselves and pass it to
later rounds. The orientation belongs to the builder's patch state, not to
`SurfacePatch` itself.
`tag_scope="a"` prefixes tags with `a:`; a swapped gadget adds `swapped:` after
that scope. Labels identify physical register slots, while returned syndrome
arrays follow the current X/Z families. Ordinals start at zero within each
function invocation.

### Memory composition

Surface-code memory and syndrome extraction follow the construction described
by Fowler, Mariantoni, Martinis, and Cleland in [arXiv:1208.0928](https://arxiv.org/abs/1208.0928).
The memory module includes factories `make_memory_z` and `make_memory_x`.
A distance-3 Z memory with three rounds takes 34 ticks and makes 37 measurements.

```python
parts = gadgets.memory_gadgets(patch, 3, "Z")
assert len(parts) == 6
steps = [step for gadget in parts for step in gadget.steps]
tc = TickCircuitRenderer().render(steps, allocation, patch, 3, "Z")
assert tc.num_measurements() == 37
assert tc.num_ticks() == 34
```

```python
source = render_surface_gadget_module(patch)
assert "def make_memory_z" in source
assert "def make_memory_x" in source
```

### Preparation in Z

`basis="Z"` prepares each data qubit in |0>. Z checks are initially fixed;
X projection establishes the complementary signs. At distance 3, preparation
in Z is one tick of nine preparations, with no measurements.

```python
gadget = gadgets.prep_gadget(patch, allocation, basis="Z")
tc = render_tick(gadget)
assert tc.num_ticks() == 1
assert tc.num_measurements() == 0
```

```python
gadget = gadgets.prep_gadget(patch, allocation, basis="Z")
lines = render_gadget_function(gadget)
assert lines[0] == "@guppy"
guard_excerpt("Preparation in Z", lines)
```

```text
@guppy
def prep_z_basis() -> SurfaceCode_3x3:
    """Prepare logical |0_L> state."""
    data = array(qubit() for _ in range(9))
    return SurfaceCode_3x3(data)
```

### Preparation in X

`basis="X"` applies H to every newly allocated data qubit, preparing |+>.
X checks are initially fixed; Z projection follows. At distance 3, preparation
in X takes two ticks and makes no measurements.

```python
gadget = gadgets.prep_gadget(patch, allocation, basis="X")
tc = render_tick(gadget)
assert tc.num_ticks() == 2
assert tc.num_measurements() == 0
```

```python
gadget = gadgets.prep_gadget(patch, allocation, basis="X")
lines = render_gadget_function(gadget)
assert lines[0] == "@guppy"
guard_excerpt("Preparation in X", lines)
```

```text
@guppy
def prep_x_basis() -> SurfaceCode_3x3:
    """Prepare logical |+_L> state."""
    data = array(qubit() for _ in range(9))
    for i in range(9):
        h(data[i])
    return SurfaceCode_3x3(data)
```

### Preparation in Y

`basis="Y"` applies H then physical S on every data qubit. Neither check
family is initially deterministic. The projected state carries logical Y only
for odd `dx` and `dz`; its sign depends on syndrome outcomes. On an even square
patch, the all-X product is a stabilizer and the projected state has no logical-Y
content, although the preparation gadget still renders.
`add_sz_via_teleportation` requires both ancilla dimensions to be odd. Preparation in Y takes
three ticks and makes no measurements at distances 3 and 4. The builder
rejects the distance-4 ancilla for S teleportation.

```python
gadget = gadgets.prep_gadget(patch, allocation, basis="Y")
tc = render_tick(gadget)
assert tc.num_ticks() == 3
assert tc.num_measurements() == 0

p4 = SurfacePatch.create(distance=4)
a4 = gadgets.default_allocation(p4)
y4 = gadgets.prep_gadget(p4, a4, basis="Y")
tc4 = TickCircuitRenderer().render(list(y4.steps), a4, p4, 0, "Y")
assert tc4.num_ticks() == 3
assert tc4.num_measurements() == 0
assert "        s(data[i])" in render_gadget_function(y4)
builder = LogicalCircuitBuilder()
builder.add_patch(p4, "D")
builder.add_patch(p4, "A", qubit_offset=p4.geometry.num_qubits)
try:
    builder.add_sz_via_teleportation("D", "A", 2, 2)
except ValueError as error:
    assert str(error) == "Injection ancilla 'A' requires odd dx and dz for encoded logical-Y content"
else:
    raise AssertionError("Even-distance teleportation ancilla was accepted")
```

```python
gadget = gadgets.prep_gadget(patch, allocation, basis="Y")
lines = render_gadget_function(gadget)
assert lines[0] == "@guppy"
guard_excerpt("Preparation in Y", lines)
```

```text
@guppy
def prep_y_basis() -> SurfaceCode_3x3:
    """Prepare logical |+i_L> state."""
    data = array(qubit() for _ in range(9))
    for i in range(9):
        h(data[i])
    for i in range(9):
        s(data[i])
    return SurfaceCode_3x3(data)
```

### Initial syndrome projection

`basis="Z"` measures X checks; `basis="X"` measures Z checks. `round_order`
selects the CX schedule and `x_z_swapped` selects the orientation. This separate
projection belongs to the memory module, not the protocol factories. At
distance 3, Z-basis initialization takes eight ticks and measures four ancillas;
X-basis initialization takes six ticks and also measures four ancillas. These
counts hold in either orientation.
The Guppy excerpt shows the first CX layer and the complete readout tail;
ancilla allocation, Hadamards, and the other CX layers are omitted.

```python
gadget = gadgets.init_syndrome_gadget(patch, allocation, basis="Z")
tc = render_tick(gadget)
assert tc.num_ticks() == 8
assert tc.num_measurements() == 4

for basis in ("Z", "X"):
    for swapped in (False, True):
        gadget = gadgets.init_syndrome_gadget(patch, allocation, basis=basis, x_z_swapped=swapped)
        tc = render_tick(gadget)
        assert tc.num_ticks() == (8 if basis == "Z" else 6)
        assert tc.num_measurements() == 4
        lines = render_gadget_function(gadget, tag_scope="a")
        assert sum('output("a:' in line for line in lines) == 4
```

```python
gadget = gadgets.init_syndrome_gadget(patch, allocation, basis="Z")
lines = render_gadget_function(gadget)
assert lines[0] == "@guppy"
guard_excerpt("Initial syndrome projection", lines)
```

```text
@guppy
def init_z_basis(surf: SurfaceCode_3x3) -> array[bool, 4]:
    cx(ax1, surf.data[2])
    cx(ax2, surf.data[4])
    cx(ax3, surf.data[8])
    ...
    sx0 = measure(ax0).read()
    output("sx0:init:meas:0", sx0)
    sx1 = measure(ax1).read()
    output("sx1:init:meas:1", sx1)
    sx2 = measure(ax2).read()
    output("sx2:init:meas:2", sx2)
    sx3 = measure(ax3).read()
    output("sx3:init:meas:3", sx3)
    return array(sx0, sx1, sx2, sx3)
```

### Syndrome round

`round_index` is zero-based and labels the step list and `TickCircuit` round
metadata only. Guppy output tags carry no round index and repeat across
invocations. `round_order` selects the schedule. Set `x_z_swapped=True` after one transversal H to reverse CX
directions and exchange which ancillas receive H. At distance 3, a syndrome
round takes eight ticks and makes eight measurements in either orientation.
The Guppy excerpt shows the first CX layer, the last ancilla readout, and
the returned syndrome arrays; allocation and intermediate operations are omitted.

```python
gadget = gadgets.syndrome_round_gadget(patch, allocation, round_index=0, x_z_swapped=True)
tc = render_tick(gadget)
assert tc.num_ticks() == 8
assert tc.num_measurements() == 8

for swapped in (False, True):
    gadget = gadgets.syndrome_round_gadget(patch, allocation, round_index=1, x_z_swapped=swapped)
    tc = render_tick(gadget)
    assert tc.num_ticks() == 8
    assert tc.num_measurements() == 8
    assert sum('output("' in line for line in render_gadget_function(gadget)) == 8
```

```python
gadget = gadgets.syndrome_round_gadget(patch, allocation, round_index=0, x_z_swapped=True)
lines = render_gadget_function(gadget)
assert lines[0] == "@guppy"
guard_excerpt("Syndrome round", lines)
```

```text
@guppy
def syndrome_extraction_swapped(surf: SurfaceCode_3x3) -> Syndrome_3x3:
    cx(az0, surf.data[3])
    cx(surf.data[2], ax1)
    cx(az1, surf.data[1])
    cx(surf.data[4], ax2)
    cx(az2, surf.data[5])
    cx(surf.data[8], ax3)
    ...
    sx3 = measure(ax3).read()
    output("swapped:sx3:meas:7", sx3)
    synx = array(sz0, sz1, sz2, sz3)
    synz = array(sx0, sx1, sx2, sx3)
    return Syndrome_3x3(synx, synz)
```

### Measure-out

`basis="Z"` directly measures data; `basis="X"` applies H first. Both
consume the patch and return data bits. Y readout is unsupported. At distance 3,
X readout takes two ticks and Z readout takes one; each makes nine measurements.

```python
gadget = gadgets.measure_out_gadget(patch, allocation, basis="X")
tc = render_tick(gadget)
assert tc.num_ticks() == 2
assert tc.num_measurements() == 9

for basis in ("Z", "X"):
    gadget = gadgets.measure_out_gadget(patch, allocation, basis=basis)
    tc = render_tick(gadget)
    assert tc.num_ticks() == (1 if basis == "Z" else 2)
    assert tc.num_measurements() == 9
    assert "    return collect_measurements(measure_array(surf.data))" in render_gadget_function(gadget)
```

```python
gadget = gadgets.measure_out_gadget(patch, allocation, basis="X")
lines = render_gadget_function(gadget)
assert lines[0] == "@guppy"
guard_excerpt("Measure-out", lines)
```

```text
@guppy
def measure_x_basis(surf: SurfaceCode_3x3 @ owned) -> array[bool, 9]:
    """Destructively measure in X basis."""
    for i in range(9):
        h(surf.data[i])
    return collect_measurements(measure_array(surf.data))
```

### Logical Pauli

`pauli="X"` or `pauli="Z"` applies the geometry's logical Pauli string without
measurements or allocation; its effect depends on the state and on the current
orientation. At distance
3, either string applies three Pauli gates in one tick, with no measurements.

```python
gadget = gadgets.logical_pauli_gadget(patch, allocation, pauli="X")
tc = render_tick(gadget)
assert tc.num_ticks() == 1
assert tc.num_measurements() == 0

for pauli in ("X", "Z"):
    gadget = gadgets.logical_pauli_gadget(patch, allocation, pauli=pauli)
    assert len(gadget.steps) == 3
    tc = render_tick(gadget)
    assert tc.num_ticks() == 1
    assert tc.num_measurements() == 0
    assert sum(line.strip().startswith(pauli.lower() + "(") for line in render_gadget_function(gadget)) == 3
```

```python
gadget = gadgets.logical_pauli_gadget(patch, allocation, pauli="X")
lines = render_gadget_function(gadget)
assert lines[0] == "@guppy"
guard_excerpt("Logical Pauli", lines)
```

```text
@guppy
def apply_logical_x(surf: SurfaceCode_3x3) -> None:
    """Apply logical X (string along left edge)."""
    x(surf.data[0])
    x(surf.data[3])
    x(surf.data[6])
```

### Transversal H

`gate="H"` applies H to every data qubit of a square patch and exchanges
the X/Z orientation. Later standalone extraction needs the updated
`x_z_swapped` flag. At distance 3, transversal H applies nine gates in one tick,
with no measurements.

```python
gadget = gadgets.transversal_layer_gadget(patch, allocation, gate="H")
tc = render_tick(gadget)
assert tc.num_ticks() == 1
assert tc.num_measurements() == 0
```

```python
gadget = gadgets.transversal_layer_gadget(patch, allocation, gate="H")
lines = render_gadget_function(gadget)
assert lines[0] == "@guppy"
guard_excerpt("Transversal H", lines)
```

```text
@guppy
def transversal_h(surf: SurfaceCode_3x3) -> None:
    """Apply physical transversal H."""
    for i in range(9):
        h(surf.data[i])
```

### Physical S and S-dagger layers

`gate="SZ"` and `gate="SZDG"` apply physical S and S-dagger to every data
qubit. These are not logical S gates on surface-code patches of distance at
least 3; a distance-1 patch is the trivial case. The builder's
`add_transversal_sz` and `add_transversal_szdg` require a square patch, while
the standalone `transversal_layer_gadget` accepts rectangles. At distance 3, each
layer takes one tick and makes no measurements.

```python
for gate in ("SZ", "SZDG"):
    gadget = gadgets.transversal_layer_gadget(patch, allocation, gate=gate)
    tc = render_tick(gadget)
    assert tc.num_ticks() == 1
    assert tc.num_measurements() == 0
```

```python
lines = []
for gate in ("SZ", "SZDG"):
    gadget = gadgets.transversal_layer_gadget(patch, allocation, gate=gate)
    rendered = render_gadget_function(gadget)
    expected = "s" if gate == "SZ" else "sdg"
    assert f"        {expected}(surf.data[i])" in rendered
    lines.extend(rendered)
guard_excerpt("Physical S and S-dagger layers", lines)
```

```text
@guppy
def physical_sz_layer(surf: SurfaceCode_3x3) -> None:
    """Apply physical transversal SZ."""
    for i in range(9):
        s(surf.data[i])
```

```text
@guppy
def physical_szdg_layer(surf: SurfaceCode_3x3) -> None:
    """Apply physical transversal SZDG."""
    for i in range(9):
        sdg(surf.data[i])
```

### Transversal CX

Pass two patches with identical static geometry, disjoint allocations, and
the same current X/Z orientation. The builder rejects an orientation mismatch;
the standalone `transversal_cx_gadget` does not check orientation.
Corresponding data qubits undergo control-to-target CX. Transversal CNOT is
logical CNOT for any CSS code; see Gottesman, section 4.3 of
[arXiv:0904.2557](https://arxiv.org/abs/0904.2557).
Use the builder to render the two-patch program with detector annotations.
The distance-3 program below runs two rounds on each side of CX, taking
36 ticks and making 82 measurements across the two patches.

```python
builder = LogicalCircuitBuilder()
builder.add_patch(patch, "D")
builder.add_patch(patch, "A", qubit_offset=patch.geometry.num_qubits)
builder.add_memory(["D", "A"], 2, "Z")
builder.add_transversal_cx("D", "A")
builder.add_memory(["D", "A"], 2, "Z")
tc = builder.to_tick_circuit()
assert tc.num_measurements() == 82
assert tc.num_ticks() == 36
```

```python
offset = patch.geometry.num_qubits
target = QubitAllocation(
    [q + offset for q in allocation.data_qubits],
    [q + offset for q in allocation.x_ancilla_qubits],
    [q + offset for q in allocation.z_ancilla_qubits],
)
gadget = gadgets.transversal_cx_gadget(patch, allocation, patch, target)
lines = render_gadget_function(gadget)
guard_excerpt("Transversal CX", lines)
assert len(gadget.steps) == 9
```

```text
@guppy
def transversal_cx(ctrl: SurfaceCode_3x3, tgt: SurfaceCode_3x3) -> None:
    """Apply physical transversal CX."""
    for i in range(9):
        cx(ctrl.data[i], tgt.data[i])
```

## Protocols in Guppy

`render_surface_protocol_module(patch)` returns source;
`load_surface_protocol_module(patch)` returns a cached dictionary of functions.
These four factories use full syndrome rounds without a separate initial
projection. Each example places the factory next to the corresponding builder
program. The two forms share operation order and measurement partition, not
physical scheduling or allocation: the builder schedules patches together with
dedicated ancillas, while the Guppy functions run patch by patch.
The optional `logical_x` (H) and `control_x` (CX) factory arguments insert a
logical X after preparation; the corresponding builder programs below use
their default `False`.

```hidden-python
from pecos.guppy_gen import load_surface_protocol_module, render_surface_protocol_module

module = load_surface_protocol_module(patch)


def new_builder(two_patches=False):
    builder = LogicalCircuitBuilder()
    builder.add_patch(patch, "D")
    if two_patches:
        builder.add_patch(patch, "A", qubit_offset=patch.geometry.num_qubits)
    return builder
```

### H experiment

```python
program = module["make_h_experiment"](num_rounds=2)
builder = new_builder()
builder.add_memory("D", 2, "Z")
builder.add_transversal_h("D")
builder.add_memory("D", 2, "X")
assert builder.to_tick_circuit().num_measurements() == 41
assert callable(program.compile)
```

```python
source = render_surface_protocol_module(patch)
guard_excerpt("H experiment", source.splitlines())
```

```text
def make_h_experiment(num_rounds: int, *, logical_x: bool = False):
    """Prepare Z, extract full rounds, apply H, extract swapped rounds, read X."""
    def h_experiment() -> None:
        """Logical H experiment."""
        a = prep_z_basis()
        if comptime(logical_x):
            apply_logical_x(a)
        for _ in range(comptime(num_rounds)):
            syn = syndrome_extraction_a(a)
            output("synx_a", syn.synx)
            output("synz_a", syn.synz)
        transversal_h(a)
        for _ in range(comptime(num_rounds)):
            syn = syndrome_extraction_swapped_a(a)
            output("synx_a", syn.synx)
            output("synz_a", syn.synz)
        final = measure_x_basis(a)
        output("final_a", final)
    return guppy(variant_scoped(h_experiment, num_rounds, logical_x))
```

### CX experiment

```python
program = module["make_transversal_cx"](num_rounds=2)
builder = new_builder(two_patches=True)
builder.add_memory(["D", "A"], 2, "Z")
builder.add_transversal_cx("D", "A")
builder.add_memory(["D", "A"], 2, "Z")
assert builder.to_tick_circuit().num_measurements() == 82
assert callable(program.compile)
```

### S teleportation

For background on gate teleportation in general, see Gottesman, section 4.5
of [arXiv:0904.2557](https://arxiv.org/abs/0904.2557).
Here the resource is a projected logical-Y state whose sign depends on the
projection outcomes. It is prepared by H followed by a physical S layer on
every ancilla data qubit, then syndrome projection. The implementation
performs no frame processing or conditional correction.

```python
program = module["make_sz_teleportation"](2, 2, 2)
builder = new_builder(two_patches=True)
builder.add_sz_via_teleportation("D", "A", 2, 2)
builder.add_memory("D", 2, "Z")
assert builder.to_tick_circuit().num_measurements() == 98
assert callable(program.compile)
```

### T injection stand-in

```python
program = module["make_t_injection"](2, 2)
builder = new_builder(two_patches=True)
builder.add_t_via_injection("D", "A", 2, 2)
assert builder.to_tick_circuit().num_measurements() == 82
assert callable(program.compile)
```

!!! note "Resource and correction limits"
    These factories generate uncorrected protocol experiments: the S factory
    uses a projected Y resource with no conditional correction, and the T
    factory is entirely Clifford (no T gate, no magic resource).
    The builder records injection readouts in the circuit metadata and in
    `build_algorithm_descriptor()`; those records do not apply a correction.
    The Guppy factories emit measurement outputs only.
    Chen, Chen, Lu and Pan ([arXiv:2412.01391](https://arxiv.org/abs/2412.01391))
    describe a mid-cycle fold-transversal logical S, which these gadgets do not
    implement.

### Cross-form measurement check

This checks actual measurement membership by patch, family, and round, including
swapped orientation and the S protocol's trailing syndrome rounds on the data
patch alone. The helpers compare measurement-ordinal groups from a Guppy trace
and from the builder's circuit.

<!--mark.slow-->
```python
from pecos.testing import (
    measurement_partition_from_builder,
    measurement_partition_from_trace,
    assert_same_measurement_partition,
)

for recipe in ("h", "cx", "sz", "t"):
    builder = new_builder(two_patches=recipe != "h")
    if recipe == "h":
        program = module["make_h_experiment"](2)
        builder.add_memory("D", 2, "Z")
        builder.add_transversal_h("D")
        builder.add_memory("D", 2, "X")
        scopes = {"a": "D"}
    elif recipe == "cx":
        program = module["make_transversal_cx"](2)
        builder.add_memory(["D", "A"], 2, "Z")
        builder.add_transversal_cx("D", "A")
        builder.add_memory(["D", "A"], 2, "Z")
        scopes = {"ctrl": "D", "tgt": "A"}
    elif recipe == "sz":
        program = module["make_sz_teleportation"](2, 2, 2)
        builder.add_sz_via_teleportation("D", "A", 2, 2)
        builder.add_memory("D", 2, "Z")
        scopes = {"data": "D", "anc": "A"}
    else:
        program = module["make_t_injection"](2, 2)
        builder.add_t_via_injection("D", "A", 2, 2)
        scopes = {"data": "D", "anc": "A"}
    expected = measurement_partition_from_builder(builder)
    actual = measurement_partition_from_trace(program, 17 if recipe == "h" else 26, scopes)
    assert_same_measurement_partition(actual, expected)
```

The automatic surface-memory DEM path expects the unscoped memory output
schema and derives no protocol boundary detectors from scoped outputs.
`pecos.qec.surface_memory_dem_spec` supplies detector and observable
specifications for `make_surface_code` memory experiments. Generic traced DEM
construction with `pecos.qec.build_dem_from_guppy` or
`pecos.qec.DetectorErrorModel.from_guppy`, given explicitly supplied detector
and observable specifications, is not restricted to one patch.
The builder's `to_stim`, `build_dem`, and `build_algorithm_descriptor` provide
the circuit-analysis route independently.

## Validating a gadget

Gadgets in this library are validated with four complementary checks; a
contribution should carry the same: a literature citation for the physical
construction, a stabilizer oracle for its state action, measurement-partition
agreement, and a DEM fault-distance test. The following counts the operations
of the default distance-3 extraction round; it does not validate the schedule.

```python
from pecos.qec.surface.circuit_builder import OpType

round_gadget = gadgets.syndrome_round_gadget(patch, allocation, round_index=0)
assert sum(step.op_type == OpType.CX for step in round_gadget.steps) == 24
assert sum(step.op_type == OpType.MEASURE for step in round_gadget.steps) == 8
```

For the stabilizer oracle, prepare Y and execute its first full syndrome round.
The product of the geometry's logical X and Z supports gives logical Y up to
phase. Check both signs because projection outcomes determine the encoded sign.

```python
from pecos.testing import stabilizer_generators_after, group_contains

builder = new_builder()
builder.add_memory("D", 1, "Y")
builder.add_memory("D", 1, "Z")
tc = builder.to_tick_circuit()
ancillas = set(allocation.x_ancilla_qubits + allocation.z_ancilla_qubits)
first_readout = next(
    i
    for i in range(tc.num_ticks())
    for gate in tc.get_tick(i).gate_batches()
    if gate.gate_type.name == "MZ" and ancillas.intersection(gate.qubits)
)
generators = stabilizer_generators_after(tc, first_readout + 1)
lx = set(patch.geometry.logical_x.data_qubits)
lz = set(patch.geometry.logical_z.data_qubits)
body = "".join("Y" if q in lx & lz else "X" if q in lx else "Z" if q in lz else "I" for q in range(17))
assert group_contains(generators, "+" + body) or group_contains(generators, "-" + body)
```

The executed [cross-form measurement check](#cross-form-measurement-check)
above checks measurement-partition agreement; equal counts alone would not
establish equal partitions. Measurement-partition agreement does not establish
circuit or state-action equivalence. Finally, the distance-3 Z memory has observable fault distance
3 under the following circuit noise model:

```python
from pecos.qec import DetectorErrorModel

builder = new_builder()
builder.add_memory("D", 3, "Z")
dem = DetectorErrorModel.from_circuit(builder.to_tick_circuit(), p1=0.001, p2=0.001, p_meas=0.001, p_prep=0.001)
distances = dem.per_observable_fault_distances(3)
assert len(distances) == 1
assert distances[0] is not None and distances[0].distance == 3
```

## What the detector layer guarantees

For valid `LogicalCircuitBuilder` programs, every emitted detector is
deterministic in noiseless execution. Boundary detectors come from backward
propagation of stabilizer terms through gates, earlier measurements, and
preparation. Standalone `TickCircuitRenderer` annotations follow the memory
template and inherit no such guarantee.

Propagation resolves each check against earlier measurements of the same type
on the same physical register, never against products of other checks or
jointly against a partner's same-round measurement. Some deterministic
parities are therefore not emitted. The concrete shapes are listed in the
docstring of `_propagate_stabilizer_terms` in
`pecos/qec/surface/logical_circuit.py`.

Generation rejects a gate before a patch's first preparation or after its
final readout. Registration rejects odd-weight checks because the propagation
model drops check-level signs only for even weights.

```python
from pecos.testing import simulate_tick_circuit

builder = new_builder()
builder.add_memory("D", 3, "Z")
_, fired, _ = simulate_tick_circuit(builder.to_tick_circuit(), seed=0)
assert fired == 0
for before_preparation in (True, False):
    invalid = new_builder()
    if before_preparation:
        invalid.add_transversal_h("D")
        invalid.add_memory("D", 1, "X")
    else:
        invalid.add_memory("D", 1, "Z")
        invalid.add_transversal_h("D")
    try:
        invalid.to_tick_circuit()
    except ValueError as error:
        assert "precedes" in str(error) if before_preparation else "after final" in str(error)
    else:
        raise AssertionError("Invalid gate placement was accepted")
```

## API reference

| Name | Module | Purpose |
|---|---|---|
| `SurfacePatch.create` | `pecos.qec.surface` | Construct square or rectangular geometry |
| `Gadget`, `GadgetKind` | `pecos.qec.surface.gadgets` | Physical definition and role |
| `default_allocation` | `pecos.qec.surface.gadgets` | Data and dedicated ancilla registers |
| `prep_gadget`, `init_syndrome_gadget` | `pecos.qec.surface.gadgets` | Product preparation and complementary projection |
| `syndrome_round_gadget`, `measure_out_gadget` | `pecos.qec.surface.gadgets` | Check extraction and destructive data readout |
| `logical_pauli_gadget`, `transversal_layer_gadget` | `pecos.qec.surface.gadgets` | Logical strings and physical layers |
| `transversal_cx_gadget`, `memory_gadgets` | `pecos.qec.surface.gadgets` | Two-patch CX and memory composition |
| `TickCircuitRenderer`, `QubitAllocation`, `SurfaceCircuitStep` | `pecos.qec.surface.circuit_builder` | Render physical operations with register mapping |
| `LogicalCircuitBuilder` | `pecos.qec.surface` | Compose protocols and export circuits, DEMs, and descriptors |
| `render_gadget_function`, `render_surface_gadget_module` | `pecos.guppy_gen.gadget_render` | Render one function or the memory module |
| `render_surface_protocol_module`, `load_surface_protocol_module` | `pecos.guppy_gen` | Render or load the four protocol factories |
| `simulate_tick_circuit`, `stabilizer_generators_after`, `group_contains` | `pecos.testing` | Noiseless simulation and signed stabilizer oracles |
| `measurement_partition_from_builder`, `measurement_partition_from_trace`, `assert_same_measurement_partition` | `pecos.testing` | Measurement-partition agreement, not circuit or state-action equivalence |
| `DetectorErrorModel.from_circuit` | `pecos.qec` | Circuit fault analysis and observable distances |

## Next steps

- [QEC with Guppy](qec-guppy.md) covers memory generation and simulation.
- [QEC Geometry](qec-geometry.md) explains patch construction.
- [Detector Error Models from Guppy](dem-from-guppy.md) covers the traced DEM workflow.
