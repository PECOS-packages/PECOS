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

Rotated square and rectangular patches are supported throughout the
single-patch routes. Fold-transversal S requires a square rotated patch
with distance at least 2.
The non-rotated layout renders standalone with a
serialised CX layer and is rejected by the builder. All tick counts on this
page are for the rotated layout.
Transversal H requires a square patch. CX requires identical static geometry,
disjoint allocations, and the same current X/Z orientation on both patches;
the builder checks orientation, while the standalone gadget does not.
The logical builder requires even-weight checks. The protocol module accepts
only odd square rotated patches with distance at least 3.

Y product preparation renders at any distance, but the teleportation helper
requires both ancilla dimensions to be odd; the [Y preparation example](#preparation-in-y)
explains the projected state and shows both odd- and even-distance preparation.

Run the setup below before the gadget examples. Default `TickCircuitRenderer`
detector annotations are a memory-experiment template, valid only for a full
memory circuit; use the builder's annotations for builder programs. Disable
the template when rendering a lone gadget.

<!--setup-->
```python
from pecos.qec.surface import SurfacePatch, LogicalCircuitBuilder, gadgets
from pecos.qec.surface.circuit_builder import TickCircuitRenderer, QubitAllocation
from pecos.guppy_gen.gadget_render import render_gadget_function, render_surface_gadget_module

patch = SurfacePatch.create(distance=3)
allocation = gadgets.default_allocation(patch)


def render_tick(gadget):
    return TickCircuitRenderer(add_detectors=False).render(
        list(gadget.steps), allocation, patch, 0, gadget.basis or "Z"
    )
```

The builder rejects the non-rotated layout during circuit generation:

```python
non_rotated = SurfacePatch.create(distance=3, rotated=False)
builder = LogicalCircuitBuilder()
builder.add_patch(non_rotated, "D")
builder.add_memory("D", 1, "Z")
try:
    builder.to_tick_circuit()
except ValueError as error:
    assert "repeated CX on qubit 3 in a parallel layer" in str(error)
else:
    raise AssertionError("Non-rotated builder circuit was accepted")
```

## The gadget library

`Gadget` is a frozen dataclass with `kind`, `name`, `steps`, `allocations`,
`dimensions`, `basis`, `x_z_swapped`, and `fold` (`"S"`, `"SDG"`, or `None`). Its steps are `SurfaceCircuitStep`
operations.
All single-patch functions below take `patch, allocation` first unless shown
otherwise. `round_order=None` uses the default schedule.

| Function and remaining parameters | `GadgetKind` | Effect | Guppy sideband tags |
|---|---|---|---|
| `prep_gadget(..., basis=)` | `PREP` | Product preparation in Z, X, or Y; projection follows separately | None |
| `init_syndrome_gadget(..., basis=, round_order=None, x_z_swapped=False, ancilla_budget=None, ancilla_schedule=None)` | `INIT_SYNDROME` | Project the complementary family after Z or X preparation | `<label>:init:meas:<ordinal>` |
| `syndrome_round_gadget(..., round_index=, round_order=None, x_z_swapped=False, ancilla_budget=None, ancilla_schedule=None)` | `SYNDROME_ROUND` | Measure both check families | `<label>:meas:<ordinal>` |
| `fold_s_round_gadget(..., round_index=, x_z_swapped=False, dagger=False)` | `SYNDROME_ROUND` | Logical S or S-dagger within the default round | `<label>:meas:<ordinal>` |
| `measure_out_gadget(..., basis=)` | `MEASURE_OUT` | Destructive Z or X data measurement | None; returns an array |
| `logical_pauli_gadget(..., pauli=)` | `LOGICAL_PAULI` | Apply the geometry's logical X or Z string | None |
| `transversal_layer_gadget(..., gate=)` | `TRANSVERSAL` | H exchanges X/Z orientation; SZ and SZDG are physical S layers | None |
| `transversal_cx_gadget(ctrl_patch, ctrl_allocation, tgt_patch, tgt_allocation)` | `TWO_PATCH` | CX between corresponding data indices | None |
| `memory_gadgets(patch, num_rounds, basis, allocation=None, round_order=None, ancilla_budget=None, ancilla_schedule=None)` | List of gadget kinds | Z/X prep, initial projection, full rounds, readout | Constituent gadget tags |

During generation the builder toggles its tracked orientation after a
transversal H; standalone callers track orientation themselves and pass it to
later rounds. The orientation belongs to the builder's patch state, not to
`SurfacePatch` itself.
`tag_scope="a"` prefixes tags with `a:`; a swapped gadget adds `swapped:` after
that scope. Labels identify physical register slots, while returned syndrome
arrays follow the current X/Z families. Ordinals start at zero within each
function invocation.
On square patches where the two families have unequal sizes, swapped rounds return
`Syndrome_<dx>x<dz>_swapped`; the memory module supplies both orientation types.

### Memory composition

For background on the memory cycle in the unrotated planar code, see Fowler,
Mariantoni, Martinis, and Cleland, [arXiv:1208.0928](https://arxiv.org/abs/1208.0928).
For background on rotated patches and four-layer CX schedules, see Tomita and
Svore, *Low-distance surface codes under realistic quantum noise*,
[arXiv:1404.3747](https://arxiv.org/abs/1404.3747), and Horsman, Fowler, Devitt
and Van Meter, *Surface code quantum computing by lattice surgery*,
[arXiv:1111.4022](https://arxiv.org/abs/1111.4022).
`memory_gadgets` returns gadgets and emits nothing. The memory factories
`make_memory_z` and `make_memory_x` in `render_surface_gadget_module` emit
scalar `final:meas:<index>` tags and aggregate `init_synx`/`init_synz`,
`synx`/`synz`, and `final` outputs in addition to the constituent gadget tags.
A distance-3 Z memory with three rounds takes 34 ticks and makes 37 measurements.

```python
parts = gadgets.memory_gadgets(patch, 3, "Z")
assert len(parts) == 6
steps = [step for gadget in parts for step in gadget.steps]
tc = TickCircuitRenderer().render(steps, allocation, patch, 3, "Z")
assert tc.num_measurements() == 37
assert tc.num_ticks() == 34
```

`ancilla_budget` caps live ancillas without changing data-qubit lifetimes.
`None` keeps dedicated ancillas. With a smaller budget, each batch allocates
pool slots, rotates its X ancillas, runs the four CX layers filtered to its
stabilizers, rotates back, measures, and ticks before the next allocation.
`ancilla_schedule` selects `"default"` or `"balanced-data-v1"` using the shared
batching contract. An explicit `QubitAllocation` must map stabilizers to pool
slots for the same budget and schedule; `default_allocation` accepts both.
The Guppy module derives its batching schedule from the CX `check_plan`;
the gadget API takes `ancilla_budget` and `ancilla_schedule` separately.
Returned syndrome arrays retain stabilizer-index order, while
scalar tag ordinals follow the physical measurement order.

```python
from pecos.guppy_gen import get_num_qubits

budgeted = gadgets.memory_gadgets(patch, 2, "Z", ancilla_budget=2)
budget_allocation = budgeted[0].allocations[0]
assert budget_allocation.data_qubits == allocation.data_qubits
budget_tc = TickCircuitRenderer().render(
    [step for part in budgeted for step in part.steps], budget_allocation, patch, 2, "Z"
)
assert budget_tc.num_ticks() == 94
assert budget_tc.num_measurements() == 29
live = set()
peak = 0
for tick_index in range(budget_tc.num_ticks()):
    for gate in budget_tc.get_tick(tick_index).gate_batches():
        if gate.gate_type.name == "QAlloc":
            assert not live.intersection(gate.qubits)
            live.update(gate.qubits)
            peak = max(peak, len(live))
        elif gate.gate_type.name == "MeasureFree":
            live.difference_update(gate.qubits)
assert peak == get_num_qubits(3, ancilla_budget=2) == 11
```

```python
source = render_surface_gadget_module(patch)
assert "def make_memory_z" in source
assert "def make_memory_x" in source
```

### Certified Guppy memory

`make_surface_memory` accepts a `SurfacePatch` and returns a compiled
gadget memory definition with a program-bound measurement-layout certificate.
Memory modules require nonempty X and Z stabilizer families because Guppy
cannot infer the type of the empty syndrome arrays emitted otherwise.
Its layout comes from the gadget steps, including batch order when ancillas are
reused. `build_dem_from_guppy`, `GuppyDemBuilder.build`, and
`DetectorErrorModel.from_guppy` need this certificate for programs whose compiled
form contains loops or conditionals. The abstract-circuit route and the builder
route (`LogicalCircuitBuilder.to_tick_circuit()` then
`DetectorErrorModel.from_circuit`) do not need a certificate.

```python
from pecos.guppy_gen import get_num_qubits, make_surface_code, make_surface_memory
from pecos.qec import DetectorErrorModel
from pecos.qec.surface import SurfacePatch
from pecos.qec.surface.circuit_builder import generate_tick_circuit_from_patch

memory_patch = SurfacePatch.create(distance=3)
memory_circuit = generate_tick_circuit_from_patch(memory_patch, 2, "Z", ancilla_budget=2)
dem_options = dict(
    num_qubits=get_num_qubits(patch=memory_patch, ancilla_budget=2),
    detectors_json=memory_circuit.get_meta("detectors"),
    observables_json=memory_circuit.get_meta("observables"),
    p1=0.001,
    p2=0.001,
    p_meas=0.001,
    p_prep=0.001,
)
gadget_memory = make_surface_memory(memory_patch, 2, "Z", ancilla_budget=2)
legacy_memory = make_surface_code(3, 2, "Z", ancilla_budget=2)
gadget_dem = DetectorErrorModel.from_guppy(gadget_memory, **dem_options)
legacy_dem = DetectorErrorModel.from_guppy(legacy_memory, **dem_options)
assert gadget_dem.to_string() == legacy_dem.to_string()
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
assert tc.get_meta("detectors") is None
```

```python
gadget = gadgets.prep_gadget(patch, allocation, basis="Z")
lines = render_gadget_function(gadget)
assert "    data = array(qubit() for _ in range(9))" in lines
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
assert "        h(data[i])" in lines
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
tc4 = TickCircuitRenderer(add_detectors=False).render(list(y4.steps), a4, p4, 0, "Y")
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
assert "        h(data[i])" in lines
assert "        s(data[i])" in lines
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
assert '    output("sx0:init:meas:0", sx0)' in lines
```

```text
...
@guppy
def init_z_basis(surf: SurfaceCode_3x3) -> array[bool, 4]:
    ...
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

`round_index` is zero-based. It labels the step list and `TickCircuit` round metadata only.
Guppy output tags carry no round index and repeat across invocations.
`round_order` selects the schedule. Set `x_z_swapped=True` after one transversal
H to reverse CX directions and exchange which ancillas receive H. At distance 3, a syndrome
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
assert '    output("swapped:sz0:meas:0", sz0)' in lines
assert "    synx = array(sz0, sz1, sz2, sz3)" in lines
```

```text
...
@guppy
def syndrome_extraction_swapped(surf: SurfaceCode_3x3) -> Syndrome_3x3:
    ...
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

### Fold-transversal S

`fold_s_round_gadget` inserts one disjoint gate layer after CX layer 2 of the
default syndrome round. Cartesian transpose `(x, y) -> (y, x)` pairs data and
bulk ancillas for CZ. Along the diagonal, data coordinates are odd and bulk
ancilla coordinates are even, so S on data and S-dagger on ancillas alternate.
Exterior ancillas are untouched. Square rotated patches with distance at least 2
are supported, including even distances. Set `dagger=True` to reverse every fixed-point phase, or
`x_z_swapped=True` for the current orientation after transversal H.

The exact round flow is `X_L -> +Y_L * product(current Z checks)` and
`Z_L -> Z_L`, where `+Y_L = i X_L Z_L`. This uses SparseStab's `Y = iXZ`
convention and PECOS `SZ = diag(1, i)`. For an X-prepared patch, the output
logical Y sign is the parity of this round's Z outcomes: even gives +Y and odd
gives -Y. The dagger variant reverses that sign.

X records also carry Z-check information. On input, a bottom-row bulk X
ancilla measures its X check times the left-boundary Z check at `(0, x_j)`; other
X records measure bare checks. On output, X record j together with the Z
record at `(y_j + 2, x_j)` certifies X check j (coordinates are `(x, y)`). Without that partner, the X
record alone certifies the check. These coordinates use the current frame,
transposed after transversal H.

Under circuit noise the X-sector fault distance is reduced because a Y fault
before the fold becomes a Z pair on a mirror pair. In their benchmark with
separated S rounds, Chen, Chen, Lu and Pan observe two to three times the
memory's logical error rate; see [arXiv:2412.01391](https://arxiv.org/abs/2412.01391).
For the half-cycle description see McEwen, Bacon and Gidney,
[arXiv:2302.02192](https://arxiv.org/abs/2302.02192).

The fold has `d(d-1)/2` data CZ pairs and `(d-1)(d-2)/2` bulk-ancilla CZ
pairs, totaling `(d-1)^2`, plus d data and d-1 ancilla fixed points. At
distance 3 the complete round takes nine ticks and makes eight measurements,
one tick more than the default round. The Tick and Stim renderers refuse
detector annotation for this standalone gadget. Use `LogicalCircuitBuilder.add_logical_s`
for circuits with fold-aware detectors and observables.

```python
gadget = gadgets.fold_s_round_gadget(patch, allocation, round_index=0)
tc = render_tick(gadget)
assert tc.num_ticks() == 9
assert tc.num_measurements() == 8
assert {gate.gate_type.name for gate in tc.get_tick(4).gate_batches()} == {"CZ", "SZ", "SZdg"}
assert tc.get_meta("detectors") is None

from pecos.qec.surface.circuit_builder import OpType

d = patch.dx
fold_ops = [step for step in gadget.steps if step.op_type in {OpType.CZ, OpType.SZ, OpType.SZDG}]
data = set(allocation.data_qubits)
cz_pairs = [step.qubits for step in fold_ops if step.op_type == OpType.CZ]
fixed_points = [step.qubits[0] for step in fold_ops if step.op_type != OpType.CZ]
assert sum(a in data for a, b in cz_pairs) == d * (d - 1) // 2
assert sum(a not in data for a, b in cz_pairs) == (d - 1) * (d - 2) // 2
assert len(cz_pairs) == (d - 1) ** 2
assert sum(q in data for q in fixed_points) == d
assert sum(q not in data for q in fixed_points) == d - 1
```

The builder models the fold as one syndrome segment. Preparation and final
readout are explicit memories; either can request zero plain rounds. X-check
detectors use the fold round's input and output Z partners. A logical Z readout
passes through unchanged. An X readout crossing one fold is random; an
S then S-dagger restores a deterministic X observable whose parity includes
both fold rounds' Z records. This observable is a frame parity; the descriptor's
`SGate` boundary gate carries the logical frame update. The adjacent X-prepared
pair has fault distance 2 at d=3, while Z memory retains distance 3.

A zero-round final memory reads data immediately after the previous segment.
Its final checks therefore compare with that segment's last syndrome round,
including across H. This also fills a detector gap in programs without folds:
`M(2,Z); M(0,Z)` and `M(2,Z); H; M(0,X)` each have 16 detectors at d=3.

```python
import json
from pecos.qec.surface import LogicalCircuitBuilder, SurfacePatch

for with_h in (False, True):
    memory = LogicalCircuitBuilder()
    memory.add_patch(SurfacePatch.create(distance=3), "A")
    memory.add_memory("A", 2, "Z")
    if with_h:
        memory.add_transversal_h("A")
    memory.add_memory("A", 0, "X" if with_h else "Z")
    assert len(json.loads(memory.to_tick_circuit().get_meta("detectors"))) == 16
```

Fold rounds create hyperedges that `build_decoder`'s matching route
(`LogicalSubgraphDecoder`) skips. Use a hypergraph decoder such as the
[Tesseract route](decoders.md) for fold circuits.

```python
import json
from pecos.qec import DetectorErrorModel
from pecos.qec.surface import LogicalCircuitBuilder, SurfacePatch
from pecos.testing import simulate_tick_circuit

fold_builder = LogicalCircuitBuilder()
fold_builder.add_patch(SurfacePatch.create(distance=3), "D")
fold_builder.add_memory("D", 1, "Z")
fold_builder.add_logical_s("D")
fold_builder.add_memory("D", 1, "Z")
fold_tc = fold_builder.to_tick_circuit()
assert len(json.loads(fold_tc.get_meta("detectors"))) == 24
assert len(json.loads(fold_tc.get_meta("observables"))) == 1
for seed in range(8):
    _, fired, observables = simulate_tick_circuit(fold_tc, seed=seed)
    assert fired == 0
    assert observables == {0: 0}
fold_dem = DetectorErrorModel.from_circuit(fold_tc, p1=0.001, p2=0.001, p_meas=0.001, p_prep=0.001)
assert fold_dem.per_observable_fault_distances(3)[0].distance == 3
```

Use `add_logical_s("D", dagger=True)` or `add_logical_sdg("D")` for
S-dagger. Both variants emit `SGate` in the algorithm descriptor because it
tracks sign-free Pauli frames: both propagate an X frame bit into X and Z.

Observables are defined relative to the noiseless reference, as in Stim.
Raw parity encodes the sign with which the program's net logical Clifford maps
the readout Pauli back onto the prepared eigenstate: positive gives 0, negative
gives 1. With folds it can be 1 noiselessly: S,S before an X readout, S,S,H
before a Z readout, or S-dagger pairs through a CX. Stim's reference-relative
sampling reports zero observable flips for these noiseless programs.
Observable metadata has no sign field, so both raw-parity consumers,
`pecos.testing.simulate_tick_circuit` and
`pecos.qec.surface.extract_detection_events_and_observables`, must account for
this reference. S then S-dagger before X readout has raw parity 0. A readout with no
supported logical image, as under a physical S layer, produces no observable
at all.

```python
import stim
from pecos.qec import DetectorErrorModel
from pecos.testing import simulate_tick_circuit
from pecos.qec.surface.circuit_builder import tick_circuit_to_stim

for dagger, expected_raw in ((False, 1), (True, 0)):
    pair = LogicalCircuitBuilder()
    pair.add_patch(SurfacePatch.create(distance=3), "D")
    pair.add_memory("D", 1, "X")
    pair.add_logical_s("D")
    pair.add_logical_s("D", dagger=dagger)
    pair.add_memory("D", 1, "X")
    tc_pair = pair.to_tick_circuit()
    for seed in range(8):
        _, fired, raw = simulate_tick_circuit(tc_pair, seed=seed)
        assert fired == 0
        assert raw == {0: expected_raw}
    circuit = stim.Circuit(tick_circuit_to_stim(tc_pair))
    detectors, flips = circuit.compile_detector_sampler(seed=0).sample(256, separate_observables=True)
    assert not detectors.any()
    assert not flips.any()
    dem_pair = DetectorErrorModel.from_circuit(tc_pair, p1=0.001, p2=0.001, p_meas=0.001, p_prep=0.001)
    assert dem_pair.per_observable_fault_distances(3)[0].distance == 2
```

Render the Guppy function independently; the memory module does not include
it. When assembling a Guppy module, also import `cz`, `s`, and `sdg` from
`guppylang.std.quantum`. The protocol module provides
`make_logical_s_experiment(rounds_before, rounds_after, dagger=False)`.

```python
gadget = gadgets.fold_s_round_gadget(patch, allocation, round_index=0)
lines = render_gadget_function(gadget)
assert "    cz(az1, az2)" in lines
assert "    sdg(ax1)" in lines
assert "    s(surf.data[2])" in lines
assert "    # fold-transversal S layer" in lines
assert "syndrome_extraction_fold_s" not in render_surface_gadget_module(patch)
```

```text
@guppy
def syndrome_extraction_fold_s(surf: SurfaceCode_3x3) -> Syndrome_3x3:
    """Extract full syndrome with the fold-transversal logical S between CX layers 2 and 3."""
    ...
    # fold-transversal S layer
    s(surf.data[6])
    cz(surf.data[3], surf.data[7])
    cz(surf.data[0], surf.data[8])
    sdg(ax2)
    cz(az1, az2)
    s(surf.data[4])
    cz(surf.data[1], surf.data[5])
    sdg(ax1)
    s(surf.data[2])
    ...
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
assert "    return collect_measurements(measure_array(surf.data))" in lines
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

`pauli="X"` or `pauli="Z"` applies the geometry's string regardless of
orientation, without measurements or allocation. Apply it in the unswapped
orientation: after a transversal H, that string is not a logical operator of
the swapped patch. The protocol factories apply it directly after preparation.
At distance 3, either string applies three Pauli gates in one tick, with no measurements.

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
assert "    x(surf.data[0])" in lines
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
assert "        h(surf.data[i])" in lines
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
for gate in ("SZ", "SZDG"):
    gadget = gadgets.transversal_layer_gadget(patch, allocation, gate=gate)
    rendered = render_gadget_function(gadget)
    expected = "s" if gate == "SZ" else "sdg"
    assert f"        {expected}(surf.data[i])" in rendered
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
Corresponding data qubits undergo control-to-target CX. Transversal CNOT is logical CNOT between two blocks of the same CSS code; see Gottesman, section 4.3 of
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
assert "        cx(ctrl.data[i], tgt.data[i])" in lines
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
`load_surface_protocol_module(patch)` returns the loaded module's namespace and
caches per patch geometry. Scoped tags support measurement-provenance checks.
These five factories (H, CX, logical S/S-dagger, SZ teleportation, and T
injection) use full syndrome rounds without a separate initial
projection. Each example places the factory next to the corresponding builder
program. The two forms share operation order and measurement partition, not
physical scheduling or allocation: the builder schedules patches together with
dedicated ancillas, while the Guppy functions run patch by patch.
The optional `logical_x` (H) and `control_x` (CX) factory arguments insert a
logical X after preparation; the factory calls below use the default `False`.

<!--setup-->
```python
from pecos.qec.surface import SurfacePatch, LogicalCircuitBuilder, gadgets
from pecos.guppy_gen import load_surface_protocol_module, render_surface_protocol_module

patch = SurfacePatch.create(distance=3)
allocation = gadgets.default_allocation(patch)
module = load_surface_protocol_module(patch)


def new_builder(two_patches=False):
    builder = LogicalCircuitBuilder()
    builder.add_patch(patch, "D")
    if two_patches:
        builder.add_patch(patch, "A", qubit_offset=patch.geometry.num_qubits)
    return builder
```

### Logical S experiment

The factory prepares Z, runs the requested plain rounds around one fold
round, and reads Z. Both variants use scoped syndrome tags on `a`.
The fold round is included in `synx_a`, but its X records are not bare X
syndromes: bottom-row records carry input-side Z-check information. Consumers
must combine X records with the partner Z records specified by
`fold_s_round_gadget` to recover the corresponding check parity.

```python
from pecos.testing import (
    assert_same_measurement_partition,
    measurement_partition_from_builder,
    measurement_partition_from_trace,
)

for dagger in (False, True):
    program = module["make_logical_s_experiment"](1, 1, dagger=dagger)
    builder = new_builder()
    builder.add_memory("D", 1, "Z")
    builder.add_logical_s("D", dagger=dagger)
    builder.add_memory("D", 1, "Z")
    assert program.compile() is not None
    assert builder.to_tick_circuit().num_measurements() == 33
    assert_same_measurement_partition(
        measurement_partition_from_trace(program, patch.geometry.num_qubits, {"a": "D"}),
        measurement_partition_from_builder(builder),
    )
```

The single-patch peak is `patch.geometry.num_qubits`. The two-patch Guppy
peak is `patch.geometry.num_data + patch.geometry.num_qubits`, not twice the
single-patch count: the factories run patches one at a time and reuse ancilla
registers while retaining both data arrays.

<!--mark.slow-->
```python
from pecos import selene_engine, sim, stabilizer
from pecos.guppy_gen import get_transversal_num_qubits

num_qubits = patch.geometry.num_data + patch.geometry.num_qubits
assert num_qubits == get_transversal_num_qubits("surface", 3)
program = module["make_transversal_cx"](num_rounds=2)
results = sim(program).classical(selene_engine()).quantum(stabilizer()).qubits(num_qubits).seed(42).run(10)
columns = results.to_dict()
assert len(columns["final_ctrl"]) == len(columns["final_tgt"]) == 10
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
assert "            syn = syndrome_extraction_swapped_a(a)" in source.splitlines()
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
syndrome projection outcomes. It is prepared by H followed by a physical S
layer on every ancilla data qubit, then syndrome projection. Both forms prepare
the data in |0_L>, an S eigenstate, so this experiment cannot distinguish S
from identity. The builder records the readout the correction depends on
(the ancilla's final logical-Z bits, in `injection_readouts` in the circuit
metadata and in `build_algorithm_descriptor()`) and nothing applies the
correction: with parity 1 (for the +Y resource sign) a logical Z on the data
patch would be required.
Neither the resource sign nor this correction is processed by any decoder.

```python
program = module["make_sz_teleportation"](2, 2, 2)
builder = new_builder(two_patches=True)
builder.add_sz_via_teleportation("D", "A", 2, 2)
builder.add_memory("D", 2, "Z")
assert builder.to_tick_circuit().num_measurements() == 98
assert callable(program.compile)

import json

tc = builder.to_tick_circuit()
readout = json.loads(tc.get_meta("injection_readouts"))[0]
assert (readout["ancilla_patch"], readout["data_patch"], readout["basis"]) == ("A", "D", "Z")
assert len(readout["meas_ids"]) == len(patch.geometry.logical_z.data_qubits)
# Measurement IDs index the readout stream; map them to the ancilla's data register.
measurement_keys = json.loads(tc.get_meta("measurement_keys"))
ancilla_data_ids = {
    meas_id
    for label, qubit, meas_id in measurement_keys["data"]
    if label == readout["ancilla_patch"] and qubit in allocation.data_qubits
}
assert set(readout["meas_ids"]) <= ancilla_data_ids
descriptor = builder.build_algorithm_descriptor()
assert descriptor["injection_readouts"][0]["meas_ids"] == readout["meas_ids"]
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
    describe a mid-cycle fold-transversal logical S, available separately as
    the [fold round gadget](#fold-transversal-s), outside these protocol factories.

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
    num_qubits = patch.geometry.num_qubits
    if recipe != "h":
        num_qubits += patch.geometry.num_data
    actual = measurement_partition_from_trace(program, num_qubits, scopes)
    assert_same_measurement_partition(actual, expected)
```

The Guppy protocol factories on this page cannot be traced into a DEM:
they contain `comptime` loops and carry no trusted measurement-layout
certificate. Their scoped tags serve `measurement_partition_from_trace` only,
not DEM construction. `make_surface_code` and `make_surface_memory` programs
have a generator certificate; these protocol factories do not. Use
`LogicalCircuitBuilder.to_tick_circuit()` with `DetectorErrorModel.from_circuit`,
or the builder's `build_dem`, for protocol DEMs.

```python
from pecos.qec import DetectorErrorModel, build_dem_from_guppy

for factory, arguments in (
    ("make_h_experiment", (2,)),
    ("make_transversal_cx", (2,)),
    ("make_sz_teleportation", (2, 2, 2)),
    ("make_t_injection", (2, 2)),
):
    program = module[factory](*arguments)
    num_qubits = patch.geometry.num_qubits
    if factory != "make_h_experiment":
        num_qubits += patch.geometry.num_data
    for build, specs in (
        (build_dem_from_guppy, {"detectors": [], "observables": []}),
        (DetectorErrorModel.from_guppy, {"detectors_json": "[]", "observables_json": "[]"}),
    ):
        try:
            build(program, num_qubits=num_qubits, **specs)
        except ValueError as error:
            assert str(error).startswith(
                "GuppyDemBuilder requires a statically straight-line Guppy program "
                "unless it carries a trusted generator-owned measurement layout"
            )
        else:
            raise AssertionError(f"{factory} unexpectedly accepted for a traced DEM")
```

## Validating a gadget

Four complementary checks validate a gadget: a literature citation for the
physical construction, a stabilizer oracle for its state action, measurement-partition
agreement, and a DEM fault-distance test. The following counts the operations
of the default distance-3 extraction round; it does not validate the schedule.

```python
from pecos.qec.surface.circuit_builder import OpType

round_gadget = gadgets.syndrome_round_gadget(patch, allocation, round_index=0)
assert sum(step.op_type == OpType.CX for step in round_gadget.steps) == 24
assert sum(step.op_type == OpType.MEASURE for step in round_gadget.steps) == 8
```

A Y memory segment cannot be a patch's final segment: generation raises
`NotImplementedError: Y readout is unsupported`. The oracle example follows Y
with a Z segment for that reason, then stops after the first full syndrome round.
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
body = "".join(
    "Y" if q in lx & lz else "X" if q in lx else "Z" if q in lz else "I" for q in range(patch.geometry.num_qubits)
)
assert group_contains(generators, "+" + body) or group_contains(generators, "-" + body)
```

The executed [cross-form measurement check](#cross-form-measurement-check)
above checks measurement-partition agreement; equal counts alone would not
establish equal partitions. Measurement-partition agreement does not establish
circuit or state-action equivalence.

Finally, the distance-3 Z memory has observable fault distance 3 under the
following circuit noise model:

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
deterministic: a stabilizer term that cannot be resolved to an earlier
same-type measurement or a preparation yields no detector, so the model's
known limitations lose detectors rather than emit non-deterministic ones.
Boundary detectors come from backward
propagation of stabilizer terms through gates, earlier measurements, and
preparation. Standalone `TickCircuitRenderer` annotations follow the memory
template and inherit no such guarantee.

Propagation resolves each check against earlier measurements of the same type
on the same physical register, never against products of other checks or
jointly against a partner's same-round measurement. Some deterministic
parities are therefore not emitted. The propagation walk in
`pecos.qec.surface.logical_circuit` documents the concrete shapes.

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
| `default_allocation` | `pecos.qec.surface.gadgets` | Data and dedicated or budgeted ancilla registers |
| `prep_gadget`, `init_syndrome_gadget` | `pecos.qec.surface.gadgets` | Product preparation and complementary projection |
| `syndrome_round_gadget`, `measure_out_gadget` | `pecos.qec.surface.gadgets` | Check extraction and destructive data readout |
| `LogicalCircuitBuilder.add_logical_s`, `add_logical_sdg` | `pecos.qec.surface` | Fold syndrome segments with detectors and logical parity records |
| `make_logical_s_experiment` | Loaded surface protocol namespace | Z memory with one fold S or S-dagger round |
| `deterministic_parity_basis` | `pecos.testing` | Basis of parities constant across noiseless measurement shots |
| `fold_s_round_gadget` | `pecos.qec.surface.gadgets` | Fold-transversal logical S or S-dagger inside a default round |
| `logical_pauli_gadget`, `transversal_layer_gadget` | `pecos.qec.surface.gadgets` | Logical strings and physical layers |
| `transversal_cx_gadget`, `memory_gadgets` | `pecos.qec.surface.gadgets` | Two-patch CX and memory composition |
| `TickCircuitRenderer`, `QubitAllocation`, `SurfaceCircuitStep` | `pecos.qec.surface.circuit_builder` | Render physical operations with register mapping |
| `LogicalCircuitBuilder` | `pecos.qec.surface` | Compose protocols and export circuits, DEMs, and descriptors |
| `render_gadget_function`, `render_surface_gadget_module` | `pecos.guppy_gen.gadget_render` | Render one function or the memory module |
| `make_surface_memory` | `pecos.guppy_gen` | Compile certified single-patch gadget memory for Guppy DEM construction |
| `render_surface_protocol_module`, `load_surface_protocol_module` | `pecos.guppy_gen` | Render or load the four protocol factories |
| `simulate_tick_circuit`, `stabilizer_generators_after`, `group_contains` | `pecos.testing` | Noiseless simulation and signed stabilizer oracles |
| `measurement_partition_from_builder`, `measurement_partition_from_trace`, `assert_same_measurement_partition` | `pecos.testing` | Measurement-partition agreement, not circuit or state-action equivalence |
| `DetectorErrorModel.from_circuit` | `pecos.qec` | Circuit fault analysis and observable distances |

## Next steps

- [QEC with Guppy](qec-guppy.md) covers memory generation and simulation.
- [QEC Geometry](qec-geometry.md) explains patch construction.
- [Detector Error Models from Guppy](dem-from-guppy.md) covers the traced DEM workflow.
