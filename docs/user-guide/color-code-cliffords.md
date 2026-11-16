# Six-state memories and logical Clifford gadgets

The triangular 6.6.6 constructions support all six encoded Pauli eigenstates
and static circuits of transversal H, S, S†, X, Z, and CX. These implement
circuit ingredients from [Lavasani et al., arXiv:2610.24077v1](https://arxiv.org/abs/2610.24077v1),
including Bell-flag preparation, Clifford rotations and SWAP-superdense cycles.
They use the [named gadget interface](named-qec-gadgets.md) throughout.

## Run a six-state memory

```python
from pecos import sim, stabilizer
from pecos.qec.color import (
    ColorExtraction,
    LogicalState,
    color_code_spec,
    color_code_implementation,
    color_state_memory_program,
)
from pecos.qec.generic import SchedulePolicy

implementation = color_code_implementation(
    color_code_spec("666", 5),
    ColorExtraction.SWAP_SUPERDENSE,
)
experiment = color_state_memory_program(
    implementation,
    rounds=4,
    state=LogicalState.MINUS_I,
    start_phase=2,
)
print(experiment.describe())
plan = experiment.compile(SchedulePolicy.REFERENCE)
columns = sim(plan.guppy_program()).qubits(plan.num_qubits).quantum(stabilizer()).seed(42).run(32).to_dict()
for shot in range(32):
    detectors, observables = plan.evaluate_named({tag: values[shot] for tag, values in columns.items()})
    assert not any(detectors)
    assert not any(observables)
```

| State | String argument | Enum |
| --- | --- | --- |
| Logical zero | `"+Z"` | `LogicalState.ZERO` |
| Logical one | `"-Z"` | `LogicalState.ONE` |
| Logical plus | `"+X"` | `LogicalState.PLUS` |
| Logical minus | `"-X"` | `LogicalState.MINUS` |
| Logical +i | `"+Y"` | `LogicalState.PLUS_I` |
| Logical −i | `"-Y"` | `LogicalState.MINUS_I` |

Every state starts with Bell-flag projection of logical zero, followed by its
preparation Clifford. The final circuit undoes that Clifford before Z readout,
so its observable is a **failure parity**, expected to be zero for all six
states. `color_state_memory(...)` directly returns the compiled plan. Both
superdense and SWAP-superdense extraction are supported.

These are separate recipes from `color_memory(..., basis="X"/"Z")`, whose
original physical circuits and reference fixtures remain unchanged.

## Compose multiple logical blocks

<!--continuation-->
```python
from pecos.qec.color import ColorCliffordExperiment

experiment = ColorCliffordExperiment(implementation, blocks=("alice", "bob"))
experiment.prepare("alice", "+X", start_phase=2)
experiment.prepare("bob", "+Z", start_phase=1)
experiment.extract("alice", rounds=2)
experiment.extract("bob", rounds=3)
experiment.gate("CX", "alice", "bob")
experiment.extract("alice")
experiment.extract("bob")

print(experiment.frame("alice"))
print(experiment.next_phase("bob"))
a = experiment.readout("alice", basis="Z")
b = experiment.readout("bob", basis="Z")
experiment.observable("Bell ZZ", a ^ b)
plan = experiment.compile()
```

This prepares an encoded Bell state and checks its ZZ correlation. Readout in
X on both blocks instead checks XX. All blocks use the same canonical code and
extraction implementation. Each has its own next phase and reference carrier
history. Block names reserve `:` and `/` for generated scope identifiers.
A block is prepared once and cannot be used after readout.

`gate("H", block)`, `gate("S", block)`, `gate("S_DAG", block)`, `gate("X", block)`
and `gate("Z", block)` apply single-block operations. `gate("CX", control, target)`
requires two distinct live blocks. The builder updates detector references and
logical sign corrections as each operation is appended. It does not infer
which ideal logical correlations you want to measure; declare those with
`observable(...)`.

`readout` returns the **logical measurement value**, including the measurement
frame correction. A direct readout of the prepared negative eigenstate returns
one. This differs from the normalized failure parity returned by the six-state
memory recipe. XORing a constant is supported in Python parity evaluation;
the current Stim/DEM exporters explicitly reject affine constants. For memory
DEMs, the inverse-preparation recipe avoids those constants.

## Inspect and reuse the logical gadgets

<!--continuation-->
```python
from pecos.qec.color import logical_clifford_gadget

print(logical_clifford_gadget(implementation.code, "S").describe())
cx = logical_clifford_gadget(implementation.code, "CX")
```

These return ordinary `Gadget` objects. Single-block ports are `data[i]`; CX
ports are `control[i]` and `target[i]`. A lower-level `GadgetProgram` can bind
them directly, but then its author must compose the surrounding detector/frame
relations. `ColorCliffordExperiment` handles those relations for the supported
canonical 6.6.6 constructions.

Logical S uses S on sublattice A and S† on B. Each face has balanced A/B support,
while the whole triangular patch has one extra A site. This implements the
signed logical action X→Y, Y→−X, Z→Z while preserving the stabilizer group.
The inverse gadget reverses the two physical phases.

The logical Y convention is `Y_L = i X_L Z_L`. For all-data X/Z representatives,
this can differ in sign from the product of all physical Y operators. The
readout rotations and tests use the logical convention explicitly.

`ColorFrame.check_x/check_z` hold expected named stabilizer parities.
`x_sign/z_sign` hold logical sign deviations from the intended ideal state;
they are not state eigenvalues. H exchanges the X/Z references, S and S† mix
them, and CX propagates them across blocks. Y deviations combine both signs.
The gadget's physical Hadamard boundary convention remains distinct from these
measurement-dependent Pauli signs.

## Verification and scope

Independent Stim conjugation checks verify exact signed logical Pauli images
and stabilizer images for H/S/S†/X/Z at distances 3/5/7/9, plus the full two-block
CX Pauli action at 3/5/7. Tests check each prepared state's eigenvalues before
any inverse rotation, and compare interleaved logical circuits with an ideal
two-qubit simulator across all logical Pauli products and random syndrome branches.
Six-state memories cover distances 3/5/7, both extraction implementations, all
starting phases, and both schedule policies. Actual Guppy checks include all
six states, distance-five Y byproducts, signed readout, two-block correlations,
and an audited noisy DEM.

These circuits remain static and use the functional Guppy timing contract.
They do not implement randomized benchmarking sampling/fitting, lattice surgery,
magic-state injection, hardware leakage/repump models, or benchmark error rates.
The 4.8.8 family is explicitly rejected by these new logical-gadget factories.

Run the complete example with:

```sh
uv run python examples/python_examples/color_clifford_memories.py --distance 3 --output /tmp/color-cliffords
```
