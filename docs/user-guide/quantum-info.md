# Quantum Information Primitives

PECOS exposes Rust-backed channel representations and measure functions for
workflows that connect process characterization to QEC simulation. These APIs
live at `pecos.quantum_info` in Python and in the `pecos-quantum` crate in Rust.

Use these types when you need exact channel data, validation, or process
metrics before reducing a model to Pauli rates, a detector error model, or a
fault catalog.

## Channel Representations

PECOS currently provides seven concrete channel representations:

| Type | Purpose |
| ---- | ------- |
| `PauliChannel` | Sparse Pauli error probabilities. |
| `Ptm` | Dense Pauli transfer matrix. |
| `KrausOps` | Kraus-operator representation. |
| `ChoiMatrix` | Choi-matrix representation for channel validation and tomography. |
| `SuperOp` | Dense column-stacked superoperator. |
| `ChiMatrix` | Process matrix in the Pauli basis. |
| `Stinespring` | Stinespring isometry. |

Use the representation that matches the question you are asking:

| Goal | Start with |
| ---- | ---------- |
| Small pure-state examples | State vectors |
| Noisy states and entanglement measures | Density matrices |
| Operational noise construction | `KrausOps` |
| Complete positivity, trace preservation, and tomography reconstruction | `ChoiMatrix` |
| Pauli-basis channel diagnostics | `Ptm` |
| Sparse stochastic Pauli noise | `PauliChannel` |
| Environment/isometry models | `Stinespring` |

PECOS uses these conventions consistently:

| Convention | Meaning |
| ---------- | ------- |
| Qubit order | Little-endian: qubit 0 is the least-significant computational-basis bit. |
| Dense Pauli labels | Highest-numbered qubit first, so `IX` on two qubits means I on qubit 1 and X on qubit 0. |
| Sparse Pauli strings | Constructor and sparse text forms use explicit qubit IDs, e.g. `X(0) & Z(3)` or `"X0 Z3"`. |
| PTM basis order | Dense Pauli labels in PECOS basis-label order. |
| Superoperator order | Column-stacked operator vectorization. |
| Choi matrix | Built from PECOS's column-stacked superoperator convention. A trace-preserving channel has output partial trace equal to identity. |
| Subsystem order | Subsystem 0 is the fastest-varying tensor factor. Qubit helpers follow the same little-endian rule. |

```python
from pecos.quantum_info import PauliChannel, process_fidelity

channel = PauliChannel.one_qubit(px=0.001, py=0.0005, pz=0.002)
print(channel.probabilities())
print(channel.total_error_rate())

ptm = channel.to_ptm()
identity = type(ptm).identity(1)
print(process_fidelity(ptm, identity))
```

For Pauli channels, PECOS also provides exact dependency-free diamond norm and
diamond distance helpers:

```python
from pecos.quantum_info import (
    PauliChannel,
    pauli_channel_diamond_distance,
    pauli_channel_diamond_norm,
)

left = PauliChannel.one_qubit(0.001, 0.0, 0.0)
right = PauliChannel.one_qubit(0.0, 0.0, 0.001)

print(pauli_channel_diamond_norm(left, right))
print(pauli_channel_diamond_distance(left, right))
```

Arbitrary-channel diamond norm is intentionally not exposed yet. General
channels require a semidefinite program, and PECOS will only expose that API
once the solver and SDP assembly live in Rust with PECOS-owned validation. The
current Rust groundwork covers the exact Pauli-channel formula plus internal
linear-algebra helpers for future SDP assembly.

For multi-qubit Pauli channels, pass a label-to-probability map:

```python
from pecos.quantum_info import PauliChannel

from_labels = PauliChannel.from_probabilities(
    2,
    {
        "II": 0.98,
        "IX": 0.01,
        "ZI": 0.01,
    },
)
```

You can also use `PauliString` keys when the channel is written in PECOS's typed
Pauli style:

```python
from pecos.quantum import X, Z
from pecos_rslib import PauliString
from pecos.quantum_info import PauliChannel

from_paulis = PauliChannel.from_probabilities(
    2,
    {
        PauliString.I(): 0.98,
        X(0): 0.01,
        Z(1): 0.01,
    },
)

assert from_paulis.probabilities() == {
    "II": 0.98,
    "IX": 0.01,
    "ZI": 0.01,
}
```

## Choi Validation

`ChoiMatrix` exposes checks that are useful when importing reconstructed
processes from tomography or generated channels from another model:

```python
from pecos.quantum_info import Ptm

choi = Ptm.identity(1).to_choi()
assert choi.is_completely_positive()
assert choi.is_trace_preserving()
assert choi.is_cptp()
assert choi.is_unital()

trace_check = choi.partial_trace_output()
```

For PECOS's Choi convention, a trace-preserving channel satisfies
`partial_trace_output() == I`.

## Process Tomography Helpers

`ProcessTomographyDesign.matrix_unit(n)` gives the complete computational
matrix-unit operator basis used by PECOS Choi reconstruction. This is a
linear-inversion design for exact channel characterization and simulator
validation; it is not a physical state-preparation recipe.

```python
from pecos.quantum_info import ProcessTomographyDesign, Ptm

design = ProcessTomographyDesign.matrix_unit(1)
assert design.input_metadata_all() == [
    (0, 0, 0),  # |0><0|
    (1, 1, 0),  # |1><0|
    (2, 0, 1),  # |0><1|
    (3, 1, 1),  # |1><1|
]

choi = Ptm.identity(1).to_choi()
outputs = design.simulate_outputs(choi)
reconstructed = design.reconstruct_choi(outputs)
assert reconstructed.matrix() == choi.matrix()
```

Physical process tomography is a separate layer. A physical workflow prepares
experimentally realizable input states, measures in chosen bases, aggregates
shot counts, and reconstructs an estimated channel before converting it into
`Ptm`, `ChoiMatrix`, or another channel representation. The current
`ProcessTomographyDesign` is the Rust-backed exact reconstruction primitive
that those higher-level experiment-design helpers should build on.

The dense channel forms convert through the same Rust-backed validation path:

```python
from pecos.quantum_info import Ptm

ptm = Ptm.identity(1)
superop = ptm.to_superop()
chi = ptm.to_chi()
stinespring = ptm.to_kraus().to_stinespring()

assert superop.to_ptm().matrix() == ptm.matrix()
assert chi.to_ptm().matrix() == ptm.matrix()
assert stinespring.to_kraus().is_trace_preserving()
```

## State and Process Measures

State measures accept Python lists of complex values:

```python
from pecos.quantum_info import (
    entropy,
    hellinger_distance,
    negativity,
    partial_trace_qubits,
    partial_trace_subsystems,
    purity,
    schmidt_decomposition,
    shannon_entropy,
    state_fidelity,
)

zero = [1.0 + 0.0j, 0.0 + 0.0j]
plus = [2.0**-0.5, 2.0**-0.5]

assert state_fidelity(zero, zero) == 1.0
assert abs(state_fidelity(zero, plus) - 0.5) < 1e-12

rho_zero = [[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]]
assert purity(rho_zero) == 1.0
assert entropy(rho_zero) == 0.0

bell = [2.0**-0.5 + 0.0j, 0.0j, 0.0j, 2.0**-0.5 + 0.0j]
bell_rho = [
    [0.5 + 0.0j, 0.0j, 0.0j, 0.5 + 0.0j],
    [0.0j, 0.0j, 0.0j, 0.0j],
    [0.0j, 0.0j, 0.0j, 0.0j],
    [0.5 + 0.0j, 0.0j, 0.0j, 0.5 + 0.0j],
]

assert abs(negativity(bell_rho, [2, 2], 1) - 0.5) < 1e-12
assert len(schmidt_decomposition(bell, [2, 2], [0])) == 2
assert partial_trace_qubits(bell_rho, 2, [1]) == [
    [0.5 + 0.0j, 0.0 + 0.0j],
    [0.0 + 0.0j, 0.5 + 0.0j],
]
assert partial_trace_subsystems(bell_rho, [2, 2], [1]) == [
    [0.5 + 0.0j, 0.0 + 0.0j],
    [0.0 + 0.0j, 0.5 + 0.0j],
]

assert shannon_entropy([0.5, 0.5], 2.0) == 1.0
assert hellinger_distance([1.0, 0.0], [0.0, 1.0]) == 1.0
```

Process measures operate on `Ptm` values:

```python
from pecos.quantum_info import Ptm, average_gate_fidelity, gate_error

ideal = Ptm.identity(1)
actual = Ptm.identity(1)

assert average_gate_fidelity(actual, ideal) == 1.0
assert gate_error(actual, ideal) == 0.0
```

## Random Generators

Seeded random generators are available for tests and examples:

```python
from pecos.quantum_info import random_density_matrix, random_quantum_channel

rho = random_density_matrix(num_qubits=1, seed=123)
channel = random_quantum_channel(num_qubits=1, num_kraus=2, seed=123)
assert channel.is_trace_preserving()
```

`random_density_matrix` samples Hilbert-Schmidt random density matrices.
`random_quantum_channel` samples CPTP channels through a random Stinespring
isometry.

## Relationship to QEC APIs

These channel and measure APIs are exact quantum-information tools. They do not
replace detector error models, fault catalogs, or decoders. A typical workflow
is:

1. Characterize or construct a channel as `KrausOps`, `ChoiMatrix`, `Ptm`, or
   `PauliChannel`.
2. Validate the channel and compute state or process measures.
3. Reduce the channel to the noise model needed by a QEC simulation.
4. Build a DEM or fault catalog and estimate logical error rate with a decoder.

This separation keeps exact channel analysis distinct from the compressed fault
models used for large-scale QEC studies.
