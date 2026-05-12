# PECOS Concepts

PECOS keeps a few related QEC concepts separate because they answer different
questions. The implementation can share propagation machinery underneath, but
the public API should make the distinction clear.

## Quick Map

| Concept | Meaning | Typical source | Decoder role |
|---|---|---|---|
| Detector | A parity check on measurement records | Measurement record metadata | Syndrome bit |
| Observable | A measured logical or experiment output | Measurement record metadata | Logical class decoded from the syndrome |
| Tracked Pauli | A Pauli operator inserted as a non-physical probe | Circuit annotation | Analysis output, ignored by ordinary DEM decoders |
| Gate | An ideal operation in a circuit | Circuit builder | No noise unless a noise model attaches it |
| Channel | A physical CPTP map, often noise | Noise model or explicit channel op | Source of stochastic or coherent faults |
| Fault location | One independent place a modeled fault can occur | Fault catalog | Unit of fault enumeration and sampling |

## Detectors, Observables, and Tracked Operators

**Detectors** are syndrome bits. A detector is defined by a parity expression
over measurement records, such as "the previous ancilla measurement differs
from the same check in the prior round." Decoders consume detector flips as the
syndrome.

**Observables** are measurement-defined experiment outputs. In QEC workflows
these are often logical measurement outcomes. They are still defined through
measurement records, so they are things the experiment can observe directly or
infer from recorded measurement data. Logical error rate terminology in PECOS
continues to refer to errors in these logical observables.

**Tracked Paulis** are Pauli strings placed at a circuit point as probes.
They are not measured by that annotation and do not become detector syndrome
bits. Fault-analysis code asks whether propagated faults anticommute with the
tracked Pauli at that point. A tracked Pauli might be a logical operator,
a stabilizer, or another tracked Pauli useful for analysis.

Error events can therefore flip three independent kinds of output:

- detectors: what syndrome bits changed
- observables: what measured logical or experiment outputs changed
- tracked Paulis: which tracked Paulis anticommute with the propagated fault

Do not merge observable IDs and tracked-Pauli IDs. Observable `0` is always
observable `0`; tracked Paulis have their own ID space and their own metadata.

## Operator Construction

Use the most structured representation that fits the situation:

1. Typed constructors, such as `X(0) & Z(3)`, for ordinary code.
2. Sparse strings, such as `"X0 Z3"`, for compact text input with explicit
   qubit indices.
3. Dense strings, such as `"XIIZ"`, for table-like input where character
   position is the qubit index.

When writing Rust code, the constructor style is the default:

```rust
use pecos_core::pauli::*;
use pecos_core::PauliOperator;

let logical_x = X(0) & X(1) & X(2);
let z_probe = Z(3);

assert_eq!(logical_x.weight(), 3);
assert!(logical_x.commutes_with(&z_probe));
```

String parsing is still useful when reading checks, user input, test fixtures,
or text formats:

```rust
use pecos_core::{PauliOperator, PauliString};

let stabilizer: PauliString = "Z0 Z1 Z4 Z5".parse().unwrap();
assert_eq!(stabilizer.weight(), 4);
```

Python exposes the same idea. Use `X(0) & Z(3)` for inline construction,
`PauliString.from_sparse_str(...)` for explicit sparse text, and
`PauliString.from_dense_str(...)` for dense text. `PauliString.from_str(...)`
auto-detects sparse versus dense notation.

```python
from pecos.quantum import PauliString, X, Z

probe = X(0) & X(1) & Z(3)
from_text = PauliString.from_str("X0 X1 Z3")

assert probe == from_text
assert probe.to_sparse_str() == "+X0 X1 Z3"
assert probe.to_dense_str() == "+XXIZ"
```

In Pauli-algebra contexts, `X`, `Y`, and `Z` construct `PauliString` values.
In circuit-building contexts, use circuit APIs such as `Gate.x(...)`,
`TickCircuit.tick().x(...)`, or the corresponding builder methods.

## Gates, Channels, Noise, and Idle Locations

A **gate** is an ideal circuit operation. A **channel** is a physical map. Noise
models attach channels to selected circuit locations.

`Idle` is a scheduling marker unless idle noise is attached explicitly. Adding
an `Idle` gate records timing structure; it should not silently inherit
single-qubit gate noise. To model idle decoherence, use a noise model or channel
API that explicitly targets idle locations.

This keeps two actions separate:

- changing circuit timing: add or remove idle locations
- changing the physical noise model: attach idle noise explicitly

## DEMs and Fault Catalogs

PECOS detector-error models represent detector and observable effects that
ordinary decoders consume. PECOS-specific metadata can also carry tracked
Paulis for analysis, but tracked Paulis are not logical observables and
ordinary DEM decoders should ignore them.

The fault catalog gives the most detailed per-location view:

- `affected_measurements`: raw measurement flips
- `affected_detectors`: syndrome flips
- `affected_observables`: measurement-defined logical or experiment outputs
- `affected_tracked_paulis`: tracked Paulis flipped by anticommutation

## Recommended Surface-Code Memory Path

For standard surface-code memory experiments:

1. Build the patch and circuit with `SurfacePatch` and the surface circuit
   builders.
2. Generate the circuit-level DEM with the surface decoder helpers.
3. Sample and decode with the native DEM sampler or a matching decoder backend.
4. Use the fault catalog when you need per-location fault anatomy, targeted
   lookup decoding, or probability-weighted explanations for a syndrome.

Use the lower-level `TickCircuit`, `DagCircuit`, `DemBuilder`, and fault-catalog
APIs when you are developing new circuit families, new analysis tools, or new
decoder integrations.
