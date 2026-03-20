# Stabilizer Code Architecture

This document describes the architecture of the stabilizer code types in `pecos-qec` and how they relate to the Pauli algebra types in `pecos-core` and `pecos-quantum`. For the broader operator type system (Cliffords, unitaries, channels, `Op`), see [Operator Type System Architecture](OPERATOR_TYPE_SYSTEM.md).

## Type Hierarchy

The stabilizer code system is built in layers, from low-level Pauli algebra up to fault tolerance analysis:

```
pecos-core                    pecos-quantum              pecos-qec
┌──────────────────┐         ┌─────────────────────┐    ┌──────────────────────────┐
│  PauliString     │         │ PauliStabilizerGroup │    │ StabilizerCode           │
│  Pauli (X,Y,Z,I) │────────▶│ PauliCollection      │───▶│ (mathematical definition)│
│  QuarterPhase    │         │ F2Matrix             │    ├──────────────────────────┤
│  CliffordRep     │         └─────────────────────┘    │ StabilizerCodeSpec       │
└──────────────────┘                                    │ (operational spec)       │
                                                        ├──────────────────────────┤
                                                        │ StabilizerFlipChecker    │
                                                        │ PauliPropChecker         │
                                                        │ (fault tolerance)        │
                                                        └──────────────────────────┘
```

## Two Stabilizer Code Types

There are two distinct stabilizer code types in `pecos-qec`, serving different roles:

### `StabilizerCode` -- The Mathematical Definition

**File:** `crates/pecos-qec/src/stabilizer_code.rs`

A lightweight type that wraps a `PauliStabilizerGroup` together with an explicit `num_qubits`. This is the mathematical definition of a stabilizer code: a subgroup of the Pauli group that defines a code space.

```rust
pub struct StabilizerCode {
    group: PauliStabilizerGroup,
    num_qubits: usize,
}
```

**Purpose:** On-demand QEC analysis. Given just the stabilizer generators, it can compute:

- `num_logical_qubits()` -- `n - rank` via GF(2) linear algebra
- `logical_operators()` -- centralizer computation over GF(2)
- `distance()` -- coset enumeration (exponential, small codes only)
- `syndrome(error)` -- commutation check against each generator
- `apply_clifford(C)` -- conjugate all generators by a Clifford

**Key design decisions:**

- **Explicit `num_qubits`**: Stabilizer generators may not touch all physical qubits. For example, `ZZ` on a 4-qubit system defines a `[[4, 3]]` code, not a `[[2, 1]]` code. The explicit qubit count determines the code parameters.
- **Computed on demand**: Nothing is precomputed or cached. Each call to `logical_operators()` or `distance()` redoes the computation. This keeps the type simple and stateless.
- **Standard constructors**: `repetition(n)`, `steane()`, `five_qubit()`, `shor()`, `four_two_two()`, `toric(l)` provide well-known codes.

### `StabilizerCodeSpec` -- The Operational Specification

**File:** `crates/pecos-qec/src/stabilizer_code_spec.rs`

A heavier type that stores explicit stabilizers, destabilizers, paired logical operators, and optional distance. Used by the fault tolerance analysis stack.

```rust
pub struct StabilizerCodeSpec {
    num_qubits: usize,
    stabilizers: Vec<PauliString>,
    destabilizers: Vec<PauliString>,
    logical_zs: Vec<PauliString>,
    logical_xs: Vec<PauliString>,
    distance: Option<usize>,
}
```

**Purpose:** Verification and fault tolerance analysis. Provides:

- **Verification methods**: `verify()` checks all commutation relations (stabilizers commute, logicals commute with stabilizers, X/Z pairs anticommute, cross-logical commutation).
- **Column-indexed lookups**: `build_stabilizer_index()` creates an O(weight) anticommutation index for efficient syndrome computation.
- **Builder pattern**: Fluent API for constructing codes with explicit logical operators.
- **Logical pairing**: Stores matched (X_i, Z_i) pairs, unlike `StabilizerCode` which returns an unpaired basis.

**Key design decisions:**

- **Paired logicals**: Fault tolerance analysis needs to know which logical X goes with which logical Z. `StabilizerCodeSpec` enforces this pairing.
- **Destabilizers**: Stored explicitly for use in stabilizer simulation and error correction.
- **Verification-first**: The `verify()` method checks all algebraic constraints before the code is used for analysis. This catches bugs in code definitions early.

### Why Two Types?

They serve different roles:

| Feature | `StabilizerCode` | `StabilizerCodeSpec` |
|---|---|---|
| Input | Generators only | Generators + logicals + destabilizers |
| Computation | On-demand (centralizer, coset) | Pre-stored, verified |
| Logical operators | Unpaired basis | Paired (X_i, Z_i) |
| Verification | None (algebraic correctness assumed) | Full commutation verification |
| Column indexing | No | Yes (O(weight) lookups) |
| Used by | Exploratory analysis, code discovery | Fault tolerance stack |

### Conversion

`StabilizerCode` can be converted to `StabilizerCodeSpec` via:

```rust
// Direct conversion (discovers logicals via stabilizer simulation)
let spec = StabilizerCodeSpec::from_stabilizer_code(&code)?;

// Via builder (for adding manual logicals)
let spec = StabilizerCodeSpecBuilder::from_stabilizer_code(&code)
    .logical_z(Zs([0, 1, 2]))
    .logical_x(Xs([0]))
    .build()
    .unwrap();
```

The `from_stabilizer_code` method uses `discover_logicals()` which runs a stabilizer simulation to find properly paired (X_i, Z_i) logical operators and their corresponding destabilizers.

## Supporting Types

### `PauliStabilizerGroup` (pecos-quantum)

A purely algebraic type: a collection of commuting Pauli strings with real phases (+1 or -1). No QEC interpretation. Provides:

- `rank()` -- GF(2) rank of the generator matrix
- `row_reduce()` -- reduced row echelon form over GF(2)
- `centralizer_in(n)` -- centralizer of the group in the n-qubit Pauli group
- `to_symplectic_matrix()` -- binary symplectic representation
- `apply_clifford(C)` -- conjugate all elements

### `F2Matrix` (pecos-quantum)

Matrix over GF(2) for symplectic linear algebra. Used internally for rank computation, row reduction, and centralizer calculation.

### `CliffordRep` (pecos-core)

Sparse representation of a Clifford unitary as conjugation rules on single-qubit Paulis. Used by `StabilizerCode::apply_clifford()` to transform code generators.

## Fault Tolerance Integration

The fault tolerance module in `pecos-qec` consumes `StabilizerCodeSpec`:

- **`StabilizerFlipChecker`**: Code-level analysis. Takes a `StabilizerCodeSpec` and checks whether faults of a given weight can cause undetectable logical errors. Works without a circuit.

- **`PauliPropChecker`**: Circuit-level analysis. Takes a syndrome extraction circuit, then propagates Pauli faults through it to verify fault tolerance of a specific implementation.

- **`GadgetChecker`**: Gadget-level analysis. Extends `PauliPropChecker` with explicit input/output qubit tracking for analyzing gadgets in composed QEC protocols. Enforces the s + r <= t constraint (input fault weight + internal fault weight).

`StabilizerFlipChecker` uses the column-indexed anticommutation structure provided by `StabilizerCodeSpec` for efficient syndrome computation.

## Module Organization

```
crates/pecos-qec/src/
  lib.rs                    -- crate root, re-exports
  stabilizer_code.rs        -- StabilizerCode (mathematical definition)
  stabilizer_code_spec.rs   -- StabilizerCodeSpec (operational spec)
  logical_discovery.rs      -- discover_logicals() via stabilizer simulation
  distance.rs               -- distance calculation algorithms
  geometry.rs               -- physical layout types
  surface.rs                -- surface code geometry
  fault_tolerance/           -- fault tolerance analysis
    mod.rs
    stabilizer_flip_checker.rs
    pauli_prop_checker.rs
    gadget_checker.rs        -- gadget-level fault tolerance (input/output tracking)
    dem_builder.rs           -- detector error model construction
    ...
```
