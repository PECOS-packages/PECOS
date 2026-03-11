# Operator Type System Architecture

This document describes the quantum operator type system in PECOS: the four algebraic levels (Pauli, Clifford, Unitary, Channel), their representations, the unified `Op` type with automatic promotion, and the Pauli collection hierarchy.

## Design Principles

1. **Tightest possible representation**: Each operator is represented at the most specific level that can express it. A Hadamard gate is Clifford, not "just a unitary". This enables level-specific optimizations (e.g., fast Pauli conjugation via tableaux).

2. **Automatic promotion**: When combining operators from different levels, the result is promoted to the tightest level that can represent the combination. Pauli & Clifford -> Clifford.

3. **Lazy expression trees**: At the Unitary and Channel levels, operators are stored as symbolic expression trees, not eagerly evaluated matrices. Composition (`*`) and tensor (`&`) build tree nodes.

4. **Sparse by default**: Pauli strings store only non-identity entries. Clifford tableaux store only the images of generators that change. This handles large qubit counts efficiently.

## Operator Levels

```
Level 0: Pauli      PauliString         Exact, finite group, symplectic algebra
Level 1: Clifford   CliffordRep         Heisenberg picture, O(n) Pauli conjugation
Level 2: Unitary    UnitaryRep          Lazy expression tree, any unitary
Level 3: Channel    ChannelExpr         Non-unitary: measurement, noise, reset
```

### Level 0: Pauli (`PauliString`)

**File:** `crates/pecos-core/src/pauli/pauli_string.rs`

```rust
pub struct PauliString {
    phase: QuarterPhase,              // {+1, -1, +i, -i}
    paulis: Vec<(Pauli, QubitId)>,    // sparse: only non-identity entries
}
```

The n-qubit Pauli group has 4^(n+1) elements (4 single-qubit Paulis, 4 phases). `PauliString` represents them sparsely -- a Pauli acting non-trivially on 3 out of 1000 qubits uses O(3) storage.

**Key properties:**
- Exact arithmetic (no floating point)
- O(w) commutation check where w = total weight of both operators
- Multiplication produces another `PauliString` with computed phase
- The `PauliOperator` trait unifies `PauliString`, `PauliBitmap`, and `PauliSparse<T>`

**Constructor modules:**
- `pecos_core::pauli::constructors` -- `X(q)`, `Y(q)`, `Z(q)`, `Xs(qs)`, `Ys(qs)`, `Zs(qs)`
- `pecos_core::pauli::algebra` -- operator overloading: `&` (tensor), `*` (multiply), `-` (negate), `i *` (phase)

**Alternative representations:**
- `PauliBitmap` -- u64 bitmasks for X and Z positions. Limited to 64 qubits, but bitwise operations are very fast.
- `PauliSparse<T>` -- generic over the set type `T` for X/Z positions. Allows custom set implementations.

### Level 1: Clifford (`CliffordRep`)

**File:** `crates/pecos-core/src/clifford_rep.rs`

```rust
pub struct CliffordRep {
    num_qubits: usize,
    x_images: Vec<PauliString>,    // X_i -> PauliString
    z_images: Vec<PauliString>,    // Z_i -> PauliString
}
```

Represents a Clifford gate via the Heisenberg picture: how it conjugates each single-qubit Pauli generator. A Clifford on n qubits is fully specified by 2n generator images.

**Why this representation:**
- Conjugating a weight-w Pauli string through a Clifford is O(w * n), not O(4^n)
- Composition of two Cliffords is O(n^2)
- Natural for stabilizer simulation and code analysis

**Key methods:**
- `identity(n)`, `h(q)`, `cx(c, t)`, `s(q)`, etc. -- standard gate constructors
- `compose(&other)` -- C1 * C2
- `apply(&pauli)` -- C * P * C^dag
- `inverse()` -- C^dag
- `from(PauliString)` -- every Pauli is a Clifford

**Relation to `Clifford` enum:** The `Clifford` enum (`crates/pecos-core/src/clifford.rs`) lists the 24 single-qubit and 14 two-qubit Clifford primitives by name. `CliffordRep` is the algebraic representation used for computation.

### Level 2: Unitary (`UnitaryRep`)

**File:** `crates/pecos-core/src/unitary_rep.rs`

```rust
pub enum UnitaryRep {
    Pauli(PauliString),
    Rotation { rotation_type: RotationType, angle: Angle64, qubits: SmallVec<[usize; 2]> },
    Gate { gate_type: GateType, qubits: SmallVec<[usize; 3]> },
    Tensor(Vec<UnitaryRep>),
    Compose(Vec<UnitaryRep>),
    Adjoint(Box<UnitaryRep>),
    Phase { phase: Angle64, inner: Box<UnitaryRep> },
}
```

A lazy expression tree that can represent any unitary, including non-Clifford gates (T, Rz(theta), etc.). Operations build tree nodes rather than evaluating eagerly.

**Why an expression tree:**
- Composition and tensor product are O(1) (just wrap in a new node)
- The tree can be analyzed symbolically (e.g., `is_clifford()` checks)
- Different backends can evaluate the tree differently (matrix, stabilizer sim, etc.)

**Constructor functions:** Defined in `crates/pecos-core/src/unitary_rep.rs`:
- Single-qubit: `X(q)`, `Y(q)`, `Z(q)`, `H(q)`, `SX(q)`, `SZ(q)`, `T(q)`, etc.
- Rotation: `RX(angle, q)`, `RY(angle, q)`, `RZ(angle, q)`, `RXX(angle, q0, q1)`, etc.
- Two-qubit: `CX(c, t)`, `CZ(q0, q1)`, `SWAP(q0, q1)`, `ISWAP(q0, q1)`, etc.
- Pluralized: `Hs([q0, q1, ...])`, `CXs([(c0,t0), (c1,t1)])` -- tensor multiple gates

### Level 3: Channel (`ChannelExpr`)

**File:** `crates/pecos-core/src/op.rs`

```rust
pub enum ChannelExpr {
    Prep { basis: Basis, qubit: usize },
    Measure { basis: Basis, qubit: usize },
    Unitary(UnitaryRep),
    MixedUnitary(Vec<(f64, UnitaryRep)>),
    AmplitudeDamping { gamma: f64, qubit: usize },
    PhaseDamping { lambda: f64, qubit: usize },
    Erasure { prob: f64, qubit: usize },
    Reset { qubit: usize },
    Leakage { rate: f64, qubit: usize },
    Tensor(Vec<ChannelExpr>),
    Compose(Vec<ChannelExpr>),
}
```

Non-unitary quantum operations. These compose and tensor like unitaries but are not invertible (no `dg()`).

**Notable variants:**
- `MixedUnitary` -- covers Pauli channels, depolarizing, dephasing, bit-flip via weighted sums of unitaries
- `AmplitudeDamping` / `PhaseDamping` -- explicit Kraus-operator channels for T1/T2 processes
- `Erasure` -- heralded error channel
- `Leakage` -- transition to non-computational states

## The Unified `Op` Type

**File:** `crates/pecos-core/src/op.rs`

```rust
pub enum Op {
    Pauli(PauliString),
    Clifford(CliffordRep, UnitaryRep),
    Unitary(UnitaryRep),
    Channel(ChannelExpr),
}
```

`Op` wraps all four levels and provides automatic promotion via the `&` (tensor) and `*` (composition) operators. When combining two `Op` values, the result is at the maximum level of the operands.

### Dual Representation at Clifford Level

The `Clifford` variant stores both a `CliffordRep` (for efficient Pauli conjugation) and a `UnitaryRep` (for promotion to the Unitary level). This avoids information loss when mixing Clifford and non-Clifford operations.

### Promotion Rules

```
Pauli & Pauli       -> Pauli
Pauli & Clifford    -> Clifford
Pauli & Unitary     -> Unitary
Clifford & Unitary  -> Unitary
anything & Channel  -> Channel
```

Same rules apply for composition (`*`).

### Extraction Methods

`Op` provides both borrowing and consuming extractors:

- `as_pauli()` / `into_pauli()` -- returns `None` for non-Pauli
- `as_clifford()` / `into_clifford()` -- Pauli promotes to Clifford; Unitary/Channel return `None`
- `as_unitary()` / `into_unitary()` -- Pauli and Clifford promote; Channel returns `None`
- `into_channel()` -- always succeeds (lower levels wrap in `ChannelExpr::Unitary`)

### Constructor Functions

`Op`-level constructors live in `crates/pecos-core/src/op.rs` and mirror the `UnitaryRep` constructors but return `Op` at the tightest level:

- `X(q)`, `Z(q)` -- return `Op::Pauli`
- `H(q)`, `CX(c,t)`, `SZ(q)` -- return `Op::Clifford`
- `T(q)`, `RZ(angle, q)` -- return `Op::Unitary`
- `MZ(q)`, `PZ(q)`, `Depolarizing(p, q)` -- return `Op::Channel`

**Important:** The constructors in `pecos_core::op` and `pecos_core::unitary_rep` have the same names (`X`, `H`, `T`, etc.) but return different types (`Op` vs `UnitaryRep`). Use `use pecos_core::op::*` for the unified `Op` algebra, or `use pecos_core::unitary_rep::*` for the `UnitaryRep`-only algebra. Similarly, `use pecos_core::pauli::constructors::*` gives `PauliString`-level constructors.

## Pauli Collection Types

**Crate:** `pecos-quantum`

Four collection types with increasing algebraic constraints:

```
PauliSequence ──(validate commutativity)──> PauliGroup ──(validate real phases)──> PauliStabilizerGroup

PauliSet (separate: unordered, deduplicated)
```

### PauliSequence

**File:** `crates/pecos-quantum/src/pauli_sequence.rs`

Ordered list of `PauliString`s with no constraints. Provides GF(2) symplectic analysis:

- `rank()` -- number of linearly independent generators
- `row_reduce()` -- independent generator subset
- `contains(&pauli)` -- membership in GF(2) span (ignoring phase)
- `is_abelian()` -- check mutual commutativity
- `to_symplectic_matrix()` -- binary representation for linear algebra

### PauliSet

**File:** `crates/pecos-quantum/src/pauli_set.rs`

Unordered set of unique `PauliString`s backed by `BTreeSet`. Phase-sensitive equality (+XZ and -XZ are distinct). Standard set operations (union, intersection, difference).

### PauliGroup

**File:** `crates/pecos-quantum/src/pauli_group.rs`

Wraps `PauliSequence` with validated commutativity. Generators may have any `QuarterPhase`. When a generator has phase +i or -i, it has order 4 and the group contains -I (so it cannot stabilize any quantum state).

### PauliStabilizerGroup

**File:** `crates/pecos-quantum/src/stabilizer_group.rs`

Wraps `PauliGroup` with the additional constraint that all generators have `Sign` phases (+1 or -1). This is the standard stabilizer group for QEC: every element squares to +I, and the group defines a valid code space.

### Conversion Safety

Widening conversions (dropping constraints) always succeed via `From`:
```
PauliStabilizerGroup -> PauliGroup -> PauliSequence
```

Narrowing conversions (adding constraints) are fallible via `TryFrom`:
```
PauliSequence -> PauliGroup          (validates commutativity)
PauliGroup -> PauliStabilizerGroup   (validates real phases)
PauliSequence -> PauliStabilizerGroup (validates both)
```

All types also have `from_generators_unchecked()` for internal use when constraints are known to hold.

## File Organization

```
crates/pecos-core/src/
  pauli.rs                    -- Pauli enum, PauliOperator trait
  pauli/
    pauli_string.rs           -- PauliString (primary Pauli type)
    pauli_bitmap.rs           -- PauliBitmap (<=64 qubits, fast)
    pauli_sparse.rs           -- PauliSparse<T> (generic)
    constructors.rs           -- X(), Y(), Z(), Xs(), Ys(), Zs()
    algebra.rs                -- operator overloading (&, *, -, i*)
  clifford.rs                 -- Clifford enum (named gate primitives)
  clifford_rep.rs             -- CliffordRep (Heisenberg picture)
  unitary_rep.rs              -- UnitaryRep (expression tree)
  op.rs                       -- Op (unified type), ChannelExpr

crates/pecos-quantum/src/
  pauli_sequence.rs           -- PauliSequence, F2Matrix
  pauli_set.rs                -- PauliSet
  pauli_group.rs              -- PauliGroup
  stabilizer_group.rs         -- PauliStabilizerGroup
```

## Relation to QEC Types

The Pauli collection types feed into the QEC layer in `pecos-qec`:

- `PauliStabilizerGroup` + `num_qubits` -> `StabilizerCode` (mathematical definition, on-demand analysis)
- `StabilizerCode` -> `StabilizerCodeSpec` (verified, with paired logicals, for fault tolerance)

See [Stabilizer Code Architecture](STABILIZER_CODE_ARCHITECTURE.md) for details.
