# Quantum Operator Algebra

This guide covers PECOS's quantum operator type system: from single-qubit Paulis up through Cliffords, general unitaries, and channels, plus the collection types for algebraic analysis. All types live in `pecos-core` (individual operators) and `pecos-quantum` (collections/groups).

## What You'll Learn

- The four operator levels: Pauli, Clifford, Unitary, Channel
- How to build operators with ergonomic constructor functions
- Automatic type promotion via the unified `Op` type
- Collection types for Pauli analysis: sequences, sets, groups
- Conversions between all these types

## Overview: The Four Levels

PECOS organizes quantum operators into a strict hierarchy where each level is a subset of the next:

```text
Pauli  ⊂  Clifford  ⊂  Unitary  ⊂  Gate  ⊂  Channel
```

Each level has its own representation optimized for what it can express:

| Level | Type | What it represents | Key capability |
|---|---|---|---|
| Pauli | `PauliString` | Tensor products of I, X, Y, Z with phase | Exact commutation, symplectic algebra |
| Clifford | `CliffordRep` | Gates that map Paulis to Paulis | Fast conjugation via Heisenberg picture |
| Unitary | `UnitaryRep` | Any unitary (including non-Clifford) | Lazy expression tree with algebraic ops |
| Gate | `GateExpr` | Ideal circuit operations (unitary, preparation, measurement, reset) | Compose and tensor circuit operations |
| Channel | `ChannelExpr` | General CPTP maps and noise/decoherence operations | Compose and tensor arbitrary physical maps |

The unified `Op` type wraps all five and automatically promotes when you combine operators from different levels.

---

## Level 1: Pauli Operators

### PauliString

The primary Pauli type. Stores a sparse list of non-identity single-qubit Paulis with a global phase from `{+1, -1, +i, -i}`.

```rust
use pecos_core::pauli::*;
use pecos_core::PauliOperator;

// Single-qubit constructors
let p = X(0);          // X on qubit 0
let q = Z(3);          // Z on qubit 3

// Multi-qubit constructors (same Pauli type on multiple qubits)
let zz = Zs([0, 1]);          // Z(0) & Z(1)
let xxx = Xs([0, 1, 2]);      // X(0) & X(1) & X(2)
let yy = Ys(0..2);            // also accepts ranges

// Mixed Paulis via tensor product (&)
let xz = X(0) & Z(1);         // X on qubit 0, Z on qubit 1
let xzzx = X(0) & Z(1) & Z(2) & X(3);

// Pauli multiplication (*)
let xy = X(0) * Z(0);         // X * Z = -iY (same qubit)
let tensor = X(0) * Z(1);     // different qubits -> tensor product

// Phases
let neg = -X(0);              // -X
```

```rust
use pecos_core::pauli::algebra::i;
use pecos_core::pauli::*;

let imag = i * X(0);          // iX
let neg_imag = -i * Y(1);     // -iY
```

### Key Operations

```rust
use pecos_core::pauli::*;
use pecos_core::PauliOperator;

let a = X(0) & Z(1);
let b = Z(0) & X(1);

// Commutation
// X(0)Z(1) and Z(0)X(1) anticommute on each qubit, but 2 anticommutations = commute
assert!(a.commutes_with(&b));
assert!(a.commutes_with(&a));   // self-commutes

// Weight (number of non-identity sites)
assert_eq!(a.weight(), 2);

// Phase
use pecos_core::QuarterPhase;
assert_eq!(X(0).phase(), QuarterPhase::PlusOne);
assert_eq!((-X(0)).phase(), QuarterPhase::MinusOne);

// Qubit positions
assert_eq!(a.x_positions(), vec![0]);  // X on qubit 0
assert_eq!(a.z_positions(), vec![1]);  // Z on qubit 1
```

### Alternative Pauli Representations

`PauliString` is the primary type, but two alternatives exist for specialized use cases:

- **`PauliBitmap`**: Compact bitmap representation for systems with at most 64 qubits. Uses bitwise operations for fast arithmetic.
- **`PauliSparse<T>`**: Generic sparse representation parameterized by the set type.

All three implement the `PauliOperator` trait, which provides `multiply()`, `weight()`, `commutes_with()`, `phase()`, `x_positions()`, and `z_positions()`.

### Parsing from Strings

Use constructors for ordinary code. Use sparse strings when explicit qubit
indices make text input clearer, and dense strings when positional notation is
useful for compact tables.

```rust
use pecos_core::PauliString;

let sparse: PauliString = "X0 Z3".parse().unwrap();
let dense: PauliString = "XIIZ".parse().unwrap();
let explicit_sparse = PauliString::from_sparse_str("X0 Z3").unwrap();
let explicit_dense = PauliString::from_dense_str("XIIZ").unwrap();

assert_eq!(sparse, dense);
assert_eq!(sparse, explicit_sparse);
assert_eq!(dense, explicit_dense);
assert_eq!(sparse.to_sparse_str(), "+X0 Z3");
assert_eq!(sparse.to_dense_str(None), "+XIIZ");
```

---

## Level 2: Clifford Gates

### CliffordRep

Represents a Clifford gate via the Heisenberg picture: how it transforms each Pauli generator X_i and Z_i. This enables O(n) conjugation of Pauli strings.

```rust
use pecos_core::clifford_rep::CliffordRep;
use pecos_core::pauli::*;

// Hadamard on qubit 0: X -> Z, Z -> X
let h = CliffordRep::h(0);
assert_eq!(*h.x_image(0), Z(0));  // H X H^dag = Z
assert_eq!(*h.z_image(0), X(0));  // H Z H^dag = X

// Composition
let hh = h.compose(&h);  // H^2 = I (up to phase)

// Transform a Pauli string through a Clifford
let stabilizer = Zs([0, 1]);
let transformed = h.apply(&stabilizer);
// H on qubit 0 turns Z(0)Z(1) into X(0)Z(1)
```

### Clifford Enum

The 24 single-qubit Clifford gates and 14 two-qubit entangling gates are also available as a named enum (`Clifford`) for gate-level work, but `CliffordRep` is more useful for algebraic manipulation.

---

## Level 3: Unitary Expression Tree

### UnitaryRep

A lazy expression tree that can represent any quantum unitary, including non-Clifford gates (T, arbitrary rotations). Supports composition, tensor products, and adjoint algebraically.

```rust
use pecos_core::unitary::*;
use pecos_core::Angle64;

// Named gates
let circuit = T(1) * CX(0, 1) * H(0);  // apply H, then CX, then T

// Check Clifford-ness
assert!(!circuit.is_clifford());    // T is not Clifford
assert!((CX(0, 1) * H(0)).is_clifford());

// Tensor product
let parallel = H(0) & H(1);  // H on both qubits simultaneously

// Adjoint
let inv = circuit.dg();  // T^dag * CX^dag * H^dag

// Rotation gates
let rz = RZ(Angle64::HALF_TURN / 4, 0);  // Rz(pi/4) on qubit 0

// Multi-qubit gate constructors
let cx_pair = CXs([(0, 1), (2, 3)]);  // CX on two pairs
let h_all = Hs([0, 1, 2]);            // H on three qubits
```

**Composition order**: `A * B` means "apply B first, then A" (matrix multiplication order).

---

## Level 4: Gates (Ideal Circuit Operations)

### GateExpr

Represents ideal circuit operations: unitaries, measurements, preparations, resets, and their compositions.

```rust
use pecos_core::op::*;

// Measurement, preparation, and reset
let mz = MZ(0);           // Z-basis measurement on qubit 0
let mx = MX(1);           // X-basis measurement on qubit 1
let pz = PZ(0);           // Prepare |0> on qubit 0
let reset = Reset(0);     // Reset to |0>
assert!(mz.is_gate());
```

---

## Level 5: Channels (Physical Maps)

### ChannelExpr

Represents general CPTP maps, which PECOS usually uses for noise and decoherence.

```rust
use pecos_core::op::*;

// Noise channels
let depol = Depolarizing(0.01, 0);            // 1% depolarizing on qubit 0
let deph = Dephasing(0.02, 1);                // 2% dephasing on qubit 1
let amp_damp = AmplitudeDamping(0.05, 0);     // T1 decay
let phase_damp = PhaseDamping(0.03, 0);       // T2 dephasing
let erasure = Erasure(0.01, 0);               // Erasure channel
let leak = Leakage(0.001, 0);                 // Leakage to non-computational state

// Custom Pauli channel
let pauli_ch = PauliChannel(0.01, 0.01, 0.01, 0);  // px, py, pz
assert!(depol.is_channel());
```

---

## The Unified `Op` Type

`Op` wraps all five levels and automatically promotes when you combine operators:

```rust
use pecos_core::op::*;

// Pauli & Pauli stays Pauli
let p = X(0) & Y(3);
assert!(p.is_pauli());

// Pauli & Clifford promotes to Clifford
let c = X(0) & H(3);
assert!(c.is_clifford());

// Adding a non-Clifford promotes to Unitary
let u = X(0) & H(3) & T(5);
assert!(u.is_unitary());

// Adding a measurement promotes to Gate
let g = H(0) & MZ(1);
assert!(g.is_gate());

// Adding noise promotes to Channel
let ch = g & Depolarizing(0.01, 2);
assert!(ch.is_channel());
```

### Extracting Inner Types

```rust
use pecos_core::op::*;

let p = X(0) & Z(1);
let ps = p.as_pauli().unwrap();  // borrow the inner PauliString

let c = H(0) & X(1);
let cr = c.as_clifford().unwrap();  // borrow the inner CliffordRep

// Consuming extraction with promotion
let u = (X(0) & H(1)).into_unitary().unwrap();  // promotes Clifford to UnitaryRep

// Every Op can become a Channel
let ch = (H(0) & T(1)).into_channel();  // always succeeds
```

### Adjoint

```rust
use pecos_core::op::*;

let circuit = T(1) * CX(0, 1) * H(0);
let inverse = circuit.dg();  // works for Pauli, Clifford, Unitary

// Channels are not invertible -- dg() panics, use try_dg()
let m = MZ(0);
assert!(m.try_dg().is_none());
```

### Qubit Query

```rust
use pecos_core::op::*;

let circuit = CX(0, 3) & H(5);
assert_eq!(circuit.num_qubits(), 6);     // matrix span is qubits 0..5
assert_eq!(circuit.qubits(), vec![0, 3, 5]);  // actual support
```

---

## Pauli Collection Types (pecos-quantum)

For working with multiple Pauli strings, `pecos-quantum` provides four collection types with increasing constraints:

```text
PauliSequence          -- ordered, no constraints
    ↓ validate commutativity
PauliGroup             -- commuting generators, any phase
    ↓ validate real phases
PauliStabilizerGroup   -- commuting generators, +/-1 phase only

PauliSet               -- unordered, deduplicated (separate hierarchy)
```

### PauliSequence (No Constraints)

An ordered list of Pauli strings with GF(2) symplectic analysis. No commutativity or phase constraints.

```rust
use pecos_quantum::PauliSequence;
use pecos_core::pauli::*;

let seq = PauliSequence::new(vec![
    Zs([0, 1]),
    Zs([1, 2]),
    Zs([0, 2]),  // linearly dependent on the first two
]);

// GF(2) rank: number of linearly independent Paulis
assert_eq!(seq.rank(), 2);

// Membership in GF(2) span (ignoring phase)
assert!(seq.contains(&Zs([0, 2])));   // Z(0)Z(2) = product of first two
assert!(!seq.contains(&X(0)));         // X not in Z-span

// Commutativity check
assert!(seq.is_abelian());  // all Z-type, so commute

// Row reduction to independent generators
let reduced = seq.row_reduce();
assert_eq!(reduced.len(), 2);
```

### PauliSet (Unordered, Unique)

A set of distinct Pauli strings. Two strings are equal only if they have the same operators AND the same phase (+XZ and -XZ are distinct).

```rust
use pecos_quantum::PauliSet;
use pecos_core::pauli::*;

let mut set = PauliSet::new();
set.insert(&X(0));
set.insert(&Z(1));
set.insert(&X(0));  // duplicate, ignored

assert_eq!(set.len(), 2);
assert!(set.contains(&X(0)));

// Set operations
let other = PauliSet::from_iter(vec![X(0), Y(2)]);
let union = set.union(&other);
assert_eq!(union.len(), 3);  // X(0), Z(1), Y(2)
```

### PauliGroup (Abelian, Any Phase)

A commuting subgroup of the Pauli group. Generators may carry any `QuarterPhase` (+1, -1, +i, -i). When a generator has imaginary phase, its order is 4 (not 2), and the group may contain -I.

```rust
use pecos_quantum::PauliGroup;
use pecos_core::pauli::*;
use pecos_core::pauli::algebra::i;

// Generators with imaginary phase
let group = PauliGroup::new(vec![
    i * X(0) & Y(1),   // phase +i, order 4
    Z(2),               // phase +1, order 2
]).unwrap();

assert_eq!(group.rank(), 2);
assert!(group.contains_minus_identity());  // (iXY)^2 = -I

// Validation: anticommuting generators are rejected
let err = PauliGroup::new(vec![X(0), Z(0)]);
assert!(err.is_err());
```

### PauliStabilizerGroup (Abelian, Real Phase Only)

The standard stabilizer group for QEC. All generators must commute and have phase +1 or -1 (so every element squares to +I).

```rust
use pecos_quantum::PauliStabilizerGroup;
use pecos_core::pauli::*;

// Repetition code stabilizers
let stab = PauliStabilizerGroup::new(vec![
    Zs([0, 1]),
    Zs([1, 2]),
]).unwrap();

assert_eq!(stab.rank(), 2);

// GF(2) membership (is ZIZ in the stabilizer group?)
assert!(stab.contains(&Zs([0, 2])));

// Phase-aware membership
assert!(stab.contains_with_phase(&Zs([0, 2])));   // +ZIZ is in the group
assert!(!stab.contains_with_phase(&(-Zs([0, 2])))); // -ZIZ is not

// Clifford transformation: conjugate all generators
use pecos_core::clifford_rep::CliffordRep;
let h = CliffordRep::h(0);
let transformed = stab.apply_clifford(&h);
// Z(0)Z(1) becomes X(0)Z(1) under H on qubit 0
```

### Conversions Between Collections

```rust
use pecos_quantum::{PauliSequence, PauliGroup, PauliStabilizerGroup, PauliSet};
use pecos_core::pauli::*;

// Upward (widening) -- always succeeds
let stab = PauliStabilizerGroup::new(vec![Zs([0, 1])]).unwrap();
let group: PauliGroup = stab.clone().into();        // drop phase constraint
let seq: PauliSequence = group.into();               // drop commutativity constraint
let set: PauliSet = PauliSet::from_iter(vec![Zs([0, 1])]);

// Downward (narrowing) -- validates constraints
let seq = PauliSequence::new(vec![Zs([0, 1]), Zs([1, 2])]);
let group: PauliGroup = seq.try_into().unwrap();     // checks commutativity
let stab: PauliStabilizerGroup = group.try_into().unwrap(); // checks real phases

// From strings
let stab = PauliStabilizerGroup::from_strs(&["ZZI", "IZZ"]).unwrap();
let group = PauliGroup::from_strs(&["ZZI", "IZZ"]).unwrap();
let seq = PauliSequence::from_strs(&["XZ", "ZX"]).unwrap();  // can anticommute
```

---

## From Collections to Codes

The collection types feed into the QEC types in `pecos-qec`:

```rust
use pecos_quantum::PauliStabilizerGroup;
use pecos_qec::StabilizerCode;
use pecos_core::pauli::*;

// Build a stabilizer group
let group = PauliStabilizerGroup::new(vec![
    Zs([0, 1]),
    Zs([1, 2]),
]).unwrap();

// Wrap in a StabilizerCode for QEC analysis
let code = StabilizerCode::from_group(group);
assert_eq!(code.num_logical_qubits(), 1);
assert_eq!(code.distance(), Some(1));

// Or use standard constructors directly
let steane = StabilizerCode::steane();
assert_eq!(steane.distance(), Some(3));
```

For details on `StabilizerCode` and `StabilizerCodeSpec`, see the [Stabilizer Codes guide](stabilizer-codes.md).

---

## Summary: Which Type to Use

| I want to... | Use |
|---|---|
| Build a single Pauli operator | `PauliString` via constructors (`X`, `Z`, `Xs`, etc.) |
| Build a Clifford gate and transform Paulis | `CliffordRep` |
| Build a circuit with non-Clifford gates | `UnitaryRep` |
| Mix unitaries with measurements/noise | `Op` (auto-promotes) |
| Analyze a collection of Paulis (rank, independence) | `PauliSequence` |
| Store unique Paulis with set operations | `PauliSet` |
| Represent a commuting Pauli subgroup | `PauliGroup` or `PauliStabilizerGroup` |
| Define a QEC code and compute distance/logicals | `StabilizerCode` |
| Verify and do fault tolerance analysis on a code | `StabilizerCodeSpec` |
