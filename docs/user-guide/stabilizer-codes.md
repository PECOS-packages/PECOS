# Stabilizer Codes

This guide covers working with Pauli strings and stabilizer codes in PECOS's Rust API (`pecos-qec`). These types are also available in Python via `pecos_rslib`.

## What You'll Learn

- Building Pauli strings with the constructor API
- Defining stabilizer codes from generators
- Computing code parameters, logical operators, and distance
- Verifying code definitions
- Using standard code constructors
- Converting between code types

```hidden-rust
use pecos_core::pauli::*;
use pecos_core::PauliOperator;

fn main() {
    // CODE
}
```

## Building Pauli Strings

Pauli strings are the fundamental building block. PECOS provides a concise constructor API:

=== ":fontawesome-brands-rust: Rust"

    ```rust
    use pecos_core::pauli::*;

    // Single-qubit Paulis
    let x0 = X(0);       // X on qubit 0
    let z3 = Z(3);       // Z on qubit 3
    let y1 = Y(1);       // Y on qubit 1

    // Multi-qubit (same type) via array constructors
    let zz = Zs([0, 1]);         // ZZ on qubits 0,1
    let xxxx = Xs([0, 1, 2, 3]); // XXXX on qubits 0-3

    // Mixed Paulis via the & operator
    let xzzx = X(0) & Z(1) & Z(2) & X(3);  // XZZXI...

    // Commutation checks
    // X and Z on different qubits commute:
    assert!(x0.commutes_with(&z3));
    // X and Z on the SAME qubit anticommute:
    assert!(!X(0).commutes_with(&Z(0)));
    ```

=== ":fontawesome-brands-python: Python"

    ```python
    from pecos.quantum import X, Z

    # Constructor notation
    p = X(0) & Z(1)
    q = Z(0) & X(1)

    # Commutation check
    print(p.commutes_with(q))  # False (anticommute)
    ```

## Creating a Stabilizer Code

A `StabilizerCode` is defined by its stabilizer generators and a qubit count:

=== ":fontawesome-brands-rust: Rust"

    ```rust
    use pecos_qec::StabilizerCode;
    use pecos_core::pauli::*;

    // 3-qubit bit-flip repetition code: generators ZZI, IZZ
    let group = pecos_quantum::PauliStabilizerGroup::new(vec![
        Zs([0, 1]),
        Zs([1, 2]),
    ]).unwrap();

    let code = StabilizerCode::from_group(group);

    // Or use the built-in constructor:
    let code = StabilizerCode::repetition(3);
    ```

=== ":fontawesome-brands-python: Python"

    ```python
    from pecos_rslib import StabilizerCode

    code = StabilizerCode.repetition(3)
    ```

## Code Parameters

Once you have a code, query its parameters:

=== ":fontawesome-brands-rust: Rust"

    ```rust
    use pecos_qec::StabilizerCode;

    let code = StabilizerCode::steane();

    println!("n = {}", code.num_qubits());          // 7
    println!("k = {}", code.num_logical_qubits());   // 1
    println!("{}", code.code_parameters());           // [[7, 1]]
    ```

=== ":fontawesome-brands-python: Python"

    ```python
    from pecos_rslib import StabilizerCode

    code = StabilizerCode.steane()
    print(code.num_qubits())  # 7
    print(code.num_logical_qubits())  # 1
    print(code.code_parameters())  # [[7, 1]]
    ```

## Logical Operators

Compute a basis for the logical operator space:

```rust
use pecos_qec::StabilizerCode;

let code = StabilizerCode::steane();
let logicals = code.logical_operators();

// [[7, 1]] code has 2k = 2 independent logical directions
assert_eq!(logicals.len(), 2);

for (i, op) in logicals.iter().enumerate() {
    println!("Logical {i}: {op}");
}
```

The returned operators form a basis but are not paired into (X, Z) pairs. For paired logicals, use `StabilizerCodeSpec` (see below).

## Code Distance

Compute the minimum weight of a non-trivial logical operator:

```rust
use pecos_qec::StabilizerCode;

let code = StabilizerCode::five_qubit();
assert_eq!(code.distance(), Some(3));

let code = StabilizerCode::steane();
assert_eq!(code.distance(), Some(3));

// Returns None if there are no logical qubits
let trivial = StabilizerCode::repetition(2);
// (still has distance for this case)
assert_eq!(trivial.distance(), Some(1));
```

The distance computation is exact but exponential: O(2^(k+r)) where k is the number of logical operators and r is the stabilizer rank. Only use it for small codes (k + r <= 30).

## Syndrome Computation

Check which stabilizers an error anticommutes with:

```rust
use pecos_qec::StabilizerCode;
use pecos_core::pauli::*;

let code = StabilizerCode::repetition(3);

// X error on qubit 0: triggers first stabilizer (ZZI)
assert_eq!(code.syndrome(&X(0)), vec![true, false]);

// X error on qubit 1: triggers both (ZZI and IZZ)
assert_eq!(code.syndrome(&X(1)), vec![true, true]);

// Z error: commutes with all Z-stabilizers (undetectable)
assert_eq!(code.syndrome(&Z(0)), vec![false, false]);
```

## Standard Code Constructors

PECOS includes constructors for well-known codes:

| Constructor | Code | Parameters |
|---|---|---|
| `repetition(n)` | Bit-flip repetition | [[n, 1, 1]] |
| `steane()` | Steane (Hamming-based CSS) | [[7, 1, 3]] |
| `five_qubit()` | Perfect code | [[5, 1, 3]] |
| `shor()` | Shor code | [[9, 1, 3]] |
| `four_two_two()` | Error-detecting code | [[4, 2, 2]] |
| `toric(l)` | Toric code on L x L torus | [[2L^2, 2, L]] |

```rust
use pecos_qec::StabilizerCode;

let toric = StabilizerCode::toric(2);
assert_eq!(toric.num_qubits(), 8);        // 2 * 2^2
assert_eq!(toric.num_logical_qubits(), 2);
assert_eq!(toric.distance(), Some(2));
```

## Explicit Qubit Count

When stabilizer generators don't touch all qubits, the explicit `num_qubits` matters:

```rust
use pecos_qec::StabilizerCode;
use pecos_quantum::PauliStabilizerGroup;
use pecos_core::pauli::Zs;

// ZZ on qubits 0,1 -- but we declare 4 physical qubits
let group = PauliStabilizerGroup::new(vec![
    Zs([0, 1]),
]).unwrap();

let code = StabilizerCode::new(group, 4);

// k = n - rank = 4 - 1 = 3 logical qubits (not 1!)
assert_eq!(code.num_logical_qubits(), 3);
```

## StabilizerCodeSpec: Verified Code Definitions

For fault tolerance analysis, use `StabilizerCodeSpec`. This stores explicit paired logical operators and supports verification:

```rust
use pecos_qec::StabilizerCodeSpec;
use pecos_core::pauli::{Xs, Zs};

// Build a 3-qubit bit-flip code with explicit logicals
let code = StabilizerCodeSpec::builder(3)
    .check(Zs([0, 1]))
    .check(Zs([1, 2]))
    .logical_z(Zs([0, 1, 2]))
    .logical_x(Xs([0, 1, 2]))
    .build()
    .unwrap();

// Full verification: checks all commutation relations
code.verify().unwrap();

assert_eq!(code.num_qubits(), 3);
assert_eq!(code.num_logical_qubits(), 1);
```

### Converting StabilizerCode to StabilizerCodeSpec

You can convert from the lightweight `StabilizerCode` to the full `StabilizerCodeSpec`. This automatically discovers paired logical operators:

```rust
use pecos_qec::{StabilizerCode, StabilizerCodeSpec};

let code = StabilizerCode::steane();
let spec = StabilizerCodeSpec::from_stabilizer_code(&code).unwrap();

// Logicals were discovered and paired
assert_eq!(spec.logical_xs().len(), 1);
assert_eq!(spec.logical_zs().len(), 1);

// Full verification passes
spec.verify().unwrap();
```

## Fault Tolerance Analysis

`StabilizerCodeSpec` integrates with the fault tolerance checkers:

```rust
use pecos_qec::{StabilizerCodeSpec, StabilizerFlipChecker};
use pecos_core::pauli::{Xs, Zs};

let code = StabilizerCodeSpec::builder(7)
    .check(Xs([0, 2, 4, 6]))
    .check(Xs([1, 2, 5, 6]))
    .check(Xs([3, 4, 5, 6]))
    .check(Zs([0, 2, 4, 6]))
    .check(Zs([1, 2, 5, 6]))
    .check(Zs([3, 4, 5, 6]))
    .logical_z(Zs([0, 2, 4, 6]))
    .logical_x(Xs([0, 2, 4, 6]))
    .build()
    .unwrap();

// Check: can a weight-1 fault cause an undetectable logical error?
let checker = StabilizerFlipChecker::new(&code);
let analysis = checker.analyze_weight_with_types(1, true, true, true);
assert_eq!(analysis.undetectable_logical, 0);  // distance-3: safe against weight-1
```

## Architecture Summary

PECOS separates stabilizer code concerns into layers:

- **`PauliString`** (pecos-core): Individual Pauli operators with phase.
- **`PauliStabilizerGroup`** (pecos-quantum): Commuting Pauli group, purely algebraic.
- **`StabilizerCode`** (pecos-qec): Mathematical code definition, on-demand analysis.
- **`StabilizerCodeSpec`** (pecos-qec): Operational specification with verification and fault tolerance integration.

For architecture details, see `design/STABILIZER_CODE_ARCHITECTURE.md` in the repository root.
