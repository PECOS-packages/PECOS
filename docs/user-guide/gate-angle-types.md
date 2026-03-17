# Gate and Angle Types

This guide covers the foundational types that underpin PECOS's quantum operator system: the phase hierarchy, exact angle arithmetic, gate type classification, and the gate registry for custom gate decomposition.

## Phase Types

PECOS uses a three-level phase type hierarchy, mirroring the mathematical structure:

```text
Sign  ⊂  QuarterPhase  ⊂  GlobalPhase
(+/-1)   (+/-1, +/-i)     (any e^{i*theta})
```

### Sign

The simplest phase: +1 or -1. Required for stabilizer group generators (every stabilizer must square to +I).

```rust
use pecos_core::Sign;

let plus = Sign::PlusOne;
let minus = Sign::MinusOne;

// Multiplication is XOR (extremely fast)
assert_eq!(plus * minus, Sign::MinusOne);
assert_eq!(minus * minus, Sign::PlusOne);
```

### QuarterPhase

Fourth roots of unity: {+1, -1, +i, -i}. The natural closure of single-qubit Pauli multiplication (X * Y = iZ, etc.).

```rust
use pecos_core::QuarterPhase;

let phase = QuarterPhase::PlusI;
assert!(!phase.is_real());

// Multiplication
assert_eq!(QuarterPhase::PlusI * QuarterPhase::PlusI, QuarterPhase::MinusOne);

// Conjugation
assert_eq!(QuarterPhase::PlusI.conjugate(), QuarterPhase::MinusI);
```

Every `PauliString` carries a `QuarterPhase`. Every `Sign` widens to `QuarterPhase` losslessly; narrowing from `QuarterPhase` to `Sign` fails on imaginary phases.

### GlobalPhase

Arbitrary phase e^{i*theta}. Internally stored as either a `QuarterPhase` (for exact fourth-roots) or an `Angle64` (for general angles).

```rust
use pecos_core::phase::GlobalPhase;
use pecos_core::Angle64;

// Quarter phases stored exactly
let p = GlobalPhase::i();
assert_eq!(p.as_quarter(), Some(pecos_core::QuarterPhase::PlusI));

// Arbitrary phase
let p = GlobalPhase::from_angle(Angle64::from_turns(0.125)); // e^{i*pi/4}
```

## Angle Types

PECOS uses fixed-point angle arithmetic to avoid floating-point errors common in quantum computing.

### Angle64

The primary angle type. Represents angles as fixed-point fractions of a full turn (2*pi), not radians. This means multiples of pi/4 (T gate), pi/2 (S gate), etc. are represented exactly.

```rust
use pecos_core::Angle64;

// Named constants
let zero = Angle64::ZERO;              // 0
let quarter = Angle64::QUARTER_TURN;   // pi/2
let half = Angle64::HALF_TURN;         // pi
let full = Angle64::FULL_TURN;         // 2*pi (wraps to 0)

// From turns (fractional rotations)
let eighth = Angle64::from_turns(0.125);  // pi/4

// Exact rational construction
let third = Angle64::from_turn_ratio(1, 3);  // 2*pi/3

// Arithmetic (wrapping modular)
assert_eq!(quarter + quarter, half);
assert_eq!(half + half, zero);  // wraps around

// Conversion to radians (for interop)
let radians: f64 = quarter.to_radians();
```

### Angle Macros

Two macros provide compile-time exact angle construction:

```rust
use pecos_core::{angle, turn};

// angle! -- pi-based syntax
let a = angle!(pi);         // pi (half turn)
let a = angle!(pi / 2);    // pi/2 (quarter turn)
let a = angle!(pi / 4);    // pi/4 (T gate angle)
let a = angle!(2 * pi / 3); // 2*pi/3

// turn! -- turn-based syntax (often more intuitive)
let a = turn!(1 / 8);  // 1/8 turn = pi/4 (T gate)
let a = turn!(1 / 4);  // 1/4 turn = pi/2 (S gate)
let a = turn!(1 / 2);  // 1/2 turn = pi (Z gate)
```

### Phase Macros for Gate Expressions

The `phase!` and `phase_turn!` macros wrap angles into `PhaseValue` for use with operator expressions:

```rust
use pecos_core::{phase, phase_turn};
use pecos_core::unitary_rep::X;

// Global phase applied to a gate
let op = phase!(pi / 4) * X(0);      // e^{i*pi/4} * X
let op = phase_turn!(1 / 8) * X(0);  // same thing
```

### Other Angle Sizes

For memory-constrained applications, smaller angle types are available:

| Type | Bits | Precision | Use case |
|---|---|---|---|
| `Angle8` | 8 | 1/256 turn | Coarse angles, minimal memory |
| `Angle16` | 16 | 1/65536 turn | Moderate precision |
| `Angle32` | 32 | ~10^-10 turn | Good precision |
| `Angle64` | 64 | ~10^-19 turn | Default, nearly exact |
| `Angle128` | 128 | ~10^-38 turn | Extreme precision |

## GateType

The `GateType` enum classifies quantum gates for circuit representation and simulation dispatch. It is FFI-friendly (`#[repr(u8)]`) and supports string parsing.

### Gate Categories

**Single-qubit Clifford:**
`I`, `X`, `Y`, `Z`, `H`, `SX`, `SXdg`, `SY`, `SYdg`, `SZ`, `SZdg`, `F`, `Fdg`

**Two-qubit Clifford:**
`CX`, `CY`, `CZ`, `SWAP`, `SXX`, `SXXdg`, `SYY`, `SYYdg`, `SZZ`, `SZZdg`, `ISWAP`, `ISWAPdg`

**Parameterized (non-Clifford):**
`RX`, `RY`, `RZ`, `RXX`, `RYY`, `RZZ`, `CRZ`, `T`, `Tdg`, `U`, `R1XY`

**Three-qubit:**
`CCX` (Toffoli)

**Measurement/Preparation:**
`MZ`, `PZ`

**Lifecycle:**
`QAlloc`, `QFree`, `Idle`

### Gate Introspection

```rust
use pecos_core::GateType;

let gate = GateType::CX;
assert_eq!(gate.quantum_arity(), 2);  // two-qubit gate
assert!(!gate.is_parameterized());    // no angle parameter
assert!(gate.is_two_qubit());

let gate = GateType::RZ;
assert_eq!(gate.angle_arity(), 1);    // one angle parameter
assert!(gate.is_parameterized());

// String parsing (case-insensitive, with aliases)
let gate: GateType = "CNOT".parse().unwrap();  // alias for CX
let gate: GateType = "H".parse().unwrap();
```

## Gate Registry

The `GateRegistry` allows defining custom gates as decompositions into base gates. Decomposition is lazy -- it happens at simulation time, not circuit construction.

### Defining Custom Gates

```rust
use pecos_core::{GateRegistry, GateDefinitionBuilder, GateType, AngleSource, Angle64};

let mut registry = GateRegistry::new();

// Define a custom "RZX" gate that decomposes to H-CX-RZ-CX-H
GateDefinitionBuilder::new("RZX", 2)
    .angle_arity(1)
    .step(GateType::H, &[1])                              // H on target
    .step(GateType::CX, &[0, 1])                          // CX
    .step_with_angles(GateType::RZ, &[1], &[AngleSource::Input(0)])  // RZ(theta)
    .step(GateType::CX, &[0, 1])                          // CX
    .step(GateType::H, &[1])                              // H on target
    .build()
    .register_into(&mut registry);

assert!(registry.contains("RZX"));
```

### Angle Sources

When decomposing parameterized gates, `AngleSource` specifies where each angle comes from:

- `AngleSource::Input(i)` -- forward the i-th input angle
- `AngleSource::Fixed(angle)` -- use a fixed angle constant
- `AngleSource::NegInput(i)` -- negate the i-th input angle

### Python API

```python
from pecos_rslib import GateRegistry, GateDefBuilder, AngleSource

registry = GateRegistry()

# Define a custom gate
(
    GateDefBuilder()
    .define("RZX", quantum_arity=2)
    .angle_arity(1)
    .step("H", [1])
    .step("CX", [0, 1])
    .step_with_angles("RZ", [1], [AngleSource.input(0)])
    .step("CX", [0, 1])
    .step("H", [1])
    .register_into(registry)
)

# Decompose
steps = registry.decompose("RZX", [0, 1], [0.5])
```

## How These Types Connect

The gate and angle types form the foundation for the operator algebra:

- `Angle64` is used by `UnitaryRep::Rotation` for rotation gate angles
- `QuarterPhase` is carried by every `PauliString` for exact Pauli algebra
- `GateType` is used by `UnitaryRep::Gate` for named gate nodes in expression trees
- `GlobalPhase` is used by `UnitaryRep::Phase` for arbitrary global phases
- `GateRegistry` decomposes custom gates into sequences of base `GateType` operations during simulation

For how these feed into the operator type system, see the [Quantum Operator Algebra guide](quantum-operator-algebra.md).
