# Gate Reference

This guide provides a comprehensive reference for all quantum gates supported by PECOS simulators.

## Setup

All examples in this guide use the following setup:

=== ":fontawesome-brands-python: Python"

    ```python
    from pecos.simulators import SparseStab

    # Create a stabilizer simulator with 5 qubits
    state = SparseStab(num_qubits=5)

    # Qubit indices for examples
    q = 0
    q0, q1 = 0, 1
    ```

=== ":fontawesome-brands-rust: Rust"

    <!--skip-->
    ```rust
    use pecos::prelude::*;
    use pecos::simulators::{SparseStab, CliffordGateable};

    let mut sim = SparseStab::new(5);
    let q = QubitId(0);
    let (q0, q1) = (QubitId(0), QubitId(1));
    ```

```hidden-python
from pecos.simulators import SparseStab

# Create a stabilizer simulator with 5 qubits
state = SparseStab(num_qubits=5)

# Qubit indices for examples
q = 0
q0, q1 = 0, 1
```

```hidden-rust
use pecos::prelude::*;
use pecos::simulators::{SparseStab, CliffordGateable};

fn main() {
    let mut sim = SparseStab::new(5);
    let q = QubitId(0);
    let q0 = QubitId(0);
    let q1 = QubitId(1);
    let q2 = QubitId(2);
    let control = QubitId(0);
    let target = QubitId(1);
    // CODE
}
```

## Overview

PECOS supports two categories of quantum gates:

- **Clifford Gates**: Gates that map Pauli operators to Pauli operators. These can be efficiently simulated using stabilizer simulators like `SparseStab`.
- **Non-Clifford Gates**: Rotation gates and other operations that require state vector simulation.

## Quick Reference

### Single-Qubit Gates

| Gate | Type | Description |
|------|------|-------------|
| I | Clifford | Identity |
| X, Y, Z | Clifford | Pauli gates |
| H | Clifford | Hadamard |
| SX, SY, SZ | Clifford | Square root of Pauli gates |
| SX†, SY†, SZ† | Clifford | Adjoint square root gates |
| F, F2, F3, F4 | Clifford | Face gates (cyclic Pauli permutations) |
| H2-H6 | Clifford | Hadamard variants |
| T, T† | Non-Clifford | π/8 phase gates |
| RX, RY, RZ | Non-Clifford | Arbitrary rotations |
| U | Non-Clifford | General single-qubit unitary |

### Two-Qubit Gates

| Gate | Type | Description |
|------|------|-------------|
| CX (CNOT) | Clifford | Controlled-X |
| CY | Clifford | Controlled-Y |
| CZ | Clifford | Controlled-Z |
| SWAP | Clifford | Swap two qubits |
| iSWAP | Clifford | Swap with phase |
| SXX, SYY, SZZ | Clifford | Square root of Pauli-Pauli interactions |
| G | Clifford | Two-qubit Clifford |
| RXX, RYY, RZZ | Non-Clifford | Two-qubit rotations |

### Measurements and Preparations

| Operation | Description |
|-----------|-------------|
| MX, MY, MZ | Measure in X, Y, Z basis |
| MNX, MNY, MNZ | Measure in -X, -Y, -Z basis |
| PX, PY, PZ | Prepare in +X, +Y, +Z eigenstate |
| PNX, PNY, PNZ | Prepare in -X, -Y, -Z eigenstate |
| MPX, MPY, MPZ | Measure and prepare in + eigenstate |
| MPNX, MPNY, MPNZ | Measure and prepare in - eigenstate |

## Setup

The examples below use a simulator instance. Run this setup code first:

=== ":fontawesome-brands-python: Python"

    ```python
    from pecos.simulators import SparseStab

    # Create a stabilizer simulator with 5 qubits
    state = SparseStab(num_qubits=5)

    # Qubit indices for examples
    q = 0
    q0, q1 = 0, 1
    ```

=== ":fontawesome-brands-rust: Rust"

    ```rust
    use pecos::prelude::*;
    use pecos::simulators::{SparseStab, CliffordGateable};

    let mut sim = SparseStab::new(5);
    let q = QubitId(0);
    let (q0, q1) = (QubitId(0), QubitId(1));
    ```

## Clifford Gates

The Clifford group consists of quantum operations that map Pauli operators to Pauli operators under conjugation. For a Clifford operation C and Pauli operator P:

```
C P C† = P'
```

where P' is another Pauli operator (possibly with a phase of ±1 or ±i).

### Pauli Gates

#### Identity (I)

The identity gate leaves the state unchanged.

**Pauli Transformation:**
```
X → X
Y → Y
Z → Z
```

**Matrix:**
```
I = [[1, 0],
     [0, 1]]
```

=== ":fontawesome-brands-python: Python"
    ```python
    state.run_gate("I", {q})
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    sim.identity(&[q]);
    ```

---

#### Pauli X

The X gate is equivalent to a classical NOT operation in the computational basis. It performs a π rotation around the X axis of the Bloch sphere.

**Pauli Transformation:**
```
X → X
Y → -Y
Z → -Z
```

**Matrix:**
```
X = [[0, 1],
     [1, 0]]
```

=== ":fontawesome-brands-python: Python"
    ```python
    state.run_gate("X", {q})
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    sim.x(&[q]);
    ```

---

#### Pauli Y

The Y gate performs a π rotation around the Y axis of the Bloch sphere.

**Pauli Transformation:**
```
X → -X
Y → Y
Z → -Z
```

**Matrix:**
```
Y = [[ 0, -i],
     [+i,  0]]
```

=== ":fontawesome-brands-python: Python"
    ```python
    state.run_gate("Y", {q})
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    sim.y(&[q]);
    ```

---

#### Pauli Z

The Z gate applies a phase flip in the computational basis.

**Pauli Transformation:**
```
X → -X
Y → -Y
Z → Z
```

**Matrix:**
```
Z = [[1,  0],
     [0, -1]]
```

=== ":fontawesome-brands-python: Python"
    ```python
    state.run_gate("Z", {q})
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    sim.z(&[q]);
    ```

---

### Hadamard Gates

#### Hadamard (H)

The Hadamard gate creates an equal superposition of basis states. It transforms between the X and Z bases.

**Pauli Transformation:**
```
X → Z
Y → -Y
Z → X
```

**Matrix:**
```
H = 1/√2 [[1,  1],
          [1, -1]]
```

=== ":fontawesome-brands-python: Python"
    ```python
    state.run_gate("H", {q})
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    sim.h(&[q]);
    ```

---

#### Hadamard Variants (H2-H6)

PECOS provides additional Hadamard-like gates that perform basis transformations in different planes of the Bloch sphere.

| Gate | Pauli Transformation | Matrix |
|------|---------------------|--------|
| H2 | X→-Z, Y→-Y, Z→-X | 1/√2 [[1, -1], [-1, 1]] |
| H3 | X→Y, Y→X, Z→-Z | 1/√2 [[1, i], [i, 1]] |
| H4 | X→-Y, Y→-X, Z→-Z | 1/√2 [[1, -i], [-i, 1]] |
| H5 | X→-X, Y→Z, Z→Y | 1/√2 [[-1, 1], [1, 1]] |
| H6 | X→-X, Y→-Z, Z→-Y | 1/√2 [[-1, -1], [-1, 1]] |

=== ":fontawesome-brands-rust: Rust"
    ```rust
    sim.h2(&[q]);  // H2 variant
    sim.h3(&[q]);  // H3 variant
    sim.h4(&[q]);  // H4 variant
    sim.h5(&[q]);  // H5 variant
    sim.h6(&[q]);  // H6 variant
    ```

---

### Square Root Gates

#### Square Root of X (SX)

The SX gate is equivalent to a π/2 rotation around the X axis of the Bloch sphere.

**Pauli Transformation:**
```
X → X
Y → -Z
Z → Y
```

**Matrix:**
```
SX = 1/2 [[1+i, 1-i],
          [1-i, 1+i]]
```

=== ":fontawesome-brands-python: Python"
    ```python
    state.run_gate("SX", {q})
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    sim.sx(&[q]);
    sim.sxdg(&[q]);  // Adjoint (inverse)
    ```

---

#### Square Root of Y (SY)

The SY gate is equivalent to a π/2 rotation around the Y axis of the Bloch sphere.

**Pauli Transformation:**
```
X → -Z
Y → Y
Z → X
```

**Matrix:**
```
SY = 1/√2 [[1, -1],
           [1,  1]]
```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    sim.sy(&[q]);
    sim.sydg(&[q]);  // Adjoint (inverse)
    ```

---

#### Square Root of Z (SZ / S Gate)

The SZ gate (also known as the S or P gate) is equivalent to a π/2 rotation around the Z axis of the Bloch sphere.

**Pauli Transformation:**
```
X → Y
Y → -X
Z → Z
```

**Matrix:**
```
SZ = [[1, 0],
      [0, i]]
```

=== ":fontawesome-brands-python: Python"
    ```python
    state.run_gate("SZ", {q})
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    sim.sz(&[q]);
    sim.szdg(&[q]);  // Adjoint (inverse): [[1, 0], [0, -i]]
    ```

---

### Face Gates

The Face gates perform cyclic permutations of the Pauli operators with various sign combinations.

#### Face Gate (F)

**Pauli Transformation:**
```
X → Y
Y → Z
Z → X
```

**Matrix:**
```
F = 1/√2 [[1, -i],
          [i,  1]]
```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    sim.f(&[q]);
    sim.fdg(&[q]);  // Adjoint: X→Z, Y→X, Z→Y
    ```

#### Face Gate Variants

| Gate | Pauli Transformation |
|------|---------------------|
| F2 | X→-Z, Y→-X, Z→Y |
| F2† | X→-Y, Y→Z, Z→-X |
| F3 | X→Y, Y→-Z, Z→-X |
| F3† | X→-Z, Y→X, Z→-Y |
| F4 | X→Z, Y→-Z, Z→-X |
| F4† | X→-Y, Y→Z, Z→-X |

=== ":fontawesome-brands-rust: Rust"
    ```rust
    sim.f2(&[q]);   sim.f2dg(&[q]);
    sim.f3(&[q]);   sim.f3dg(&[q]);
    sim.f4(&[q]);   sim.f4dg(&[q]);
    ```

---

### Two-Qubit Clifford Gates

#### Controlled-X (CX / CNOT)

The CX gate flips the target qubit if the control qubit is in state |1⟩.

**Pauli Transformation:**
```
XI → XX
IX → IX
ZI → ZI
IZ → ZZ
```

**Matrix:**
```
CX = [[1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 0, 1],
      [0, 0, 1, 0]]
```

=== ":fontawesome-brands-python: Python"
    ```python
    state.run_gate("CX", {(q0, q1)})
    # or
    state.run_gate("CNOT", {(q0, q1)})
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    sim.cx(&[(control, target)]);
    ```

---

#### Controlled-Y (CY)

The CY gate applies a Y operation on the target qubit if the control qubit is in state |1⟩.

**Pauli Transformation:**
```
XI → XY
IX → IX
ZI → ZI
IZ → ZZ
```

**Matrix:**
```
CY = [[1,  0,  0,  0],
      [0,  1,  0,  0],
      [0,  0,  0, -i],
      [0,  0, +i,  0]]
```

=== ":fontawesome-brands-python: Python"
    ```python
    state.run_gate("CY", {(q0, q1)})
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    sim.cy(&[(control, target)]);
    ```

---

#### Controlled-Z (CZ)

The CZ gate applies a phase of -1 when both qubits are in state |1⟩. It is symmetric under qubit exchange.

**Pauli Transformation:**
```
XI → XZ
IX → ZX
ZI → ZI
IZ → IZ
```

**Matrix:**
```
CZ = [[1,  0,  0,  0],
      [0,  1,  0,  0],
      [0,  0,  1,  0],
      [0,  0,  0, -1]]
```

=== ":fontawesome-brands-python: Python"
    ```python
    state.run_gate("CZ", {(q0, q1)})
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    sim.cz(&[(q1, q2)]);
    ```

---

#### SWAP

The SWAP gate exchanges the quantum states of two qubits.

**Pauli Transformation:**
```
XI → IX
IX → XI
ZI → IZ
IZ → ZI
```

**Matrix:**
```
SWAP = [[1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]]
```

=== ":fontawesome-brands-python: Python"
    ```python
    state.run_gate("SWAP", {(q0, q1)})
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    sim.swap(&[(q1, q2)]);
    ```

---

#### iSWAP

The iSWAP gate swaps states with an additional i phase on the swapped states.

**Pauli Transformation:**
```
XI → -ZY
IX → YZ
ZI → IZ
IZ → ZI
```

**Matrix:**
```
iSWAP = [[1, 0, 0, 0],
         [0, 0, i, 0],
         [0, i, 0, 0],
         [0, 0, 0, 1]]
```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    sim.iswap(&[(q1, q2)]);
    ```

---

#### Square Root of XX (SXX)

The SXX gate implements evolution under XX coupling for time π/4.

**Pauli Transformation:**
```
XI → XI
IX → IX
ZI → -YX
IZ → -XY
```

**Matrix:**
```
SXX = 1/√2 [[1,  0,  0, -i],
            [0,  1, -i,  0],
            [0, -i,  1,  0],
            [-i, 0,  0,  1]]
```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    sim.sxx(&[(q1, q2)]);
    sim.sxxdg(&[(q1, q2)]);  // Adjoint
    ```

---

#### Square Root of YY (SYY)

The SYY gate implements evolution under YY coupling for time π/4.

**Pauli Transformation:**
```
XI → -ZY
IX → -YZ
ZI → XY
IZ → YX
```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    sim.syy(&[(q1, q2)]);
    sim.syydg(&[(q1, q2)]);  // Adjoint
    ```

---

#### Square Root of ZZ (SZZ)

The SZZ gate implements evolution under ZZ coupling for time π/4.

**Pauli Transformation:**
```
XI → YZ
IX → ZY
ZI → ZI
IZ → IZ
```

**Matrix:**
```
SZZ = e^(-iπ/4) [[1,  0,  0,  0],
                 [0, -i,  0,  0],
                 [0,  0, -i,  0],
                 [0,  0,  0,  1]]
```

=== ":fontawesome-brands-python: Python"
    ```python
    state.run_gate("SZZ", {(q0, q1)})
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    sim.szz(&[(q1, q2)]);
    sim.szzdg(&[(q1, q2)]);  // Adjoint
    ```

---

#### G Gate

The G gate is a symmetric two-qubit Clifford that implements a particular permutation of single-qubit Paulis.

**Pauli Transformation:**
```
XI → IX
IX → XI
ZI → XZ
IZ → ZX
```

**Matrix:**
```
G = 1/2 [[ 1,  1,  1, -1],
         [ 1, -1,  1,  1],
         [ 1,  1, -1,  1],
         [-1,  1,  1,  1]]
```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    sim.g(&[(q1, q2)]);
    ```

---

## Non-Clifford Gates

```hidden-python
import pecos as pc
from pecos.simulators import StateVec

# StateVec supports non-Clifford gates
state = StateVec(num_qubits=5)
q = 0
q0, q1 = 0, 1
theta = pc.f64.pi / 4
phi = pc.f64.pi / 8
lam = pc.f64.pi / 6
```

```hidden-rust
use pecos::prelude::*;
use pecos::simulators::{StateVec, ArbitraryRotationGateable, CliffordGateable};
use std::f64::consts::PI;

fn main() {
    let mut sim = StateVec::new(5);
    let q = QubitId(0);
    let q0 = QubitId(0);
    let q1 = QubitId(1);
    let q2 = QubitId(2);
    let theta = Angle64::from_radians(PI / 4.0);
    let phi = Angle64::from_radians(PI / 8.0);
    let lam = Angle64::from_radians(PI / 6.0);
    // CODE
}
```

Non-Clifford gates include arbitrary rotation gates that cannot be efficiently simulated with stabilizer methods. These require state vector or other universal simulators.

**Setup for Non-Clifford Gates:**

=== ":fontawesome-brands-python: Python"

    ```python
    from pecos.simulators import StateVec

    # StateVec supports non-Clifford gates
    state = StateVec(num_qubits=5)
    q = 0
    q0, q1 = 0, 1
    ```

=== ":fontawesome-brands-rust: Rust"

    ```rust
    use pecos::prelude::*;
    use pecos::simulators::{StateVec, ArbitraryRotationGateable, CliffordGateable};
    use std::f64::consts::PI;

    let mut sim = StateVec::new(5);
    let q = QubitId(0);
    let (q0, q1) = (QubitId(0), QubitId(1));
    let theta = Angle64::from_radians(PI / 4.0);
    ```

### Single-Qubit Rotations

#### RX (X-axis Rotation)

Rotation around the X-axis by angle θ.

**Definition:** RX(θ) = exp(-i θ X/2) = cos(θ/2) I - i·sin(θ/2) X

**Matrix:**
```
RX(θ) = [[cos(θ/2),    -i·sin(θ/2)],
         [-i·sin(θ/2),  cos(θ/2)  ]]
```

=== ":fontawesome-brands-python: Python"
    ```python
    import pecos as pc

    state.run_gate("RX", {q}, angles=(pc.f64.pi / 4,))
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    sim.rx(theta, &[q]);
    ```

---

#### RY (Y-axis Rotation)

Rotation around the Y-axis by angle θ.

**Definition:** RY(θ) = exp(-i θ Y/2) = cos(θ/2) I - i·sin(θ/2) Y

**Matrix:**
```
RY(θ) = [[cos(θ/2), -sin(θ/2)],
         [sin(θ/2),  cos(θ/2)]]
```

=== ":fontawesome-brands-python: Python"
    ```python
    state.run_gate("RY", {q}, angles=(theta,))
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    sim.ry(theta, &[q]);
    ```

---

#### RZ (Z-axis Rotation)

Rotation around the Z-axis by angle θ.

**Definition:** RZ(θ) = exp(-i θ Z/2) = cos(θ/2) I - i·sin(θ/2) Z

**Matrix:**
```
RZ(θ) = [[e^(-iθ/2),     0     ],
         [    0,      e^(iθ/2) ]]
```

=== ":fontawesome-brands-python: Python"
    ```python
    state.run_gate("RZ", {q}, angles=(theta,))
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    sim.rz(theta, &[q]);
    ```

---

#### T Gate (π/8 Gate)

The T gate is a π/4 rotation around the Z-axis (equivalent to RZ(π/4)).

**Matrix:**
```
T = [[1,        0     ],
     [0, e^(iπ/4)     ]]
```

=== ":fontawesome-brands-python: Python"
    ```python
    state.run_gate("T", {q})
    state.run_gate("Tdg", {q})  # Adjoint
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    sim.t(&[q]);
    sim.tdg(&[q]);  // Adjoint (T†)
    ```

---

#### U Gate (General Single-Qubit Unitary)

The U gate is a general single-qubit unitary with three parameters.

**Definition:** U(θ, φ, λ) = RZ(φ) · RY(θ) · RZ(λ)

**Matrix:**
```
U(θ,φ,λ) = [[        cos(θ/2),      -e^(iλ)·sin(θ/2)],
            [e^(iφ)·sin(θ/2), e^(i(λ+φ))·cos(θ/2)]]
```

=== ":fontawesome-brands-python: Python"
    ```python
    state.run_gate("U", {q}, angles=(theta, phi, lam))
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    sim.u(theta, phi, lam, &[q]);
    ```

---

#### R1XY (X-Y Plane Rotation)

An X-Y plane rotation gate with a specified angle and axis.

=== ":fontawesome-brands-rust: Rust"
    ```rust
    sim.r1xy(theta, phi, &[q]);
    ```

---

### Two-Qubit Rotations

#### RXX (XX Rotation)

Two-qubit rotation implementing evolution under the XX interaction.

**Definition:** RXX(θ) = exp(-i θ XX/2)

=== ":fontawesome-brands-rust: Rust"
    ```rust
    sim.rxx(theta, &[(q1, q2)]);
    ```

---

#### RYY (YY Rotation)

Two-qubit rotation implementing evolution under the YY interaction.

**Definition:** RYY(θ) = exp(-i θ YY/2)

The YY coupling generates entanglement through the Y⊗Y interaction. For example, RYY(π/2) transforms:
- |00⟩ → (|00⟩ - i|11⟩)/√2
- |11⟩ → (|11⟩ - i|00⟩)/√2
- |01⟩ → (|01⟩ + i|10⟩)/√2
- |10⟩ → (|10⟩ + i|01⟩)/√2

=== ":fontawesome-brands-rust: Rust"
    ```rust
    sim.ryy(theta, &[(q1, q2)]);
    ```

---

#### RZZ (ZZ Rotation)

Two-qubit rotation implementing evolution under the ZZ interaction.

**Definition:** RZZ(θ) = exp(-i θ ZZ/2)

The ZZ coupling is diagonal in the computational basis:
- |00⟩ → e^(-iθ/2)|00⟩
- |11⟩ → e^(-iθ/2)|11⟩
- |01⟩ → e^(iθ/2)|01⟩
- |10⟩ → e^(iθ/2)|10⟩

**Matrix:**
```
RZZ(θ) = [[e^(-iθ/2),     0,          0,          0       ],
          [    0,      e^(iθ/2),      0,          0       ],
          [    0,         0,      e^(iθ/2),       0       ],
          [    0,         0,          0,      e^(-iθ/2)  ]]
```

=== ":fontawesome-brands-python: Python"
    ```python
    state.run_gate("RZZ", {(q0, q1)}, angles=(theta,))
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    sim.rzz(theta, &[(q1, q2)]);
    ```

---

## Measurements

### Z-Basis Measurement (MZ)

Projects the state into either |0⟩ or |1⟩.

**Semantics:**
- Outcome `false` (0): projected to |0⟩
- Outcome `true` (1): projected to |1⟩

=== ":fontawesome-brands-python: Python"
    ```python
    result = state.run_gate("MZ", {q})
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    let results = sim.mz(&[q]);
    // results[0].outcome: true if |1⟩, false if |0⟩
    // results[0].is_deterministic: true if already in eigenstate
    ```

---

### X-Basis Measurement (MX)

Projects the state into either |+⟩ or |-⟩, where |±⟩ = (|0⟩ ± |1⟩)/√2.

=== ":fontawesome-brands-rust: Rust"
    ```rust
    let results = sim.mx(&[q]);   // Measure +X
    let results = sim.mnx(&[q]);  // Measure -X
    ```

---

### Y-Basis Measurement (MY)

Projects the state into either |+i⟩ or |-i⟩, where |±i⟩ = (|0⟩ ± i|1⟩)/√2.

=== ":fontawesome-brands-rust: Rust"
    ```rust
    let results = sim.my(&[q]);   // Measure +Y
    let results = sim.mny(&[q]);  // Measure -Y
    ```

---

## State Preparations

### Prepare in Z Eigenstates

=== ":fontawesome-brands-rust: Rust"
    ```rust
    sim.pz(&[q]);   // Prepare |0⟩ (eigenstate of +Z)
    sim.pnz(&[q]);  // Prepare |1⟩ (eigenstate of -Z)
    ```

### Prepare in X Eigenstates

=== ":fontawesome-brands-rust: Rust"
    ```rust
    sim.px(&[q]);   // Prepare |+⟩ = (|0⟩ + |1⟩)/√2
    sim.pnx(&[q]);  // Prepare |-⟩ = (|0⟩ - |1⟩)/√2
    ```

### Prepare in Y Eigenstates

=== ":fontawesome-brands-rust: Rust"
    ```rust
    sim.py(&[q]);   // Prepare |+i⟩ = (|0⟩ + i|1⟩)/√2
    sim.pny(&[q]);  // Prepare |-i⟩ = (|0⟩ - i|1⟩)/√2
    ```

### Measure and Prepare (MP*)

These operations measure and then prepare the qubit in a specific eigenstate regardless of the measurement outcome.

=== ":fontawesome-brands-rust: Rust"
    ```rust
    sim.mpz(&[q]);   // Measure Z, prepare |0⟩
    sim.mpnz(&[q]);  // Measure -Z, prepare |1⟩
    sim.mpx(&[q]);   // Measure X, prepare |+⟩
    sim.mpnx(&[q]);  // Measure -X, prepare |-⟩
    sim.mpy(&[q]);   // Measure Y, prepare |+i⟩
    sim.mpny(&[q]);  // Measure -Y, prepare |-i⟩
    ```

---

## Simulator Compatibility

| Simulator | Clifford Gates | Non-Clifford Gates | Notes |
|-----------|---------------|-------------------|-------|
| **SparseStab** | All | None | Default, fastest for QEC |
| **StateVec** | All | All | Pure Rust state vector |
| **CuStateVec** | All | All | GPU-accelerated (requires CUDA) |
| **MPS** | All | All | Tensor network (requires CUDA) |
| **PauliProp** | All | None | Error propagation tracking |

For guidance on choosing a simulator, see the [Simulators](simulators.md) guide.

## See Also

- [Simulators](simulators.md) - Choosing the right simulation backend
- [QASM Simulation](qasm-simulation.md) - Running quantum circuits
- [Getting Started](getting-started.md) - Introduction to PECOS
