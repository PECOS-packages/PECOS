# Levels of Abstraction in QEC

QEC can be described at multiple levels, each with its own concerns:

```
Level 4: Programs
    Dynamic circuits, adaptive protocols, real-time decoding
    ↓
Level 3: Circuits
    Syndrome extraction circuits, gate sequences, fault locations
    ↓
Level 2: Geometry
    Qubit layout, check schedules, connectivity constraints
    ↓
Level 1: Abstract Code
    Stabilizer generators, logical operators, code parameters [[n,k,d]]
```

## Level 1: Abstract Stabilizer Codes

At the most abstract level, a stabilizer code is defined purely algebraically:

- **Stabilizer group S**: Abelian subgroup of the Pauli group
- **Code space**: Joint +1 eigenspace of all stabilizers
- **Logical operators**: Paulis that commute with S but aren't in S
- **Code parameters**: [[n, k, d]] notation

This level answers questions like:
- What are the code's stabilizer generators?
- What logical operators exist?
- What is the code distance?
- Which errors are detectable vs. cause logical errors?

```rust
use pecos_qec::{StabilizerCode, StabilizerFlipChecker};
use pecos_core::{Xs, Zs};

// Define a code abstractly
let code = StabilizerCode::builder(7)
    .check(Xs([0, 2, 4, 6]))  // X-type stabilizers
    .check(Xs([1, 2, 5, 6]))
    .check(Xs([3, 4, 5, 6]))
    .check(Zs([0, 2, 4, 6]))  // Z-type stabilizers
    .check(Zs([1, 2, 5, 6]))
    .check(Zs([3, 4, 5, 6]))
    .logical_x(Xs([0, 1, 2, 3, 4, 5, 6]))
    .logical_z(Zs([0, 1, 2, 3, 4, 5, 6]))
    .build()
    .unwrap();

// Analyze code properties - no circuit needed
let checker = StabilizerFlipChecker::new(&code);
let distance = checker.compute_distance(5);  // [[7,1,3]] Steane code
```

**Key insight**: At this level, we can determine fault tolerance properties without any circuit - just by checking which errors anti-commute with which stabilizers.

## Level 2: Code Geometry

Geometry bridges abstract codes and physical implementations. It can be viewed two ways:

**Abstract geometry**: Logical structure of how stabilizers overlap
- Which qubits participate in which checks
- Check scheduling (parallel vs. sequential measurement)
- Logical operator support

**Physical geometry**: Actual layout on a device
- 2D grid, heavy-hex, etc.
- Qubit connectivity constraints
- Physical distance between qubits

```rust
use pecos_qec::SurfaceCode;

// Surface code has inherent 2D geometry
let surface = SurfaceCode::rotated(3);  // Distance-3 rotated surface code

// Geometry determines check schedule
let schedule = surface.check_schedule();
```

The geometry level is where abstract codes meet physical constraints. A [[7,1,3]] code is abstract; the Steane code with its specific qubit arrangement is geometric.

## Level 3: Circuits

Circuits are explicit gate sequences that implement QEC operations:

- **State preparation**: Initialize logical qubits
- **Syndrome extraction**: Measure stabilizers via ancilla qubits
- **Logical gates**: Implement operations on encoded data
- **Measurement**: Read out logical information

This is "textbook QEC" - static circuits that repeat in a fixed pattern.

```rust
use pecos_quantum::TickCircuit;

// Syndrome extraction circuit for 3-qubit code
let mut circuit = TickCircuit::new();
circuit.tick().pz(&[3, 4]);           // Prepare ancillas
circuit.tick().cx(&[(0, 3)]);          // CNOT from data to ancilla
circuit.tick().cx(&[(1, 3)]);
circuit.tick().cx(&[(1, 4)]);
circuit.tick().cx(&[(2, 4)]);
circuit.tick().mz(&[3, 4]);            // Measure ancillas
```

At this level, we analyze:
- Where can faults occur? (spacetime locations)
- How do faults propagate through gates?
- Does the circuit preserve fault tolerance?

## Level 4: Programs (Beyond Textbook QEC)

Modern and future QEC goes beyond static circuits:

**Dynamic circuits**: Measurement outcomes control subsequent operations
```
measure ancilla -> if syndrome != 0: apply correction -> continue
```

**Adaptive protocols**: Strategy changes based on observed errors
```
if high error rate detected: switch to more conservative decoding
```

**Real-time decoding**: Decoder runs concurrently with quantum operations
```
stream syndromes to decoder -> receive corrections -> apply mid-circuit
```

**Lattice surgery**: Dynamic code deformations for logical gates
```
merge codes -> measure boundary -> split codes (logical CNOT)
```

This level requires:
- Tracking stabilizer state through conditional operations
- Handling measurement-dependent control flow
- Reasoning about programs, not just circuits

**Why stabilizer-level analysis matters here**: When operations are conditional, measurements may or may not occur. Tracking "which stabilizers are flipped" is more fundamental than "which measurements fired" because stabilizer state persists regardless of whether we measure.

## Code Validation vs. Circuit Verification

These are related but distinct concerns:

**Code Validation** (Level 1):
- Do stabilizers commute?
- Do logical operators anti-commute properly?
- What is the code distance?
- Which errors are correctable?

**Circuit Verification** (Level 3):
- Does this specific circuit correctly extract syndromes?
- Do gate-level faults propagate to cause logical errors?
- Is the circuit t-fault tolerant?

A valid code can have a non-fault-tolerant circuit implementation. Circuit verification ensures the implementation preserves the code's theoretical properties.

## Module Mapping

| Module | Level | Purpose |
|--------|-------|---------|
| `stabilizer_code` | 1 | Define and verify stabilizer codes |
| `distance` | 1 | Calculate code distance |
| `logical_discovery` | 1 | Discover logical operators |
| `geometry` | 2 | Physical layout structures |
| `surface` | 2 | Surface code implementations |
| `fault_tolerance` | 3-4 | Fault tolerance analysis |
