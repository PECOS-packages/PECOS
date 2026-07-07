# Zlup Tutorial: Getting Started

This tutorial will guide you through writing your first Zlup quantum programs. By the end, you'll understand the core concepts: allocators, gates, measurement, and control flow.

## What is Zlup?

Zlup is the **low-level complement to Guppy** in PECOS. While Guppy provides a high-level, Pythonic experience for QEC researchers, Zlup is designed for:

- **Quantum orchestration**: Gate sequences, syndrome extraction, correction application
- **Rust integration**: Calling decoders and simulation backends via FFI
- **Reliable, production-grade code** following NASA Power of 10 principles
- **Simulation infrastructure**: Noise modeling, error injection, state tracking

Complex classical algorithms (MWPM decoders, etc.) belong in **Rust**—Zlup handles the quantum side and calls Rust for heavy computation.

If you're a QEC researcher who prefers Python, **use Guppy**. Zlup is for the systems layer connecting quantum operations to classical backends.

## Prerequisites

- Zlup compiler installed (`cargo install --path .` from the zlup directory)
- Basic understanding of quantum computing concepts (qubits, gates, measurement)

## Your First Program: Hello Quantum

Create a file called `hello.zlp`:

```zlup
/// My first Zlup program - creates a superposition state
pub fn main() -> unit {
    // Allocate a single qubit
    q := qalloc(1);

    // Prepare (reset) the qubit to |0⟩
    pz q;

    // Apply Hadamard gate to create superposition
    h q[0];

    // Measure the qubit
    result: u1 = mz(u1) q[0];

    return;
}
```

Compile it:

```bash
zlup compile hello.zlp -o hello.json
```

Let's break down what each line does.

## Allocators: Managing Qubits

In Zlup, qubits are managed through **allocators**. This explicit resource management (inspired by Zig) ensures you always know where your qubits come from.

```zlup_fragment
// Allocate 4 qubits
q := qalloc(4);

// Access individual qubits by index
h q[0];      // First qubit
h q[1];      // Second qubit
h q[3];      // Fourth qubit (last)
```

The allocator capacity is fixed at compile time - you can't dynamically grow it. This is intentional: bounded resources make programs predictable and analyzable.

### Child Allocators

For complex algorithms, you can partition allocators:

```zlup_fragment
// Main allocator with 10 qubits
mut main := qalloc(10);

// Create child allocators from it
data := main.child(4);      // First 4 qubits for data
ancilla := main.child(6);   // Remaining 6 for ancillas
```

Note: The parent must be declared `mut` to create children.

## Preparing Qubits

Before using qubits, prepare (reset) them to the |0⟩ state with `pz`:

```zlup_fragment
q := qalloc(4);

// Prepare all qubits at once
pz q;

// Or prepare specific qubits
pz q[0];
pz {q[1], q[2]};  // Batch prepare
```

## Gates: Quantum Operations

### Single-Qubit Gates

```zlup_fragment
// Pauli gates
x q[0];    // X (NOT) gate
y q[0];    // Y gate
z q[0];    // Z gate

// Hadamard
h q[0];    // Creates superposition

// T gates (fourth root of Z)
t q[0];    // T gate
tdg q[0];  // T-dagger (inverse of T)

// Square root gates (sz is the S gate)
sx q[0];   // sqrt(X)
sy q[0];   // sqrt(Y)
sz q[0];   // sqrt(Z) - this is the S gate
sxdg q[0]; // sqrt(X) dagger
sydg q[0]; // sqrt(Y) dagger
szdg q[0]; // sqrt(Z) dagger - this is S-dagger
```

### Rotation Gates (Parameterized)

Rotation gates require an angle with explicit units:

```zlup_nocheck
std := @import("std");

// Preferred: use turns (native unit)
rx(1/8 turns) q[0];              // Rotate by 1/8 turn (pi/4 rad) around X
ry(1/4 turns) q[0];              // Rotate by 1/4 turn (pi/2 rad) around Y
rz(1/2 turns) q[0];              // Rotate by 1/2 turn (pi rad) around Z

// Or use a64 constants
rz(std.a64.t_angle turns) q[0];  // T-gate (1/8 turn)

// Or use radians with f64 constants
rx(std.f64.pi/4 rad) q[0];       // Same as 1/8 turns
```

### Two-Qubit Gates

Two-qubit gates use tuple syntax for (control, target):

```zlup_fragment
// CNOT (controlled-X)
cx (q[0], q[1]);    // q[0] controls, q[1] is target

// Other controlled gates
cy (q[0], q[1]);    // Controlled-Y
cz (q[0], q[1]);    // Controlled-Z
ch (q[0], q[1]);    // Controlled-Hadamard

// Swap gates
swap (q[0], q[1]);
iswap (q[0], q[1]);

// Ising gates
sxx (q[0], q[1]);   // sqrt(XX)
syy (q[0], q[1]);   // sqrt(YY)
szz (q[0], q[1]);   // sqrt(ZZ)

// Parameterized two-qubit
rzz(1/8 turns) (q[0], q[1]);
```

### Batch Operations

Apply the same gate to multiple qubits in parallel using set syntax:

```zlup_fragment
// Apply H to multiple qubits (order doesn't matter)
h {q[0], q[1], q[2]};

// Batch two-qubit gates
cx {(q[0], q[1]), (q[2], q[3])};

// Batch rotation
rx(1/4 turns) {q[0], q[1], q[2]};
```

## Measurement

Measurement extracts classical information from qubits:

```zlup_fragment
// Measure single qubit into a u1 (single bit)
r: u1 = mz(u1) q[0];

// Measure multiple qubits into an array
results: [4]u1 = mz([4]u1) [q[0], q[1], q[2], q[3]];

// Pack measurements into a byte
syndrome: u8 = mz(pack u8) [q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7]];
```

The `pack` modifier packs bits sequentially into the target type.

## Emitting Results

To output values from your quantum program, use `result()`:

```zlup_fragment
// Emit a single measurement
result("final_bit", r);

// Emit array of measurements
result("syndrome", results);

// Use namespaced tags for organization
result("qec/round_1/syndrome", syndrome);
```

Results are collected by the quantum runtime and never elided. Entry/main functions
return `unit` and use explicit `result()` calls rather than returning values.

## Creating a Bell State

Let's create a Bell state - the simplest entangled state:

```zlup
/// Creates a Bell state |00⟩ + |11⟩ (unnormalized)
pub fn main() -> unit {
    q := qalloc(2);
    pz q;

    // Create superposition on first qubit
    h q[0];

    // Entangle with second qubit
    cx (q[0], q[1]);

    // Measure both qubits
    r0: u1 = mz(u1) q[0];
    r1: u1 = mz(u1) q[1];

    // Emit results - r0 and r1 will always be correlated:
    // both 0 or both 1
    result("qubit_0", r0);
    result("qubit_1", r1);

    return;
}
```

## Variables and Bindings

Zlup uses Pascal/Go-style binding syntax:

```zlup_fragment
// Immutable binding (most common)
x := 42;           // Type inferred
y: u32 = 100;      // Type explicit

// Mutable binding
mut count := 0;
count = count + 1;

// Constants
pi := 3.14159;     // Immutable by default
```

## Control Flow

### If Statements

```zlup_nocheck
// Simple if
if x > 10 {
    // do something
}

// If-else
if condition {
    // true branch
} else {
    // false branch
}

// If-else if chain
if x == 0 {
    // zero
} else if x < 0 {
    // negative
} else {
    // positive
}
```

### For Loops (Bounded)

All loops in Zlup must have bounded iteration (NASA Power of 10 rule):

```zlup_nocheck
// Range loop
for i in 0..10 {
    h q[i];
}

// Iterating with index
for i, item in items {
    // use i and item
}
```

### Switch Statements

```zlup_nocheck
switch (value) {
    0 => { /* handle 0 */ },
    1, 2 => { /* handle 1 or 2 */ },
    3..10 => { /* handle range 3-10 */ },
    else => { /* default */ },
}
```

## Functions

```zlup_nocheck
// Simple function
fn add(a: u32, b: u32) -> u32 {
    return a + b;
}

// Function with no return value
fn apply_gates(q: *Allocator) -> unit {
    h q[0];
    return;
}

// Public function (exported)
pub fn main() -> unit {
    result := add(2, 3);
    return;
}
```

## Working with the Standard Library

Import and use standard library modules:

```zlup_nocheck
std := @import("std");

pub fn main() -> unit {
    // Angle constants from std.a64
    angle := std.a64.t_angle;

    q := qalloc(8);
    pz q;

    // Apply rotation (using turns)
    rz(angle turns) q[0];

    // Measure and compute parity
    syndrome: u8 = mz(pack u8) [q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7]];
    parity := std.parity_u8(syndrome);

    return;
}
```

## Tick Blocks: Parallel Execution

Group operations that should execute in parallel:

```zlup_fragment
q := qalloc(4);
pz q;

// These gates execute in the same time step
tick {
    h q[0];
    h q[1];
    h q[2];
    h q[3];
}

// Next time step
tick {
    cx (q[0], q[1]);
    cx (q[2], q[3]);
}
```

## Error Handling

Zlup distinguishes between **faults** (physical layer, quantum hardware) and **errors**
(logical layer, classical software). This distinction is crucial for QEC:

- **Faults** are expected hardware imperfections - collected and analyzed
- **Errors** are unexpected software problems - stop execution immediately

### Quick Example

```zlup_nocheck
// Classical errors stop execution
fn divide(a: u32, b: u32) -> DivError!u32 {
    if b == 0 {
        return error.DivisionByZero;
    }
    return a / b;
}

// Handle errors with catch
result := divide(10, 2) catch 0;  // Default to 0 on error

// Quantum faults are collected (QEC pattern)
QuantumFault := fault { Leakage, QubitLoss };

fn qec_round(q: []qubit) try -> []QuantumFault!Syndrome {
    cx (q[0], q[1]);  // Fault? Recorded, continues
    cx (q[1], q[2]);  // Fault? Recorded, continues
    return mz([2]u1) [q[3], q[4]];
}

// Caller receives both faults and result
faults, syndrome := qec_round(q);
```

### Two Error Handling Modes

| Mode | Quantum Fault | Classical Error |
|------|---------------|-----------------|
| `try!` | Stops immediately | Stops immediately |
| `try` | Collected, continues | Stops execution |

For a comprehensive guide to error handling including practical QEC examples, fault
promotion, and the explicit handling philosophy, see the
**[Error Handling Tutorial](tutorial-error-handling.md)**.

## Complete Example: GHZ State

A GHZ state is a maximally entangled state of N qubits:

```zlup_nocheck
/// Creates a 4-qubit GHZ state: |0000⟩ + |1111⟩
pub fn main() -> unit {
    std := @import("std");

    q := qalloc(4);
    pz q;

    // Put first qubit in superposition
    h q[0];

    // Entangle each subsequent qubit
    for i in 1..4 {
        cx (q[0], q[i]);
    }

    // Measure all qubits
    results: [4]u1 = mz([4]u1) [q[0], q[1], q[2], q[3]];

    // Emit results - all will be the same: either all 0 or all 1
    result("measurements", results);

    return;
}
```

## Next Steps

- Read the [Standard Library Reference](stdlib.md) for available functions
- Check out the [examples](../examples/) directory for more programs
- See [design.md](design.md) for language design rationale
- Try `zlup eval "2 + 2"` for quick expression testing

## Quick Reference

| Concept | Syntax |
|---------|--------|
| Allocate qubits | `q := qalloc(N);` |
| Prepare qubits | `pz q;` |
| Single gate | `h q[0];` |
| Rotation gate | `rz(angle) q[0];` |
| Two-qubit gate | `cx (q[0], q[1]);` |
| Batch gates | `h {q[0], q[1]};` |
| Measure one | `r: u1 = mz(u1) q[0];` |
| Measure many | `r: [N]u1 = mz([N]u1) [...];` |
| Pack measure | `r: u8 = mz(pack u8) [...];` |
| Emit result | `result("tag", value);` |
| Immutable var | `x := value;` |
| Mutable var | `mut x := value;` |
| For loop | `for i in 0..N { }` |
| If statement | `if cond { } else { }` |
| Function | `fn name(args) -> T { }` |
| Import | `std := @import("std");` |
