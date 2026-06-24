# Custom Gate Design Notes

> **Status:** Design exploration (February 2026)

## Summary

This document defines Zlup's gate system. The core design decisions are:

### All Quantum Operations Are Target-Provided

Zlup has **no built-in quantum operations**. Everything that touches qubits - gates,
preparation, measurement - is declared and provided by the compilation target.

- `std.gates` declares the "standard" set (`h`, `cx`, `pz`, `mz`, etc.)
- Targets implement what they support
- Compile error if you use an unsupported operation
- **No fallbacks, no implicit decomposition**

### Two Types of Gates

| Type | Syntax | Provider | Use Case |
|------|--------|----------|----------|
| **Target** | `declare gate name(...)` | Hardware/simulator/noise model | Native operations |
| **Composite** | `gate name(...) { body }` | Zlup code | Abstractions, subroutines |

### Only Two Built-ins

| Built-in | Purpose |
|----------|---------|
| `qalloc` | Resource allocation (doesn't touch quantum state) |
| `result` | Output emission (classical) |

Everything else requires `use std.gates.*` or custom declarations.

### Composite Gates Are Full Subroutines

Composites can include preparation, measurement, classical logic, control flow,
and return values:

```zlup_nocheck
gate measure_reset(q: qubit) -> u1 {
    r := mz(u1) q;
    if r == 1 { x q; }
    return r;
}
```

### Compile-Time Target Validation

```bash
zlup compile program.zlp --target trapped_ion
```

The compiler validates all gate usage against the target's gate set. Composites
are validated recursively.

### IDE Support via Project Config

```toml
# zlup.toml
[target]
default = "trapped_ion"
```

IDE reads config and validates accordingly. `@import("target")` gives access to
current target's definitions.

---

## Table of Contents

- [Core Principle: All Gates Are Target-Provided](#core-principle-all-gates-are-target-provided)
- [Two Types of Gates](#two-types-of-gates)
- [Why This Design](#why-this-design)
- [Proposed Syntax](#proposed-syntax)
- [Noise Model Integration](#noise-model-integration)
- [Composite Gate Generalization](#composite-gate-generalization)
- [Compilation Model](#compilation-model)
- [IDE / Language Server Support](#ide--language-server-support)
- [Open Questions](#open-questions)

---

## Core Principle: All Gates Are Target-Provided

Zlup has **no built-in gates**. Every gate - including "standard" gates like `h`, `cx`,
`rz` - is declared and must be provided by the compilation target.

The standard library (`std.gates`) declares the common gate set that most targets support.
But these are not special - they use the same `declare gate` mechanism as any other gate.

```zlup_nocheck
// In std/gates.zlp - the "standard" operation set
// These are declarations, not implementations

// Preparation (reset)
pub declare gate pz(q: qubit);  // Prepare in |0⟩ (Z basis)
pub declare gate px(q: qubit);  // Prepare in |+⟩ (X basis)
pub declare gate py(q: qubit);  // Prepare in |+i⟩ (Y basis)

// Measurement
pub declare gate mz(q: qubit) -> u1;  // Measure in Z basis
pub declare gate mx(q: qubit) -> u1;  // Measure in X basis
pub declare gate my(q: qubit) -> u1;  // Measure in Y basis

// Single-qubit gates
pub declare gate h(q: qubit);
pub declare gate x(q: qubit);
pub declare gate y(q: qubit);
pub declare gate z(q: qubit);
pub declare gate t(q: qubit);
pub declare gate tdg(q: qubit);
pub declare gate sx(q: qubit);
pub declare gate sy(q: qubit);
pub declare gate sz(q: qubit);

// Parameterized single-qubit
pub declare gate rx(theta: a64)(q: qubit);
pub declare gate ry(theta: a64)(q: qubit);
pub declare gate rz(theta: a64)(q: qubit);

// Two-qubit gates
pub declare gate cx(ctrl: qubit, tgt: qubit);
pub declare gate cy(ctrl: qubit, tgt: qubit);
pub declare gate cz(ctrl: qubit, tgt: qubit);
pub declare gate swap(a: qubit, b: qubit);

// Parameterized two-qubit
pub declare gate rzz(theta: a64)(a: qubit, b: qubit);
pub declare gate crz(theta: a64)(ctrl: qubit, tgt: qubit);

// Three-qubit
pub declare gate ccx(a: qubit, b: qubit, c: qubit);
```

**Targets must implement these operations.** A target that doesn't support `h` or
`pz` will fail at compile time when code uses them.

## Two Types of Gates

### Target Gates (Declared)

Only the signature is declared. The target provides the implementation.

**Who provides implementations:**
- **Hardware**: Native gates the device supports
- **Simulator**: Matrix implementations, optimized algorithms
- **Noise model**: Gates with specific error characteristics

```zlup_nocheck
// User declares additional target gates beyond std.gates
declare gate ms(theta: a64)(a: qubit, b: qubit);
declare gate sqrt_iswap(a: qubit, b: qubit);
```

**Key property:** No Zlup-level implementation exists. If target doesn't support
the gate, compilation fails. No fallbacks, no hidden decomposition.

### Composite Gates (Defined)

Defined in Zlup as sequences of target gates (from `std.gates` or custom declarations)
or other composite gates.

```zlup_nocheck
// Uses target gates from std.gates (h, cx, rz)
gate rzx(theta: a64)(ctrl: qubit, tgt: qubit) {
    h tgt;
    cx(ctrl, tgt);
    rz(theta) tgt;
    cx(ctrl, tgt);
    h tgt;
}

// Uses custom target gate (ms) - only works on targets that support ms
gate ion_entangle(a: qubit, b: qubit) {
    ms(1/4 turns) (a, b);
}

// Uses another composite gate
gate double_rzx(theta: a64)(ctrl: qubit, tgt: qubit) {
    rzx(theta) (ctrl, tgt);
    rzx(theta) (ctrl, tgt);
}
```

**Key property:** Zlup knows the decomposition. Composite gates are portable across
any target that supports their constituent target gates. A composite using only
`std.gates` works on any standard-compliant target. A composite using `ms` only
works on targets that provide `ms`.

## Why This Design?

### Explicit Over Implicit (Zig-style, No Prelude)

Following Zig, Zlup has **no prelude**. Nothing is automatically imported.

**Built into the language (always available):**
- Primitive types: `u8`, `i32`, `bool`, `f64`, `a64`, `qubit`, `unit`, etc.
- Keywords: `if`, `for`, `fn`, `gate`, `declare`, `pub`, etc.
- Built-in functions: `@import`, `@size_of`, `@type_info`, etc.
- Resource management: `qalloc` (allocation, not a quantum operation)
- Output: `result` (classical emission, not a quantum operation)

**Requires explicit import (target-provided):**
- All quantum operations: `h`, `cx`, `rz`, `pz`, `mz`, etc.
- Library functions: `std.math`, `std.bits`, etc.

```zlup_nocheck
std := @import("std");
use std.gates.*;  // h, cx, rz, pz, mz, ...

pub fn main() -> unit {
    q := qalloc(2);   // built-in (resource allocation)
    pz q;             // imported (target-provided)
    h q[0];           // imported (target-provided)
    cx (q[0], q[1]);  // imported (target-provided)
    r := mz(u1) q[0]; // imported (target-provided)
    result("out", r); // built-in (output)
    return;
}
```

The distinction: `qalloc` and `result` are resource/IO operations that don't touch
quantum state. Everything that interacts with qubits (gates, prep, measurement) is
target-provided because targets implement these differently and noise models need
to reason about them.

This means you can always answer "where did this come from?" by looking at imports.

### Fail Fast

If a target doesn't support a gate you use, compilation fails immediately with a
clear error message. No silent fallbacks or hidden decompositions that might
introduce unexpected behavior or performance characteristics.

### Target Flexibility

Different targets have different native gate sets:
- Trapped ion: `ms`, `gpi`, `gpi2`
- Superconducting: `sqrt_iswap`, `sycamore`
- Photonic: `beamsplitter`, `phase_shift`

Rather than trying to support all of these as "built-ins", each target declares
what it supports. Users declare additional gates as needed.

### Noise Model Control

Noise models are targets too. A noise model might:
- Provide standard gates with calibrated error rates
- Provide synthetic gates for testing (`perfect_cx`, `very_noisy_cx`)
- Reject gates it doesn't have noise data for

## Proposed Syntax

### Importing Standard Gates

All quantum operations (including `pz` and `mz`) must be imported before use:

```zlup_nocheck
// Import all standard operations
std := @import("std");
use std.gates.*;  // pz, mz, h, x, cx, rz, etc.

pub fn main() -> unit {
    q := qalloc(2);
    pz q;             // from std.gates
    h q[0];           // from std.gates
    cx (q[0], q[1]);  // from std.gates
    r := mz(u1) q[0]; // from std.gates
    result("out", r);
    return;
}
```

Or import selectively:

```zlup_nocheck
std := @import("std");
use std.gates.{pz, mz, h, cx};  // only import what you need

pub fn main() -> unit {
    q := qalloc(2);
    pz q;
    h q[0];
    cx (q[0], q[1]);
    r := mz(u1) q[0];
    result("out", r);
    // x q[0];  // Error: 'x' not imported
    return;
}
```

Or use qualified names:

```zlup_nocheck
std := @import("std");

pub fn main() -> unit {
    q := qalloc(2);
    std.gates.pz q;
    std.gates.h q[0];
    std.gates.cx (q[0], q[1]);
    r := std.gates.mz(u1) q[0];
    result("out", r);
    return;
}
```

### Composite Gates

```zlup_nocheck
// Basic composite gate with angle parameter
gate rzx(theta: a64)(ctrl: qubit, tgt: qubit) {
    h tgt;
    cx (ctrl, tgt);
    rz(theta) tgt;
    cx (ctrl, tgt);
    h tgt;
}

// Fixed gate (no parameters)
gate echo(q: qubit) {
    x q;
    x q;
}

// Multi-qubit gate
gate toffoli(a: qubit, b: qubit, c: qubit) {
    h c;
    cx (b, c); tdg c;
    cx (a, c); t c;
    cx (b, c); tdg c;
    cx (a, c);
    t b; t c; h c;
    cx (a, b);
    t a; tdg b;
    cx (a, b);
}

// Public gate (exported from module)
pub gate logical_h(data: [9]qubit) {
    inline for i in 0..9 {
        h data[i];
    }
}
```

### Target Gates (Declared)

```zlup_nocheck
// Declare a gate the target must provide
declare gate ms(theta: a64)(a: qubit, b: qubit);

// No angle parameters
declare gate sqrt_swap(a: qubit, b: qubit);

// Three-qubit native gate
declare gate native_toffoli(a: qubit, b: qubit, c: qubit);

// With target hint (documentation/validation)
@target("trapped_ion")
declare gate ms(theta: a64)(a: qubit, b: qubit);

@target("simulator:pecos")
declare gate optimized_toffoli(a: qubit, b: qubit, c: qubit);

@target("noise_model:depolarizing")
declare gate noisy_cx(ctrl: qubit, tgt: qubit);
```

### Usage

```zlup_nocheck
std := @import("std");
use std.gates.*;  // pz, mz, h, cx, rz, ...

// Declare additional target gate
declare gate ms(theta: a64)(a: qubit, b: qubit);

// Define composite gate (uses std.gates)
gate rzx(theta: a64)(ctrl: qubit, tgt: qubit) {
    h tgt;
    cx (ctrl, tgt);
    rz(theta) tgt;
    cx (ctrl, tgt);
    h tgt;
}

gate echo(q: qubit) {
    x q;
    x q;
}

pub fn main() -> unit {
    q := qalloc(4);
    pz q;

    // Use standard gate
    h q[0];

    // Use composite gate
    rzx(1/4 turns) (q[0], q[1]);

    // Use target gate (target must support)
    ms(1/8 turns) (q[2], q[3]);

    // Batch syntax works too
    echo {q[0], q[1], q[2]};

    // Measure
    r := mz(u1) q[0];
    result("out", r);

    return;
}
```

## Noise Model Integration

The key question: how should noise models treat composite gates?

### Option A: Atomic by Default

Composite gates are treated as single units for noise purposes. The noise model
sees "one rzx gate" not "h, cx, rz, cx, h".

```zlup_nocheck
// Default: noise model treats as atomic
gate rzx(theta: a64)(ctrl: qubit, tgt: qubit) {
    h tgt;
    cx (ctrl, tgt);
    rz(theta) tgt;
    cx (ctrl, tgt);
    h tgt;
}

// Override: apply noise to each constituent
@decomposed
gate debug_rzx(theta: a64)(ctrl: qubit, tgt: qubit) {
    // Same body, but noise applied per-gate
}
```

**Rationale:** Most custom gates are defined precisely because users want to
reason about them as units. Debugging/analysis can use `@decomposed`.

### Option B: Decomposed by Default

Noise applied to each constituent gate. Mark atomic explicitly.

```zlup_nocheck
// Default: noise per constituent
gate rzx(theta: a64)(ctrl: qubit, tgt: qubit) { ... }

// Override: treat as single unit
@atomic
gate atomic_rzx(theta: a64)(ctrl: qubit, tgt: qubit) { ... }
```

**Rationale:** More conservative; explicit atomicity prevents surprises.

### Option C: Noise Model Decides

Gate definition doesn't specify; noise model configuration lists atomic gates.

```json
// noise_config.json
{
  "atomic_gates": ["rzx", "logical_h", "echo"],
  "decomposed_gates": ["debug_toffoli"]
}
```

**Rationale:** Same gate can be atomic or decomposed depending on simulation needs.

### Recommendation

**Option A (atomic by default)** with **Option C (noise model override)**.

- Gates defined with `gate` keyword are atomic by default
- `@decomposed` attribute forces per-constituent noise
- Noise model config can override either direction

```zlup_nocheck
// Atomic by default
gate rzx(theta: a64)(ctrl: qubit, tgt: qubit) { ... }

// Force decomposed (ignores noise model config)
@decomposed
gate always_decomposed(q: qubit) { ... }

// Force atomic (ignores noise model config)
@atomic
gate always_atomic(q: qubit) { ... }
```

## Inlining Behavior

Separate from noise, inlining affects optimization:

```zlup_nocheck
// Default: compiler decides based on size/usage
gate small_gate(q: qubit) { x q; }

// Force inline (always expand at call site)
@inline
gate must_inline(q: qubit) { ... }

// Prevent inline (preserve structure in output)
@noinline
gate preserve_structure(q: qubit) { ... }
```

**Interaction with noise:**
- `@inline` + atomic noise = inline then apply atomic noise
- `@noinline` + decomposed noise = keep structure, noise per constituent

## Arity and Type Checking

The compiler tracks gate signatures for type checking:

```zlup_nocheck
gate rzx(theta: a64)(ctrl: qubit, tgt: qubit) { ... }
//       ^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//       1 angle     2 qubits

// Valid
rzx(1/4 turns) (q[0], q[1]);

// Error: wrong number of angle parameters
rzx (q[0], q[1]);  // Missing angle

// Error: wrong number of qubits
rzx(1/4 turns) (q[0]);  // Needs 2 qubits

// Error: type mismatch
rzx(42) (q[0], q[1]);  // 42 is not an angle
```

### Target Gate Validation

For target gates, validation happens at codegen time:

```zlup_nocheck
declare gate ms(theta: a64)(a: qubit, b: qubit);

// At compile time: type-checked against declaration
ms(1/4 turns) (q[0], q[1]);  // OK

// At codegen time: check if target supports "ms"
// Error if target doesn't have "ms" with matching signature
```

**Target capabilities:**

Different targets support different gates:

| Target | Example Supported Gates |
|--------|------------------------|
| Hardware (trapped ion) | `ms`, `gpi`, `gpi2` |
| Hardware (superconducting) | `sqrt_iswap`, `sycamore` |
| Simulator (statevector) | Any gate with provided matrix |
| Noise model | Gates with calibrated noise parameters |

The `@target` hint helps catch mismatches early:

```zlup_nocheck
@target("trapped_ion")
declare gate ms(theta: a64)(a: qubit, b: qubit);

// Compiling with --target superconducting:
// Warning: gate 'ms' declared for 'trapped_ion' but targeting 'superconducting'
```

## AST Representation

```rust
/// Custom gate definition (composite)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateDecl {
    /// Gate name
    pub name: String,
    /// Angle/value parameters (e.g., theta: a64)
    pub params: Vec<GateParam>,
    /// Qubit parameters with names
    pub qubits: Vec<QubitParam>,
    /// Gate body (sequence of statements)
    pub body: Block,
    /// Whether this gate is public
    pub is_pub: bool,
    /// Noise behavior
    pub noise_mode: NoiseMode,
    /// Inlining behavior
    pub inline_mode: InlineMode,
    pub location: Option<SourceLocation>,
}

/// Target gate (provided by compilation target)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetGate {
    /// Gate name (must match target's implementation)
    pub name: String,
    /// Angle/value parameters
    pub params: Vec<GateParam>,
    /// Qubit count and names
    pub qubits: Vec<QubitParam>,
    /// Optional target hint (hardware, simulator, noise_model)
    pub target_hint: Option<String>,
    pub location: Option<SourceLocation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateParam {
    pub name: String,
    pub ty: Type,  // Usually Type::Angle (a64)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QubitParam {
    pub name: String,
    // Future: could support qubit arrays [N]qubit
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub enum NoiseMode {
    #[default]
    Atomic,      // Treat as single unit for noise
    Decomposed,  // Apply noise to each constituent
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub enum InlineMode {
    #[default]
    Auto,        // Compiler decides
    Always,      // @inline
    Never,       // @noinline
}
```

## SLR Output

Composite gates can be output in two ways:

### Expanded (Default)

Gate body is inlined at each call site:

```json
{
  "type": "GateOp",
  "kind": "H",
  "targets": [{"allocator": "q", "index": 1}]
},
{
  "type": "GateOp",
  "kind": "CX",
  "targets": [
    {"allocator": "q", "index": 0},
    {"allocator": "q", "index": 1}
  ]
}
// ... etc
```

### Preserved Structure

When `@noinline` or for backends that support custom gates:

```json
{
  "type": "CustomGateOp",
  "name": "rzx",
  "params": [{"type": "angle", "value": 0.25, "unit": "turns"}],
  "targets": [
    {"allocator": "q", "index": 0},
    {"allocator": "q", "index": 1}
  ],
  "atomic_noise": true
}
```

### Target Gates

Always output as target gate ops (no body to expand):

```json
{
  "type": "TargetGateOp",
  "name": "ms",
  "params": [{"type": "angle", "value": 0.125, "unit": "turns"}],
  "targets": [
    {"allocator": "q", "index": 0},
    {"allocator": "q", "index": 1}
  ],
  "target_hint": "trapped_ion"
}
```

The target (hardware, simulator, or noise model) interprets this gate according to
its own implementation.

## Examples

### QEC: Logical Gates

```zlup_nocheck
std := @import("std");
use std.gates.{h, cx, pz};

// Distance-3 surface code logical Hadamard
pub gate logical_h(data: [9]qubit) {
    // Transversal H
    inline for i in 0..9 {
        h data[i];
    }
}

// Logical CNOT between two codes
pub gate logical_cx(ctrl: [9]qubit, tgt: [9]qubit) {
    inline for i in 0..9 {
        cx (ctrl[i], tgt[i]);
    }
}

pub fn main() -> unit {
    mut base := qalloc(18);
    code1 := base.child(9);
    code2 := base.child(9);

    pz code1;
    pz code2;

    logical_h(code1);
    logical_cx(code1, code2);

    return;
}
```

### Target: Hardware (Trapped Ion)

```zlup_nocheck
std := @import("std");
use std.gates.pz;

// Mølmer-Sørensen gate (trapped ion native)
@target("trapped_ion")
declare gate ms(theta: a64)(a: qubit, b: qubit);

// Global rotation (all ions)
@target("trapped_ion")
declare gate gpi(theta: a64)(q: qubit);

pub fn ion_bell_state() -> unit {
    q := qalloc(2);
    pz q;

    // Native trapped-ion Bell state preparation
    ms(1/4 turns) (q[0], q[1]);

    return;
}
```

### Target: Simulator

```zlup_nocheck
std := @import("std");
use std.gates.pz;

// Simulator provides optimized implementation
@target("simulator")
declare gate optimized_toffoli(a: qubit, b: qubit, c: qubit);

// Simulator-specific controlled rotation (avoids decomposition)
@target("simulator:pecos")
declare gate crx(theta: a64)(ctrl: qubit, tgt: qubit);

pub fn use_simulator_gates() -> unit {
    q := qalloc(3);
    pz q;

    // Simulator applies this directly (no decomposition overhead)
    optimized_toffoli(q[0], q[1], q[2]);

    return;
}
```

### Target: Noise Model

```zlup_nocheck
std := @import("std");
use std.gates.pz;

// Noise model provides gate with calibrated error characteristics
@target("noise_model:device_xyz")
declare gate calibrated_cx(ctrl: qubit, tgt: qubit);

// Noise model might define entirely synthetic gates for testing
@target("noise_model:test")
declare gate perfect_cx(ctrl: qubit, tgt: qubit);  // No noise
@target("noise_model:test")
declare gate very_noisy_cx(ctrl: qubit, tgt: qubit);  // High error rate

pub fn noise_comparison() -> unit {
    q := qalloc(4);
    pz q;

    // Compare different noise characteristics
    calibrated_cx(q[0], q[1]);  // Realistic noise from calibration data
    perfect_cx(q[2], q[3]);     // Idealized for debugging

    return;
}
```

### Dynamical Decoupling

```zlup_nocheck
std := @import("std");
use std.gates.{h, x, y, pz};

// Echo sequence for noise suppression
@atomic  // Noise model sees this as one "echo" operation
gate echo_xy(q: qubit) {
    x q;
    y q;
    x q;
    y q;
}

// CPMG sequence
@atomic
gate cpmg(n: comptime usize)(q: qubit) {
    inline for _ in 0..n {
        x q;
        x q;
    }
}

pub fn protected_operation() -> unit {
    q := qalloc(1);
    pz q;

    h q[0];
    echo_xy q[0];  // Protect during idle time
    h q[0];

    return;
}
```

### Debugging with Decomposed Noise

```zlup_nocheck
std := @import("std");
use std.gates.{h, cx, rz};

// Atomic noise (default)
gate rzx(theta: a64)(ctrl: qubit, tgt: qubit) {
    h tgt;
    cx (ctrl, tgt);
    rz(theta) tgt;
    cx (ctrl, tgt);
    h tgt;
}

// Same gate, but see noise on each component
@decomposed
gate debug_rzx(theta: a64)(ctrl: qubit, tgt: qubit) {
    h tgt;
    cx (ctrl, tgt);
    rz(theta) tgt;
    cx (ctrl, tgt);
    h tgt;
}

// Compare noise behavior
pub fn compare_noise() -> unit {
    q := qalloc(4);
    pz q;

    // Atomic noise
    rzx(1/4 turns) (q[0], q[1]);

    // Per-gate noise (for analysis)
    debug_rzx(1/4 turns) (q[2], q[3]);

    return;
}
```

## Composite Gate Generalization

Composite gates need to be more than just sequences of other gates. Real quantum
operations often include preparation, measurement, classical logic, and feedforward.

### What Composites Need to Support

```zlup_nocheck
std := @import("std");
use std.gates.*;

// Measure-and-reset: returns measurement, leaves qubit in |0⟩
gate measure_reset(q: qubit) -> u1 {
    r := mz(u1) q;
    if r == 1 {
        x q;  // flip back to |0⟩
    }
    return r;
}

// Teleportation protocol
gate teleport(psi: qubit, epr0: qubit, epr1: qubit) -> (u1, u1) {
    // Bell measurement
    cx (psi, epr0);
    h psi;
    m1 := mz(u1) psi;
    m2 := mz(u1) epr0;

    // Feedforward corrections
    if m2 == 1 { x epr1; }
    if m1 == 1 { z epr1; }

    return (m1, m2);
}

// Repeat-until-success T gate
gate rus_t(q: qubit, ancilla: qubit) -> bool {
    for attempt in 0..10 {  // bounded attempts
        pz ancilla;
        h ancilla;
        t ancilla;
        cx (q, ancilla);
        h ancilla;

        r := mz(u1) ancilla;
        if r == 0 {
            return true;  // success
        }
        // Undo partial operation and retry
        sz q;  // correction
    }
    return false;  // failed after max attempts
}

// Syndrome extraction round
gate extract_syndrome(data: [4]qubit, ancilla: qubit) -> u1 {
    pz ancilla;
    h ancilla;

    inline for i in 0..4 {
        cx (ancilla, data[i]);
    }

    h ancilla;
    return mz(u1) ancilla;
}
```

### What This Means

Composite gates can include:

| Feature | Example | Purpose |
|---------|---------|---------|
| Target gates | `h q; cx (a, b);` | Core operations |
| Preparation | `pz ancilla;` | Initialize qubits mid-circuit |
| Measurement | `r := mz(u1) q;` | Extract classical information |
| Return values | `-> u1`, `-> (u1, u1)` | Return measurement results |
| Classical variables | `r := mz(...); count += 1;` | Track state |
| Control flow | `if r == 1 { x q; }` | Feedforward corrections |
| Bounded loops | `for i in 0..n { ... }` | Repeated operations |
| Other composites | `measure_reset(q);` | Composition |

### Gate vs Function: What's the Difference?

If gates can do all this, how are they different from functions?

| Aspect | `gate` | `fn` |
|--------|--------|------|
| **Semantics** | "This is a quantum operation" | General code |
| **Noise model** | Treated as unit (by default) | No special treatment |
| **Target override** | Target can provide native impl | No target override |
| **Inlining** | Controlled by `@atomic`/`@decomposed` | Normal inlining rules |
| **Scheduling** | May have timing implications | No timing semantics |

The key distinction: a `gate` declaration says "this is a quantum operation that
targets and noise models should reason about". A function is just code.

```zlup_nocheck
// Gate: noise model can treat as atomic "teleport" operation
gate teleport(psi: qubit, epr0: qubit, epr1: qubit) -> (u1, u1) { ... }

// Function: just code, no special quantum semantics
fn run_teleportation_experiment(shots: u32) -> unit { ... }
```

### Noise Model Implications

When a composite gate includes measurement and feedforward:

**Atomic mode (`@atomic` or default):**
- Noise model sees one "teleport" operation
- Applies noise characteristic of the whole operation
- Internal structure hidden from noise model

**Decomposed mode (`@decomposed`):**
- Noise model sees each constituent operation
- `cx`, `h`, `mz`, conditional `x`, conditional `z` each get noise
- Full visibility into structure

```zlup_nocheck
// Atomic: noise model sees "measure_reset" as one operation
gate measure_reset(q: qubit) -> u1 { ... }

// Decomposed: noise model sees mz + conditional x
@decomposed
gate debug_measure_reset(q: qubit) -> u1 { ... }
```

## Compilation Model

When compiling Zlup code, you specify a target. The target declares what gates it
supports. The compiler validates your code against the target's gate set.

### Target Gate Sets

Each target provides a gate set definition:

```
# trapped_ion.target
gates:
  - pz, px
  - mz, mx
  - gpi, gpi2
  - ms
  - rz

# superconducting.target
gates:
  - pz
  - mz
  - h, x, y, z
  - rx, ry, rz
  - cx, cz
  - sqrt_iswap

# simulator_pecos.target
gates:
  - [all of std.gates]
  - optimized_toffoli
  - any_unitary  # simulator can do arbitrary unitaries
```

### Compile-Time Validation

```bash
# Compile for trapped ion target
zlup compile program.zlp --target trapped_ion
```

The compiler:
1. Loads target's gate set
2. Checks every gate usage in your code
3. Errors if you use a gate the target doesn't support

```zlup_nocheck
std := @import("std");
use std.gates.*;

pub fn main() -> unit {
    q := qalloc(2);
    pz q;
    h q[0];           // Error on trapped_ion: 'h' not supported
    cx (q[0], q[1]);  // Error on trapped_ion: 'cx' not supported
    return;
}
```

```
error: gate 'h' not supported by target 'trapped_ion'
  --> program.zlp:7:5
   |
 7 |     h q[0];
   |     ^ target 'trapped_ion' does not provide 'h'
   |
   = help: trapped_ion supports: pz, px, mz, mx, gpi, gpi2, ms, rz
   = help: consider decomposing 'h' using supported gates
```

### Composites Are Validated Recursively

When you define a composite gate, the compiler checks that all gates it uses are
supported by the target:

```zlup_nocheck
// This composite uses h and cx
gate bell(a: qubit, b: qubit) {
    h a;
    cx (a, b);
}

pub fn main() -> unit {
    q := qalloc(2);
    pz q;
    bell(q[0], q[1]);  // Error: bell uses 'h' and 'cx', not supported
    return;
}
```

The error traces through:
```
error: gate 'h' not supported by target 'trapped_ion'
  --> program.zlp:4:5
   |
 4 |     h a;
   |     ^ in composite gate 'bell'
   |
  --> program.zlp:11:5
   |
11 |     bell(q[0], q[1]);
   |     ^^^^ called here
```

### Writing Portable Code

To write code that works on multiple targets:

**Option 1: Use only common gates**
```zlup_nocheck
// Only use gates that all your targets support
// This limits expressiveness but maximizes portability
```

**Option 2: Target-specific modules**
```zlup_nocheck
// trapped_ion/bell.zlp
gate bell(a: qubit, b: qubit) {
    ms(1/4 turns) (a, b);
    // ... ion-native implementation
}

// superconducting/bell.zlp
gate bell(a: qubit, b: qubit) {
    h a;
    cx (a, b);
}

// main.zlp - import based on target
bell := @import(@target_name() ++ "/bell.zlp");
```

**Option 3: Comptime target checking**
```zlup_nocheck
gate bell(a: qubit, b: qubit) {
    if (comptime @target_has("ms")) {
        ms(1/4 turns) (a, b);
    } else if (comptime @target_has("cx")) {
        h a;
        cx (a, b);
    } else {
        @compile_error("No known bell implementation for this target");
    }
}
```

### No Implicit Fallbacks

The compiler never silently substitutes one gate for another. If you write `h q[0]`
and the target doesn't support `h`, you get a compile error. Period.

This is intentional:
- **Explicit**: You know exactly what gates your code uses
- **Predictable**: No surprise decompositions affecting performance/noise
- **Auditable**: Code review can verify target compatibility

## IDE / Language Server Support

The IDE needs to know which target you're developing for to provide proper
errors, autocomplete, and diagnostics. Several approaches:

### Option A: Project Configuration

Like Rust's `Cargo.toml`, a project config specifies the default target:

```toml
# zlup.toml
[project]
name = "my_qec_code"

[target]
default = "trapped_ion"

# Optional: define multiple targets for the project
[target.trapped_ion]
definition = "targets/trapped_ion.toml"

[target.simulator]
definition = "targets/pecos_simulator.toml"
```

The IDE reads `zlup.toml` and uses the default target for validation. This is
similar to how rust-analyzer reads `Cargo.toml`.

### Option B: Import Target Gate Set

Make the target's gate set explicitly importable:

```zlup_nocheck
// Import current target's definitions (set by zlup.toml or CLI)
target := @import("target");
use target.gates.*;  // IDE knows exactly which gates are available

pub fn main() -> unit {
    q := qalloc(2);
    pz q;       // IDE: ✓ available in target
    h q[0];     // IDE: ✗ not available in trapped_ion (red squiggle)
    return;
}
```

Or import a specific target explicitly (useful during development):

```zlup_nocheck
// Explicitly develop against trapped_ion
ion := @import("targets/trapped_ion");
use ion.gates.*;

pub fn main() -> unit {
    q := qalloc(2);
    pz q;
    ms(1/4 turns) (q[0], q[1]);  // IDE knows ms is available
    return;
}
```

### Option C: Builtin Target Info (Zig-style)

Like Zig's `@import("builtin")`:

```zlup_nocheck
builtin := @import("builtin");

// Access target info at comptime
const target_name = builtin.target.name;  // "trapped_ion"
const has_cx = builtin.target.has_gate("cx");  // false

// Conditional compilation
if (comptime builtin.target.has_gate("cx")) {
    cx (q[0], q[1]);
} else {
    // Alternative implementation
}
```

### Option D: Per-File Target Annotation

For files that are target-specific:

```zlup_nocheck
//! target: trapped_ion
//! This module contains trapped-ion specific implementations

std := @import("std");
use std.gates.*;

// IDE knows to validate against trapped_ion
pub gate ion_bell(a: qubit, b: qubit) {
    ms(1/4 turns) (a, b);
}
```

### Recommended Approach

Combine several mechanisms:

1. **Project config (`zlup.toml`)**: Default target for the project, IDE reads this
2. **Explicit target import**: `target := @import("target")` makes dependencies clear
3. **Per-file annotations**: Override for target-specific files
4. **Builtin info**: `@import("builtin").target` for comptime decisions

```zlup_nocheck
// Most files: use project default via @import("target")
target := @import("target");
use target.gates.*;

// Target-specific file: annotate at top
//! target: trapped_ion
ion := @import("targets/trapped_ion");
use ion.gates.*;

// Portable code: check at comptime
builtin := @import("builtin");
if (comptime builtin.target.has_gate("ms")) {
    // Use native MS gate
} else {
    // Use decomposition
}
```

### How Zig and Rust Handle This

**Zig:**
```zig
const builtin = @import("builtin");
const native_arch = builtin.cpu.arch;

if (native_arch == .x86_64) {
    // x86-specific code
}
```
- Build system (`build.zig`) specifies target
- IDE reads build.zig
- `@import("builtin")` gives target info in code

**Rust:**
```rust
#[cfg(target_arch = "x86_64")]
fn optimized_impl() { ... }

#[cfg(not(target_arch = "x86_64"))]
fn optimized_impl() { ... }
```
- `Cargo.toml` + `.cargo/config.toml` specify target
- rust-analyzer reads config
- `cfg` attributes for conditional compilation
- `cfg!` macro for runtime checks (but resolved at compile time)

### Target Definition Files

Targets are defined in TOML files:

```toml
# targets/trapped_ion.toml
[target]
name = "trapped_ion"
description = "Generic trapped ion quantum computer"

[gates]
# Preparation and measurement
prep = ["pz", "px"]
meas = ["mz", "mx"]

# Native single-qubit
single = ["gpi", "gpi2", "rz"]

# Native two-qubit
two = ["ms"]

# Not natively supported (would need decomposition)
# h, x, y, z, cx, cy, cz - not listed, so not available
```

```toml
# targets/pecos_simulator.toml
[target]
name = "pecos_simulator"
description = "PECOS statevector simulator"

[gates]
# Simulator supports everything in std.gates
include = ["std.gates.*"]

# Plus some simulator-specific operations
extra = ["any_unitary", "state_snapshot", "density_matrix"]
```

The IDE loads the target definition and validates accordingly.

## Open Questions

1. **Qubit array parameters:** Should gates support `[N]qubit` parameters?
   ```zlup_nocheck
   gate transversal_h(data: [N]qubit) { ... }  // Generic over N?
   gate logical_h(data: [9]qubit) { ... }      // Fixed size?
   ```

2. **Classical parameters:** Should gates support non-angle parameters?
   ```zlup_nocheck
   gate conditional_x(condition: bool)(q: qubit) {
       if condition { x q; }
   }
   ```

3. **Return values:** Should gates return measurement results?
   ```zlup_nocheck
   gate measure_reset(q: qubit) -> u1 {
       result := mz(u1) q;
       pz q;
       return result;
   }
   ```
   Or should this remain a function, not a gate?

4. **Gate composition:** Should gates call other custom gates?
   ```zlup_nocheck
   gate double_echo(q: qubit) {
       echo q;
       echo q;
   }
   ```

5. **Verification:** Should we allow assertions about gate behavior?
   ```zlup_nocheck
   @unitary_check  // Verify at comptime that this is unitary
   gate my_gate(q: qubit) { ... }
   ```

6. **Target capability checking:** Compile error at codegen time if target doesn't
   support the gate. No fallbacks - explicit failure is the only option.

7. **Multiple target hints:** What if code uses gates from multiple targets?
   ```zlup_nocheck
   @target("trapped_ion")
   declare gate ms(theta: a64)(a: qubit, b: qubit);

   @target("superconducting")
   declare gate sqrt_iswap(a: qubit, b: qubit);

   // Compile error if targeting trapped_ion and using sqrt_iswap
   // Clear message: "gate 'sqrt_iswap' declared for 'superconducting'
   //                 but compiling for 'trapped_ion'"
   ```

8. **Composite portability:** Should composites be allowed to use target-specific gates?
   ```zlup_nocheck
   // This composite only works on trapped_ion targets
   gate ion_bell(a: qubit, b: qubit) {
       ms(1/4 turns) (a, b);  // ms is trapped_ion specific
   }
   // Should compiler warn? Infer target requirement? Just fail at codegen?
   ```

## Implementation Plan

### Phase 1: Gate Declaration Infrastructure
- Add `declare gate` syntax to parser
- Move standard gates to `std/gates.zlp` as declarations
- Target capability registry (what gates each target supports)
- Compile-time validation: error if target doesn't support used gate

### Phase 2: Composite Gates
- Add `gate` keyword for composite definitions
- Inline expansion at call sites
- Type checking for arity (angles, qubits)

### Phase 3: Noise Attributes
- Add `@atomic` / `@decomposed` attributes for composites
- SLR output with noise hints
- Noise model integration in PECOS

### Phase 4: Advanced Features
- Qubit array parameters
- Composite gates calling other composites
- Inlining control (`@inline`, `@noinline`)
- `@target` hint attribute for documentation/validation

---

## Summary

Zlup has **no built-in quantum operations**. All quantum operations are either:

| Type | Syntax | Body | Provider |
|------|--------|------|----------|
| Target | `declare gate name(...)` | None | Hardware, simulator, or noise model |
| Composite | `gate name(...) { }` | Required | Zlup code (sequences of other gates) |

**Target gates** are provided by the compilation target. The standard library
(`std.gates`) declares the common operation set that most targets support:
- Preparation: `pz`, `px`, `py`
- Measurement: `mz`, `mx`, `my`
- Gates: `h`, `x`, `cx`, `rz`, etc.

If a target doesn't support an operation, compilation fails - no fallbacks, no hidden magic.

**Composite gates** are defined in Zlup as sequences of other gates. They're portable
across any target that supports their constituent gates. Noise models can treat them
as atomic units or decompose them.

**Only two things are built-in:**
- `qalloc` - resource allocation (doesn't touch quantum state)
- `result` - output emission (classical)

Key design choices:
- **All quantum operations are target-provided** - even "standard" ones
- **One import for everything quantum** - `use std.gates.*`
- **No fallbacks** - explicit failure if target doesn't support an operation
- **Fail fast** - compile-time errors, not runtime surprises
- **Atomic noise by default** for composites
- **Explicit attributes** to override noise/inlining behavior
