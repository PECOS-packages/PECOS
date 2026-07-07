# Zlup: Zig Semantics with Rust/Python Syntax

## Status

> **EXPERIMENTAL / EXPLORATORY** - Zlup is a research experiment exploring whether a quantum language with these design goals is feasible. It is not a production language—do not use it for real projects. The value is in what we learn from building it.

---

## Overview

Zlup is the **low-level reflection of Guppy** in the PECOS ecosystem. Where Guppy provides a high-level, Pythonic experience with linear types for safety, Zlup explores the opposite end of the design spectrum: **simple, explicit, low-level control**.

### The Guppy-Zlup Duality

| Concern | Guppy | Zlup |
|---------|-------|------|
| **Abstraction** | High-level, hide complexity | Low-level, expose control |
| **Safety model** | Linear type system | Simplicity + constraints |
| **Resource management** | Implicit (type system tracks) | Explicit (allocators) |
| **Target audience** | QEC researchers, algorithm developers | Low-level QEC developers, systems programmers |
| **Integration** | Python-embedded | Rust-native |
| **Philosophy** | "Make it easy" | "Make it clear" |

### When to Use Zlup

Zlup is designed for **quantum orchestration and Rust integration**:

- **Quantum control flow**: Gate sequences, syndrome extraction, correction application
- **FFI to Rust backends**: Calling decoders, simulators, and classical algorithms
- **Simulation infrastructure**: Noise modeling, error injection, state tracking
- **Compilation target**: Guppy programs can target Zlup for Rust execution

**Complex classical algorithms (MWPM, ML decoders, etc.) belong in native code**—Rust is preferred for safety, but C/C++/Zig are supported via FFI (C ABI). Zlup orchestrates the quantum side and calls native backends for heavy classical computation.

To make Rust integration ergonomic, Zlup provides the **`zlup-ffi` crate**—a Rust library with:
- `Decoder`, `NoiseModel`, `Simulator` traits that decoders and backends implement
- `#[zlup_export]` proc macro that generates C ABI wrappers automatically
- FFI-safe types (`PackedBits`, `QubitId`, `GateType`) for type-safe interop
- `zlup bindgen` command to generate Zlup declarations from Rust code

See [docs/rust-integration.md](rust-integration.md) for the full guide.

Most QEC researchers who prefer Python workflows should use **Guppy**. Zlup is for the systems layer that connects quantum operations to classical backends.

Design influences include:

- **Rust's safety goals**: Inspired by Rust's commitment to safety, but pursuing it through
  simplicity and constraints rather than a complex type system
- **More constrained than Zig**: Zig demonstrates that expressivity doesn't require complexity—
  Zlup pushes further, adding constraints so that complex safety mechanisms become unnecessary
- **Rust/Python syntax**: Familiar surface syntax for broader accessibility
- **NASA Power of 10**: Bounded loops, fixed resource limits, predictable execution

### Safety Through Constraints, Not Complexity

The core philosophy: **if the language is constrained enough, you don't need sophisticated type systems for safety**.

| Approach | How It Achieves Safety | Complexity |
|----------|------------------------|------------|
| Rust | Borrow checker, lifetimes, ownership | High (but powerful) |
| Guppy | Linear types | Medium-high |
| Zig | Explicit, no hidden behavior | Medium |
| **Zlup** | Constraints + simplicity | Low |

Zlup explores whether scoping rules, bounded resources, and explicit control flow can be designed so that:
- The "obvious" code is the safe code
- Unsafe patterns are structurally impossible, not just flagged by a type checker
- You don't need to understand complex type theory to write correct programs

This is **explicit over implicit**—no magic, no hidden behavior, no surprises. The constraints aren't limitations on expressivity; they're guardrails that make unsafe patterns impossible to express in the first place.

Zlup compiles to multiple targets:

- **SLR-AST**: JSON bridge enabling integration with Guppy and Python/PECOS
- **HUGR**: For hardware backends (shared with Guppy)
- **PHIR**: For simulator targeting (future)

---

## Design Principles

- **Expressivity through simplicity**: Powerful programs without complexity or magic
- **Safety through constraints**: Restrictions that make unsafe patterns impossible
- **Reliability and predictability**: Bounded, analyzable programs for QEC

---

## Relationship to Guppy

Zlup complements Guppy by exploring a different point in the design space. Guppy's
linear type system provides strong safety guarantees with an elegant Python-embedded
experience. Zlup explores whether similar safety can be achieved through explicit
allocator-based resource management, which may be useful for:

- Rust-native simulation workflows
- Cases where explicit low-level control is preferred
- Compiling Guppy programs to a Rust-friendly representation

| Aspect            | Guppy                    | Zlup                 |
|-------------------|--------------------------|----------------------|
| Type safety       | Linear types             | Allocator model      |
| Integration       | Python-embedded          | Standalone / Rust    |
| Abstraction       | Higher-level             | Lower-level          |
| Resource tracking | Type system              | Structural lifetimes |
| Primary use       | User-facing programs     | Compilation target   |

### Why Explore This Approach?

Zlup is inspired by Rust's commitment to safety but pursues it through simplicity and
constraints. Where Rust achieves safety through a sophisticated type system, Zlup explores
whether similar guarantees can emerge from a simpler language with stricter constraints.

Zig's design principles—particularly its commitment to no magic and its demonstration that
expressivity doesn't require complexity—align well with this goal. Zlup may push even
further toward simplicity given the NASA Power of 10 emphasis, while preserving the
expressivity needed for complex QEC algorithms.

Zig's principles map well to quantum computing needs:

| Zig Principle         | Quantum Application                     |
|-----------------------|-----------------------------------------|
| No hidden allocations | Explicit qubit lifecycle                |
| Comptime              | Circuit generation, parameterized codes |
| Simple and explicit   | Readable QEC protocols                  |
| Safety without GC     | Resource safety via allocators          |

### NASA Power of 10 Alignment

Zlup follows NASA's Power of 10 rules—designed for mission-critical systems where failure is not an option. **Production QEC infrastructure has the same requirements**: a subtle bug in a decoder, an unbounded loop during syndrome processing, or unexpected memory allocation could corrupt an entire quantum computation.

These constraints aren't limitations—they're what mature, reliable QEC infrastructure demands:

| Rule                           | Zlup Implementation                       |
|--------------------------------|-------------------------------------------|
| 1. Simple control flow         | No goto, no while, no recursion           |
| 2. Fixed loop bounds           | Only bounded `for i in 0..n` loops        |
| 3. No dynamic alloc after init | Base allocator in main, children derive   |
| 4. Small functions             | Encouraged by module system               |
| 5. Assertions                  | Built-in `test` blocks, comptime checks   |
| 6. Minimal scope               | Block scoping, automatic release          |
| 7. Check returns               | Error unions, optional types              |
| 8. Limited preprocessor        | Comptime replaces macros                  |
| 9. Limited pointers            | Allocator refs, not raw pointers          |
| 10. All warnings               | Strict mode enabled by default            |

**Safe by Constraint**: Zlup enforces safety unconditionally—recursion is always forbidden, references to local variables cannot escape functions, and dangling pointers are structurally impossible. These aren't optional "strict mode" checks; they're fundamental to the language's memory model.

---

## Language Design

### Core Concepts

#### 1. Allocator-Based Qubit Management

```zlup
pub fn main() -> unit {
    // Simple case: gates don't require mut
    q := qalloc(4);
    pz q;
    h q[0];
    cx (q[0], q[1]);
    return;
}

pub fn partitioned_example() -> unit {
    // Partitioning: mut required on parent for .child()
    mut base := qalloc(100);

    // Children track resources from parent (base is mutated)
    // but children themselves don't need mut for gate application
    data := base.child(9);
    ancilla := base.child(8);

    pz data;
    h data[0];
    cx (data[0], ancilla[0]);

    // Automatic release when scope ends
    return;
}
```

#### 2. Two Slot States

Qubits have exactly two states (simplicity):

```
┌────────────┐  prepare()  ┌──────────┐
│ unprepared │ ──────────> │ prepared │
└────────────┘             └──────────┘
      ^                          │
      │           mz()           │
      └──────────────────────────┘
```

- **unprepared**: Initial state, or after measurement
- **prepared**: Ready for gate operations

Gates on unprepared slots are compile-time errors.

#### 3. Comptime Metaprogramming

```zlup_nocheck
// Parameterized surface code
pub fn surface_code(comptime distance: u32) -> type {
    num_data := distance * distance;
    num_ancilla := (distance - 1) * (distance - 1) * 2;

    return struct {
        data: qalloc(num_data),
        ancilla: qalloc(num_ancilla),

        // Self is implicitly available (Rust-style)

        pub fn init(base: *Alloc) -> Self {
            return Self {
                data: base.child(num_data),
                ancilla: base.child(num_ancilla),
            };
        }

        pub fn syndrome_round(&mut self) -> [num_ancilla]bit {
            pz self.ancilla;

            // Compile-time loop unrolling
            inline for i in 0..num_ancilla / 2 {
                self.measure_x_stabilizer(i);
            }
            inline for i in 0..num_ancilla / 2 {
                self.measure_z_stabilizer(i);
            }

            // return mz([num_ancilla]u1) self.ancilla[..];
        }
    };
}

// Usage
pub fn main() -> unit {
    mut base := qalloc(50);
    mut code := surface_code(3).init(&base);

    for _ in 0..1000 {
        syndrome := code.syndrome_round();
        // decode and correct...
    }
}
```

#### 4. Error Handling

Zlup follows Zig's error-as-values philosophy with extensions for quantum error correction.
The design is **explicit over implicit** — you must acknowledge every potential error,
aligning with NASA Power of 10 requirements for checking all return values.

##### Faults vs Errors

Zlup distinguishes **faults** from **errors** based on their origin and typical handling pattern:

| Aspect | Fault | Error |
|--------|-------|-------|
| **Origin** | Quantum hardware (physical layer) | Classical logic (software layer) |
| **Nature** | Expected imperfection we're designed to handle | Unexpected problem that means something is wrong |
| **Handling** | Collect for later analysis (QEC pattern) | Stop execution immediately |
| **Keyword** | `fault` | `error` |

The distinction is not about severity — a fault can absolutely cause a logical error. Rather,
it's about **where the problem originates** and **how we typically want to handle it**:

- **Faults happen at the physical layer.** The quantum hardware did something imperfect:
  a gate didn't apply cleanly, a qubit leaked, a measurement was noisy. These are expected —
  QEC is designed to handle a certain fault rate. Stopping on every fault would make
  error correction impossible. A QEC round might collect 50 faults and that's fine;
  that's what error correction is for.

- **Errors happen at the logical layer.** Something in the classical algorithm went wrong:
  the decoder couldn't find a valid correction, a file wasn't found, an invalid state was
  reached. These indicate either a bug or an unrecoverable situation. If the decoder says
  "I can't figure out a correction," that's an error — the logical algorithm failed.

**Mental model:**
- Faults = "expected badness we're designed to handle"
- Errors = "unexpected badness that means something is wrong"

**Quantum faults** — physical events from the quantum hardware:
```zlup
// Use `fault` keyword for quantum/physical faults
QuantumFault := fault {
    Leakage,
    QubitLoss,
    GateFailure,
};
```

**Classical errors** — logical problems in the software:
```zlup
// Use `error` keyword for classical/logical errors
DecodeError := error {
    SyndromeAmbiguous,
    WeightTooHigh,
};

IoError := error {
    FileNotFound,
    PermissionDenied,
};
```

| Category | Keyword | Behavior in `try` | Behavior in `try!` | Examples |
|----------|---------|-------------------|-------------------|----------|
| Quantum Fault | `fault` | Collected, continues | Stops immediately | Leakage, qubit loss, gate failure |
| Classical Error | `error` | Stops immediately | Stops immediately | Decoder failure, I/O error, invalid state |

##### Explicit Handling Required

Unlike languages that allow exceptions to propagate silently or errors to be ignored,
Zlup requires **explicit handling of both faults and errors**. This aligns with NASA Power
of 10 Rule 7: "Check the return value of all non-void functions."

**Learning from Rust's `?` operator criticism:**

Rust has been criticized for making error propagation *too easy*. The `?` operator lets
developers mechanically propagate errors without thinking about them:

```rust
// Rust: ? makes it easy to just bubble errors up without thought
fn do_something() -> Result<T, E> {
    let x = step1()?;  // Just propagate, don't think
    let y = step2()?;  // Just propagate, don't think
    let z = step3()?;  // Just propagate, don't think
    Ok(z)
}
```

While this is "explicit" in that you must write `?`, it becomes so automatic that
developers stop thinking about error handling. The errors bubble up, but nobody along
the way considered what to do about them.

**Zlup's approach: deliberate handling over mechanical propagation**

In QEC, errors and faults carry crucial diagnostic information. Mechanical propagation
loses context and makes debugging difficult. Zlup encourages *deliberate* handling:

```zlup_nocheck
// INVALID: ignoring the return value
qec_round(q);  // Compile error: unhandled faults/errors

// DISCOURAGED: mechanical propagation without thought
// (Zlup intentionally doesn't have a `?` operator)

// ENCOURAGED: deliberate handling with context
faults, result := qec_round(q);
result catch |err| {
    log("QEC round failed at step {}: {}", step, err);
    log("Faults before failure: {}", faults);
    return err;  // Propagate, but with added context
};

// VALID: explicitly discard if truly not needed
_ = qec_round(q);  // Explicit discard - you meant to ignore it
```

The absence of a `?`-like operator is intentional. When you propagate an error, Zlup
wants you to think about it — add context, log diagnostics, or transform it. This
makes error handling visible, auditable, and meaningful rather than mechanical.

**Why this matters for quantum computing:**

In classical computing, a propagated error eventually reaches a handler somewhere.
In quantum computing, by the time an error surfaces, the quantum state may be
irretrievably corrupted. Understanding *where* and *why* things went wrong is
essential for:
- Debugging QEC implementations
- Tuning error thresholds
- Identifying systematic hardware issues
- Post-mortem analysis of failed computations

Silent or mechanical error propagation loses this crucial information.

##### Promoting Faults to Errors

Sometimes accumulated faults cross a threshold where they should become a logical error.
Zlup supports **promoting faults to errors** when the situation warrants it:

```zlup_nocheck
fn qec_round(q: []qubit) try -> ([]QuantumFault, QecError!Syndrome) {
    // Run the circuit, collecting faults
    faults, syndrome := run_stabilizers(q);

    // Too many faults? Promote to a classical error
    if (faults.len > max_correctable) {
        return error.TooManyFaults;  // Stops execution, returns collected faults
    }

    // Faults within tolerance - continue
    return syndrome;
}

// Caller receives both the faults that occurred AND the error/result
faults, result := qec_round(q);

result catch |err| {
    // err might be TooManyFaults - we still have access to `faults`
    // to see what happened before the threshold was crossed
    log("QEC failed with {} faults: {}", faults.len, err);
    return;
};
```

This pattern allows:
- **Graceful degradation**: Collect faults until a threshold, then fail cleanly
- **Diagnostic information**: Even on failure, you know what faults occurred
- **Policy flexibility**: Different QEC codes can set different thresholds

The key insight: faults don't automatically become errors. Your code decides when
accumulated faults constitute a logical failure, making the policy explicit and tunable.

##### Return Type Syntax

```zlup_nocheck
// Single error union: either E or T (Zig style)
E!T

// Quantum faults + classical errors with value (QEC pattern)
// Returns: ([]QuantumFault, ClassicalError!T)
([]QuantumFault, DecodeError!T)
```

##### Two Error Handling Modes

**`try!` — Stop on First Error/Fault (Traditional/Strict)**

Matches Zig/Rust semantics. Any error or fault stops execution immediately.

```zlup_nocheck
fn strict_circuit(q: []qubit) try! -> QuantumFault!unit {
    h q[0];           // if fault occurs, return immediately
    cx (q[0], q[1]);    // only runs if h succeeded
}

// Caller handles single fault
strict_circuit(q) catch |fault| {
    log("Failed: {} on qubit {}", fault.type, fault.qubit);
};
```

**`try` — Collect Faults, Stop on Errors (QEC Pattern)**

Quantum-specific extension:
- Quantum faults: collected into array, execution continues
- Classical errors: stops execution, returns collected faults + error

##### Summary: Behavior by Mode

| Mode | Quantum Fault | Classical Error | Return Type |
|------|---------------|-----------------|-------------|
| `try!` | **Stops immediately** | **Stops immediately** | `E!T` |
| `try` | Collected, continues | Stops, returns collected faults | `([]QuantumFault, ClassicalE!T)` |

```zlup_nocheck
fn qec_round(q: []qubit) try -> ([]QuantumFault, DecodeError!Syndrome) {
    // Quantum faults - collected, continues
    cx (q[0], q[1]);     // Leakage detected → recorded, keeps going
    cx (q[1], q[2]);     // QubitLoss detected → recorded, keeps going

    syndrome := mz([2]u1) [q[3], q[4]];

    // Classical error - stops execution, returns collected faults
    correction := decode(syndrome);  // WeightTooHigh → STOP

    apply(correction, q);
    return syndrome;
}
```

##### Caller Side

```zlup_nocheck
faults, result := qec_round(q);

// faults: []QuantumFault - all quantum faults detected during execution
// result: DecodeError!Syndrome - either error that stopped us, or final value

result catch |err| {
    log("Classical error: {}", err);
    log("Quantum faults before failure: {}", faults);
    return;
};

// Success path - unwrap result
syndrome := result.!;
if (faults.len > 0) {
    // Faults may or may not have caused logical errors
    corrections := analyze_faults(faults);
    apply_corrections(q, corrections);
}
```

##### Block Syntax

Error handling can also be scoped to blocks within functions:

```zlup_nocheck
fn complex_circuit(q: []qubit) -> unit {
    // Strict section - any error stops
    try! {
        prepare_logical_zero(q);
        logical_h(q);
    } catch |err| {
        abort("Logical prep failed: {}", err);
    }

    // QEC section - soft errors collected, hard errors stop
    soft_errors, result := try {
        cx (q[0], q[3]);
        cx (q[1], q[3]);
        syndrome := mz([2]u1) [q[3], q[4]];
        decode(syndrome)  // hard error stops here
    };

    result catch |hard_err| {
        log("Decode failed: {}", hard_err);
        return;
    };

    if (soft_errors.len > 0) {
        apply_corrections(q, soft_errors);
    }
}
```

##### Rich Error Context

Errors automatically carry context (compiler-injected):

```zlup_nocheck
soft_errors, result := qec_round(q);

for (soft_errors) |err| {
    switch (err) {
        .Leakage => |ctx| {
            log("Leakage in {} on qubit {}", ctx.gate, ctx.qubit);
            reset_qubit(q[ctx.qubit]);
        },
        .QubitLoss => |ctx| {
            log("Lost qubit {} during {}", ctx.qubit, ctx.gate);
            flag_qubit_lost(ctx.qubit);
        },
        else => log_error(err),
    }
}
```

##### Classical-Only Functions

Functions without quantum operations use standard Zig-style error handling:

```zlup_nocheck
fn decode(syndrome: []const bit) -> DecodeError!Correction {
    if (weight(syndrome) > threshold) {
        return error.WeightTooHigh;
    }
    return compute_correction(syndrome);
}

fn run_with_recovery(code: *SurfaceCode) -> unit {
    syndrome := code.syndrome_round();

    correction := decode(syndrome) catch |err| switch (err) {
        error.WeightTooHigh => Correction.identity,
        error.SyndromeAmbiguous => {
            return run_with_recovery(code);
        },
    };

    apply(correction, code.data);
}
```

##### Explicit Returns Required

Zlup requires the `return` keyword for all function returns. Unlike Rust, which uses
implicit trailing expressions (the last expression without a semicolon becomes the
return value), Zlup makes returns explicit:

```zlup_nocheck
// INVALID in Zlup (valid in Rust): implicit return
fn add(a: i32, b: i32) -> i32 {
    a + b  // Rust would return this implicitly - Zlup requires explicit return
}

// VALID in Zlup: explicit return
fn add(a: i32, b: i32) -> i32 {
    return a + b;
}
```

**Why require explicit returns?**

1. **Clarity of intent**: An explicit `return` makes it unambiguous that you intend
   to exit the function with a value. In Rust, forgetting a semicolon can accidentally
   change a statement into a return expression.

2. **Consistency**: Every return looks the same, whether it's at the end of the function,
   in the middle, or inside a conditional. No special rules for "trailing position."

3. **NASA Power of 10 alignment**: Rule 1 emphasizes simple control flow. Explicit
   returns make control flow obvious — you can grep for `return` to find all exit points.

4. **Error handling clarity**: When combined with error handling, explicit returns
   make it clear what value is being returned:

```zlup_nocheck
fn qec_round(q: []qubit) try -> ([]QuantumFault, DecodeError!Syndrome) {
    faults, syndrome := run_stabilizers(q);

    if (faults.len > threshold) {
        return error.TooManyFaults;  // Clearly returning an error
    }

    return syndrome;  // Clearly returning success value
}
```

5. **Auditable code**: In safety-critical quantum computing, code reviewers can easily
   verify that every code path has an explicit return with the correct type.

**Block expressions vs function returns:**

Note that block expressions (like `if` expressions used for assignment) can still
evaluate to values — the explicit return requirement applies to *function* returns:

```zlup_nocheck
// Block expression for assignment - this is fine
x := if (condition) { 42 } else { 0 };

// But function must use explicit return
fn get_value(condition: bool) -> i32 {
    return if (condition) { 42 } else { 0 };
}
```

**Unit functions require explicit return:**

All functions must explicitly return their value, including unit functions. For unit
functions, `return;` is shorthand for `return unit;` and is the preferred style:

```zlup_nocheck
// INVALID: unit function without return
fn do_work() -> unit {
    process_data();
}

// VALID: unit function with explicit return
fn do_work() -> unit {
    process_data();
    return;  // Preferred: return; is shorthand for return unit;
}
```

This requirement serves several purposes:

1. **Uniform control flow**: Every function has an explicit exit point, making code flow
   analysis and review easier.

2. **Prevent accidental fallthrough**: Without explicit returns, it's easy to forget that
   control reaches the end of a function. With `return;`, you're forced to think about it.

3. **NASA Power of 10 compliance**: All control flow is explicit. There's no implicit
   "fall off the end" behavior.

4. **Exception**: Only `never` functions (functions that never return normally, like
   `panic()` or `abort()`) are exempt from this requirement.

> **Note:** `return;` without a value is only valid in functions returning `unit`.
> Using `return;` in a function with a non-unit return type is a compile error.

> **Implementation note:** The semantic analyzer should enforce that:
> 1. All functions have explicit `return` statements on all code paths
> 2. Trailing expressions in function bodies are not treated as implicit returns
> 3. Missing returns produce clear compile-time errors
> 4. `return;` is only allowed in unit-returning functions

##### Design Rationale

| Principle | Implementation |
|-----------|----------------|
| Explicit over implicit | Must use `try`/`try!`/`catch` — no silent error dropping |
| Explicit returns | Use `return` keyword, not implicit trailing expressions |
| NASA Power of 10 | All return values checked, errors are values |
| QEC-friendly | Quantum faults collected, classical errors stop execution |
| Faults vs Errors | `fault` for faults (physical), `error` for errors (logical) |
| Rich diagnostics | Faults/errors carry gate, qubit, and location context |
| Zig-aligned | `E!T` syntax, `catch` handling, error sets |

#### 5. Module System

```zlup_nocheck
// lib/qec/surface.zlp
std := @import("std");

pub Distance := enum(u32) {
    d3 = 3,
    d5 = 5,
    d7 = 7,
};

pub SurfaceCode := struct {
    distance: Distance,
    data: Alloc,
    ancilla: Alloc,

    pub fn init(base: *Alloc, distance: Distance) -> SurfaceCode {
        d := @enumToInt(distance);
        return .{
            distance,
            data: base.child(d * d),
            ancilla: base.child((d-1) * (d-1) * 2),
        };
    }
};

// main.zlp
surface := @import("qec/surface.zlp");

pub fn main() -> unit {
    mut base := qalloc(100);
    mut code := surface.SurfaceCode.init(&base, .d5);
    // ...
}
```

---

## Syntax Reference

### Declarations

Zlup uses `:=` (walrus operator) for type-inferred declarations and `: T =` for explicit types.

```zlup_nocheck
// Constants (type inferred from value)
pi := 3.14159;                    // f64 inferred
num_qubits := 17;                 // integer inferred

// Constants (explicit type)
pi: f64 = 3.14159;
num_qubits: u32 = 17;

// Mutable variables
mut count := 0;                   // type inferred
mut count: u32 = 0;               // explicit type
mut syndrome: [8]bit = undefined;

// Type declarations (constructor makes type obvious)
Point := struct { x: f64, y: f64 };
Color := enum { Red, Green, Blue };
QubitError := error { Leakage, QubitLoss, GateFailure };

// Set literals
targets := set { q[0], q[1], q[2] };

// Public exports
pub Config := struct { ... };
pub fn process() -> unit { ... }
```

#### Declaration Mental Model

| Syntax | Meaning |
|--------|---------|
| `name := value` | Type inferred from value |
| `name: T = value` | Explicit type annotation |
| `mut name := value` | Mutable, type inferred |
| `mut name: T = value` | Mutable, explicit type |
| `Name := struct { }` | Type definition (constructor infers "type") |

### Types

```zlup_nocheck
// Primitives - arbitrary-width integers (1-128 bits, like Zig)
u1, u2, u3, ..., u128       // Unsigned N-bit integers
i1, i2, i3, ..., i128       // Signed N-bit integers
usize, isize                // Pointer-sized integers
f32, f64                     // Floats
a64                          // Angle (backed by PECOS Angle64)
bool                         // Boolean
unit                         // Unit type (single value)

// Quantum types
qubit                        // Single qubit (abstract)
bit                          // Classical bit (measurement result)
Alloc                        // Qubit allocator
qalloc(N)                     // Allocator with capacity N (comptime)
// Classical data uses standard array types:
[N]bit                       // Array of classical bits
[N]u8                        // Array of bytes

// Compound types
[N]T                         // Array of N elements of type T
[]T                          // Slice (runtime-sized view)
*T                           // Pointer to T
?T                           // Optional (T or none)
E!T                          // Error union (error E or value T) - Zig style
[]E!T                        // Collected errors plus value (QEC extension)
Set(T)                       // Unordered set of unique elements

// User-defined
struct { ... }
enum { ... }
union(enum) { ... }
error { ... }                // Error set definition
```

### Type Ascription

Type ascription uses a space-separated postfix syntax, providing a clean and consistent way to specify types on expressions:

```zlup
// Literal type ascription
x := 42 u32;           // 42 as u32
y := 100 i64;          // 100 as i64
pi := 3.14159 f64;     // Float as f64

// Expression type ascription (evaluates, then converts)
half := 1/2 f64;       // 0.5 (division yields float, then typed)
quarter := 1/4 f64;    // 0.25 (non-exact division automatically floats)

// Angle unit suffix (same pattern)
angle := 1/4 turns;    // Quarter turn
theta := 3.14159 rad;  // Radians
```

**Design rationale:**

- **Unified syntax**: Type suffixes (`42 u32`), angle units (`1/4 turns`), and type ascription all use the same postfix pattern
- **Reads naturally**: "42 as u32", "one quarter turns"
- **No truncation surprises**: Integer division `1/4` produces `0.25` (float) when the result isn't exact, not `0` (truncated)
- **Explicit over implicit**: The type/unit is always visible at the expression site

This differs from Zig's `@as(u32, 42)` builtin in favor of the more readable postfix form.

### Angle Literals

Angles require explicit units—no implicit radians or degrees:

```zlup_nocheck
// Turns (native unit) - 1 turn = full rotation
rz(1/4 turns) q[0];      // Quarter turn
rz(1/8 turns) q[0];      // T gate (eighth turn)
rz(0.5 turns) q[0];      // Half turn (Z gate equivalent)

// Radians (for those who prefer mathematical convention)
rz(std.f64.pi/4 rad) q[0];   // pi/4 radians = 1/8 turns
rz(std.f64.pi/2 rad) q[0];   // pi/2 radians = 1/4 turns

// Fractions preferred over decimals (exact representation)
rz(1/4 turns) q[0];      // Exact
rz(0.25 turns) q[0];     // Also works, but 1/4 is clearer
```

**Design rationale:**

- **Explicit units prevent bugs**: No confusion about radians vs degrees vs turns (Mars Climate Orbiter!)
- **Turns as native**: Common QEC angles are simple fractions (1/4, 1/8, 1/2)
- **Backed by PECOS Angle64**: Uses fixed-point representation that's exact for fractions of turns
- **Fractions encouraged**: `1/4 turns` is both more readable and more precise than `0.25 turns`

**Angle64 Internal Representation:**

The `a64` type uses a 64-bit fixed-point representation where the full range [0, 2^64) maps to [0, 1) turns:

| Angle | Fixed-Point Value | Common Use |
|-------|------------------|------------|
| 0 turns | 0 | Identity |
| 1/8 turns | 2^61 | T-gate (π/4 rad) |
| 1/4 turns | 2^62 | S-gate (π/2 rad) |
| 1/2 turns | 2^63 | Z-gate (π rad) |

This representation provides:
- **Exact arithmetic** for all dyadic fractions (powers of 2 in denominator)
- **Wrapping at full turn** via natural integer overflow
- **Efficient operations** using integer arithmetic
- **No floating-point precision issues** for common quantum angles

Pre-defined constants in `zlup-ffi`:
```rust
Angle64::ZERO          // 0 turns
Angle64::EIGHTH_TURN   // 1/8 turns (T-gate)
Angle64::QUARTER_TURN  // 1/4 turns (S-gate)
Angle64::HALF_TURN     // 1/2 turns (Z-gate)
```

### Control Flow

```zlup_nocheck
// Conditionals
if condition {
    // ...
} else if other {
    // ...
} else {
    // ...
}

// If as expression
max := if a > b { a } else { b };

// Bounded for loop (preferred - NASA Rule 2)
for i in 0..n {
    process(i);
}

// For with collection
for item in items {
    use(item);
}

// For with index (enumerate)
for i, item in items {
    use(i, item);
}

// Switch
switch (value) {
    0 => handle_zero(),
    1..10 => handle_small(),
    else => handle_other(),
}

// Labeled blocks (for break with value)
result := blk: {
    if early_exit { break :blk default_value; }
    break :blk computed_value;
};
```

### Quantum Operations

```zlup_nocheck
// Allocator operations
mut base := qalloc(100);           // mut needed - will call .child()
data := base.child(9);             // no mut needed - just applying gates
pz data;                          // Prepare all slots

// Single-qubit gates (lowercase names)
h data[0];
x data[1];
z data[2];

// Parameterized gates with explicit angle units
rz(1/4 turns) data[0];           // Quarter turn (T² equivalent)
rz(1/8 turns) data[0];           // T gate angle
rx(1/4 turns) data[1];           // Quarter turn around X
rz(std.f64.pi/4 rad) data[0];    // Same as 1/8 turns, in radians

// Two-qubit gates
cx (data[0], data[1]);
cz(ancilla[0], data[0]);

// Batch operations with set semantics (unordered)
h { data[0], data[1], data[2] };   // Apply H to multiple qubits
cx { (data[0], data[1]), (data[2], data[3]) };  // Multiple CX gates

// Typed measurements with array semantics (ordered)
r := mz(u1) data[0];                        // Single qubit → u1
results := mz([2]u1) [data[0], data[1]];    // Multiple qubits → [2]u1 (explicit size)
results := mz([4]u1) slice;                 // From slice (size must match)

// Conditional operations
if r == 1 {
    x data[1];
}
```

#### Batch vs Array Semantics

Zlup distinguishes between unordered and ordered operations:

| Syntax | Semantics | Use Case |
|--------|-----------|----------|
| `{ }` | Set/unordered | Batch gates (order doesn't matter) |
| `[ ]` | Array/ordered | Measurements (result order matters) |

```zlup_nocheck
// Batch gate: order doesn't matter, all applied "simultaneously"
h { q[0], q[1], q[2] };

// Measurement: order matters, results[0] corresponds to q[0]
results := mz([3]u1) [q[0], q[1], q[2]];
```

### Tick Blocks (Parallel Layers)

Tick blocks represent atomic time slices where operations execute in parallel. They act as **optimization barriers** - the optimizer cannot move operations across tick boundaries.

```zlup_nocheck
// Basic tick block
tick {
    h data[0];
    h data[1];
}

// Labeled tick
tick syndrome_round {
    cx({(data[0], ancilla[0]), (data[1], ancilla[1])});
}
```

**Note**: Nested tick blocks are disallowed. A tick represents an atomic time slice, and nesting would create ambiguity about timing semantics. Use sequential ticks instead:

```zlup_nocheck
// Sequential ticks (correct)
tick layer1 { h({data[0], data[1]}); }
tick layer2 { cx({(data[0], data[2]), (data[1], data[3])}); }
```

### Attributes

Metadata can be attached to ticks and gates:

```zlup_fragment
// Single attribute
@attr(round, 0)
tick syndrome_check {
    cx (data[0], ancilla[0]);
}

// Multiple attributes
@attrs({round: 0, kind: "syndrome"})
tick syndrome_check {
    cx (data[0], ancilla[0]);
}

// Gate attributes
@attrs({syndrome: "X", ancilla: true})
cx (data[0], ancilla[0]);

// Inline attributes on ticks
tick @attr(round, 1) layer1 {
    h data[0];
}
```

### Optimization Barriers

Zlup provides several mechanisms to control optimization, particularly important for QEC where gate ordering and timing can affect error correction.

#### Preserve Attributes

These attributes prevent the optimizer from modifying or removing operations:

| Attribute | Purpose | Use Case |
|-----------|---------|----------|
| `@preserve` | Prevent any optimization | Debugging, calibration sequences |
| `@timing` | Preserve timing relationships | Time-sensitive QEC protocols |
| `@identity` | Keep intentional identity operations | Noise characterization, benchmarking |
| `@noopt` | Disable all optimizations in scope | Development, debugging |

```zlup_nocheck
// Prevent gate cancellation
@preserve
h q[0];
@preserve
h q[0];  // Both H gates preserved, won't cancel

// Preserve timing-critical sequence
@timing {
    cx (data[0], ancilla[0]);
    mz(u1) ancilla[0];
}

// Keep intentional identity
@identity {
    x q[0];
    x q[0];  // Won't be optimized away
}

// Disable optimization in block
@noopt {
    h q[0];
    h q[0];
    z q[0];  // Nothing optimized
}
```

#### QEC Round Tracking

The `@round(n)` attribute tracks QEC syndrome rounds, preventing gate cancellation across round boundaries:

```zlup_nocheck
// Gates in different rounds won't cancel
@round(0) {
    h ancilla[0];
    mz(u1) ancilla[0];
}

@round(1) {
    h ancilla[0];  // Same gate, different round - won't cancel with round 0
    mz(u1) ancilla[0];
}
```

**Design rationale**: In QEC, the same gate sequence may appear in multiple rounds, but each round's operations must execute independently. `@round(n)` prevents the optimizer from incorrectly combining gates across logical round boundaries.

#### Tick Blocks as Barriers

Tick blocks are always optimization barriers - operations cannot be moved across tick boundaries:

```zlup_fragment
tick {
    h q[0];
}
// Optimizer cannot move this H into the tick above
h q[0];
tick {
    h q[0];
}
```

This ensures that timing-critical sequences remain intact even when optimization is enabled.

### Functions

```zlup_nocheck
// Basic function
fn add(a: i32, b: i32) -> i32 {
    return a + b;
}

// Function with error return (Zig-style)
fn divide(a: f64, b: f64) -> error{DivByZero}!f64 {
    if b == 0 { return error.DivByZero; }
    return a / b;
}

// Comptime parameters
fn make_array(comptime T: type, comptime N: usize) -> [N]T {
    return [_]T{0} ** N;
}

// Method syntax (Rust-style receivers)
Counter := struct {
    value: u32,

    fn increment(&mut self) -> unit {
        self.value += 1;
    }
};
```

### Error-Handling Functions

```zlup_nocheck
// try! function: stop on first fault/error (traditional/strict)
fn strict_circuit(q: []qubit) try! -> QuantumFault!unit {
    h q[0];           // if fault occurs, return immediately
    cx (q[0], q[1]);    // only runs if h succeeded
}

// try! with return value
fn strict_measure(q: []qubit) try! -> QuantumFault![]u1 {
    h q[0];
    mz([2]u1) [q[0], q[1]]
}

// try function: collect faults, stop on errors (QEC pattern)
fn qec_round(q: []qubit) try -> []QuantumFault!unit {
    cx (q[0], q[3]);    // fault recorded, continues
    cx (q[1], q[3]);    // runs regardless of previous faults
}

// try with return value
fn qec_measure(q: []qubit) try -> []QuantumFault!u1 {
    cx (q[0], q[3]);
    mz(u1) q[3]
}
```

### Error Handling

```zlup_nocheck
// Catch single fault/error
strict_circuit(q) catch |fault| {
    log("Fault: {}", fault);
};

// Destructure collected faults and value
faults, result := qec_measure(q);

// Error blocks within functions
try! {
    h q[0];
    cx (q[0], q[1]);
} catch |err| {
    handle_error(err);
};

syndromes := try {
    cx (q[0], q[3]);
    cx (q[1], q[3]);
};
```

---

## Implementation Architecture

### Parsing Strategy: Recursive Descent

We use recursive descent parsing rather than the visitor pattern for several reasons:

1. **Simplicity**: Direct mapping from grammar to code
2. **Explicitness**: Clear control flow, no hidden dispatch
3. **Debuggability**: Easy to step through
4. **NASA Power of 10**: Predictable call structure

```rust
// Example parser structure
impl Parser {
    fn parse_program(&mut self) -> Result<Program> {
        let mut decls = Vec::new();

        while !self.at_end() {
            decls.push(self.parse_top_level_decl()?);
        }

        Ok(Program { declarations: decls })
    }

    fn parse_top_level_decl(&mut self) -> Result<Declaration> {
        if self.check(Token::Const) {
            self.parse_const_decl()
        } else if self.check(Token::Var) {
            self.parse_var_decl()
        } else if self.check(Token::Fn) {
            self.parse_fn_decl()
        } else if self.check(Token::Struct) {
            self.parse_struct_decl()
        } else {
            Err(self.error("expected declaration"))
        }
    }

    fn parse_statement(&mut self) -> Result<Statement> {
        // Direct dispatch based on current token
        match self.current().kind {
            Token::Const => self.parse_const_decl().map(Statement::Const),
            Token::Var => self.parse_var_decl().map(Statement::Var),
            Token::If => self.parse_if_stmt(),
            Token::For => self.parse_for_stmt(),
            Token::Return => self.parse_return_stmt(),
            _ => self.parse_expr_stmt(),
        }
    }

    // ... etc
}
```

### Compilation Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Zlup Source (.zlp)                        │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Lexer (pest grammar → tokens)                    │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│              Recursive Descent Parser → Zlup AST (Rust)           │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Semantic Analysis                              │
│  - Type checking                                                    │
│  - Allocator validation (capacity, lifecycle)                       │
│  - Qubit state validation (unprepared/prepared)                     │
│  - Comptime evaluation                                              │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
            ▼                   ▼                   ▼
┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
│  SLR-AST Path     │ │  HUGR Path        │ │  PHIR Path        │
│  (Python)         │ │  (Experiments)    │ │  (Simulators)     │
│                   │ │                   │ │                   │
│  Zlup AST       │ │  Zlup AST       │ │  Zlup AST       │
│      │            │ │      │            │ │      │            │
│      ▼            │ │      ▼            │ │      ▼            │
│  SLR-AST          │ │  HUGR             │ │  PHIR             │
│      │            │ │      │            │ │      │            │
│      ▼            │ │      ▼            │ │      ▼            │
│  ┌──────────┐     │ │  ┌──────────┐     │ │  ┌──────────┐     │
│  │ Guppy    │     │ │  │ Hardware │     │ │  │ PECOS    │     │
│  │ codegen  │     │ │  │ backends │     │ │  │ sims     │     │
│  │ QASM     │     │ │  │ TKET2    │     │ │  │          │     │
│  │ codegen  │     │ │  └──────────┘     │ │  └──────────┘     │
│  └──────────┘     │ │                   │ │                   │
└───────────────────┘ └───────────────────┘ └───────────────────┘
```

### HUGR vs PHIR: MLIR-Inspired IRs

Both HUGR and PHIR are MLIR-inspired intermediate representations, but target different use cases:

**HUGR (Hierarchical Unified Graph Representation)**
- Used by TKET2 compiler and Guppy
- Targets hardware/experimental backends
- Rich optimization framework
- Serializable for tool interop

**PHIR (Program Hierarchical IR)**
- Targets simulator backends
- Optimized for simulation semantics
- Used by PECOS simulation engines

Compiling to both enables:
1. **Hardware path**: Zlup → HUGR → TKET2 → Hardware
2. **Simulation path**: Zlup → PHIR → PECOS simulators
3. **Python path**: Zlup → SLR-AST → Guppy/QASM

### AST Design (Rust)

The Rust AST mirrors the Python SLR-AST for easy conversion:

```rust
// src/ast.rs

#[derive(Debug, Clone, PartialEq)]
pub struct SourceLocation {
    pub line: u32,
    pub column: u32,
    pub file: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Program {
    pub name: String,
    pub declarations: Vec<Declaration>,
    pub location: Option<SourceLocation>,
}

#[derive(Debug, Clone)]
pub enum Declaration {
    Const(ConstDecl),
    Var(VarDecl),
    Fn(FnDecl),
    Struct(StructDecl),
    Enum(EnumDecl),
    Allocator(AllocatorDecl),
}

#[derive(Debug, Clone)]
pub struct AllocatorDecl {
    pub name: String,
    pub capacity: u32,
    pub parent: Option<String>,
    pub location: Option<SourceLocation>,
}

#[derive(Debug, Clone)]
pub enum Statement {
    Const(ConstDecl),
    Var(VarDecl),
    Assign(AssignStmt),
    If(IfStmt),
    While(WhileStmt),
    For(ForStmt),
    Return(ReturnStmt),
    Break(BreakStmt),
    Continue(ContinueStmt),
    Defer(DeferStmt),
    Block(Block),
    Expr(ExprStmt),

    // Quantum operations
    Gate(GateOp),
    Prepare(PrepareOp),
    Measure(MeasureOp),
}

#[derive(Debug, Clone)]
pub struct GateOp {
    pub kind: GateKind,
    pub targets: Vec<SlotRef>,
    pub params: Vec<Expression>,
    pub location: Option<SourceLocation>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateKind {
    // Single-qubit
    X, Y, Z, H, T, Tdg,
    SX, SY, SZ, SXdg, SYdg, SZdg,
    RX, RY, RZ,
    RXY1q,

    // Two-qubit
    CX, CY, CZ, CH,
    SXX, SYY, SZZ, SXXdg, SYYdg, SZZdg,
    RZZ,

    // Face rotations
    F, Fdg, F4, F4dg,
}

// ... etc
```

---

## Example Programs

### Hello Quantum World

```zlup
// hello.zlp
pub fn main() -> unit {
    q := qalloc(2);  // no mut needed - just applying gates
    pz q;

    // Create Bell state
    h q[0];
    cx (q[0], q[1]);

    // Measure with explicit type
    results := mz([2]u1) [q[0], q[1]];
    return;
}
```

### Teleportation Protocol

```zlup_nocheck
// teleport.zlp
pub fn main() -> unit {
    mut base := qalloc(10);

    // Prepare initial state
    mut psi := base.child(1);
    pz psi;
    ry(0.7) psi[0];   // Some arbitrary state

    // Create EPR pair
    mut epr := base.child(2);
    pz epr;
    h epr[0];
    cx (epr[0], epr[1]);

    // Bell measurement
    cx (psi[0], epr[0]);
    h psi[0];

    m1 := mz(u1) psi[0];
    m2 := mz(u1) epr[0];

    // Classical corrections
    if m2 == 1 { x epr[1]; }
    if m1 == 1 { z epr[1]; }

    // epr[1] now holds the teleported state
}
```

### Surface Code QEC

```zlup_nocheck
// surface_qec.zlp
std := @import("std");

pub fn SurfaceCode(comptime distance: u32) -> type {
    d := distance;
    num_data := d * d;
    num_x_ancilla := (d - 1) * d;
    num_z_ancilla := d * (d - 1);
    num_ancilla := num_x_ancilla + num_z_ancilla;

    return struct {
        data: qalloc(num_data),
        ancilla: qalloc(num_ancilla),
        syndrome: [num_ancilla]bit = undefined,

        pub fn init(base: *Alloc) -> Self {
            return .{
                data: base.child(num_data),
                ancilla: base.child(num_ancilla),
                syndrome: undefined,
            };
        }

        pub fn prepare_logical_zero(&mut self) -> unit {
            pz self.data;
            // All data qubits start in |0⟩
        }

        pub fn syndrome_round(&mut self) -> unit {
            pz self.ancilla;

            // X stabilizers
            inline for i in 0..num_x_ancilla {
                a := i;
                h self.ancilla[a];

                // Connect to neighboring data qubits
                neighbors := comptime x_stabilizer_neighbors(i);
                inline for n in neighbors {
                    cx (self.ancilla[a], self.data[n]);
                }

                h self.ancilla[a];
            }

            // Z stabilizers
            inline for i in 0..num_z_ancilla {
                a := num_x_ancilla + i;

                neighbors := comptime z_stabilizer_neighbors(i);
                inline for n in neighbors {
                    cx (self.data[n], self.ancilla[a]);
                }
            }

            // Measure all ancilla (typed measurement)
            // self.syndrome = mz([num_ancilla]u1) self.ancilla[..];
        }

        pub fn decode_and_correct(&mut self) -> unit {
            correction := mwpm_decode(self.syndrome);
            apply_correction(self.data, correction);
        }

        // Comptime helper functions
        fn x_stabilizer_neighbors(comptime i: u32) -> [4]u32 {
            // ... compute neighbors at compile time
        }

        fn z_stabilizer_neighbors(comptime i: u32) -> [4]u32 {
            // ... compute neighbors at compile time
        }
    };
}

pub fn main() -> unit {
    mut base := qalloc(50);
    mut code := SurfaceCode(3).init(&base);

    code.prepare_logical_zero();

    // QEC rounds
    for round in 0..1000 {
        code.syndrome_round();
        code.decode_and_correct();

        // Optional: inject errors for testing
        if round % 100 == 0 {
            x code.data[0];  // Inject X error
        }
    }

    // Final logical measurement
    // logical_result := mz([num_data]u1) code.data[..];
}
```

---

## Implementation Plan

### Phase 1: Core Language (MVP) ✓ COMPLETE

1. **Lexer/Parser**: Pest grammar → Rust AST ✓
2. **Basic types**: Integers, bools, arrays, a64 angles ✓
3. **Control flow**: if, for (bounded only, no while) ✓
4. **Functions**: Basic fn declarations ✓
5. **Quantum ops**: Gates, typed measurements, allocators ✓
6. **Batch operations**: Set literals for parallel gates ✓
7. **Tick blocks**: Parallel layers with labels (nesting disallowed) ✓
8. **Attributes**: `@key(value)` metadata on ticks/gates ✓
9. **Optimization**: Constant folding, dead code elimination, gate cancellation ✓
10. **Optimization barriers**: `@preserve`, `@timing`, `@identity`, `@noopt`, `@round(n)` ✓
11. **Build system**: `build.zlp` infrastructure with targets and steps ✓

### Phase 2: Type System ✓ COMPLETE

1. **Type checking**: Basic static type analysis ✓
2. **Allocator validation**: Capacity and lifecycle ✓
3. **Qubit state tracking**: Compile-time state validation ✓
4. **Error unions**: Zig-style `E!T` with `catch` and `.!` unwrap ✓
5. **Collected errors**: QEC-style `[]E!T` for fault collection ✓
6. **Try blocks**: `try { }` (collect) and `try! { }` (propagate) syntax ✓
7. **Try functions**: `fn foo() try -> []E!T` syntax ✓
8. **Fault sets**: Quantum-specific `fault { Leakage, QubitLoss }` ✓

### Phase 3: Comptime ✓ COMPLETE

1. **Inline for loop unrolling**: `inline for i in 0..N { }` unrolls at compile time ✓
   - Semantic validation: errors for non-comptime ranges, break/continue in inline for
   - Optimization pass: variable substitution, recursive unrolling for nested loops
2. **Advanced builtins**: Type reflection with snake_case naming ✓
   - `@type_info(T)` - returns structured type information
   - `@field_names(T)` - returns array of struct field names
   - `@enum_fields(T)` - returns array of enum variant names
   - `@type_from_info(info)` - constructs type from TypeInfo struct
3. **Generic type instantiation**: Functions with comptime params get specialized ✓
   - `fn make_array(comptime T: type, comptime N: u32) -> [N]T`
   - Automatic mangling and caching of instantiated functions
4. **Comptime function memoization**: Cache comptime function results ✓
   - Structural type serialization for cache keys (handles anonymous structs)
   - Avoids redundant evaluation for same arguments

### Phase 4: Integration (IN PROGRESS)

1. **SLR-AST bridge**: Convert to Python AST ✓
2. **PyO3 bindings**: Use from Python ✓
3. **QASM codegen**: Direct QASM output ✓
4. **HUGR codegen**: Direct HUGR output ✓

### Phase 5: Tooling (IN PROGRESS)

1. **LSP server**: IDE support - see [IDE Setup Guide](ide-setup.md) ✓
2. **Formatter**: `zlup fmt` for canonical code style ✓
3. **Linter**: `zlup lint` with auto-fix capabilities ✓
4. **Documentation generator**: From doc comments (planned)
5. **Test runner**: Built-in test support (planned)

---

## Open Questions

1. ~~**Recursion policy**: Disallow entirely (strict Power of 10) or allow with depth limits?~~ → Resolved: Recursion is unconditionally disallowed. This is fundamental to Zlup's "safe by constraint" memory model—not a strict mode option. Use `inline for` with comptime bounds or regular `for` with runtime bounds instead. Recursive algorithms should be implemented iteratively or in native Rust code called via FFI.

2. ~~**Parallel blocks**: How to express `parallel { }` from SLR?~~ → Resolved: `tick { }` blocks

3. ~~**Interop with Guppy**: Can we call Guppy functions from Zlup?~~ → Design doc: [future/guppy-compat.md](future/guppy-compat.md)
   - Strategy: Guppy linter enforcing NASA Power of 10 constraints ("Reliable Guppy" subset)
   - Mechanical conversion to Zlup when code passes lint
   - The linter is valuable independently, even without Zlup adoption

4. ~~**Standard library**: What should be in `@import("std")`?~~ → Design doc: [future/stdlib-design.md](future/stdlib-design.md)
   - Modules: math, bits, mem, qec, testing, ffi
   - Zig-style import semantics with Rust/Python syntax
   - Comptime-first, bounded containers, QEC-focused

5. ~~**Build system**: Zig uses `build.zig`, what should we use?~~ → Implemented in `src/build.rs`
   - `build.zlp` - the build system IS the language
   - Full comptime power for build configuration
   - Integrated Rust FFI library building
   - Supports targets, optimization levels, build steps, and `-Dname=value` options

6. ~~**Quantum fault handling**: How to handle gate faults (leakage, loss, etc.)?~~ → Resolved:
   - `try { }` collects quantum faults (QEC pattern), returns `([]QuantumFault, ClassicalE!T)`
   - `try! { }` stops on first fault/error (traditional), returns `E!T`
   - `fn foo() try -> ...` and `fn foo() try! -> E!T` function syntax
   - `fault { }` for faults (physical), `error { }` for errors (logical)
   - Faults/errors carry rich context (gate, qubits, location)

---

## Summary

Zlup is an experimental language combining Zig semantics with Rust/Python-flavored syntax.
Inspired by Rust's commitment to safety and Zig's demonstration that expressivity doesn't
require complexity or magic, it explores achieving powerful, safe programs through simplicity
and constraints.

It complements Guppy by providing:

- **Expressivity through simplicity**: Powerful programs without hidden behavior or magic
- **Safety through constraints**: A naturally safe language given its restrictions
- **Rust-native workflows**: Direct integration with PECOS's Rust simulation backends
- **QEC reliability**: NASA Power of 10 principles for the reliability and predictability
  that large-scale quantum error correction demands

Zlup bridges to Python via SLR-AST JSON and targets hardware via HUGR (shared with Guppy),
enabling interoperability across the PECOS ecosystem while Guppy remains the primary
user-facing quantum programming language.
