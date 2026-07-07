# Error Handling in Zlup: A Practical Guide

This tutorial covers Zlup's error handling system, designed specifically for quantum error
correction workflows. You'll learn the difference between faults and errors, how to collect
quantum faults while stopping on classical errors, and how to write robust QEC code.

## Table of Contents

- [The Faults vs Errors Distinction](#the-faults-vs-errors-distinction)
- [Error Sets and Fault Sets](#error-sets-and-fault-sets)
- [Error Union Syntax](#error-union-syntax)
- [Handling Errors with catch](#handling-errors-with-catch)
- [Try Blocks: Collect vs Propagate](#try-blocks-collect-vs-propagate)
- [Try Functions](#try-functions)
- [The Explicit Handling Philosophy](#the-explicit-handling-philosophy)
- [Practical QEC Examples](#practical-qec-examples)
- [Summary](#summary)

---

## The Faults vs Errors Distinction

Zlup distinguishes between **faults** and **errors** based on where they originate and how
we typically want to handle them:

| Aspect | Fault | Error |
|--------|-------|-------|
| **Origin** | Quantum hardware (physical layer) | Classical logic (software layer) |
| **Nature** | Expected imperfection | Unexpected problem |
| **Handling** | Collect for later analysis | Stop execution immediately |
| **Keyword** | `fault` | `error` |

### Why This Matters

**Faults happen at the physical layer.** The quantum hardware did something imperfect:
a gate didn't apply cleanly, a qubit leaked to a non-computational state, a measurement
was noisy. These are *expected* - QEC is designed to handle a certain fault rate.

```zlup_nocheck
// A QEC round might see 50 faults - that's normal!
// Stopping on every fault would make error correction impossible.
faults, syndrome := syndrome_round(q);
if faults.len < threshold {
    // Still correctable - this is what QEC is for
    apply_correction(decode(syndrome), q);
}
```

**Errors happen at the logical layer.** Something in the classical algorithm went wrong:
the decoder couldn't find a valid correction, a file wasn't found, an invalid state was
reached. These indicate bugs or unrecoverable situations.

```zlup_nocheck
// If the decoder says "I can't figure out a correction" - that's an error
correction := decode(syndrome) catch |err| {
    log("Decoder failed: {}", err);
    return;  // Can't continue
};
```

**Mental model:**
- Faults = "expected badness we're designed to handle"
- Errors = "unexpected badness that means something is wrong"

---

## Error Sets and Fault Sets

Define your own error and fault types using the `error` and `fault` keywords:

### Classical Error Sets

```zlup
// Define an error set for decoder failures
DecodeError := error {
    SyndromeAmbiguous,
    WeightTooHigh,
    NoValidCorrection,
};

// Error sets for I/O operations
IoError := error {
    FileNotFound,
    PermissionDenied,
    ConnectionLost,
};

// Create error values
err := error.SyndromeAmbiguous;
```

### Quantum Fault Sets

```zlup
// Define a fault set for quantum hardware faults
QuantumFault := fault {
    Leakage,       // Qubit leaked to non-computational state
    QubitLoss,     // Qubit physically lost
    GateFailure,   // Gate didn't apply correctly
    MeasurementError,  // Measurement gave wrong result
};

// Create fault values
f := fault.Leakage;
```

### When to Use Each

| Situation | Use |
|-----------|-----|
| Decoder can't find correction | `error` |
| File I/O failed | `error` |
| Invalid function argument | `error` |
| Gate applied with noise | `fault` |
| Qubit leaked to |2⟩ | `fault` |
| Measurement bit flip | `fault` |

---

## Error Union Syntax

Error unions express that a function can either return a value or an error/fault.

### Basic Error Union: `E!T`

The `E!T` syntax means "either error type E, or value type T":

```zlup_nocheck
// Function that might fail
fn divide(a: f64, b: f64) -> DivError!f64 {
    if b == 0.0 {
        return error.DivisionByZero;
    }
    return a / b;
}

// Multiple error types
fn read_config(path: []const u8) -> IoError!Config {
    if !file_exists(path) {
        return error.FileNotFound;
    }
    // ... parse and return config
    return config;
}
```

### Collected Faults: `[]E!T`

For QEC patterns, you often want to collect all faults that occurred, plus either
an error or the final value:

```zlup_nocheck
// This function collects quantum faults, might return a classical error
fn qec_round(q: []qubit) try -> []QuantumFault!Syndrome {
    // Faults collected, execution continues
    cx (q[0], q[1]);  // Might fault
    cx (q[1], q[2]);  // Might fault, still runs

    syndrome := mz([3]u1) [q[3], q[4], q[5]];

    // Classical error stops execution
    correction := decode(syndrome);  // If this errors, we stop

    return syndrome;
}

// Caller receives both faults and result
faults, result := qec_round(q);
```

---

## Handling Errors with catch

The `catch` keyword handles errors when they occur:

### Basic catch

```zlup_nocheck
// Provide a default value
result := divide(10.0, x) catch 0.0;

// Handle with a block
result := divide(10.0, x) catch |err| {
    log("Division failed: {}", err);
    return 0.0;
};

// Transform the error
result := read_config("settings.json") catch |err| {
    log("Config load failed, using defaults");
    return Config.default();
};
```

### catch with Error Inspection

```zlup_nocheck
fn load_or_create(path: []const u8) -> Config {
    config := read_config(path) catch |err| switch (err) {
        error.FileNotFound => {
            // File doesn't exist, create default
            return Config.default();
        },
        error.PermissionDenied => {
            log("Cannot read {}: permission denied", path);
            abort();
        },
        else => {
            log("Unexpected error: {}", err);
            abort();
        },
    };
    return config;
}
```

### Unwrapping with `.!`

When you're certain the value isn't an error, use `.!` to unwrap:

```zlup_nocheck
// Only use when you know it will succeed
result := divide(10.0, 2.0).!;  // We know 2.0 != 0

// Better: use catch for safety
result := divide(10.0, x) catch |_| abort();
```

---

## Try Blocks: Collect vs Propagate

Zlup provides two modes for handling errors within a scope:

### `try!` - Stop on First Error/Fault (Strict Mode)

The `try!` block stops immediately when any fault or error occurs:

```zlup_nocheck
fn strict_preparation(q: []qubit) -> QuantumFault!unit {
    try! {
        pz q;             // If this faults, stop immediately
        h q[0];           // Only runs if pz succeeded
        cx (q[0], q[1]);  // Only runs if h succeeded
    } catch |fault| {
        log("Preparation failed: {}", fault);
        return fault;
    }
    return;
}
```

Use `try!` when:
- Running calibration sequences where any fault invalidates results
- Debugging to find exactly where faults occur
- Operations must succeed completely or not at all

### `try` - Collect Faults, Stop on Errors (QEC Mode)

The `try` block collects quantum faults but stops on classical errors:

```zlup_nocheck
fn qec_syndrome_round(q: []qubit) -> ([]QuantumFault, DecodeError!Syndrome) {
    faults, result := try {
        // Quantum operations - faults collected, continues
        cx (q[0], q[3]);  // Fault? Recorded, keep going
        cx (q[1], q[3]);  // Fault? Recorded, keep going
        cx (q[2], q[3]);  // Fault? Recorded, keep going

        syndrome := mz([4]u1) [q[3], q[4], q[5], q[6]];

        // Classical operation - error stops execution
        correction := decode(syndrome);  // Error? Stop here

        apply_correction(correction, q);
        return syndrome;
    };

    // Handle the result
    result catch |err| {
        log("Decode failed after {} faults: {}", faults.len, err);
        return;
    };

    // Success - might still have faults
    if faults.len > 0 {
        log("Round completed with {} faults", faults.len);
    }
    return faults, result;
}
```

Use `try` when:
- Running QEC rounds where some faults are expected
- Collecting fault statistics for analysis
- Operations should continue despite hardware imperfections

### Behavior Summary

| Mode | Quantum Fault | Classical Error | Return Type |
|------|---------------|-----------------|-------------|
| `try!` | **Stops immediately** | **Stops immediately** | `E!T` |
| `try` | Collected, continues | Stops, returns collected faults | `([]Fault, E!T)` |

---

## Try Functions

Functions can be declared with `try` or `try!` to indicate their error handling mode:

### `try!` Functions

```zlup_nocheck
// Strict mode: any fault/error stops execution
fn calibrate_qubit(q: qubit) try! -> CalibrationFault!CalibrationData {
    pz q;
    h q;

    // Repeated measurements for statistics
    for i in 0..100 {
        m := mz(u1) q;
        record(m);
        pz q;
        h q;
    }

    return analyze_calibration();
}

// Caller handles single fault/error
data := calibrate_qubit(q[0]) catch |fault| {
    log("Calibration failed: {}", fault);
    return;
};
```

### `try` Functions

```zlup_nocheck
// QEC mode: faults collected, errors stop
fn run_qec_cycle(code: *SurfaceCode) try -> []QuantumFault!unit {
    // All operations in this function collect faults
    code.syndrome_round();
    code.decode_and_correct();
    return;
}

// Caller receives faults and result
faults, result := run_qec_cycle(&code);

result catch |err| {
    log("QEC cycle failed: {}", err);
    log("Faults before failure: {}", faults);
    return;
};

// Success path
if faults.len > warning_threshold {
    log("Warning: {} faults in cycle", faults.len);
}
```

---

## The Explicit Handling Philosophy

Zlup intentionally does **not** have a `?` operator like Rust. This is a deliberate design
choice for quantum computing contexts.

### Why No `?` Operator?

Rust's `?` operator makes error propagation easy - perhaps too easy:

```rust
// Rust: ? makes it easy to just bubble errors up without thought
fn do_something() -> Result<T, E> {
    let x = step1()?;  // Just propagate
    let y = step2()?;  // Just propagate
    let z = step3()?;  // Just propagate
    Ok(z)
}
```

While explicit, this pattern becomes so automatic that developers stop thinking about
error handling. Errors bubble up, but nobody along the way considered what to do about them.

### Zlup's Approach: Deliberate Handling

In QEC, errors and faults carry crucial diagnostic information. Mechanical propagation
loses context and makes debugging difficult:

```zlup_nocheck
// INVALID: ignoring the return value
qec_round(q);  // Compile error: unhandled faults/errors

// ENCOURAGED: deliberate handling with context
faults, result := qec_round(q);
result catch |err| {
    log("QEC round failed at step {}: {}", step, err);
    log("Faults before failure: {}", faults);
    log("Syndrome state: {}", last_syndrome);
    return err;  // Propagate with added context
};

// VALID: explicitly discard if truly not needed
_ = qec_round(q);  // Explicit discard - you meant to ignore it
```

### Why This Matters for Quantum Computing

In classical computing, a propagated error eventually reaches a handler somewhere.
In quantum computing, by the time an error surfaces, the quantum state may be
irretrievably corrupted. Understanding *where* and *why* things went wrong is essential for:

- Debugging QEC implementations
- Tuning error thresholds
- Identifying systematic hardware issues
- Post-mortem analysis of failed computations

Silent or mechanical error propagation loses this crucial information.

---

## Practical QEC Examples

### Example 1: Basic Syndrome Extraction

```zlup_nocheck
QuantumFault := fault { Leakage, GateError, MeasurementError };
QecError := error { DecodeFailed, TooManyFaults };

fn extract_syndrome(data: []qubit, ancilla: []qubit) try -> []QuantumFault![]u1 {
    // Prepare ancilla
    pz ancilla;
    h ancilla;

    // Stabilizer measurements (faults collected)
    for i in 0..ancilla.len {
        cx (data[i * 2], ancilla[i]);
        cx (data[i * 2 + 1], ancilla[i]);
    }

    h ancilla;

    // Measure ancilla
    syndrome := mz([4]u1) ancilla;

    return syndrome;
}

pub fn main() -> unit {
    q := qalloc(8);
    pz q;

    alias data := q[0..4];
    alias ancilla := q[4..8];

    // Run syndrome extraction
    faults, syndrome_result := extract_syndrome(data, ancilla);

    syndrome := syndrome_result catch |err| {
        log("Syndrome extraction failed: {}", err);
        return;
    };

    log("Syndrome: {}, Faults: {}", syndrome, faults.len);
    result("syndrome", syndrome);
    result("fault_count", faults.len);

    return;
}
```

### Example 2: Full QEC Round with Threshold

```zlup_nocheck
QuantumFault := fault { Leakage, QubitLoss, GateFailure };
DecodeError := error { SyndromeAmbiguous, WeightTooHigh };

fn qec_round(code: *SurfaceCode, max_faults: usize) try -> []QuantumFault!DecodeError!unit {
    // Extract syndrome (collects faults)
    syndrome := code.measure_stabilizers();

    // Check if too many faults occurred
    // Note: we're inside a try block, so faults are being collected
    // We can inspect them at any point

    // Decode (classical - error stops execution)
    correction := decode(syndrome);

    // Apply correction
    code.apply_correction(correction);

    return;
}

pub fn main() -> unit {
    mut base := qalloc(100);
    mut code := SurfaceCode.init(&base, 3);

    code.prepare_logical_zero();

    // Run multiple QEC rounds
    mut total_faults: usize = 0;

    for round in 0..1000 {
        faults, result := qec_round(&code, 10);

        total_faults += faults.len;

        result catch |err| {
            log("Round {} failed: {}", round, err);
            log("Total faults so far: {}", total_faults);
            break;
        };

        // Log periodic status
        if round % 100 == 0 {
            log("Round {}: {} faults this round, {} total", round, faults.len, total_faults);
        }
    }

    // Final measurement
    logical := code.measure_logical();
    result("logical_result", logical);
    result("total_faults", total_faults);

    return;
}
```

### Example 3: Promoting Faults to Errors

Sometimes accumulated faults cross a threshold where they should become a logical error:

```zlup_nocheck
fn qec_round_with_threshold(
    q: []qubit,
    max_correctable: usize
) try -> []QuantumFault!QecError!Syndrome {
    // Run the circuit, collecting faults
    faults, syndrome := run_stabilizers(q);

    // Too many faults? Promote to a classical error
    // This is a classical decision, so it's an error, not a fault
    if faults.len > max_correctable {
        return error.TooManyFaults;
    }

    // Faults within tolerance - continue
    return syndrome;
}

// Caller receives both the faults AND the error/result
faults, result := qec_round_with_threshold(q, 5);

result catch |err| {
    // err might be TooManyFaults - we still have access to `faults`
    log("QEC failed with {} faults: {}", faults.len, err);
    return;
};

// Success - but we still know how many faults occurred
syndrome := result.!;
if faults.len > 0 {
    log("Succeeded with {} faults", faults.len);
}
```

### Example 4: Rich Fault Context

Faults carry context information that can be inspected:

```zlup_nocheck
fn analyze_faults(faults: []QuantumFault) -> unit {
    mut leakage_count: usize = 0;
    mut gate_failures: usize = 0;

    for fault in faults {
        switch (fault) {
            fault.Leakage => |ctx| {
                log("Leakage in {} on qubit {}", ctx.gate, ctx.qubit);
                leakage_count += 1;
            },
            fault.QubitLoss => |ctx| {
                log("Lost qubit {} during {}", ctx.qubit, ctx.gate);
                flag_qubit_lost(ctx.qubit);
            },
            fault.GateFailure => |ctx| {
                log("Gate {} failed on qubits {:?}", ctx.gate, ctx.qubits);
                gate_failures += 1;
            },
            else => {
                log("Unknown fault: {}", fault);
            },
        }
    }

    result("leakage_events", leakage_count);
    result("gate_failures", gate_failures);

    return;
}
```

---

## Summary

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Fault** | Physical layer issue (expected, collected) |
| **Error** | Logical layer issue (unexpected, stops execution) |
| **`E!T`** | Error union: either error E or value T |
| **`[]E!T`** | Collected faults plus error union |
| **`try!`** | Stop on first fault/error (strict) |
| **`try`** | Collect faults, stop on errors (QEC) |
| **`catch`** | Handle errors with default or block |
| **`.!`** | Unwrap (when certain no error) |

### Best Practices

1. **Use `fault` for hardware issues**, `error` for software issues
2. **Use `try` blocks for QEC rounds** where faults are expected
3. **Use `try!` blocks for calibration** where any fault invalidates results
4. **Always handle or explicitly discard** return values
5. **Add context when propagating errors** - don't just bubble up
6. **Inspect fault context** to diagnose hardware issues
7. **Set thresholds** to promote excessive faults to errors

### Quick Reference

```zlup_nocheck
// Define fault and error sets
QuantumFault := fault { Leakage, QubitLoss };
DecodeError := error { Failed, Ambiguous };

// Function that collects faults
fn qec_op(q: []qubit) try -> []QuantumFault!DecodeError!T { ... }

// Function that stops on first fault
fn strict_op(q: []qubit) try! -> QuantumFault!T { ... }

// Handle with catch
result := may_fail() catch default_value;
result := may_fail() catch |err| { handle(err); return; };

// Collect faults from try function
faults, result := qec_op(q);

// Try block (collect mode)
faults, result := try {
    // quantum ops - faults collected
    // classical ops - errors stop
};

// Try! block (strict mode)
try! {
    // any fault or error stops immediately
} catch |err| {
    handle(err);
};
```
