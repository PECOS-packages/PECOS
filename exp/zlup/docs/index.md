# Zlup

> **EXPERIMENTAL / EXPLORATORY** - Zlup is a research experiment, not a production language.

A quantum programming language for QEC research: simple, low-level, and predictable by design.

## Overview

Zlup is the **low-level complement to Guppy** in the PECOS ecosystem. While Guppy provides a high-level, Pythonic experience with linear types for safety, Zlup explores a different approach:

| | Guppy | Zlup |
|---|---|---|
| **Philosophy** | High-level, Pythonic | Low-level, explicit |
| **Safety mechanism** | Linear type system | Constraints make unsafe impossible |
| **Target users** | QEC researchers | Systems programmers |

## Quick Example

```zlup
pub fn main() -> unit {
    q := qalloc(4);
    pz q;

    // Create GHZ state
    h q[0];
    cx (q[0], q[1]);
    cx (q[1], q[2]);
    cx (q[2], q[3]);

    // Measure all qubits
    results: [4]u1 = mz([4]u1) [q[0], q[1], q[2], q[3]];

    // Emit results to runtime
    result("measurements", results);

    return;
}
```

## A Closer Look: QEC Workflow

Here's a more complete program that shows off much of the language. It implements a
simple repetition code QEC workflow: encoding, multiple syndrome extraction rounds,
decoding, correction, and result emission.

```zlup_nocheck
/// Repetition Code QEC Demo
///
/// Demonstrates Zlup's features through a 3-qubit bit-flip code:
/// structs, error/fault sets, child allocators, tick blocks,
/// measurements, bounded loops, control flow, and more.

std := @import("std");

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Packed syndrome from two stabilizer measurements.
Syndrome := struct {
    bits: u2,       // two parity bits packed into a u2

    /// Decode syndrome to a qubit index (or none if no error).
    pub fn error_location(&self) -> ?u2 {
        return switch (self.bits) {
            0b00 => none,   // no error
            0b10 => 0,      // error on data[0]
            0b11 => 1,      // error on data[1]
            0b01 => 2,      // error on data[2]
        };
    }
};

/// Stats collected across rounds.
RoundStats := struct {
    rounds_run:  u32,
    corrections: u32,
};

// ---------------------------------------------------------------------------
// Error and fault sets
// ---------------------------------------------------------------------------

/// Classical errors — something unexpected in the logic.
DecodeError := error { AmbiguousSyndrome };

/// Quantum faults — expected hardware imperfections.
HwFault := fault { Leakage, Crosstalk };

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Apply a bit-flip correction to a single data qubit.
fn apply_correction(data: *Allocator, idx: u2) -> unit {
    x data[idx];
    @emit.log.debug(f"corrected qubit {idx}");
    return;
}

/// Measure the two Z-stabilizers of the 3-qubit repetition code.
///
/// Uses child allocators to separate data and ancilla concerns,
/// tick blocks to express parallelism, and pack measurement.
fn measure_syndrome(data: *Allocator, ancilla: *Allocator) -> Syndrome {
    // Reset ancillas before each round
    pz ancilla;

    // Stabilizer circuits run in parallel where possible
    @attr(round, "syndrome")
    tick stabilizers {
        // Z₀Z₁ stabilizer
        cx (data[0], ancilla[0]);
        cx (data[1], ancilla[0]);

        // Z₁Z₂ stabilizer
        cx (data[1], ancilla[1]);
        cx (data[2], ancilla[1]);
    }

    // Measure ancillas, packing two bits into a u2
    bits: u2 = mz(pack u2) [ancilla[0], ancilla[1]];
    return Syndrome { bits };
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub fn main() -> unit {
    // Set up a reproducible simulation
    @emit.sim.send("seed", 42);
    @emit.sim.send("noise_model", "depolarizing");
    @emit.sim.send("noise_rate", 0.001);

    // --- Allocate qubits with child partitioning ---
    mut q := qalloc(5);
    data    := q.child(3);  // data qubits [0..3)
    ancilla := q.child(2);  // ancilla qubits [3..5)

    // Prepare everything to |0⟩
    pz q;

    // --- Encode: create the logical |+⟩ state ---
    h data[0];
    // Spread with CNOTs (inline for unrolls at compile time)
    inline for i in 1..3 {
        cx (data[0], data[i]);
    }

    // --- Run several syndrome extraction rounds ---
    num_rounds := 4;
    mut stats := RoundStats { rounds_run: 0, corrections: 0 };

    for round in 0..num_rounds {
        syndrome := measure_syndrome(&data, &ancilla);

        @emit.log.info(f"round {round}: syndrome = 0b{syndrome.bits:02b}");

        // Decode and maybe correct
        if loc := syndrome.error_location() {
            apply_correction(&data, loc);
            stats.corrections += 1;
        }

        stats.rounds_run += 1;
    }

    // --- Final readout ---
    @emit.sim.noise_disable();  // noiseless final measurement

    final: [3]u1 = mz([3]u1) [data[0], data[1], data[2]];
    parity := std.parity_u8(final[0] ^ final[1] ^ final[2]);

    // Emit results to the runtime
    result("qec/final_readout", final);
    result("qec/parity", parity);
    result("qec/stats/rounds", stats.rounds_run);
    result("qec/stats/corrections", stats.corrections);

    return;
}
```

**Features shown above:**

| Feature | Where |
|---|---|
| Doc comments (`///`) | Top of file, on structs and functions |
| Imports (`@import`) | `std := @import("std")` |
| Structs with methods | `Syndrome`, `RoundStats` |
| Error and fault sets | `DecodeError`, `HwFault` |
| Allocators & children | `qalloc(5)`, `.child(3)`, `.child(2)` |
| Named tick blocks | `tick stabilizers { ... }` |
| Attributes | `@attr(round, "syndrome")` |
| Single & two-qubit gates | `h`, `x`, `cx` |
| `inline for` (comptime unroll) | `inline for i in 1..3 { cx ... }` |
| Bounded `for` loops | `for round in 0..num_rounds { ... }` |
| Typed measurement | `mz([3]u1)`, `mz(pack u2)` |
| Optionals & `if`-unwrap | `if loc := syndrome.error_location()` |
| `switch` expression | Inside `error_location` |
| Mutable bindings | `mut stats`, `stats.corrections += 1` |
| F-strings | `f"round {round}: syndrome = ..."` |
| Structured logging | `@emit.log.debug(...)`, `@emit.log.info(...)` |
| Simulator control | `@emit.sim.send(...)`, `@emit.sim.noise_disable()` |
| Result emission | `result("qec/final_readout", final)` |
| Namespaced result tags | `"qec/stats/rounds"` |
| Standard library call | `std.parity_u8(...)` |
| Pointer parameters | `fn apply_correction(data: *Allocator, ...)` |

For the full language reference, see [Syntax](syntax.md). For error handling
details (faults vs errors, `try`/`catch`, error unions), see the
[Error Handling Guide](tutorial-error-handling.md).

## Design Philosophy

**Simple. Explicit. No magic.**

- **Safe by constraint**: No recursion, no escaping references, no dangling pointers
- **NASA Power of 10**: Bounded loops, fixed resources, explicit control flow
- **Zig semantics, Rust/Python syntax**: Familiar surface, powerful foundations

## Getting Started

- [Tutorial](tutorial.md) - Learn the basics
- [Error Handling Guide](tutorial-error-handling.md) - Faults vs errors, QEC patterns
- [CLI Reference](cli.md) - Command-line interface
- [Language Syntax](syntax.md) - Complete reference
- [IDE Setup](ide-setup.md) - Editor configuration

## Key Features

### Type System
- Arrays `[N]T` vs Slices `[]T` - distinct types
- Slice syntax: `arr[0..5]`, `arr[2..]`, `arr[..5]`, `arr[..]`
- Aliases for safe slice views: `alias data := q[0..4]`

### Error Handling
- **Faults** (physical/quantum) vs **Errors** (logical/classical)
- `try` blocks collect faults, stop on errors (QEC pattern)
- `try!` blocks stop on first fault or error (strict mode)
- Error unions `Error!T` for explicit error handling

### Quantum Operations
- Allocator-based qubit management: `q := qalloc(4)`
- Batch operations: `h {q[0], q[1], q[2]}`
- Typed measurements: `result: u8 = mz(pack u8) qubits`

### Safety Model
- Escape analysis prevents returning references to locals
- Recursion unconditionally forbidden
- Duplicate qubit detection in parallel operations

## Learn More

- [Design Philosophy](design.md) - Why Zlup exists
- [Rust Integration](rust-integration.md) - FFI and native backends
- [Standard Library](stdlib.md) - Available modules
- [Error Reference](errors.md) - Compiler error messages
- [Development Notes](dev-notes.md) - Recent changes and implementation details

## Advanced Topics

- **Parallelism Analysis** - Use `zlup analyze` to detect parallelizable operations. See [CLI Reference](cli.md#parallelism-analysis) and [Development Notes](dev-notes.md).
- **Aliases** - The `alias` keyword creates safe slice views with overlap detection. See [Alias Design](future/alias-design.md) for details.

## Future Designs

- [Custom Gates](future/custom-gates-design.md) - User-defined composite gates and target-provided gates
