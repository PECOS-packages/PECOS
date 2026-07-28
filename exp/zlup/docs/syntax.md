# Zlup Language Reference

Complete syntax reference for the Zlup quantum programming language.

## Table of Contents

- [Comments](#comments)
- [Literals](#literals)
- [Variables and Bindings](#variables-and-bindings)
- [Types](#types)
- [Operators](#operators)
- [Control Flow](#control-flow)
- [Functions](#functions)
- [Structs and Enums](#structs-and-enums)
- [Error Handling](#error-handling)
- [Quantum Operations](#quantum-operations)
- [Attributes](#attributes)
- [Modules](#modules)
- [Logging](#logging)
- [Result Emission](#result-emission)
- [Simulator Control](#simulator-control)
- [Compilation Targets](#compilation-targets)

---

## Comments

```zlup_nocheck
// Single-line comment

/* Multi-line
   comment */

/// Documentation comment (for declarations)
```

---

## Literals

### Numbers

```zlup_nocheck
// Integers
42              // Decimal
0xFF            // Hexadecimal
0b1010          // Binary
0o755           // Octal
1_000_000       // Underscores for readability

// With type suffix
42_u32          // Explicit u32
255_u8          // Explicit u8

// Floats
3.14            // f64 by default
3.14_f32        // Explicit f32
1.5e10          // Scientific notation
1e-5            // Scientific without decimal

// Angles
1.57_a64        // Angle type for rotations
```

### Strings

```zlup_nocheck
// Regular string (escape sequences processed)
"hello\nworld"

// Raw string (no escape processing)
r"C:\Users\path"
r"\d+\.\d+"     // Regex pattern

// Multi-line string (triple-quoted)
"""
Line 1
Line 2
"""

// F-string (interpolation)
f"Value: {x}"
f"Pi = {pi:.2f}"        // With format specifier
f"Padded: {n:08d}"      // Zero-padded
```

### Escape Sequences

| Sequence | Character |
|----------|-----------|
| `\n` | Newline |
| `\r` | Carriage return |
| `\t` | Tab |
| `\\` | Backslash |
| `\"` | Double quote |
| `\'` | Single quote |
| `\0` | Null |
| `\{` | Literal `{` in f-strings |
| `\}` | Literal `}` in f-strings |
| `\xNN` | Hex byte |

### Other Literals

```zlup_nocheck
true, false     // Booleans
none            // Optional null value
undefined       // Uninitialized value
unit            // Unit type value
'a'             // Character literal
```

---

## Variables and Bindings

### Immutable Bindings

```zlup_fragment
x := 42;                    // Type inferred
y: u32 = 100;              // Type explicit
z := "hello";              // String
```

### Mutable Bindings

```zlup_fragment
mut count := 0;            // Mutable, type inferred
mut buffer: [10]u8 = undefined;  // Mutable array
count = count + 1;         // Assignment
count += 1;                // Compound assignment
```

### Aliases

Aliases create named views into existing data with overlap checking:

```zlup_nocheck
arr: [8]u32 = undefined;

// Create non-overlapping aliases
alias data := arr[0..4];
alias ancilla := arr[4..8];

// Use aliases like slices
process(data);
h ancilla[0];

// Overlapping aliases are compile-time errors:
alias overlap := arr[2..6];  // ERROR: overlaps with 'data'
```

**Alias constraints:**
- Source must be a slice expression (e.g., `arr[0..4]`)
- Range bounds must be comptime-evaluable for overlap checking
- Aliases are immutable (cannot be reassigned)
- Overlapping ranges on the same source are errors

### Assignment Operators

| Operator | Description |
|----------|-------------|
| `=` | Assignment |
| `+=` | Add and assign |
| `-=` | Subtract and assign |
| `*=` | Multiply and assign |
| `/=` | Divide and assign |
| `&=` | Bitwise AND and assign |
| `\|=` | Bitwise OR and assign |
| `^=` | Bitwise XOR and assign |

---

## Types

### Primitive Types

| Type | Description |
|------|-------------|
| `bool` | Boolean (true/false) |
| `uN` | Unsigned N-bit integer (N = 1-128), e.g., `u1`, `u7`, `u32`, `u128` |
| `iN` | Signed N-bit integer (N = 1-128), e.g., `i8`, `i32`, `i64` |
| `usize`, `isize` | Pointer-sized integers |
| `f16`, `f32`, `f64`, `f128` | Floating point |
| `a64` | Angle type (for gate rotations) |
| `unit` | Unit type (no value) |
| `type` | Type as a value (comptime) |
| `anytype` | Any type (comptime) |

Like Zig, Zlup supports arbitrary-width integers from 1 to 128 bits. This is useful for:
- Bit-packed measurement results: `u1`, `u2`, `u4`
- Syndrome values: `u3` for 3-bit syndromes
- Efficient storage: use exactly the bits you need

### Quantum Types

| Type | Description |
|------|-------------|
| `qubit` | Single qubit |
| `bit` | Classical bit |

### Compound Types

```zlup_nocheck
// Arrays (fixed size, known at compile time)
[4]u32                      // Array of 4 u32
[_]u8                       // Size inferred from initializer

// Slices (dynamic view into contiguous memory)
[]u8                        // Slice of u8
[]const u8                  // Immutable slice
[][]i32                     // Slice of slices (2D)

// Pointers
*u32                        // Single-item pointer
[*]u8                       // Many-item pointer
[*:0]u8                     // Sentinel-terminated

// Optionals
?u32                        // Optional u32

// Error unions
Error!u32                   // u32 or Error

// Tuples
(u32, bool)                 // Tuple of u32 and bool
(u32, u32, u32)             // 3-element tuple

// Sets
Set(u32)                    // Set of u32
```

### Array and Slice Operations

```zlup_nocheck
// Indexing (returns element)
arr[0]                      // First element
matrix[0][1]                // Nested indexing

// Slicing (returns slice)
arr[0..5]                   // Elements 0 to 4 (exclusive end)
arr[2..]                    // From index 2 to end
arr[..5]                    // From start to index 4
arr[..]                     // Full slice (all elements)

// Chained slicing
arr[1..10][2..5]            // Slice of a slice

// Array to slice conversion
slice := arr[..];           // Convert array to slice
```

**Note:** Arrays (`[N]T`) and slices (`[]T`) are distinct types. To pass an array where a slice is expected, use `arr[..]` to create a slice view.

### Type Expressions

```zlup_nocheck
// Function types
fn(u32, u32) -> u32

// Comptime types
comptime T: type
```

---

## Operators

### Arithmetic

| Operator | Description |
|----------|-------------|
| `+` | Addition |
| `-` | Subtraction (or negation) |
| `*` | Multiplication |
| `/` | Division |
| `%` | Modulo |

### Comparison

| Operator | Description |
|----------|-------------|
| `==` | Equal |
| `!=` | Not equal |
| `<` | Less than |
| `>` | Greater than |
| `<=` | Less or equal |
| `>=` | Greater or equal |

### Logical

| Operator | Description |
|----------|-------------|
| `and` | Logical AND |
| `or` | Logical OR |
| `!` | Logical NOT |

### Bitwise

| Operator | Description |
|----------|-------------|
| `&` | Bitwise AND |
| `\|` | Bitwise OR |
| `^` | Bitwise XOR |
| `~` | Bitwise NOT |
| `<<` | Left shift |
| `>>` | Right shift |

### Membership

| Operator | Description |
|----------|-------------|
| `in` | Membership test |
| `not in` | Negated membership |

```zlup_nocheck
if x in items { }
if y not in set { }
```

### Optional/Error Operators

| Operator | Description |
|----------|-------------|
| `orelse` | Unwrap optional with default |
| `catch` | Handle error with default |
| `.?` | Optional unwrap (returns optional) |
| `.!` | Error unwrap |
| `try` | Propagate error |

```zlup_nocheck
value := optional orelse default;
result := fallible() catch |err| handle(err);
```

---

## Control Flow

### If Statements

```zlup_nocheck
// Simple if
if condition {
    // body
}

// If-else
if condition {
    // true branch
} else {
    // false branch
}

// If-else if chain
if a {
    // ...
} else if b {
    // ...
} else {
    // ...
}

// Optional unwrapping (walrus operator)
if value := optional {
    // value is unwrapped here
}
```

### If Expressions

```zlup_nocheck
// If as expression (requires parentheses and else)
result := if (condition) { value1 } else { value2 };
```

### For Loops

All loops must have bounded iteration.

```zlup_nocheck
// Range loop
for i in 0..10 {
    // i goes from 0 to 9
}

// With index
for i, item in items {
    // i is index, item is value
}

// Inline for (comptime unrolling)
inline for i in 0..4 {
    h q[i];
}
// Unrolls to: h q[0]; h q[1]; h q[2]; h q[3];

// Nested inline for
inline for i in 0..2 {
    inline for j in 0..3 {
        cx (q[i], q[j + 2]);
    }
}
// Unrolls to 6 cx gates
```

**Inline for constraints:**
- Range bounds must be comptime-evaluable (literals or comptime constants)
- `break` and `continue` are not allowed inside inline for bodies
- Maximum unroll limit of 1024 iterations for safety

### Switch Statements

```zlup_nocheck
switch (value) {
    0 => { /* handle 0 */ },
    1, 2 => { /* handle 1 or 2 */ },
    3..10 => { /* handle range */ },
    else => { /* default */ },
}
```

### Control Flow Keywords

```zlup_nocheck
return value;           // Return from function
return;                 // Return unit (shorthand for return unit;)
break;                  // Exit loop
break :label value;     // Break with label and value
continue;               // Next iteration
continue :label;        // Continue outer loop
```

### Labeled Blocks

```zlup_nocheck
result := blk: {
    if condition {
        break :blk value1;
    }
    break :blk value2;
};
```

### Unsafe Blocks

Unsafe blocks provide an escape hatch from strict mode constraints. They allow operations that are normally forbidden, such as recursion. Unsafe blocks are **forbidden by default** and require the `--allow-unsafe` flag.

```zlup_nocheck
// Recursion inside unsafe block (requires --allow-unsafe)
fn factorial(n: u32) -> u32 {
    unsafe {
        if n <= 1 { return 1; }
        return n * factorial(n - 1);
    }
}
```

**What unsafe allows:**
- Recursive function calls
- Gates on potentially unprepared qubits
- Other strict mode violations

**What unsafe does NOT allow:**
- Type errors (still checked)
- Undefined variables (still checked)
- Syntax errors (still checked)

Unsafe blocks follow Rust's philosophy: make potentially dangerous operations explicit and auditable. Production code can ban all unsafe by not passing `--allow-unsafe`.

---

## Functions

### Function Declaration

```zlup_nocheck
// Basic function
fn add(a: u32, b: u32) -> u32 {
    return a + b;
}

// Public function
pub fn main() -> unit {
    return;
}

// Inline function
inline fn square(x: u32) -> u32 {
    return x * x;
}

// Method with self receiver
fn increment(&mut self) -> unit {
    self.count += 1;
    return;
}

// Comptime parameters
fn make_array(comptime N: usize) -> [N]u32 {
    arr: [N]u32 = undefined;
    return arr;
}
```

### Function Types

```zlup_nocheck
fn_type := fn(u32, u32) -> u32;
```

### Anonymous Functions

```zlup_nocheck
callback := fn(x: u32) -> u32 {
    return x * 2;
};
```

---

## Structs and Enums

### Struct Declaration

```zlup_nocheck
Point := struct {
    x: f64,
    y: f64,

    pub fn distance(&self) -> f64 {
        // method implementation
    }
};
```

### Struct Initialization

```zlup_nocheck
// Named fields (Rust-style)
p := Point { x: 1.0, y: 2.0 };

// Anonymous struct
data := .{ x: 1, y: 2 };

// Shorthand (when variable name matches field)
x := 1.0;
y := 2.0;
p := Point { x, y };
```

### Enum Declaration

```zlup_nocheck
Color := enum {
    Red,
    Green,
    Blue,
};

// With explicit values
Status := enum(u8) {
    Ok = 0,
    Error = 1,
};
```

### Tagged Union

```zlup_nocheck
Value := union(enum) {
    Int: i32,
    Float: f64,
    None,
};
```

### Generic Types (Comptime)

```zlup_nocheck
Stack := fn(comptime T: type, comptime capacity: usize) -> type {
    struct {
        items: [capacity]T = undefined,
        len: usize = 0,
    }
};

// Usage
stack: Stack(u32, 64) = .{};
```

---

## Error Handling

### Error Sets

```zlup_nocheck
// Define error set
FileError := error { NotFound, PermissionDenied, IoError };

// Error value literal
err := error.NotFound;
```

### Fault Sets (Quantum)

```zlup_fragment
// Define fault set
QuantumFault := fault { Leakage, QubitLoss };

// Fault value literal
f := fault.Leakage;
```

### Error Unions

```zlup_nocheck
// Function returning error union
fn read(path: []const u8) -> FileError![]u8 {
    if !exists(path) {
        return error.NotFound;
    }
    return data;
}
```

### Handling Errors

```zlup_nocheck
// Propagate with try
data := try read("file.txt");

// Handle with catch
data := read("file.txt") catch |err| {
    // handle error
    return default;
};

// Default value
data := read("file.txt") catch "default";
```

### Try Blocks

```zlup_nocheck
// Collect all errors (QEC pattern)
errors := try {
    risky_op1();
    risky_op2();
};

// Stop on first error
result := try! {
    step1();
    step2();
} catch |err| handle(err);
```

### Defer

```zlup_nocheck
fn process() -> unit {
    resource := acquire();
    defer release(resource);  // Runs on scope exit

    // Use resource...
    return;
}

// Error-specific defer
errdefer |err| cleanup(err);
```

---

## Quantum Operations

### Allocators

```zlup_fragment
// Allocate qubits
q := qalloc(4);

// Child allocators
mut main := qalloc(10);
data := main.child(4);
ancilla := main.child(6);
```

### Prepare (Reset)

```zlup_fragment
pz q;           // Prepare all
pz q[0];        // Prepare one
pz {q[0], q[1]}; // Batch prepare
```

### Single-Qubit Gates

```zlup_fragment
// Pauli gates
x q[0];    y q[0];    z q[0];

// Hadamard
h q[0];

// T gates (fourth root of Z)
t q[0];    tdg q[0];

// Square root gates (sz is the S gate, sqrt of Z)
sx q[0];   sy q[0];   sz q[0];
sxdg q[0]; sydg q[0]; szdg q[0];
```

### Rotation Gates

```zlup_nocheck
// Parameterized rotations
rx(angle) q[0];
ry(angle) q[0];
rz(angle) q[0];
```

### Two-Qubit Gates

```zlup_fragment
// Controlled gates (control, target)
cx (q[0], q[1]);
cy (q[0], q[1]);
cz (q[0], q[1]);
ch (q[0], q[1]);

// Swap
swap (q[0], q[1]);
iswap (q[0], q[1]);

// Ising gates
sxx (q[0], q[1]);
syy (q[0], q[1]);
szz (q[0], q[1]);

// Parameterized
rzz(angle) (q[0], q[1]);
crz(angle) (q[0], q[1]);
```

### Three-Qubit Gates

```zlup_fragment
ccx (q[0], q[1], q[2]);  // Toffoli
```

### Batch Operations

```zlup_nocheck
// Apply same gate to multiple qubits
h {q[0], q[1], q[2]};

// Batch two-qubit gates
cx {(q[0], q[1]), (q[2], q[3])};

// Batch rotations
rx(angle) {q[0], q[1]};
```

### Measurement

```zlup_nocheck
// Single qubit
result: u1 = mz(u1) q[0];

// Multiple qubits into array
results: [4]u1 = mz([4]u1) [q[0], q[1], q[2], q[3]];

// Measure entire register
all_bits := mz([4]u1) q;

// Pack into integer
syndrome: u8 = mz(pack u8) [q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7]];

// Pack into custom struct (for QEC syndromes, etc.)
Syndrome := struct { x_parity: u1, z_parity: u1, flags: u2 };
syndrome := mz(pack Syndrome) [ancilla[0], ancilla[1], ancilla[2], ancilla[3]];
```

The `pack` modifier fills bits sequentially into the target type's bit layout.
Without `pack`, each qubit produces one value of type T (count must match exactly).

### Tick Blocks

```zlup_fragment
// Group parallel operations
tick {
    h q[0];
    h q[1];
}

// Named tick with attributes
@attr(round, 0)
tick syndrome_check {
    // operations
}
```

---

## Attributes

```zlup_nocheck
// Single attribute
@attr(key, value)

// Multiple attributes
@attrs({key1: value1, key2: value2})

// On declarations
@attr(inline, true)
fn fast_op() -> unit { }

// On tick blocks
@attr(round, 0)
tick { }
```

---

## Modules

### Import

```zlup_nocheck
// Import standard library
std := @import("std");

// Import local module
utils := @import("utils.zlup");
```

### Public Exports

```zlup_nocheck
// Public binding
pub x := 42;

// Public function
pub fn helper() -> unit { }

// Public type
pub Point := struct { x: f64, y: f64 };
```

### Builtins

| Builtin | Description |
|---------|-------------|
| `@import(path)` | Import module |
| `@size_of(T)` | Size of type in bytes |
| `@type_info(T)` | Returns structured type information (kind, name, fields, etc.) |
| `@type_name(T)` | Type name as string |
| `@field_names(T)` | Returns array of struct field names |
| `@enum_fields(T)` | Returns array of enum variant names |
| `@type_from_info(info)` | Construct type from TypeInfo struct (reverse of `@type_info`) |
| `@compile_error(msg)` | Compile-time error |
| `@compile_log(...)` | Compile-time debug print |

Note: Both snake_case (Rust/Python style) and camelCase names are accepted for builtins. Snake_case is preferred.

---

## Logging

Zlup provides built-in structured logging with namespace filtering.

### Basic Logging

```zlup_nocheck
// Standard log levels
@emit.log.trace(f"detailed trace");
@emit.log.debug(f"debug info");
@emit.log.info(f"general info");
@emit.log.warn(f"warning");
@emit.log.error(f"error");
```

### With Namespace

```zlup_nocheck
// Sub-namespace for filtering
@emit.log.debug("decoder", f"processing syndrome");
@emit.log.info("qec::round", f"round complete");
```

### With Structured Data

```zlup_nocheck
// Attach data for structured logging
@emit.log.debug(f"state", data: current_state);
@emit.log.info(f"result", data: measurement_results);
```

### Custom Log Levels

```zlup_nocheck
// Numeric levels for fine-grained control
@emit.log.at(15, f"between trace and debug");
@emit.log.at(25, "perf", f"timing metric");
```

### Log Levels

| Level | Priority | Description |
|-------|----------|-------------|
| `trace` | 0 | Very detailed tracing |
| `debug` | 100 | Debug information |
| `info` | 200 | General information |
| `warn` | 300 | Warnings |
| `error` | 400 | Errors |

### Runtime Filtering (ZLUP_LOG)

```bash
# All logs at debug and above
ZLUP_LOG=debug ./program

# Only errors
ZLUP_LOG=error ./program

# By namespace
ZLUP_LOG=mymodule=trace ./program
ZLUP_LOG=mymodule::decoder=debug ./program
```

### Compile-Time Elision

```bash
# Release mode - remove all logs
zlup compile --release program.zlp

# Keep only warn and error
zlup compile --log-level 300 program.zlp
```

---

## Result Emission

The `result()` function emits tagged values as program outputs. Unlike logs, results are **never elided** - they're essential for returning data from quantum programs.

### Basic Usage

```zlup_fragment
// Emit a measurement result
result("measurement", m);

// Emit computed values
result("parity", parity_check);
result("syndrome", syndrome_bits);
```

### Namespaced Tags

Use `/` to organize results hierarchically:

```zlup_fragment
// QEC results
result("qec/syndrome", syndrome);
result("qec/parity", parity);

// Per-round results
result("round_1/ancilla", ancilla_result);
result("round_2/ancilla", ancilla_result);

// Nested namespaces
result("experiment/run_5/final_state", state);
```

### Supported Value Types

```zlup_nocheck
result("int_result", 42);           // Integers
result("bool_result", true);        // Booleans
result("float_result", 3.14);       // Floats
result("array_result", [1, 2, 3]);  // Arrays
```

### Ordering

Unlike `@emit.sim.*` and `@emit.log.*`, result expressions have **flexible scheduling**. They can "slip down" during compilation - the value just needs to be eventually recorded. This means:

```zlup_nocheck
h q[0];
result("before_cx", some_value);  // Could be reordered
cx (q[0], q[1]);
result("after_cx", other_value);
```

The compiler may batch or reorder result emissions as long as data dependencies are respected. This allows optimizations that wouldn't be possible with strict ordering.

### Comparison with Guppy

Zlup's `result()` is equivalent to Guppy's `result(tag, value)`:
- Tag must be a compile-time string literal
- Returns tagged (key, value) pairs to the caller
- Essential for extracting data from quantum program execution

### Entry Function Pattern

Entry/main functions should return `unit` and use explicit `result()` calls to emit outputs:

```zlup_nocheck
fn main() -> unit {
    q := qalloc(4);
    h q[0];
    cx (q[0], q[1]);
    m := measure(q);
    result("measurements", m);  // Emit to runtime
}
```

This pattern keeps the return type simple and makes data emission explicit.

---

## Simulator Control

The `@emit.sim.*` channel sends hints and commands to the simulator. These are **elided when targeting hardware** (`--target hardware`) but active for simulator and emulator targets.

The sim channel is intentionally kept simple and flexible. Most communication uses `@emit.sim.send(key, value)` which the noise modeling interprets.

### Noise Control

Noise is **enabled by default**. Use these convenience functions to toggle:

```zlup_nocheck
@emit.sim.noise_disable();  // Turn off noise
@emit.sim.noise_enable();   // Turn on noise (default state)
```

### Generic Message Channel

Use `@emit.sim.send(key, value)` for all other simulator communication:

```zlup_nocheck
// Set RNG seed for reproducibility
@emit.sim.send("seed", 12345);

// Configure noise model
@emit.sim.send("noise_model", "depolarizing");
@emit.sim.send("noise_rate", 0.001);

// Create checkpoints
@emit.sim.send("checkpoint", "before_correction");

// Any custom key-value pairs the simulator understands
@emit.sim.send("custom_param", some_value);
```

### Full Example

```zlup_nocheck
pub fn main() -> unit {
    // Set up reproducible simulation
    @emit.sim.send("seed", 42);
    @emit.sim.send("noise_model", "depolarizing");
    @emit.sim.send("noise_rate", 0.001);

    // Allocate and prepare
    q := qalloc(3);
    @emit.sim.send("checkpoint", "initial");

    // Apply gates (noise enabled by default)
    h q[0];
    cx (q[0], q[1]);
    cx (q[0], q[2]);

    @emit.sim.send("checkpoint", "after_encoding");

    // Disable noise for measurement
    @emit.sim.noise_disable();

    // Measure
    syndrome := mz(u8) q[1..3];

    result("syndrome", syndrome);
    return;
}
```

### SLR Output

Both `result(key, value)` and `@emit.sim.send(key, value)` generate a unified `SendStmt` in SLR:

```json
{"type": "SendStmt", "channel": "result", "key": "counts", "value": {...}}
{"type": "SendStmt", "channel": "sim", "key": "seed", "value": 42}
{"type": "SendStmt", "channel": "sim", "key": "noise_enable", "value": null}
```

The `channel` field distinguishes them for downstream handling (PECOS, etc.).

### Ordering Semantics

The three channels have different ordering/scheduling behavior:

| Channel | Scheduling | Elision Behavior |
|---------|------------|------------------|
| `result(key, value)` | Flexible | Never elided - program output |
| `@emit.sim.*` | Barrier | Elided for hardware, but preserves ordering |
| `@emit.log.*` | Flexible | Can be fully elided in release mode |

**Result expressions** can "slip" during optimization - they just need to eventually record the value. They have data dependencies on their value but can be reordered relative to other operations.

**Simulator commands** (`@emit.sim.*`) act as synchronization points. `@emit.sim.noise_disable()` must happen *before* the operations it protects. When compiled for hardware targets, sim commands are elided but could optionally emit a barrier to preserve the same scheduling behavior as on simulator (currently they are completely elided).

**Log expressions** (`@emit.log.*`) can be completely elided in release mode. Unlike sim commands, logs are purely for debugging - when elided, no semantic difference should exist. This allows optimizations to move operations across where logs used to be.

**Current implementation:**
- The SLR codegen processes statements in source order
- `@emit.log.*` with elision returns no statement - complete removal, allows optimizations
- `@emit.sim.*` for hardware emits a barrier by default to preserve ordering
- Use `--elide-sim` flag for complete removal (max optimization, no ordering guarantee)
- Downstream tools (PECOS, hardware compilers) may perform their own reordering

**SimMode options:**
| Mode | Behavior | Use case |
|------|----------|----------|
| `Emit` | Output actual SimStmt | Simulator target |
| `Barrier` | Output scoped barrier | Hardware default - preserves ordering |
| `Elide` | Complete removal | `--elide-sim` flag - max optimization |

The barrier is scoped to allocators visible in the current scope, not a global barrier. This means `@emit.sim.*` commands only synchronize the qubits that are actually accessible at that point in the program.

### Target-Dependent Behavior

| Command | `--target simulator` | `--target hardware` | `--target hardware --elide-sim` |
|---------|---------------------|---------------------|--------------------------------|
| `@emit.sim.noise_enable()` | SimStmt | Barrier (no-op) | Elided |
| `@emit.sim.noise_disable()` | SimStmt | Barrier (no-op) | Elided |
| `@emit.sim.send(...)` | SimStmt | Barrier (no-op) | Elided |

By default, hardware targets preserve ordering with barriers. Use `--elide-sim` for complete removal.

---

## Compilation Targets

Zlup separates **what** you're compiling for (target) from **how** you serialize (format).

### Execution Targets

```bash
# Simulator (default): full debug, relaxed constraints
zlup compile program.zlp --target simulator

# Hardware: strict constraints, simulation artifacts removed
zlup compile program.zlp --target hardware

# Emulator: hardware-like constraints with visibility
zlup compile program.zlp --target emulator
```

### Output Formats

```bash
# SLR-AST JSON (default, for Python/PECOS)
zlup compile program.zlp --format slr

# PHIR JSON (PECOS simulator)
zlup compile program.zlp --format phir-json

# OpenQASM 2.0
zlup compile program.zlp --format qasm
```

### Build Modes

```bash
# Debug (default): all logs, permissive
zlup compile program.zlp --mode debug

# Release: optimized, logs elided, strict
zlup compile program.zlp --mode release
```

### Combined Examples

```bash
# Development workflow
zlup compile program.zlp  # simulator + slr + debug

# Production for hardware
zlup compile program.zlp --target hardware --mode release

# Simulation with QASM output
zlup compile program.zlp --target simulator --format qasm

# Full control
zlup compile program.zlp \
    --target hardware \
    --format slr \
    --mode release \
    --strict true \
    --log-level 400
```

### Effective Settings by Target + Mode

| Target | Mode | Strict | Log Elision |
|--------|------|--------|-------------|
| simulator | debug | No | None (all logs) |
| simulator | release | Yes | 100+ (debug+) |
| hardware | debug | Yes | 300+ (warn+) |
| hardware | release | Yes | 300+ (warn+) |
| emulator | debug | Yes | 200+ (info+) |
| emulator | release | Yes | 200+ (info+) |

---

## Reserved Keywords

```
and         break       catch       comptime    continue
defer       else        enum        errdefer    error
false       fault       fn          for         if
in          inline      log         mut         none
not         or          orelse      packed      pub
return      Self        set         struct      switch
test        tick        true        try         type
undefined   union       unit        unsafe
```

---

## Grammar Summary

```
program     = declaration*
declaration = binding | fn_decl | struct_decl | enum_decl | union_decl | error_set | fault_set
binding     = "pub"? "mut"? identifier (":" type)? "=" expr ";"
fn_decl     = "pub"? "inline"? "fn" name "(" params ")" ("->" type)? block
statement   = binding | assignment | if | for | switch | tick | return | break | continue | defer | block | expr ";"
expr        = binary_expr | unary_expr | primary_expr
```
