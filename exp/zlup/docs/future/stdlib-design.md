# Standard Library Design

This document outlines the design of Zlup's standard library.

## Import Syntax

Zlup uses **Zig-style import semantics** with **Rust/Python-inspired syntax**:

```zlup_nocheck
// Import entire module (Zig semantics)
std := @import("std");

// Use qualified access
angle := std.math.pi_4;
count := std.bits.popcount_u64(syndrome);

// Import specific items (Rust-style syntax, Zig semantics)
// The module is still loaded, but we bind specific names
math := @import("std").math;
pi := math.pi;

// Alternative: destructuring import (Python-style syntax)
// Still Zig semantics - the module is evaluated once
{ pi, tau, e } := @import("std").math;
```

### Why Zig Semantics?

- **Comptime evaluation**: Imports are resolved at compile time
- **No runtime overhead**: Module code runs once during compilation
- **Explicit dependencies**: Clear what each file imports
- **Deterministic**: Same import always gives same result

### Syntax Comparison

| Style | Syntax | Zlup Approach |
|-------|--------|---------------|
| Zig | `const std = @import("std");` | `std := @import("std");` |
| Rust | `use std::math::pi;` | `{ pi } := @import("std").math;` |
| Python | `from std.math import pi` | `{ pi } := @import("std").math;` |

## Standard Library Structure

```
std/
├── math.zlp       # Mathematical constants and functions
├── bits.zlp       # Bitwise operations
├── mem.zlp        # Memory utilities (bounded)
├── fmt.zlp        # Formatting utilities
├── io.zlp         # I/O (constrained, no unbounded reads)
├── debug.zlp      # Debug utilities
├── testing.zlp    # Test framework
├── qec/           # QEC-specific utilities
│   ├── syndrome.zlp
│   ├── pauli.zlp
│   └── codes.zlp
└── ffi.zlp        # FFI helpers
```

## Module Contents

### std.math

Mathematical constants and pure functions.

```zlup_nocheck
// Constants
pub pi: f64 = 3.14159265358979323846;
pub tau: f64 = 6.28318530717958647692;  // 2π
pub e: f64 = 2.71828182845904523536;
pub sqrt2: f64 = 1.41421356237309504880;
pub sqrt2_inv: f64 = 0.70710678118654752440;  // 1/√2

// Angle fractions (for quantum gates)
pub pi_2: f64 = 1.57079632679489661923;   // π/2
pub pi_4: f64 = 0.78539816339744830962;   // π/4
pub pi_8: f64 = 0.39269908169872415481;   // π/8

// Conversion
pub deg_to_rad: f64 = 0.01745329251994329577;  // π/180
pub rad_to_deg: f64 = 57.29577951308232087680; // 180/π

// Functions (comptime-evaluable where possible)
pub fn abs(x: f64) -> f64 { ... }
pub fn min(a: f64, b: f64) -> f64 { ... }
pub fn max(a: f64, b: f64) -> f64 { ... }
pub fn clamp(x: f64, lo: f64, hi: f64) -> f64 { ... }
```

### std.bits

Bitwise operations for syndrome processing.

```zlup_nocheck
// Popcount (count set bits)
pub fn popcount_u8(x: u8) -> u8 { ... }
pub fn popcount_u16(x: u16) -> u16 { ... }
pub fn popcount_u32(x: u32) -> u32 { ... }
pub fn popcount_u64(x: u64) -> u64 { ... }

// Parity (XOR of all bits)
pub fn parity_u8(x: u8) -> u1 { ... }
pub fn parity_u16(x: u16) -> u1 { ... }
pub fn parity_u32(x: u32) -> u1 { ... }
pub fn parity_u64(x: u64) -> u1 { ... }

// Bit extraction
pub fn get_bit(x: u64, index: u6) -> u1 { ... }
pub fn set_bit(x: u64, index: u6, value: u1) -> u64 { ... }
pub fn extract_bits(x: u64, start: u6, len: u6) -> u64 { ... }

// Rotation
pub fn rotl_u32(x: u32, n: u5) -> u32 { ... }
pub fn rotr_u32(x: u32, n: u5) -> u32 { ... }
pub fn rotl_u64(x: u64, n: u6) -> u64 { ... }
pub fn rotr_u64(x: u64, n: u6) -> u64 { ... }

// Leading/trailing zeros
pub fn clz_u32(x: u32) -> u6 { ... }  // count leading zeros
pub fn ctz_u32(x: u32) -> u6 { ... }  // count trailing zeros
pub fn clz_u64(x: u64) -> u7 { ... }
pub fn ctz_u64(x: u64) -> u7 { ... }
```

### std.mem

Bounded memory utilities (NASA Power of 10 compliant).

```zlup_nocheck
// Fixed-capacity stack
pub fn Stack(comptime T: type, comptime capacity: usize) -> type {
    return struct {
        items: [capacity]T = undefined,
        len: usize = 0,

        pub fn push(&mut self, item: T) -> CapacityError!unit { ... }
        pub fn pop(&mut self) -> ?T { ... }
        pub fn peek(&self) -> ?*const T { ... }
        pub fn is_empty(&self) -> bool { ... }
        pub fn is_full(&self) -> bool { ... }
    };
}

// Fixed-capacity queue (ring buffer)
pub fn Queue(comptime T: type, comptime capacity: usize) -> type { ... }

// Fixed-capacity hash map
pub fn HashMap(comptime K: type, comptime V: type, comptime capacity: usize) -> type { ... }

// Copying and comparison
pub fn copy(comptime T: type, dest: []T, src: []const T) -> usize { ... }
pub fn eql(comptime T: type, a: []const T, b: []const T) -> bool { ... }
pub fn set(comptime T: type, dest: []T, value: T) -> unit { ... }
```

### std.qec

QEC-specific utilities.

```zlup_nocheck
// Syndrome buffer for multi-round storage
pub fn SyndromeBuffer(comptime bits: usize, comptime rounds: usize) -> type {
    return struct {
        data: [rounds]u64 = undefined,
        current_round: usize = 0,

        pub fn record(&mut self, syndrome: u64) -> unit { ... }
        pub fn get(&self, round: usize) -> u64 { ... }
        pub fn diff(&self, round_a: usize, round_b: usize) -> u64 { ... }
    };
}

// Pauli frame tracking
pub fn PauliFrame(comptime num_qubits: usize) -> type {
    return struct {
        x_frame: u64 = 0,  // X corrections to track
        z_frame: u64 = 0,  // Z corrections to track

        pub fn apply_x(&mut self, qubit: usize) -> unit { ... }
        pub fn apply_z(&mut self, qubit: usize) -> unit { ... }
        pub fn propagate_cx(&mut self, control: usize, target: usize) -> unit { ... }
    };
}

// Lookup table decoder (for small codes)
pub fn LookupDecoder(comptime syndrome_bits: usize, comptime correction_bits: usize) -> type {
    return struct {
        table: [1 << syndrome_bits]u64 = undefined,

        pub fn init(table_data: [1 << syndrome_bits]u64) -> Self { ... }
        pub fn decode(&self, syndrome: u64) -> u64 { ... }
    };
}
```

### std.testing

Test framework for Zlup programs.

```zlup_nocheck
// Test declaration (comptime)
pub fn expect(ok: bool) -> TestError!unit {
    if (!ok) return error.ExpectFailed;
    return;
}

pub fn expectEq(comptime T: type, expected: T, actual: T) -> TestError!unit {
    if (expected != actual) return error.ExpectEqFailed;
    return;
}

pub fn expectApprox(expected: f64, actual: f64, tolerance: f64) -> TestError!unit {
    if (@abs(expected - actual) > tolerance) return error.ExpectApproxFailed;
    return;
}

// Test blocks in source files
test "syndrome parity" {
    syndrome: u8 = 0b10101010;
    try expect(std.bits.parity_u8(syndrome) == 0);
}
```

### std.ffi

Helpers for FFI with Rust/C.

```zlup_nocheck
// Opaque pointer wrapper
pub fn Opaque(comptime name: []const u8) -> type {
    return struct {
        ptr: *anyopaque,

        pub fn is_null(&self) -> bool {
            return self.ptr == null;
        }
    };
}

// C string utilities
pub fn c_str_len(s: [*:0]const u8) -> usize { ... }
pub fn c_str_to_slice(s: [*:0]const u8) -> []const u8 { ... }
```

## Design Principles

1. **Comptime-first**: Prefer compile-time evaluation where possible
2. **Bounded**: All containers have fixed capacity (NASA Power of 10)
3. **No allocations**: Standard library never allocates at runtime
4. **Pure functions**: Math and bit operations are pure, side-effect free
5. **QEC-focused**: Include primitives that QEC workflows commonly need
6. **FFI-friendly**: Types that work well across the Rust boundary

## Implementation Priority

| Module | Priority | Rationale |
|--------|----------|-----------|
| std.math | High | Gates need angle constants |
| std.bits | High | Syndrome processing |
| std.qec | High | Core QEC workflows |
| std.mem | Medium | Bounded containers |
| std.testing | Medium | Quality assurance |
| std.ffi | Medium | Rust integration |
| std.fmt | Lower | Nice to have |
| std.io | Lower | Constrained I/O |
| std.debug | Lower | Development aid |
