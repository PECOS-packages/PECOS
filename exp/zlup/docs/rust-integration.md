# Zlup-Rust Integration Guide

This document describes how to write Rust code that integrates with Zlup, including traits, FFI conventions, and the `zlup-ffi` crate.

## Overview

Zlup is designed to orchestrate quantum operations while delegating complex classical computation to native code. Rust is the preferred language for this native layer due to its safety guarantees.

The integration story has three parts:

1. **`zlup-ffi` crate** - Rust library providing traits, types, and macros
2. **C ABI conventions** - How Zlup calls into Rust (via `extern "C"`)
3. **Code generation** - Zlup compiler can generate Rust bindings

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Zlup Program                           │
│  syndrome: u64 = mz(pack u64) [...];                        │
│  correction := decode(syndrome);  // FFI call               │
└─────────────────────────┬───────────────────────────────────┘
                          │ extern "C" fn decode(u64) -> u64
┌─────────────────────────▼───────────────────────────────────┐
│                   zlup-ffi crate                            │
│  #[zlup_export]                                             │
│  impl Decoder for MyMWPM { ... }                            │
└─────────────────────────────────────────────────────────────┘
```

## The `zlup-ffi` Crate

### Installation

```toml
# Cargo.toml
[dependencies]
zlup-ffi = "0.1"
```

### Core Traits

#### `Decoder` Trait

The primary trait for implementing decoders:

```rust
use zlup_ffi::prelude::*;

/// A decoder that maps syndromes to corrections.
pub trait Decoder: Send + Sync {
    /// The syndrome type (typically u64 or a custom packed type)
    type Syndrome: SyndromeData;

    /// The correction type (typically u64 or a custom packed type)
    type Correction: CorrectionData;

    /// Decode a syndrome into a correction.
    fn decode(&self, syndrome: Self::Syndrome) -> Self::Correction;

    /// Optional: decode with soft information (for ML decoders)
    fn decode_soft(&self, syndrome: Self::Syndrome, soft_info: &[f32]) -> Self::Correction {
        self.decode(syndrome)  // Default: ignore soft info
    }

    /// Optional: reset decoder state between shots
    fn reset(&mut self) {}
}
```

#### `NoiseModel` Trait

For simulation backends:

```rust
/// A noise model that can be applied to quantum state.
pub trait NoiseModel: Send + Sync {
    /// Apply noise after a gate operation.
    fn apply_gate_noise(&self, gate: GateType, qubits: &[QubitId], rng: &mut dyn Rng);

    /// Apply measurement noise.
    fn apply_measurement_noise(&self, qubit: QubitId, rng: &mut dyn Rng) -> bool;

    /// Apply idle noise for a time step.
    fn apply_idle_noise(&self, qubits: &[QubitId], rng: &mut dyn Rng);
}
```

#### `Simulator` Trait

For custom simulation backends:

```rust
/// A quantum state simulator.
pub trait Simulator: Send + Sync {
    /// Apply a gate to the state.
    fn apply_gate(&mut self, gate: GateType, qubits: &[QubitId]);

    /// Measure a qubit in the Z basis.
    fn measure_z(&mut self, qubit: QubitId) -> bool;

    /// Reset a qubit to |0⟩.
    fn reset(&mut self, qubit: QubitId);

    /// Get the current state vector (for debugging).
    fn state_vector(&self) -> Option<&[Complex64]> { None }
}
```

### FFI-Safe Types

The crate provides FFI-safe equivalents of common types:

```rust
use zlup_ffi::types::*;

// Packed syndrome/correction data
pub struct PackedBits<const N: usize> { /* ... */ }
type Syndrome64 = PackedBits<64>;
type Syndrome128 = PackedBits<128>;

// Qubit identifiers
pub struct QubitId(u32);

// Gate types
#[repr(C)]
pub enum GateType {
    H, X, Y, Z, S, T, Sdg, Tdg,
    Sx, Sy, Sz,
    Rx(f64), Ry(f64), Rz(f64),
    Cx, Cy, Cz, Ch,
    Swap, Iswap,
    Sxx, Syy, Szz,
    Rzz(f64), Rxx(f64), Ryy(f64),
    Ccx,
}

// Error types for FFI
#[repr(C)]
pub struct FfiResult<T> {
    pub ok: bool,
    pub value: T,
    pub error_code: u32,
}
```

### The `#[zlup_export]` Macro

This proc macro generates the C ABI wrappers automatically:

```rust
use zlup_ffi::prelude::*;

pub struct MwpmDecoder {
    // decoder state
}

#[zlup_export]
impl Decoder for MwpmDecoder {
    type Syndrome = u64;
    type Correction = u64;

    fn decode(&self, syndrome: u64) -> u64 {
        // MWPM algorithm implementation
        todo!()
    }
}
```

The macro generates:

```rust
// Auto-generated C ABI exports
#[no_mangle]
pub extern "C" fn mwpm_decoder_new() -> *mut MwpmDecoder { /* ... */ }

#[no_mangle]
pub extern "C" fn mwpm_decoder_decode(
    decoder: *const MwpmDecoder,
    syndrome: u64,
) -> u64 { /* ... */ }

#[no_mangle]
pub extern "C" fn mwpm_decoder_free(decoder: *mut MwpmDecoder) { /* ... */ }
```

### Example: Complete MWPM Decoder

```rust
use zlup_ffi::prelude::*;

/// Minimum Weight Perfect Matching decoder for surface codes.
pub struct MwpmDecoder {
    distance: usize,
    graph: MatchingGraph,
}

impl MwpmDecoder {
    pub fn new(distance: usize) -> Self {
        Self {
            distance,
            graph: MatchingGraph::for_surface_code(distance),
        }
    }
}

#[zlup_export(name = "mwpm")]
impl Decoder for MwpmDecoder {
    type Syndrome = u64;
    type Correction = u64;

    fn decode(&self, syndrome: u64) -> u64 {
        let defects = self.syndrome_to_defects(syndrome);
        let matching = self.graph.minimum_weight_matching(&defects);
        self.matching_to_correction(matching)
    }

    fn reset(&mut self) {
        self.graph.clear_cache();
    }
}

impl MwpmDecoder {
    fn syndrome_to_defects(&self, syndrome: u64) -> Vec<DefectNode> {
        // Convert packed syndrome bits to defect graph nodes
        todo!()
    }

    fn matching_to_correction(&self, matching: Matching) -> u64 {
        // Convert matching result to correction operators
        todo!()
    }
}
```

## Using Rust Decoders from Zlup

### Declaring External Functions

In Zlup, declare the external decoder interface:

```zlup_nocheck
// Declare external decoder functions
extern "C" {
    fn mwpm_new(distance: u32) -> *Decoder;
    fn mwpm_decode(decoder: *Decoder, syndrome: u64) -> u64;
    fn mwpm_free(decoder: *Decoder) -> unit;
}
```

### Using the Decoder

```zlup_nocheck
pub fn main() -> unit {
    // Initialize decoder (typically once at program start)
    decoder := mwpm_new(5);  // distance-5 surface code
    defer mwpm_free(decoder);

    q := qalloc(25);  // 25 data qubits for d=5
    ancilla := qalloc(24);  // 24 syndrome qubits

    // QEC round
    for round in 0..100 {
        // Syndrome extraction
        pz ancilla;
        // ... stabilizer measurements ...
        syndrome: u64 = mz(pack u64) ancilla[0..24];

        // Decode (FFI call to Rust)
        correction := mwpm_decode(decoder, syndrome);

        // Apply correction
        apply_correction(q, correction);
    }

    return;
}
```

## Build Integration

### Linking Rust Libraries

When compiling Zlup programs that use Rust FFI:

```bash
# Build the Rust decoder library
cd my-decoder
cargo build --release

# Compile Zlup with the library
zlup compile program.zlp \
    --link-lib=my_decoder \
    --lib-path=./my-decoder/target/release \
    -o program
```

### Cargo Workspace Setup

Recommended project structure:

```
my-qec-project/
├── Cargo.toml           # Workspace root
├── decoder/
│   ├── Cargo.toml       # Rust decoder crate
│   └── src/
│       └── lib.rs
├── zlup/
│   ├── main.zlp         # Zlup orchestration code
│   └── lib/             # Generated Zlup bindings
└── build.rs             # Build script to coordinate
```

### Generated Bindings

The `zlup` CLI can generate Zlup declarations from Rust code:

```bash
# Generate Zlup bindings from Rust crate
zlup bindgen --rust ./decoder/src/lib.rs -o ./zlup/lib/decoder.zlp
```

This parses `#[zlup_export]` attributes and generates corresponding Zlup declarations.

## Error Handling Across FFI

### Rust Side

Use `FfiResult` for fallible operations:

```rust
#[zlup_export]
impl Decoder for MyDecoder {
    // Infallible decode - preferred
    fn decode(&self, syndrome: u64) -> u64 { /* ... */ }
}

// For fallible operations, use explicit error returns
#[no_mangle]
pub extern "C" fn decoder_init(
    config_ptr: *const u8,
    config_len: usize,
    out_decoder: *mut *mut MyDecoder,
) -> FfiResult<()> {
    // Validate inputs
    if config_ptr.is_null() {
        return FfiResult::err(ErrorCode::NullPointer);
    }

    // Safe initialization
    match MyDecoder::from_config(unsafe { std::slice::from_raw_parts(config_ptr, config_len) }) {
        Ok(decoder) => {
            unsafe { *out_decoder = Box::into_raw(Box::new(decoder)) };
            FfiResult::ok(())
        }
        Err(e) => FfiResult::err(e.into()),
    }
}
```

### Zlup Side

Handle FFI errors explicitly:

```zlup_nocheck
result := decoder_init(config.ptr, config.len, &decoder);
if !result.ok {
    switch (result.error_code) {
        1 => { /* handle null pointer */ },
        2 => { /* handle invalid config */ },
        else => { /* unknown error */ },
    }
}
```

## Performance Considerations

### Minimize FFI Calls

Batch operations when possible:

```rust
// Good: batch decode
#[no_mangle]
pub extern "C" fn mwpm_decode_batch(
    decoder: *const MwpmDecoder,
    syndromes: *const u64,
    corrections: *mut u64,
    count: usize,
) { /* ... */ }
```

```zlup_nocheck
// Zlup: batch call
syndromes: [100]u64 = collect_syndromes();
corrections: [100]u64 = undefined;
mwpm_decode_batch(decoder, &syndromes, &corrections, 100);
```

### Avoid Allocations

Pre-allocate buffers and reuse them:

```rust
pub struct MwpmDecoder {
    // Pre-allocated working memory
    defect_buffer: Vec<DefectNode>,
    matching_buffer: Vec<Edge>,
}

impl MwpmDecoder {
    fn decode(&mut self, syndrome: u64) -> u64 {
        self.defect_buffer.clear();  // Reuse allocation
        // ...
    }
}
```

### Thread Safety

Decoders implementing `Send + Sync` can be called from multiple Zlup threads:

```rust
// Thread-safe decoder with interior mutability
pub struct ThreadSafeDecoder {
    inner: RwLock<DecoderState>,
}

#[zlup_export(thread_safe)]
impl Decoder for ThreadSafeDecoder {
    // ...
}
```

## Testing

### Unit Testing in Rust

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_no_errors() {
        let decoder = MwpmDecoder::new(3);
        let syndrome = 0b0000;  // No errors
        let correction = decoder.decode(syndrome);
        assert_eq!(correction, 0);  // No correction needed
    }

    #[test]
    fn test_decode_single_error() {
        let decoder = MwpmDecoder::new(3);
        let syndrome = 0b0011;  // Single X error
        let correction = decoder.decode(syndrome);
        assert_ne!(correction, 0);  // Correction applied
    }
}
```

### Integration Testing with Zlup

```bash
# Run Zlup tests that exercise FFI
zlup test ./tests/*.zlp --link-lib=my_decoder
```

## Appendix: Supported Rust Types

| Rust Type | Zlup Type | Notes |
|-----------|-----------|-------|
| `bool` | `bool` | |
| `u8`, `u16`, `u32`, `u64` | `u8`, `u16`, `u32`, `u64` | |
| `i8`, `i16`, `i32`, `i64` | `i8`, `i16`, `i32`, `i64` | |
| `f32`, `f64` | `f32`, `f64` | |
| `usize` | `usize` | Platform-dependent |
| `*const T`, `*mut T` | `*T`, `*mut T` | Raw pointers |
| `[T; N]` | `[N]T` | Fixed arrays |
| `()` | `unit` | |

Complex types (structs, enums) must use `#[repr(C)]` for FFI safety.
