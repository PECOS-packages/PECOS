# zlup-ffi

FFI traits and types for integrating Rust code with Zlup.

## Overview

This crate provides the Rust side of Zlup's FFI story. Implement the provided traits to create decoders, noise models, and simulators that can be called from Zlup programs.

## Quick Start

```rust
use zlup_ffi::prelude::*;

pub struct MyMwpmDecoder {
    distance: usize,
}

impl Decoder for MyMwpmDecoder {
    type Syndrome = u64;
    type Correction = u64;

    fn decode(&self, syndrome: u64) -> u64 {
        // Your MWPM implementation here
        0
    }
}

// Generate C ABI exports manually (or use #[zlup_export] with macros feature)
#[no_mangle]
pub extern "C" fn mwpm_new(distance: u32) -> *mut MyMwpmDecoder {
    Box::into_raw(Box::new(MyMwpmDecoder { distance: distance as usize }))
}

#[no_mangle]
pub extern "C" fn mwpm_decode(decoder: *const MyMwpmDecoder, syndrome: u64) -> u64 {
    let decoder = unsafe { &*decoder };
    decoder.decode(syndrome)
}

#[no_mangle]
pub extern "C" fn mwpm_free(decoder: *mut MyMwpmDecoder) {
    if !decoder.is_null() {
        unsafe { drop(Box::from_raw(decoder)); }
    }
}
```

## Features

- `macros` - Enable `#[zlup_export]` proc macro for automatic C ABI generation (planned)

## Traits

| Trait | Purpose |
|-------|---------|
| `Decoder` | Map syndromes to corrections |
| `NoiseModel` | Define custom noise channels for simulation |
| `Simulator` | Create custom quantum state simulators |
| `BatchDecoder` | Efficient batch decoding |
| `StreamingDecoder` | Temporal/streaming decoding |

## Types

| Type | Description |
|------|-------------|
| `QubitId` | Opaque qubit identifier |
| `GateType` | Enum of supported gate types |
| `PackedBits<N>` | Efficient bit storage for syndromes |
| `FfiResult<T>` | FFI-safe result type |

## Documentation

See the [Zlup Rust Integration Guide](../docs/rust-integration.md) for complete documentation.

## License

Apache-2.0
