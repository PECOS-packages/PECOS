# pecos-qis-ffi-types

Shared types for QIS FFI layer.

## Purpose

Provides types shared between `pecos-qis-ffi` and `pecos-qis`. Separated to avoid circular dependencies.

## Key Types

- `OperationCollector` - Collects quantum operations during program execution
- `Operation` - Enum of all quantum operations (gates, measurements, etc.)
- `QuantumOp` - Core quantum gate operations

## Runtime build compatibility

The host and QIS runtime exchange Rust-owned operation collectors. They must be
built together with the same shared sources, Rust compiler, target, profile,
features, and build flags. `QIS_FFI_ABI_FINGERPRINT` is generated from these
inputs. Only built-in Rust targets are supported: Cargo does not expose enough
custom JSON target provenance to fingerprint its layout reliably. Custom
specifications (including ones named after built-in targets) are unsupported.
The runtime exports it through the versioned, integer-only
`pecos_qis_ffi_abi_fingerprint_v1(index)` handshake (four big-endian SHA-256
words at indices 0–3; other indices return zero).

The host checks this fingerprint before exchanging runtime-owned values. A
missing or different fingerprint is an error, including when a library was
selected explicitly with `PECOS_QIS_FFI_PATH`. Rebuild both the runtime and host
(or Python extension) together and restart the process after correcting the
library selection. The loader pins both successful loads and failures.

This is a conservative stale-build check, not a stable Rust ABI specification.
It intentionally rejects some harmless source/configuration changes and does
not certify allocator compatibility. Custom allocators or independently built
Rust libraries still require compatible ownership and allocation conventions.
