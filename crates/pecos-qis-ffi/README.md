# pecos-qis-ffi

QIS FFI layer providing `__quantum__rt__*` and `__quantum__qis__*` symbols.

## Purpose

Provides the C-compatible FFI functions that quantum programs call. These functions record operations to thread-local storage for later execution by the runtime.

## How It Works

1. Compiled quantum programs call `__quantum__rt__qubit_allocate()`, `__quantum__qis__h__body()`, etc.
2. These functions are provided by `libpecos_qis_ffi.so` (`.dylib` on macOS, `pecos_qis_ffi.dll` on Windows; this crate)
3. Operations are recorded in thread-local `OperationCollector`
4. After program execution, operations are retrieved and passed to the runtime

## Key Exports

- `__quantum__rt__qubit_allocate` - Allocate a qubit
- `__quantum__rt__qubit_release` - Release a qubit
- `__quantum__qis__h__body` - Hadamard gate
- `__quantum__qis__cnot__body` - CNOT gate
- `__quantum__qis__mz__body` - Measurement
- ... and many more

## Crate Type

This crate produces both `rlib` (for Rust) and `cdylib` (the platform-specific shared library) for dynamic loading.

## Runtime libraries in development and tests

Development builds and tests locate this library through Cargo's JSON artifact
messages, not Cargo's internal build directories. See "Runtime libraries in
development and tests" in the `pecos-qis` README for preparation and lookup order.
