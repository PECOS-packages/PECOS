# pecos-qis-ffi-types

Shared types for QIS FFI layer.

## Purpose

Provides types shared between `pecos-qis-ffi` and `pecos-qis`. Separated to avoid circular dependencies.

## Key Types

- `OperationCollector` - Collects quantum operations during program execution
- `Operation` - Enum of all quantum operations (gates, measurements, etc.)
- `QuantumOp` - Core quantum gate operations
