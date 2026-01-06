# pecos

Main PECOS metacrate that re-exports functionality from component crates.

## Purpose

Provides a unified API for PECOS users. Most users should depend on this crate rather than individual component crates.

## Key Features

- **Unified simulation API**: `sim(program).seed(42).run(100)`
- **Re-exports**: Core types, engines, programs, quantum backends
- **Feature-gated**: Enable only what you need (qasm, qis, hugr, etc.)

## Feature Flags

- `runtime` (default): Full simulation with QASM/PHIR support
- `qis`: QIS/LLVM IR execution (requires LLVM 14)
- `hugr`: HUGR program support
- `quest`, `qulacs`: Additional quantum backends
