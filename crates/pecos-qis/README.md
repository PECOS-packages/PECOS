# pecos-qis

QIS (Quantum Instruction Set) infrastructure for PECOS.

## Purpose

Provides the complete QIS execution pipeline: compiling quantum programs (LLVM IR, HUGR) and executing them via Selene's quantum simulator.

## Architecture

```
QisEngine
├── QisInterface (compiles programs, collects operations)
│   └── QisHeliosInterface (Selene Helios compiler)
└── QisRuntime (executes quantum operations)
    └── SeleneRuntime (Selene simulator)
```

## Key Types

- `QisEngine` - Classical control engine for QIS programs
- `QisInterface` trait - Program compilation interface
- `QisRuntime` trait - Quantum operation execution
- `QisHeliosInterface` - Selene Helios-based interface (feature: `selene`)
- `SeleneRuntime` - Selene simulator wrapper (feature: `selene`)

## Features

- `selene` (default): Selene-based implementation
- `llvm`: LLVM IR program support
- `hugr`: HUGR program compilation

## Usage

```rust
use pecos_qis::{qis_engine, helios_interface_builder, selene_simple_runtime};

let engine = qis_engine()
    .runtime(selene_simple_runtime()?)
    .interface(helios_interface_builder())
    .program(qis_program)
    .build()?;
```

## Acknowledgements

This crate integrates with [Selene](https://github.com/Quantinuum/selene), a quantum computer emulation platform developed by Quantinuum.
