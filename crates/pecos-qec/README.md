# pecos-qec

Quantum error correction utilities for PECOS.

## Overview

This crate provides tools for defining, verifying, and analyzing stabilizer quantum error correcting codes.

```rust
use pecos_qec::{StabilizerCode, StabilizerFlipChecker};
use pecos_core::{Xs, Zs};

// Define a code
let code = StabilizerCode::builder(3)
    .check(Zs([0, 1]))
    .check(Zs([1, 2]))
    .logical_z(Zs([0, 1, 2]))
    .logical_x(Xs([0]))
    .build()
    .unwrap();

// Analyze fault tolerance
let checker = StabilizerFlipChecker::new(&code);
let analysis = checker.analyze_weight(1);

if analysis.is_fault_tolerant() {
    println!("Code is 1-fault tolerant!");
}
```

## Modules

| Module | Purpose |
|--------|---------|
| `stabilizer_code` | Define and verify stabilizer codes |
| `distance` | Calculate code distance |
| `geometry` | Physical layout of codes |
| `surface` | Surface code implementations |
| `fault_tolerance` | Fault tolerance analysis |
| `logical_discovery` | Discover logical operators |

## Documentation

- [Levels of Abstraction](docs/levels-of-abstraction.md) - From abstract codes to dynamic programs
- [Fault Tolerance Analysis](docs/fault-tolerance.md) - Classification, analysis approaches, and symbolic infrastructure

## Key Concepts

**Fault Classification**: Errors are classified as stabilizer-equivalent (harmless), detectable (decoder can correct), or undetectable logical (fatal).

**Analysis Approaches**:
- `StabilizerFlipChecker` - Code-level analysis, no circuit needed
- `PauliPropChecker` - Circuit-level Pauli propagation
- `FaultChecker` - Full simulation with fault injection

## Related Crates

- `pecos-simulators` - Quantum simulators
- `pecos-quantum` - Circuit representation
- `pecos-decoders` - Decoder implementations
