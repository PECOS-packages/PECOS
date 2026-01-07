# pecos-hugr

Direct HUGR interpreter for PECOS.

## Purpose

Executes HUGR (Hierarchical Unified Graph Representation) programs directly without compilation to LLVM IR. Provides a classical control engine that interprets HUGR operations.

## Key Types

- `HugrEngine` - Classical control engine for HUGR programs
- `HugrEngineBuilder` - Builder pattern for engine construction
- `hugr_engine()` - Convenience function to start building

## Relationship to pecos-hugr-qis

- **pecos-hugr**: Direct interpretation of HUGR (this crate)
- **pecos-hugr-qis**: Compiles HUGR to LLVM IR for execution via QIS pipeline

## Usage

```rust
use pecos_hugr::{hugr_engine, hugr_sim};
use pecos_programs::Hugr;

let hugr = Hugr::from_file("program.hugr")?;
let results = hugr_sim(hugr).seed(42).run(100)?;
```

## Acknowledgements

This crate uses [HUGR](https://github.com/Quantinuum/hugr) (Hierarchical Unified Graph Representation), developed by Quantinuum.

**Paper:**
- Koch, M., Borgna, A., Sivarajah, S., Lawrence, A., Edgington, A., Wilson, D., Roy, C., Mondada, L., Heidemann, L., & Duncan, R. (2025). "HUGR: A Quantum-Classical Intermediate Representation." [arXiv:2510.11420](https://arxiv.org/abs/2510.11420)
