# pecos-quest

Rust wrapper for the QuEST quantum simulator, implementing PECOS quantum simulator traits.

## Features

- **Dual Simulation Modes**:
  - `QuestStateVec`: Pure state vector simulation
  - `QuestDensityMatrix`: Mixed state density matrix simulation
- **PECOS Compatible**: Drop-in replacement for other PECOS simulators
- **Thread Safe**: Independent instances for parallel Monte Carlo simulations
- **Automatic Build**: QuEST v4.0.0 is downloaded and built automatically

## Quick Start

```rust
use pecos_quest::{QuestStateVec, CliffordGateable};

// Create a 2-qubit simulator
let mut state = QuestStateVec::new(2);

// Create Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
state.h(0).cx(0, 1);

// Measure qubit 0
let result = state.mz(0);
println!("Measured: {}", result.outcome);
```

## Density Matrix Simulation

```rust
use pecos_quest::{QuestDensityMatrix, CliffordGateable};

// Create mixed state simulator
let mut state = QuestDensityMatrix::new(2);

// Apply operations
state.h(0).cx(0, 1);
```

## Parallel Execution

Each simulator instance is independent, perfect for Monte Carlo simulations:

```rust
use std::thread;
use pecos_quest::{QuestStateVec, CliffordGateable};

let handles: Vec<_> = (0..4).map(|id| {
    thread::spawn(move || {
        let mut state = QuestStateVec::with_seed(2, id);
        // Each thread runs independently
        state.h(0).cx(0, 1);
        state.mz(0)
    })
}).collect();
```

## Building

```bash
# Build
cargo build --package pecos-quest

# Test
cargo test --package pecos-quest

# Run example
cargo run --package pecos-quest --example bell_state
```

### Requirements
- C++ compiler with C++14 support
- Internet connection for first build (to download QuEST)

## API Compatibility

Implements standard PECOS traits:
- `QuantumSimulator`
- `CliffordGateable`
- `ArbitraryRotationGateable`
- `RngManageable`

## License

Apache-2.0 (PECOS project license). QuEST is MIT licensed.
