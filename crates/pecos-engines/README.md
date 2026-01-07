# pecos-engines

Simulation engine infrastructure for PECOS.

## Purpose

Provides the core simulation framework: engine traits, Monte Carlo simulation, noise models, and the unified `sim()` API.

## Key Types

- `ClassicalControlEngine` trait - Interface for classical control engines
- `MonteCarloEngine` - Multi-shot simulation with noise
- `SimBuilder` - Builder for configuring simulations
- `ShotResult`, `ShotVec` - Simulation results

## Noise Models

- `DepolarizingNoise` - Simple depolarizing channel
- `NoisyQuantumEngineBuilder` - Add noise to any quantum backend

## Usage

```rust
use pecos_engines::{sim, sparse_stab};

let results = sim(engine_builder)
    .quantum(sparse_stab())
    .seed(42)
    .run(1000)?;
```
