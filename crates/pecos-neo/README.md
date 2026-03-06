# pecos-neo

Composable quantum simulation with event-driven noise modeling.

## Quick Start

The `sim_neo` Tool API is the recommended entry point:

```rust
use pecos_neo::tool::sim_neo;
use pecos_neo::command::CommandBuilder;

let circuit = CommandBuilder::new()
    .pz(0).pz(1)
    .h(0).cx(0, 1)
    .mz(0).mz(1)
    .build();

// Run 1000 shots with depolarizing noise
let results = sim_neo(circuit)
    .depolarizing(0.01)
    .shots(1000)
    .seed(42)
    .run();

for outcome in &results.outcomes {
    println!("{:?}", outcome);
}
```

## Features

- **Composable Noise**: Event-driven channels that combine freely -- depolarizing, measurement, idle, crosstalk, leakage, and custom channels
- **Typed Commands**: `GateCommand` and `CommandQueue` with signal support for metadata alongside gates
- **Plugin System**: ECS-inspired architecture for bundling simulation functionality
- **Parallel Execution**: Monte Carlo across multiple workers with `.workers(n)`
- **Advanced Sampling**: Importance sampling and subset simulation for rare event estimation
- **Extensible Gates**: `GateId`-based system with runtime overrides and decomposition
- **Program Support**: Classical control engines (QASM, HUGR) with mid-circuit measurement and feedback
- **State Vector**: Non-Clifford gates (T, rotations) via `state_vector()` backend

## Documentation

See the [full documentation](docs/README.md) for examples, guides, and reference material.
