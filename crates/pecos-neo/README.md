# pecos-neo

Composable quantum simulation with event-driven noise modeling.

This crate provides a composable approach to quantum simulation with:

- **Typed Commands**: `GateCommand` and `CommandQueue` replacing `ByteMessage`
- **Composable Noise**: Event-driven noise channels that can be freely combined
- **Plugin System**: ECS-inspired architecture for bundling functionality
- **Program Support**: Classical control engines (QASM, HUGR) with mid-circuit measurement and feedback
- **Simple CircuitRunner**: Direct simulator execution via [`CircuitRunner`]

## Architecture

The key insight is **configuration via composition**. Instead of a monolithic noise model with dozens of parameters, you compose small, focused channels:

```
┌─────────────────────────────────────────────────────────────┐
│                   ComposableNoiseModel                      │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │
│  │ SingleQubit │ │  TwoQubit   │ │ Measurement │  ...       │
│  │   Channel   │ │   Channel   │ │   Channel   │            │
│  └─────────────┘ └─────────────┘ └─────────────┘            │
│         │               │               │                   │
│         └───────────────┴───────────────┘                   │
│                         │                                   │
│                    NoiseEvent                               │
│              (BeforeGate, AfterGate,                        │
│               AfterMeasurement, etc.)                       │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

The `sim_neo` Tool API is the recommended entry point for running simulations:

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

### Custom Noise Models

```rust
use pecos_neo::tool::sim_neo;
use pecos_neo::noise::GeneralNoiseModelBuilder;

let results = sim_neo(circuit)
    .noise(GeneralNoiseModelBuilder::new()
        .with_p1(0.001)
        .with_p2(0.01)
        .with_p_meas_symmetric(0.005))
    .shots(1000)
    .run();
```

### Parallel Execution

```rust
// Parallel Monte Carlo across 4 workers
sim_neo(circuit)
    .depolarizing(0.01)
    .workers(4)
    .shots(10000)
    .seed(42)
    .run();
```

For lower-level control, `CircuitRunner` is available directly --
see the [Noise Usage Guide](docs/noise-usage-guide.md) and [CircuitRunner Guide](docs/runner.md) docs.

## Classical Control Programs

Programs with mid-circuit measurement, conditional branching, and classical feedback
loops (e.g., multi-round QEC circuits) are supported via the `CommandSource` trait.
Any `ClassicalControlEngine` from pecos-engines -- such as `QASMEngine` or
`HugrEngine` -- can be used through the `ClassicalEngineAdapter`:

```rust
use pecos_neo::adapter::ClassicalEngineAdapter;
use pecos_neo::program::ProgramRunner;
use pecos_neo::prelude::*;
use pecos_qsim::SparseStab;

// Wrap an existing classical control engine
let mut program = ClassicalEngineAdapter::new(engine);

// ProgramRunner drives the command-measure-feedback loop
let mut runner = ProgramRunner::new(SparseStab::new(num_qubits))
    .with_noise(noise)
    .with_seed(42);

let result = runner.run_shot(&mut program);
```

The adapter translates between the engine's `ByteMessage` format and pecos-neo's
`CommandQueue`/`MeasurementOutcomes`, so classical engines get full access to
composable noise models, the plugin system, and importance sampling.

## Available Channels

| Channel | Purpose |
|---------|---------|
| `SingleQubitChannel` | Pauli/emission errors on 1Q gates |
| `TwoQubitChannel` | Pauli/emission errors on 2Q gates, angle-dependent scaling |
| `MeasurementChannel` | Asymmetric readout errors |
| `PreparationChannel` | Initialization errors and leakage |
| `IdleChannel` | T1/T2 decay, coherent/incoherent dephasing |
| `LeakageChannel` | Handles effects of leaked qubits |
| `CrosstalkChannel` | Local/global crosstalk with state-dependent transitions |

## Design Philosophy

1. **Configuration via Composition**: Build complex models from simple pieces
2. **Explicit over Implicit**: See exactly what channels are active
3. **Flexible Foundation**: The composable system supports any noise model
4. **Convenience Wrappers**: Builders provide familiar APIs for common patterns
5. **Extensible**: Easy to add custom channels without modifying core code
6. **Scalable**: DOD context enables million-qubit stabilizer simulations

## Documentation

For detailed guides and reference material, see the [docs/](docs/) directory:

- [Design Patterns](docs/design-patterns.md) -- API hierarchy, conventions, best practices
- [Noise Usage Guide](docs/noise-usage-guide.md) -- Composable noise, idle time, key types, custom channels
- [CircuitRunner](docs/runner.md) -- Custom gates, decomposition, rotation gates
- [Performance](docs/performance.md) -- Large-scale simulation benchmarks
- [Full documentation index](docs/README.md)
