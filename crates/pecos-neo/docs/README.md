# pecos-neo Documentation

This directory contains documentation for `pecos-neo`, a quantum circuit simulation framework with composable noise models and advanced sampling techniques.

## Documentation Index

### Noise Modeling

- **[noise-flow-design.md](noise-flow-design.md)** - Design document for the composable noise flow system. Describes the primitive-based approach to noise modeling with decision trees.

- **[noise-usage-guide.md](noise-usage-guide.md)** - Practical guide to using the noise system. Covers `FlowNoiseModelBuilder`, `ComposableNoiseModel`, and common noise patterns.

### Sampling and Rare Event Estimation

- **[subset-simulation.md](subset-simulation.md)** - Guide to subset simulation for estimating very rare event probabilities (e.g., logical error rates in QEC). Covers `SubsetSimulation`, `ProperSubsetSimulation`, and `QecSubsetSimulation`.

- **[importance-sampling.md](importance-sampling.md)** - Guide to importance sampling for variance reduction in Monte Carlo estimation.

## Quick Start

### Simple Noise Model

```rust
use pecos_neo::noise::flow::FlowNoiseModelBuilder;

let noise = FlowNoiseModelBuilder::new()
    .with_p1(0.001)           // 0.1% single-qubit gate error
    .with_p2(0.01)            // 1% two-qubit gate error
    .with_p_meas(0.02, 0.03)  // Asymmetric measurement error
    .build();
```

### Running with Noise

```rust
use pecos_neo::runner::ShotRunner;
use pecos_qsim::SparseStab;

let mut runner = ShotRunner::new(SparseStab::new(num_qubits))
    .with_noise(noise);

runner.run(&circuit);
let outcomes = runner.outcomes();
```

### Estimating Rare Events

```rust
use pecos_neo::sampling::subset::{SubsetSimulation, SubsetConfig};

let config = SubsetConfig::new()
    .with_samples_per_level(1000)
    .with_seed(42);

let result = SubsetSimulation::new(circuit, num_qubits, score_fn, is_failure_fn)
    .with_noise_builder(|| Some(noise.clone()))
    .with_config(config)
    .run();

println!("P(failure) = {:.2e}", result.probability());
```

## Architecture Overview

```
pecos-neo/
├── command/        # Circuit representation (CommandQueue, GateCommand)
├── noise/          # Noise modeling system
│   ├── flow/       # Composable primitive-based noise
│   │   ├── primitive.rs   # Core primitives (Prob, When, Sample, Seq)
│   │   ├── action.rs      # Terminal actions (Pauli, Leak, Seep, etc.)
│   │   ├── condition.rs   # Conditions (Leaked, OutcomeIs, etc.)
│   │   ├── channel.rs     # FlowChannel integration
│   │   └── builder.rs     # FlowNoiseModelBuilder
│   ├── composer.rs # ComposableNoiseModel
│   └── ...         # Legacy channel-based noise
├── sampling/       # Advanced sampling techniques
│   ├── subset.rs   # Subset simulation
│   ├── importance*.rs  # Importance sampling
│   └── ...
├── runner.rs       # ShotRunner for circuit execution
└── outcome.rs      # MeasurementOutcomes
```

## Key Concepts

### Noise Primitives

The noise system is built on composable primitives:

| Primitive | Purpose |
|-----------|---------|
| `prob(p, action)` | Apply action with probability p |
| `when(cond, then, else)` | Branch based on condition |
| `sample(weights, actions)` | Weighted random choice |
| `seq([a, b, c])` | Execute all in order |
| `skip_if(cond)` | Early exit if condition met |

### NoiseResponse Types

| Response | Effect |
|----------|--------|
| `None` | No noise applied |
| `InjectGates(gates)` | Add gates after current operation |
| `SkipGate` | Remove/skip the triggering gate |
| `FlipOutcomes(qubits)` | Flip measurement outcomes |
| `ForceOutcomes([(q, v)])` | Force outcomes to specific values |
| `MarkLeaked(qubits)` | Mark qubits as leaked |

### Subset Simulation

For estimating P(rare event) ~ 10^-6 or smaller:

1. Define a "score" function that increases toward failure
2. Define thresholds that decompose P(F) into conditional probabilities
3. Resample trajectories that exceed each threshold
4. Multiply conditional probabilities for final estimate
