# pecos-neo Documentation

This directory contains documentation for `pecos-neo`, a quantum circuit simulation framework with composable noise models and advanced sampling techniques.

## Documentation Index

### Getting Started

- **[design-patterns.md](design-patterns.md)** - API hierarchy, builder patterns, conventions, and best practices. **Read this first** to understand which APIs to use and how to contribute.

### Circuit Execution

- **[extended-runner.md](extended-runner.md)** - Guide to `ExtendedRunner` for GateId-based circuit execution. Covers custom gates, gate overrides, decomposition, and the unified `run()` API.

### Extensible Gate System

- **[extensible-gates-design.md](extensible-gates-design.md)** - Design document for the extensible gate system. Covers `GateId`, `GateDefinitions`, validation philosophy, and adaptor patterns.

- **[extensible-gates-test-plan.md](extensible-gates-test-plan.md)** - Test plan for the extensible gate system.

### Noise Modeling

- **[noise-flow-design.md](noise-flow-design.md)** - Design document for the composable noise flow system. Describes the primitive-based approach to noise modeling with decision trees.

- **[noise-usage-guide.md](noise-usage-guide.md)** - Practical guide to using the noise system. Covers `FlowNoiseModelBuilder`, `ComposableNoiseModel`, and common noise patterns.

### Sampling and Rare Event Estimation

- **[subset-simulation.md](subset-simulation.md)** - Guide to subset simulation for estimating very rare event probabilities (e.g., logical error rates in QEC). Covers `SubsetSimulation`, `ProperSubsetSimulation`, and `QecSubsetSimulation`.

- **[importance-sampling.md](importance-sampling.md)** - Guide to importance sampling for variance reduction in Monte Carlo estimation.

### Architecture

- **[architecture-evolution.md](architecture-evolution.md)** - How pecos-neo relates to pecos-engines. Compares DOD/functional patterns with OOP/trait patterns.

## Quick Start

### Simple Simulation with sim_neo (Recommended)

The `sim_neo` Tool API is the primary entry point for running simulations:

```rust
use pecos_neo::tool::sim_neo;
use pecos_neo::command::CommandBuilder;

let circuit = CommandBuilder::new()
    .prep(0).prep(1)
    .h(0).cx(0, 1)
    .measure(0).measure(1)
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

### Importance Sampling for Rare Events

For estimating rare event probabilities efficiently:

```rust
use pecos_neo::tool::{sim_neo, importance_sampling};

let results = sim_neo(circuit)
    .orchestrator(importance_sampling()
        .with_p1(0.001)      // Single-qubit error rate
        .with_p2(0.01)       // Two-qubit error rate
        .with_boost(10.0))   // Sample 10x more errors
    .shots(10000)
    .seed(42)
    .run();

// Compute weighted estimate of logical error rate
if let Some(error_rate) = results.weighted_mean(|outcome| {
    if check_logical_error(outcome) { 1.0 } else { 0.0 }
}) {
    println!("Logical error rate: {:.2e}", error_rate);
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

### Running with Custom Gates (ExtendedRunner)

For circuits with custom gates and decomposition:

```rust
use pecos_neo::prelude::*;
use pecos_qsim::SparseStab;

let definitions = GateDefinitions::new();
let circuit = OpBuilder::new()
    .prep_z(QubitId(0))
    .h(QubitId(0))
    .meas_z(QubitId(0), ResultId(0))
    .build();

let mut runner = ExtendedRunner::new(SparseStab::new(1), definitions)
    .with_noise(noise);

let outcomes = runner.run(&circuit)?;
```

### Subset Simulation for Very Rare Events

For probabilities below 10^-6:

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
├── tool/              # High-level Tool API (recommended entry point)
│   ├── simulation.rs  # sim_neo(), SimNeoBuilder, Orchestrator
│   ├── core.rs        # Tool, Resources, Plugin system
│   └── ...
├── command/           # Circuit representation (CommandQueue, GateCommand)
├── extensible/        # GateId-based gate system
│   ├── definitions.rs # GateDefinitions, GateSpec
│   ├── op_builder.rs  # OpBuilder for AdaptedSequence
│   ├── adaptor.rs     # Gate adaptors and decomposition
│   └── ...
├── noise/             # Noise modeling system
│   ├── flow/          # Composable primitive-based noise
│   │   ├── primitive.rs   # Core primitives (Prob, When, Sample, Seq)
│   │   ├── action.rs      # Terminal actions (Pauli, Leak, Seep, etc.)
│   │   ├── condition.rs   # Conditions (Leaked, OutcomeIs, etc.)
│   │   ├── channel.rs     # FlowChannel integration
│   │   └── builder.rs     # FlowNoiseModelBuilder
│   ├── composer.rs    # ComposableNoiseModel
│   └── ...            # Legacy channel-based noise
├── sampling/          # Advanced sampling techniques
│   ├── subset.rs      # Subset simulation
│   ├── importance*.rs # Importance sampling
│   └── ...
├── runner.rs          # ShotRunner for GateType-based execution
├── extended_runner.rs # ExtendedRunner for GateId-based execution
└── outcome.rs         # MeasurementOutcomes
```

## Key Concepts

### Orchestrators

The `sim_neo` API supports different execution strategies via orchestrators:

| Orchestrator | Use Case |
|--------------|----------|
| `Sequential` (default) | Simple sequential execution |
| `MonteCarlo { workers }` | Parallel execution for noiseless circuits |
| `ImportanceSampling` | Biased sampling for rare event estimation |

```rust
// Sequential (default)
sim_neo(circuit).shots(1000).run();

// Parallel Monte Carlo
sim_neo(circuit).workers(4).shots(1000).run();

// Importance sampling
sim_neo(circuit)
    .orchestrator(importance_sampling().with_boost(10.0))
    .shots(10000)
    .run();
```

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

### Importance Sampling

For estimating rare event probabilities (10^-3 to 10^-6):

1. Boost error rates to observe more failures
2. Track importance weights (true_rate / proposal_rate)
3. Use weighted statistics for unbiased estimates

Results include weights for proper reweighting via `weighted_mean()` or `weighted_stats()`.

### Subset Simulation

For estimating P(rare event) ~ 10^-6 or smaller:

1. Define a "score" function that increases toward failure
2. Define thresholds that decompose P(F) into conditional probabilities
3. Resample trajectories that exceed each threshold
4. Multiply conditional probabilities for final estimate
