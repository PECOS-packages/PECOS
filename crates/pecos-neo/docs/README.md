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

### Signals and Dispatch

- **[tags-and-dispatch-design.md](tags-and-dispatch-design.md)** - Design and implementation of the signal/dispatch system. Covers typed signals in the command stream, gate event handlers, `DispatchContext`, and integration with `NoiseEvent::Signal`.

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
    .pz(QubitId(0))
    .h(QubitId(0))
    .mz(QubitId(0), ResultId(0))
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
│   ├── signal_store.rs # Type-erased signal storage (SignalStore)
│   └── ...
├── extensible/        # GateId-based gate system
│   ├── definitions.rs # GateDefinitions, GateSpec
│   ├── op_builder.rs  # OpBuilder for AdaptedSequence
│   ├── adaptor.rs     # Gate adaptors and decomposition
│   └── ...
├── noise/             # Noise modeling system
│   ├── flow/          # Composable primitive-based noise
│   │   ├── primitive.rs   # Core Primitive trait and composites
│   │   ├── action.rs      # Terminal actions (Pauli, Leak, Seep, etc.)
│   │   ├── condition.rs   # Conditions (Leaked, OutcomeIs, etc.)
│   │   ├── channel.rs     # FlowChannel integration
│   │   └── builder.rs     # FlowNoiseModelBuilder
│   ├── composer.rs    # ComposableNoiseModel (Clone for parallel)
│   ├── plugin.rs      # NoisePlugin, EventHandler, ContextObserver
│   └── ...            # Channel-based noise (single/two qubit, etc.)
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
| `MonteCarlo { workers: 1 }` (default) | Single-worker execution via Tool schedule |
| `MonteCarlo { workers: n }` | Parallel execution across n workers |
| `ImportanceSampling` | Biased sampling for rare event estimation (via Tool schedule) |

All orchestrators run through the Tool/Schedule/Plugin system, so user-registered
plugins fire correctly regardless of orchestrator. `ImportanceSampling` uses
`ImportanceSamplingSimPlugin` (which replaces `UnifiedSimulationPlugin`) to run
`ImportanceSamplingRunner` inside the standard Stage pipeline.

```rust
// Default (single worker)
sim_neo(circuit).shots(1000).run();

// Parallel Monte Carlo (works with or without noise)
sim_neo(circuit).workers(4).shots(1000).run();
sim_neo(circuit).depolarizing(0.01).workers(4).shots(1000).run();

// Importance sampling
sim_neo(circuit)
    .orchestrator(importance_sampling().with_boost(10.0))
    .shots(10000)
    .run();
```

### Signals and Dispatch

The command stream can carry typed signals alongside gate operations, enabling
communication between circuit layers and noise/analysis tools:

```rust
use pecos_neo::command::CommandBuilder;
use pecos_core::Signal;

#[derive(Debug, Clone)]
struct RoundBoundary(pub usize);
impl Signal for RoundBoundary {}

let mut commands = CommandBuilder::new()
    .pz(0).h(0).cx(0, 1).mz(0).mz(1)
    .build();
commands.signal(RoundBoundary(1));  // Inject signal into command stream
```

Noise channels can observe signals via `NoiseEvent::Signal`, and gate event
handlers can react to gate operations via `DispatchContext`. See
[tags-and-dispatch-design.md](tags-and-dispatch-design.md) for full details.

### Cloneability and Parallel Execution

`ComposableNoiseModel` implements `Clone`, enabling parallel Monte Carlo
execution for noisy circuits. Each parallel worker gets its own clone of the
noise model with independent state:

```rust
// Parallel noisy simulation -- each worker clones the noise model
sim_neo(circuit)
    .depolarizing(0.01)
    .workers(4)
    .shots(10000)
    .seed(42)
    .run();
```

Custom `NoiseChannel` and `Primitive` implementations must provide a
`clone_box()` method that returns `Box<dyn NoiseChannel>` or
`Box<dyn Primitive>`. This enables cloning trait objects without requiring
`Clone` as a supertrait. See the [clone_box pattern](design-patterns.md#the-clone_box-pattern-for-trait-objects)
in the design patterns guide.

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
