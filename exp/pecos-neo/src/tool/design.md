# Tool Architecture Design

## Overview

This document describes the Bevy-inspired `Tool` architecture for pecos-neo, providing a
flexible, plugin-based system for building various quantum simulation and validation tools.

## Design Goals

1. **Bevy-inspired**: Follow `App::new()` patterns with plugins, systems, and resources
2. **Composable**: Different tools built from the same foundation via plugins
3. **Reusable**: Build once, run many times with reconfiguration between runs
4. **Convenient**: Specialized builders like `sim_neo()` mirror `sim()` ergonomics
5. **Extensible**: Easy to add new tool types (simulation, FT validation, etc.)

## Architecture

```
                                    ┌─────────────────────┐
                                    │     sim_neo()       │  Convenience entry points
                                    │  ft_validator_neo() │  (like sim() in pecos)
                                    └──────────┬──────────┘
                                               │ creates
                                               ▼
                                    ┌─────────────────────┐
                                    │   SimNeoBuilder     │  Specialized builders
                                    │   FTValidatorBuilder│  (configure Tool for purpose)
                                    └──────────┬──────────┘
                                               │ .build()
                                               ▼
                                    ┌─────────────────────┐
                                    │    Simulation       │  Reusable handles
                                    │    FTValidator      │  (wrap Tool, allow reconfig)
                                    └──────────┬──────────┘
                                               │ wraps
                                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                                   Tool                                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │  Plugins    │  │  Systems    │  │  Resources  │  │   World     │          │
│  │             │  │             │  │             │  │             │          │
│  │ Simulation  │  │ Startup     │  │ Circuit     │  │ Entities    │          │
│  │ Noise       │  │ PreShot     │  │ SimConfig   │  │ Components  │          │
│  │ Importance  │  │ Execute     │  │ NoiseModel  │  │             │          │
│  │ FTValidation│  │ PostShot    │  │ Results     │  │             │          │
│  │ Decoder     │  │ Finish      │  │ Statistics  │  │             │          │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘          │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### Tool

The foundation - a generic, Bevy-like application container.

```rust
pub struct Tool {
    world: World,                    // ECS world with entities/components
    resources: ResourceStorage,      // Typed resource storage
    plugins: Vec<Box<dyn Plugin>>,   // Registered plugins
    systems: SystemSchedule,         // Systems organized by stage
}

impl Tool {
    pub fn new() -> Self { ... }

    // Plugin management
    pub fn add_plugin<P: Plugin>(self, plugin: P) -> Self { ... }
    pub fn add_plugins<G: PluginGroup>(self, group: G) -> Self { ... }

    // Resource management
    pub fn insert_resource<R: Resource>(self, resource: R) -> Self { ... }
    pub fn resource<R: Resource>(&self) -> &R { ... }
    pub fn resource_mut<R: Resource>(&mut self) -> &mut R { ... }
    pub fn take_resource<R: Resource>(&mut self) -> R { ... }

    // System scheduling
    pub fn add_system<S: System>(self, stage: Stage, system: S) -> Self { ... }

    // Execution
    pub fn run(&mut self) { ... }
}
```

### Stages

Execution stages for quantum tool workflows:

```rust
pub enum Stage {
    Startup,    // Once at beginning (init simulators, compile circuits)
    PreShot,    // Before each shot (reset state, derive seed)
    Execute,    // Run the circuit with noise
    PostShot,   // After each shot (collect outcomes, update weights)
    Finish,     // Once at end (aggregate results, compute statistics)
}
```

### Plugin Trait

Plugins bundle related resources and systems:

```rust
pub trait Plugin: Send + Sync {
    fn build(&self, tool: &mut Tool);
}

pub trait PluginGroup {
    fn build(self, tool: &mut Tool);
}
```

### Resources

Typed data storage accessible during execution:

```rust
pub trait Resource: Send + Sync + 'static {}

// Example resources
pub struct Circuit(pub CommandQueue);
pub struct SimConfig {
    pub shots: usize,
    pub seed: Option<u64>,
}
pub struct SimulationResults { ... }
```

## Simulation Tool

### sim_neo() Entry Point

Convenience function that creates a simulation-configured builder:

```rust
/// Create a simulation builder for a circuit
///
/// # Example
/// ```
/// let results = sim_neo(circuit)
///     .noise(SingleQubitChannel::depolarizing(0.01))
///     .shots(1000)
///     .seed(42)
///     .build()
///     .run();
/// ```
pub fn sim_neo(circuit: impl Into<CommandQueue>) -> SimNeoBuilder {
    SimNeoBuilder::new(circuit.into())
}
```

### SimNeoBuilder

Builder that configures a Tool for simulation:

```rust
pub struct SimNeoBuilder {
    circuit: CommandQueue,
    noise: Option<ComposableNoiseModel>,
    shots: usize,
    seed: Option<u64>,
    workers: usize,
    importance_sampling: Option<ImportanceConfig>,
}

impl SimNeoBuilder {
    pub fn new(circuit: CommandQueue) -> Self { ... }

    // Configuration (consumes self, returns Self)
    pub fn shots(mut self, shots: usize) -> Self { ... }
    pub fn seed(mut self, seed: u64) -> Self { ... }
    pub fn workers(mut self, workers: usize) -> Self { ... }
    pub fn auto_workers(mut self) -> Self { ... }
    pub fn noise(mut self, noise: impl Into<ComposableNoiseModel>) -> Self { ... }
    pub fn importance_sampling(mut self, base_rate: f64, boost: f64) -> Self { ... }

    /// Build the simulation handle
    pub fn build(self) -> Simulation {
        let tool = Tool::new()
            .add_plugin(UnifiedSimulationPlugin { explicit_num_qubits: None })
            .insert_resource(ProgramSourceResource(source))
            .insert_resource(self.config)
            .insert_resource(QuantumBackendResource(self.quantum_backend));

        Simulation { tool, orchestrator: self.orchestrator, parallel_data }
    }
}
```

### Simulation Handle

Reusable handle that wraps a configured Tool:

```rust
pub struct Simulation {
    tool: Tool,
    orchestrator: Orchestrator,
    parallel_data: Option<ParallelExecutionData>,
}

impl Simulation {
    /// Reconfigure shots before next run
    pub fn shots(&mut self, shots: usize) -> &mut Self {
        self.tool.resource_mut::<SimConfig>().shots = shots;
        self
    }

    /// Reconfigure seed before next run
    pub fn seed(&mut self, seed: u64) -> &mut Self {
        self.tool.resource_mut::<SimConfig>().seed = Some(seed);
        self
    }

    /// Execute simulation with current configuration
    pub fn run(&mut self) -> SimulationResults {
        self.tool.run();
        self.tool.take_resource::<SimulationResults>()
    }
}
```

### Usage Patterns

```rust
// Pattern 1: One-shot (builder consumed)
let results = sim_neo(circuit)
    .noise(depolarizing(0.01))
    .shots(1000)
    .seed(42)
    .build()
    .run();

// Pattern 2: Build once, run many
let mut sim = sim_neo(circuit)
    .noise(depolarizing(0.01))
    .shots(1000)
    .build();

let results1 = sim.run();

sim.seed(123).shots(2000);
let results2 = sim.run();

// Pattern 3: Fluent reconfiguration
let results3 = sim.shots(5000).seed(456).run();
```

## Plugins

### UnifiedSimulationPlugin

Core simulation functionality. Handles both static circuits and classical engines
via a unified `QuantumRunner` + `CommandSource` abstraction. The same systems run
in both single-worker and parallel execution (each parallel worker gets its own
`Resources` and runs the shared schedule).

```rust
struct UnifiedSimulationPlugin {
    explicit_num_qubits: Option<usize>,
}

impl Plugin for UnifiedSimulationPlugin {
    fn build(&self, tool: &mut Tool) {
        tool.insert_resource(SimulationResults::new())
            .insert_resource(ExplicitNumQubits(self.explicit_num_qubits))
            .add_system(Stage::Startup, unified_simulation_startup)
            .add_system(Stage::PreShot, unified_simulation_pre_shot)
            .add_system(Stage::Execute, unified_simulation_execute)
            .add_system(Stage::PostShot, unified_simulation_post_shot);
    }
}
```

### NoisePlugin

Noise model integration:

```rust
pub struct NoisePlugin {
    noise: Option<ComposableNoiseModel>,
}

impl Plugin for NoisePlugin {
    fn build(&self, tool: &mut Tool) {
        if let Some(noise) = self.noise.take() {
            tool.insert_resource(NoiseModel(noise))
                .add_system(Stage::Execute, apply_noise);
        }
    }
}
```

### `ImportanceSamplingSimPlugin`

When the importance sampling orchestrator is selected, this plugin replaces
`UnifiedSimulationPlugin`. It uses `ImportanceSamplingRunner` internally for
biased noise with weight tracking, running through the same Stage/Schedule
system as Monte Carlo execution.

```rust
struct ImportanceSamplingSimPlugin {
    is_config: ImportanceSamplingBuilder,
    explicit_num_qubits: Option<usize>,
}

impl Plugin for ImportanceSamplingSimPlugin {
    fn build(&self, tool: &mut Tool) {
        // Registers IS-specific systems at Startup/PreShot/Execute/PostShot
        // Uses ImportanceSamplingRunner<SparseStab> for execution
    }
}
```

This means user-registered plugins and hooks fire correctly during IS execution,
just as they do for Monte Carlo.

### `ImportanceSamplingPlugin`

A standalone plugin for manual weight tracking (pre-shot reset, post-shot
recording, finish statistics). Available for users building custom Tool
configurations. Not used by the IS orchestrator path, which handles its own
weight storage via `SimulationResults.weights`.

## Results

### SimulationResults

Output from simulation runs:

```rust
pub struct SimulationResults {
    /// Per-shot measurement outcomes
    pub outcomes: Vec<MeasurementOutcomes>,
    /// Configuration used for this run
    pub config: SimConfig,
    /// Computed statistics (if applicable)
    pub statistics: Option<Statistics>,
}

impl SimulationResults {
    pub fn len(&self) -> usize { ... }
    pub fn counts(&self) -> BTreeMap<String, usize> { ... }
    pub fn probability(&self, outcome: &str) -> f64 { ... }
    pub fn success_rate<F: Fn(&MeasurementOutcomes) -> bool>(&self, predicate: F) -> f64 { ... }
}
```

## Future Extensions

### Fault Tolerance Validation

```rust
pub fn ft_validator_neo(
    circuit: CommandQueue,
    decoder: impl Decoder,
) -> FTValidatorBuilder { ... }

pub struct FTValidatorBuilder { ... }

impl FTValidatorBuilder {
    pub fn error_rate(self, rate: f64) -> Self { ... }
    pub fn rounds(self, rounds: usize) -> Self { ... }
    pub fn build(self) -> FTValidator { ... }
}

pub struct FTValidator {
    tool: Tool,
}

impl FTValidator {
    pub fn run(&mut self) -> FTValidationResults { ... }
}
```

### Custom Tools

Users can build custom tools directly:

```rust
let mut tool = Tool::new()
    .add_plugin(MyCustomPlugin)
    .insert_resource(MyConfig { ... })
    .add_system(Stage::Execute, my_custom_system);

tool.run();
let results = tool.resource::<MyResults>();
```

## Comparison with pecos-engines

| pecos-engines | pecos-neo | Purpose |
|---------------|-----------|---------|
| `sim(program)` | `sim_neo(circuit)` | Entry point |
| `SimBuilder` | `SimNeoBuilder` | Configuration builder |
| `MonteCarloEngine` | `Simulation` (wraps `Tool`) | Reusable execution handle |
| `ShotVec` | `SimulationResults` | Output type |
| - | `Tool` | Generic Bevy-like foundation |
| - | `Plugin` | Composable functionality bundles |

## Implementation Phases

### Phase 1: Core Tool Infrastructure
- [ ] `Tool` struct with resource storage
- [ ] `Plugin` trait and plugin registration
- [ ] `Stage` enum and system scheduling
- [ ] Basic `run()` execution loop

### Phase 2: Simulation Support
- [x] `UnifiedSimulationPlugin` with core systems
- [ ] `NoisePlugin` integration
- [x] `SimNeoBuilder` and `sim_neo()`
- [x] `Simulation` handle with reconfiguration
- [x] `SimulationResults` type

### Phase 3: Advanced Features
- [x] `ImportanceSamplingPlugin`
- [x] Parallel execution (MonteCarlo orchestrator with workers > 1)
- [ ] `FTValidatorBuilder` and FT validation

### Phase 4: Integration
- [ ] Python bindings via pecos-rslib
- [ ] Documentation and examples
- [ ] Migration guide from `MonteCarloRunner`
