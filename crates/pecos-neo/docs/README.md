# pecos-neo Documentation

Build a circuit, add noise, run it:

```rust
use pecos_neo::tool::sim_neo;
use pecos_neo::command::CommandBuilder;

let circuit = CommandBuilder::new()
    .pz(0).pz(1)
    .h(0).cx(0, 1)
    .mz(0).mz(1)
    .build();

let results = sim_neo(circuit)
    .depolarizing(0.01)
    .shots(1000)
    .seed(42)
    .run();
```

Everything chains off `sim_neo()`. The sections below show what you can plug in.

---

## What do you want to do?

### Add noise

The simplest option -- uniform depolarizing noise:

```rust
sim_neo(circuit).depolarizing(0.01).shots(1000).run();
```

Need per-channel control (single-qubit, two-qubit, measurement)?

```rust
sim_neo(circuit)
    .noise(GeneralNoiseModelBuilder::new()
        .with_p1(0.001)
        .with_p2(0.01)
        .with_p_meas_symmetric(0.005))
    .shots(1000)
    .run();
```

Noise is [event-driven](user-guides/events.md) -- channels respond to gate and
measurement events during simulation. The builders above wire this up for you.

Full guide: [Adding Noise](user-guides/noise.md) |
[Composable Noise Model](user-guides/noise-model.md) |
[Noise Channels](user-guides/noise-channels.md)

### Use non-Clifford gates

Switch to the state vector backend for T gates, arbitrary rotations, etc.:

```rust
sim_neo(circuit).quantum(state_vector()).shots(1000).run();
```

### Run in parallel

Add `.workers(n)` -- works with or without noise:

```rust
sim_neo(circuit).depolarizing(0.01).workers(4).shots(10000).seed(42).run();
```

### Estimate rare event probabilities

**Importance sampling** (10^-3 to 10^-6) -- boost error rates, reweight
results. **Subset simulation** (below 10^-6) -- decompose into conditional
probabilities. Both plug into `sim_neo()`.

Full guides: [Importance Sampling](user-guides/importance-sampling.md) |
[Subset Simulation](user-guides/subset-simulation.md)

### Override gate implementations

Swap how specific gates execute at runtime via `GateOverrides`. Useful for
custom decompositions or hardware-specific implementations.

### Hook into the execution pipeline

Register handlers for gate events, measurements, idle time, or custom signals.
This is the same event system that drives noise -- you can observe, respond to,
or inject behavior at any point in execution.

Full guide: [Events, Signals, and Handlers](user-guides/events.md)

### Run simple circuits or control the simulation step-by-step

`CircuitRunner` is a good fit when you just want to run a circuit without the
full `sim_neo` orchestration, or when you want direct control over the
simulation process -- stepping through gates, inspecting state between
operations, or integrating into your own execution loop.

Full guide: [CircuitRunner](user-guides/runner.md)

---

## Guides

| Guide | What it covers |
|-------|----------------|
| [Adding Noise](user-guides/noise.md) | Builder options, per-channel control, common presets |
| [Composable Noise Model](user-guides/noise-model.md) | Composing channels directly, plugins, mixing approaches |
| [Noise Channels](user-guides/noise-channels.md) | Built-in channels: single-qubit, two-qubit, measurement, idle, crosstalk |
| [Events, Signals, and Handlers](user-guides/events.md) | Event hooks, typed signals, DispatchContext |
| [CircuitRunner](user-guides/runner.md) | Simple circuits, step-by-step control, custom gates |
| [Importance Sampling](user-guides/importance-sampling.md) | Rare event estimation (10^-3 to 10^-6) |
| [Subset Simulation](user-guides/subset-simulation.md) | Very rare event estimation (below 10^-6) |
| [Performance](user-guides/performance.md) | Benchmarks and scaling (1M+ qubits) |

---

## Developer Docs

Full API details, custom implementations, and internals:

| Doc | What it covers |
|-----|----------------|
| [Design Patterns](dev/design-patterns.md) | API conventions, builder patterns, naming, testing standards |
| [Noise (full)](dev/noise.md) | Custom primitives, NoiseChannel trait, composable decision trees |
| [CircuitRunner (full)](dev/runner.md) | Decomposition, gate overrides, signal dispatch, full API reference |
| [Importance Sampling (full)](dev/importance-sampling.md) | ImportanceSamplingRunner API, SampleWeight, theory |
| [Subset Simulation (full)](dev/subset-simulation.md) | ProperSubsetSimulation, QEC variant, ECS trajectories |

## Design Documents

Architecture and design decisions:

- [Extensible Gates](design/extensible-gates.md) -- `GateId`, `GateDefinitions`, adaptor patterns
- [Extensible Gates Test Plan](design/extensible-gates-test-plan.md)
- [Noise Composite](design/noise-composite.md) -- Primitive-based noise decision trees
- [Signals and Dispatch](design/tags-and-dispatch.md) -- Typed signals, gate event handlers
- [Architecture Evolution](design/architecture-evolution.md) -- DOD/functional vs OOP/trait patterns
