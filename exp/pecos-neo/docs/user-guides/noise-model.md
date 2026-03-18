# Composable Noise Model

The builder in [Adding Noise](noise.md) covers the common case. When you want
to pick exactly which channels to include, compose a `ComposableNoiseModel`
directly.

## Building from Channels

```rust
use pecos_neo::prelude::*;
use pecos_neo::noise::*;
use pecos_neo::noise::plugins::CorePlugin;

let noise = ComposableNoiseModel::new()
    .add_plugin(CorePlugin)                                // State tracking (prep, meas, leakage)
    .add_channel(SingleQubitChannel::depolarizing(0.001))
    .add_channel(TwoQubitChannel::depolarizing(0.01)
        .with_angle_scaling(AngleScaling::linear()))
    .add_channel(MeasurementChannel::asymmetric(0.02, 0.03))
    .add_channel(IdleChannel::linear(0.0001)
        .with_linear_depolarizing());
```

## How It Works

The noise model is event-driven. During simulation, the runner emits events
at each point in execution -- `BeforeGate`, `AfterGate`, `AfterMeasurement`,
`IdleTime`, `Signal`, etc. For each event:

1. **Event handlers** run first (state tracking: which qubits are prepared, leaked, etc.)
2. **Channels** that respond to this event produce a `NoiseResponse` (inject errors, flip outcomes, skip gates)
3. **Observers** react to state changes caused by the responses (e.g., leakage side effects)

Each channel self-selects which events it cares about. A `SingleQubitChannel`
fires on `AfterGate` for single-qubit gates. A `MeasurementChannel` fires on
`AfterMeasurement`. Custom channels can respond to any event, including
user-defined [signals](events.md).

See [Noise Channels](noise-channels.md) for the full list of built-in
channels and [Events, Signals, and Handlers](events.md) for the event system.

## Mixing Builder + Custom Channels

You don't have to choose between the builder and manual composition. Start
with a builder for the basics, then add custom channels on top:

```rust
let noise = GeneralNoiseModelBuilder::new()
    .with_p1(0.001)
    .with_p2(0.01)
    .build()
    .add_channel(CrosstalkChannel::new()
        .with_global_rate(0.001)
        .with_transitions(CrosstalkTransitions::symmetric_with_leakage()));
```

This is useful when the builder covers most of what you need but you want one
or two specialized channels that it doesn't expose.

## Plugins

Plugins bundle related channels, event handlers, and observers into a single
unit. This is how `CorePlugin` provides state tracking, and how you can
package your own noise logic for reuse:

```rust
let noise = ComposableNoiseModel::new()
    .add_plugin(CorePlugin)                            // State tracking
    .add_plugin(LeakagePlugin::new())                  // Leakage handling
    .add_plugin(DepolarizingPlugin::new(0.01, 0.02));  // Simple depolarizing
```

## Running It

Pass the noise model to `sim_neo` or `CircuitRunner`:

```rust
// Via sim_neo (handles shots, parallelism, seeding)
sim_neo(circuit).noise(noise).shots(1000).seed(42).run();

// Via CircuitRunner (direct control)
let mut runner = CircuitRunner::<SparseStab>::new()
    .with_noise(noise)
    .with_seed(42);
let outcomes = runner.apply_circuit(&mut state, &circuit)?;
```

`ComposableNoiseModel` implements `Clone`, so parallel workers each get an
independent copy with their own state:

```rust
sim_neo(circuit).noise(noise).workers(4).shots(10000).run();
```

## Going Deeper

For implementing custom `NoiseChannel` and `NoisePlugin` traits, the
`NoiseContext` API, and the `EventHandler`/`ContextObserver` system, see the
[developer noise guide](../dev/noise.md).
