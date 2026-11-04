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
    .add_channel(SingleQubitChannel::depolarizing(0.001)?)
    .add_channel(TwoQubitChannel::depolarizing(0.01)?
        .with_angle_scaling(AngleScaling::linear())?)
    .add_channel(MeasurementChannel::asymmetric(0.02, 0.03)?)
    .add_channel(IdleChannel::linear(0.0001)?
        .with_linear_depolarizing());
```

## How It Works

`NoiseModelBuilder`, `GeneralNoiseModelBuilder` and `CompositeNoiseModelBuilder`
register `CorePlugin` automatically. Its `AfterPreparation` handler marks qubits
active and clears previous leakage before preparation noise runs, so a new leak
from that same preparation remains. Its `AfterMeasurement` handler marks qubits
inactive without clearing leakage. Thus crosstalk targets other prepared,
unmeasured, non-leaked qubits, and custom `active()` conditions follow this
lifecycle. A bare `ComposableNoiseModel` needs `.add_plugin(&CorePlugin)` for
these preparation and measurement state updates.

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
    .build()?
    .add_channel(CrosstalkChannel::new()
        .with_global_rate(0.001)?
        .with_transitions(CrosstalkTransitions::symmetric_with_leakage())?);
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
    .add_plugin(DepolarizingPlugin::new(0.01, 0.02)?);  // Simple depolarizing
```

## Running It

Pass the noise model to `sim_neo` or `CircuitRunner`:

```rust
// Via sim_neo (handles shots, parallelism, seeding)
sim_neo(circuit).auto().noise(noise).sampling(monte_carlo(1000)).seed(42).run();

// Via CircuitRunner (direct control)
let mut runner = CircuitRunner::<SparseStab>::new()
    .with_noise(noise)
    .with_seed(42);
let outcomes = runner.apply_circuit(&mut state, &circuit)?;
```

`ComposableNoiseModel` implements `Clone`, so parallel workers each get an
independent copy with their own state:

```rust
sim_neo(circuit).auto().noise(noise).sampling(monte_carlo(10000).workers(4)).run();
```

## Going Deeper

For implementing custom `NoiseChannel` and `NoisePlugin` traits, the
`NoiseContext` API, and the `EventHandler`/`ContextObserver` system, see the
[developer noise guide](../dev/noise.md).

Gate-event `qubits` contains executed groups only after a gate. Custom channels
that model seepage, partner noise or idle effects on suppressed groups must read
`leaked_suppressed` explicitly; `gated_qubits()` includes both sets. Composite
filters classify by `group_size` and now also fire on batched commands, with
partner conditions evaluated within each group. On noisy operations, suppressed groups run primitives
only on currently leaked members.

The seepage parameter is conditional on an emission fault: its marginal rate is
`p * emission_ratio * seepage`, where `p` includes angle scaling. Leakage scale
does not scale seepage. Two-qubit channels share the outer fault and emission
draws per pair, including through `channel_action`; composite builders
preserve the per-qubit marginal rate.

`channel_action` runs its wrapped channel once per operand group per stage. The
leader is the first eligible member at the start of each stage (the first leaked
member of a suppressed group, otherwise member 0), and remains fixed within that
stage. Probability and condition guards decide on the leader's invocation. A
wrapped skip stops the sequence on every later member it names, without emitting
duplicate skip marks; unnamed members continue. Leak and unleak marks retain all
named targets.

On geometric single-pass non-gate events, `Primitive::reads_operands` opts into
installing all operands before sampling, independently of whether member 0 is
sampled. Actions and conditions declare the same capability, and combinators
forward it from their children. Custom readers using `current_gate_qubits()`,
`other_qubit()` or partner conditions must declare it; otherwise they see no
operands on that path. Function conditions opt in with
`FnCondition::with_operand_reads()`. `channel_action` declares operand reads;
`pauli()` does not and keeps O(n*p) processing. Linear and two-pass paths always
install operands. Every path sets the current member index.

`SingleQubitChannel` and `TwoQubitChannel` discard a seeping qubit's state by
sampling I, X, Y or Z uniformly and independently for each seeping qubit. Identity
injects no gate. This fully depolarizing channel also discards correlations with
other qubits. The draw happens only after seep selection, for both executed and
leaked-suppressed groups; it does not change the seepage rate. No reset gate is
injected.
Composite `seep()` retains its configurable Pauli weights. `unleak()` and
`NoiseResponse::MarkUnleaked` clear only the leakage flag and preserve the
simulator state; use `seep()` for the physical channel.

### Noiseless operations and leakage

All three noise builders always register leakage suppression, even with zero
fault rates or custom leakage sources. Gates on leaked operands never run,
including noiseless gates. Partner Pauli noise defaults to zero; opt in on
`LeakageChannel` with `with_partner_pauli_probability(p)`. Each complete pair
with exactly one leaked member receives a probability draw, then a uniform
X/Y/Z on the healthy partner if it succeeds. This covers executed and
leaked-suppressed groups. Noiseless operations produce neither draws nor faults.

`with_noiseless_gate` and `with_noiseless_gates` suppress faults from the declared
operation: injected gates, readout flips, leakage, seepage, crosstalk and
idle-after-two-qubit noise. This includes composite primitives on suppressed
groups. A scheduled `IdleTime` still produces its configured noise. Preparation
still clears prior leakage, and lifecycle tracking still runs. A fault primitive
placed on a `BeforeGate` composite is not silenced: that event decides suppression.
Custom non-composite fault channels must check the noiseless setting themselves.

`AfterPreparation`, `BeforeMeasurement` and `AfterMeasurement` carry the actual
`gate_type`, distinguishing PZ from QAlloc and MZ from MeasureLeaked.
On `CircuitRunner`'s command queue, leaked MZ reads 1 and leaked MeasureLeaked
reports leaked, including when noiseless. `LeakageChannel` never changes an
older measurement record. `execute_measure` and `ImportanceSamplingRunner`
retain their separate measurement behavior without a leakage check.
