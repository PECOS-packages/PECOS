# Noise System Usage Guide

Composite decision-tree APIs require the `composite-noise` Cargo feature on `pecos-neo`.

This guide covers practical usage of the pecos-neo noise modeling system.

## Overview

pecos-neo provides two approaches to noise modeling:

1. **CompositeNoiseModelBuilder** (recommended) - Parameter-based builder with sensible defaults
2. **Custom primitives** - Build noise decision trees from composable primitives

Both produce a `ComposableNoiseModel` that integrates with `CircuitRunner`.

## CompositeNoiseModelBuilder

### Basic Usage

```rust
use pecos_neo::noise::composite::CompositeNoiseModelBuilder;

let noise = CompositeNoiseModelBuilder::new()
    .with_p1(0.001).unwrap()           // Single-qubit gate error rate
    .with_p2(0.01).unwrap()            // Two-qubit gate error rate
    .with_p_meas(0.02, 0.03).unwrap()  // Measurement error (0→1, 1→0)
    .build().unwrap();
```

### All Parameters

#### Single-Qubit Gates

```rust
CompositeNoiseModelBuilder::new()
    // Base error probability
    .with_p1(0.001)?

    // Of errors, what fraction are emission (vs Pauli)
    .with_p1_emission_ratio(0.1)?

    // Seepage probability (leaked qubit returns to computational space)
    .with_p1_seepage(0.05)?

    // Custom Pauli weights (default: uniform X, Y, Z)
    .with_p1_pauli_weights(PauliWeights::new(0.25, 0.25, 0.50))?  // More Z errors

    // Custom emission model
    .with_p1_emission_model(SingleQubitEmissionWeights { ... })?
```

#### Two-Qubit Gates

```rust
CompositeNoiseModelBuilder::new()
    // Base error probability
    .with_p2(0.01)?

    // Emission vs Pauli ratio
    .with_p2_emission_ratio(0.1)?

    // Seepage for leaked qubits
    .with_p2_seepage(0.05)?

    // Angle-dependent scaling (for RZZ, etc.)
    .with_p2_angle_scaling(AngleScaling::Quadratic)

    // Apply configured idle noise for one time unit after each two-qubit gate
    .with_p_idle_linear(0.001)?
    .with_idle_after_2q(1.0)?

    // Custom two-qubit Pauli model
    .with_p2_pauli_model(TwoQubitPauliWeights { ... })?
```

#### Preparation

```rust
CompositeNoiseModelBuilder::new()
    // Preparation error rate
    .with_p_prep(0.001).unwrap()

    // Of prep errors, fraction that leak (vs bit flip)
    .with_p_prep_leak_ratio(0.1).unwrap()

    // Crosstalk during preparation
    .with_p_prep_crosstalk(0.0001).unwrap()
```

#### Measurement

```rust
CompositeNoiseModelBuilder::new()
    // Asymmetric measurement error
    .with_p_meas(0.02, 0.03).unwrap()  // P(report 1 | true 0), P(report 0 | true 1)

    // Symmetric measurement error
    .with_p_meas_symmetric(0.02).unwrap()
```

#### Leakage

```rust
CompositeNoiseModelBuilder::new()
    // Leakage and seepage rates
    .with_leakage(
        0.001,  // Emission ratio within gate faults
        0.1,    // Conditional seepage probability after a gate emission fault
    ).unwrap()
```

#### Idle/Decoherence

```rust
CompositeNoiseModelBuilder::new()
    // T1/T2 style decoherence
    .with_idle_t1_t2(50e-6, 30e-6).unwrap()  // T1, T2 in seconds

    // Set time scale for idle calculations
    .with_time_scale(TimeScale::Nanoseconds)
```

#### Crosstalk

Preparation crosstalk fires only after noisy `PZ`, at its own global rate.
Measurement crosstalk fires after measurement, at independent local/global rates.
Both default to Z with probability 1/2 per fired target. Source operands and leaked
qubits are excluded; other qubits prepared this shot remain targets after measurement,
including noiselessly prepared and `QAlloc` qubits. Presets and user-composed composite
channels retain their chosen primitives. Tables retain their sampled-state behavior.

```rust
CompositeNoiseModelBuilder::new()
    .with_p_prep_crosstalk(0.001)?
    .with_p_meas_crosstalk(0.001, 0.01)?
    .with_p_meas_crosstalk_local_fn(neighbors)
    .with_p_meas_crosstalk_model(CrosstalkTransitions::symmetric_with_leakage())?
    .build()?;
```

## Using Noise with CircuitRunner

```rust
use pecos_neo::runner::CircuitRunner;
use pecos_neo::command::CommandBuilder;
use pecos_simulators::SparseStab;

// Build circuit
let circuit = CommandBuilder::new()
    .pz(0)
    .pz(1)
    .h(0)
    .cx(0, 1)
    .mz(0)
    .mz(1)
    .build();

// Build noise model
let noise = CompositeNoiseModelBuilder::new()
    .with_p1(0.001)?
    .with_p2(0.01)?
    .build()?;

// Run with noise
let mut state = SparseStab::new(2);
let mut runner = CircuitRunner::<SparseStab>::new()
    .with_noise(noise)
    .with_seed(42);

let outcomes = runner.apply_circuit(&mut state, &circuit)?;
println!("Qubit 0: {}", outcomes.get_bit(QubitId(0))?);
println!("Qubit 1: {}", outcomes.get_bit(QubitId(1))?);
```

## Custom Noise with Primitives

For advanced customization, build noise directly from primitives:

```rust
use pecos_neo::noise::composite::prelude::*;
use pecos_neo::noise::composite::{CompositeChannel, CompositeEventFilter};

// Custom single-qubit noise: skip leaked qubits, then apply depolarizing
let sq_primitive = seq(vec![
    skip_if_leaked(),
    prob(0.001, depolarize()).unwrap(),
]);

// Create channel
let channel = CompositeChannel::new("custom_sq", sq_primitive)
    .with_filter(CompositeEventFilter::SingleQubitGate);

// Add to model
let model = ComposableNoiseModel::new()
    .add_channel(channel);
```

### Available Primitives

#### Control Flow

```rust
// Probability gate
prob(0.01, action).unwrap()

// Dynamic probability based on gate
let p = Probability::new(0.01)?;
prob_fn(move |_gate| p, action)

// Conditional
when(leaked(), then_action, else_action)

// Weighted random choice
sample(vec![
    (0.8, nothing()),
    (0.15, pauli()),
    (0.05, leak()),
])

// Sequential
seq(vec![action1, action2, action3])

// Early exit
skip_if(condition)
skip_if_leaked()
```

#### Conditions

```rust
leaked()           // Qubit is in leaked state
not_leaked()       // Qubit is not leaked
outcome_is(true)   // Measurement outcome was 1
partner_leaked()   // Other qubit in 2Q gate is leaked
any_qubit_leaked() // Any qubit in multi-qubit gate is leaked
always()           // Always true
never()            // Always false
```

#### Actions

```rust
// No-op
nothing()

// Pauli errors
pauli()                    // Uniform X, Y, Z
pauli_weighted(0.3, 0.3, 0.4)?  // Custom weights
depolarize()               // Same as uniform pauli
dephase()                  // Z error only

// Leakage
leak()                     // Mark qubit as leaked
prob(0.1, seep())?          // With prob 0.1, unleak and apply uniform I/X/Y/Z
unleak()                   // Clear the flag only; keep the simulator state

// Gate injection
inject(GateCommand::x(qubit))

// Skip gate
skip_gate()

// Measurement manipulation
flip_outcome()             // Flip 0↔1
force_outcome(true)        // Force to specific value
```

## NoiseResponse Handling

When noise is applied, it produces a `NoiseResponse` that the runner processes:

| Response | Effect |
|----------|--------|
| `NoiseResponse::None` | No effect |
| `NoiseResponse::InjectGates(gates)` | Insert gates after current operation |
| `NoiseResponse::SkipGate(qubits)` | Suppress operand groups containing a listed qubit; list all operands to skip the whole command |
| `NoiseResponse::FlipOutcomes(qubits)` | Toggle measurement outcomes for qubits |
| `NoiseResponse::ForceOutcomes(pairs)` | Set outcomes to specific values |
| `NoiseResponse::MarkLeaked(qubits)` | Mark qubits as leaked in context |
| `NoiseResponse::Multiple(responses)` | Combine multiple responses |

After-gate `qubits` contains executed operand groups only. `leaked_suppressed`
contains complete groups skipped while leaked, in original order within each
slice. The partition is fixed immediately after before-gate dispatch. A later
unleak never promotes a suppressed group to executed, and a later leak never
restores a dropped group. A qubit that seeps in the same before-gate emit that
skipped it is healthy at this decision, so its group is dropped.

A command with only leaked-suppressed groups still emits, with empty `qubits`.
An all-dropped, originally non-empty command emits no after-gate event. Empty
operands retain their event. Queue after-gate handlers receive the same split;
adapted execution does not invoke user gate handlers.

Gate-error channels use `qubits`. Seepage channels explicitly read suppressed
groups. `LeakageChannel` applies partner noise there only when explicitly enabled
with `with_partner_pauli_probability(p)` (default 0). On noisy operations, idle-after-two-qubit
noise acts on healthy members of executed and suppressed groups. Dropped groups
do not receive this idle noise. Crosstalk has no gate-event sources.
`gated_qubits()` returns both slices without allocating when
there are no suppressed groups.

`group_size` on before/after events is a `NonZeroUsize`. Queue commands use
the gate type arity; adapted commands use the registered arity. A positive
arity that divides non-empty operands is used directly; otherwise the size
is the whole operand count, or one for empty operands.
`NoiseContext::set_current_gate` also takes a `NonZeroUsize`.
Composite filters classify by this size, including custom gates represented as
`GateType::I`. Primitives see only their own group for partner/leakage conditions.
On noisy operations, suppressed groups run primitives only on currently leaked members.

Single- and two-qubit seepage has marginal rate `p * r * s`: gate error `p`
(angle-scaled for two-qubit gates), raw emission ratio `r`, and conditional
seepage `s`. Leakage scale
does not enter. Two-qubit channels share the `p` and `r` draws per pair, then draw
`s` for each currently leaked member in operand order. Composite builders sample
per qubit, so their both-leaked joint rate differs. Seeded results change.

Handler leakage marks are applied with observers before combining the handler
response with noise-model responses. Observer responses are combined, but marks
inside observer responses are not applied recursively. Without a noise model,
a handler mark returns `ExecutionError::HandlerLeakageWithoutNoise`.

## Noise Context

`NoiseModelBuilder`, `GeneralNoiseModelBuilder` and `CompositeNoiseModelBuilder`
register `CorePlugin` automatically. Its `AfterPreparation` handler marks qubits
active and clears previous leakage before preparation noise runs, so a new leak
from that same preparation remains. Its `AfterMeasurement` handler marks qubits
inactive without clearing leakage. Crosstalk targets other non-leaked qubits
prepared this shot, including measured, noiselessly prepared and `QAlloc` qubits.
Custom `active()` conditions still exclude measured qubits. A bare
`ComposableNoiseModel` needs `.add_plugin(&CorePlugin)` for
these preparation and measurement state updates.

`SingleQubitChannel` and `TwoQubitChannel` discard a seeping qubit's state by
sampling I, X, Y or Z uniformly and independently for each seeping qubit. Identity
injects no gate. This fully depolarizing channel also discards correlations with
other qubits. The draw happens only after seep selection, for both executed and
leaked-suppressed groups; it does not change the seepage rate. No reset gate is
injected.
Composite `seep()` retains its configurable Pauli weights. `unleak()` and
`NoiseResponse::MarkUnleaked` clear only the leakage flag and preserve the
simulator state; use `seep()` for the physical channel.

The `NoiseContext` tracks simulation state for noise decisions:

```rust
// Check qubit state
ctx.is_leaked(qubit)
ctx.is_active(qubit)

// Get current gate info
ctx.current_gate()
ctx.current_gate().map(|g| g.angle())

// For measurement noise
ctx.current_outcome()  // The outcome being processed

// For multi-qubit gates
ctx.other_qubit()  // Partner qubit in 2Q gate
```

## Performance Considerations

### Geometric Sampling

For low error rates with many qubits, the system uses geometric sampling to skip directly to the next error instead of checking each qubit:

```rust
// Automatic for channels with probability < 0.01 and > 100 qubits
let channel = CompositeChannel::new("fast_depol", prob(0.001, depolarize()).unwrap())
    .with_probability(0.001).unwrap()  // Enables geometric sampling
    .with_filter(CompositeEventFilter::SingleQubitGate);
```

## Debugging Noise Models

### Introspection

```rust
use pecos_neo::noise::introspection::DescribeTree;

// Print primitive tree structure
println!("{}", primitive.describe_tree());

// Get model summary
let summary = model.summarize();
println!("Channels: {:?}", summary.channels_by_event);
```

### Validation

```rust
use pecos_neo::noise::validation::validate_noise_model;

let warnings = validate_noise_model(&model);
for warning in warnings {
    eprintln!("Warning: {}", warning);
}
```

## Common Patterns

### Ion Trap Style

```rust
let noise = CompositeNoiseModelBuilder::new()
    .with_p1(0.0001).unwrap()
    .with_p2(0.001).unwrap()
    .with_p1_emission_ratio(0.1).unwrap()
    .with_p2_emission_ratio(0.1).unwrap()
    .with_leakage(0.0001, 0.1).unwrap()
    .with_p_meas(0.001, 0.01).unwrap()
    .build().unwrap();
```

### Superconducting Style

```rust
let noise = CompositeNoiseModelBuilder::new()
    .with_p1(0.001).unwrap()
    .with_p2(0.01).unwrap()
    .with_idle_t1_t2(50e-6, 30e-6).unwrap()
    .with_time_scale(TimeScale::Nanoseconds)
    .with_p_meas_symmetric(0.02).unwrap()
    .build().unwrap();
```

### Pure Depolarizing (Testing)

```rust
let noise = CompositeNoiseModelBuilder::new()
    .with_p1(p).unwrap()
    .with_p2(p).unwrap()
    .build().unwrap();
```

## ComposableNoiseModel (Direct Composition)

For maximum flexibility, compose channels directly on a `ComposableNoiseModel`
instead of using a builder:

```rust
use pecos_neo::prelude::*;
use pecos_neo::noise::*;
use pecos_neo::noise::plugins::CorePlugin;

let noise = ComposableNoiseModel::new()
    .add_plugin(CorePlugin)  // State tracking
    .add_channel(SingleQubitChannel::depolarizing(0.001)?)
    .add_channel(TwoQubitChannel::depolarizing(0.01)?
        .with_angle_scaling(AngleScaling::linear())?)
    .add_channel(MeasurementChannel::asymmetric(0.02, 0.03)?)
    .add_channel(IdleChannel::linear(0.0001)?
        .with_linear_depolarizing());
```

### Mixed Approach (Builder + Custom Channels)

Start with a builder, then customize with additional channels:

```rust
use pecos_neo::noise::*;

// Start with standard configuration
let noise = GeneralNoiseModelBuilder::new()
    .with_p1(0.001)
    .with_p2(0.01)
    .build()?
    // Add custom channels
    .add_channel(CrosstalkChannel::measurement()
        .with_global_rate(0.001)?
        .with_transitions(CrosstalkTransitions::symmetric_with_leakage())?);
```

### Custom Channels

Implement your own noise channels by implementing the `NoiseChannel` trait:

```rust
use pecos_neo::noise::*;
use pecos_random::PecosRng;

#[derive(Clone)]
struct MyCustomChannel {
    error_rate: f64,
}

impl NoiseChannel for MyCustomChannel {
    fn responds_to(&self, event: &NoiseEvent<'_>) -> bool {
        matches!(event, NoiseEvent::AfterGate { .. })
    }

    fn apply(
        &self,
        event: &NoiseEvent<'_>,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> NoiseResponse {
        // Your custom noise logic here
        NoiseResponse::None
    }

    fn name(&self) -> &'static str {
        "MyCustomChannel"
    }

    fn clone_box(&self) -> Box<dyn NoiseChannel> {
        Box::new(self.clone())
    }
}
```

## Idle Time and Physical Time Units

The `IdleChannel` models T1/T2 decay during idle periods.
Time is specified in abstract time units - the interpretation (nanoseconds,
clock cycles, etc.) is defined by the noise model configuration.

```rust
use pecos_neo::prelude::*;

let commands = CommandBuilder::new()
    .pz(0)
    .idle(0, 100u64)  // 100 time units
    .mz(0)
    .build();

let noise = ComposableNoiseModel::new()
    .add_channel(IdleChannel::linear(0.001).unwrap()  // 0.1% error per time unit
        .with_linear_depolarizing());        // X/Y/Z with equal probability
```

### Physical Time Units

For physicists who prefer working with physical times (nanoseconds, microseconds),
configure the time scale at the model level:

```rust
use pecos_neo::prelude::*;
use pecos_core::TimeScale;

// Define: 1 TimeUnit = 1 nanosecond, then use physical times
let noise = ComposableNoiseModel::new()
    .with_time_scale(TimeScale::NANOSECONDS)
    .with_idle_t1_t2(50e-6, 30e-6).unwrap();  // T1=50us, T2=30us in seconds
```

Available time scales: `NANOSECONDS`, `MICROSECONDS`, `MILLISECONDS`, `SECONDS`,
or custom via `TimeScale::from_cycle_time_ns(50.0)` for gate-cycle-based timing.

You can also add precision to coarse units:

```rust
// Think in seconds, but with nanosecond precision (9 decimal places)
let scale = TimeScale::SECONDS.with_precision(9);
// Now 0.00005 seconds = 50,000 TimeUnits
```

## Plugins

Plugins bundle related functionality:

```rust
use pecos_neo::noise::plugins::*;

let noise = ComposableNoiseModel::new()
    .add_plugin(CorePlugin)                           // State tracking
    .add_plugin(LeakagePlugin::new())                 // Leakage handling
    .add_plugin(DepolarizingPlugin::new(0.01, 0.02)?); // Simple depolarizing
```

## Key Types Reference

### Pauli Weights

Control error distributions:

```rust
// Uniform (default)
PauliWeights::uniform()  // 1/3 X, 1/3 Y, 1/3 Z

// Z-biased (dephasing)
PauliWeights::z_biased(0.9)?  // 90% Z, 5% X, 5% Y

// Custom
PauliWeights::custom(0.1, 0.2, 0.7)?  // 10% X, 20% Y, 70% Z
```

### Emission Weights

Control leakage vs Pauli errors:

```rust
// Pauli only (no leakage)
SingleQubitEmissionWeights::uniform()

// Include leakage
SingleQubitEmissionWeights::uniform_with_leakage()  // 25% each X/Y/Z/leak

// Leakage only
SingleQubitEmissionWeights::leakage_only()
```

### Angle Scaling

For parameterized gates (RZZ, etc.):

```rust
// No angle dependence
AngleScaling::constant()

// Linear: error ~ |theta/pi|
AngleScaling::linear()

// Quadratic: error ~ (theta/pi)^2
AngleScaling::quadratic()

// Full polynomial: a + b*|theta/pi| + c*|theta/pi|^d
// (matches GeneralNoiseModel's p2_angle_* parameters)
AngleScaling::polynomial(a, b, c, d)

// Asymmetric: different scaling for +/- angles
// offset + linear*|theta/pi| + scale*|theta/pi|^power
AngleScaling::asymmetric(
    neg_offset, neg_linear, neg_scale,
    pos_offset, pos_linear, pos_scale,
    power
)
```

### Crosstalk Transitions

State-dependent crosstalk effects:

```rust
// Simple flip model
CrosstalkTransitions::flip_only()

// Include leakage
CrosstalkTransitions::symmetric_with_leakage()

// Custom per-state transitions
CrosstalkTransitions::custom(
    from_0_stay, from_0_flip, from_0_leak,
    from_1_stay, from_1_flip, from_1_leak,
)?
```

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
