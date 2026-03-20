# Noise System Usage Guide

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
    .with_p1(0.001)           // Single-qubit gate error rate
    .with_p2(0.01)            // Two-qubit gate error rate
    .with_p_meas(0.02, 0.03)  // Measurement error (0→1, 1→0)
    .build();
```

### All Parameters

#### Single-Qubit Gates

```rust
CompositeNoiseModelBuilder::new()
    // Base error probability
    .with_p1(0.001)

    // Of errors, what fraction are emission (vs Pauli)
    .with_p1_emission_ratio(0.1)

    // Seepage probability (leaked qubit returns to computational space)
    .with_p1_seepage(0.05)

    // Custom Pauli weights (default: uniform X, Y, Z)
    .with_p1_pauli_weights(PauliWeights::new(0.25, 0.25, 0.50))  // More Z errors

    // Custom emission model
    .with_p1_emission_model(SingleQubitEmissionWeights { ... })
```

#### Two-Qubit Gates

```rust
CompositeNoiseModelBuilder::new()
    // Base error probability
    .with_p2(0.01)

    // Emission vs Pauli ratio
    .with_p2_emission_ratio(0.1)

    // Seepage for leaked qubits
    .with_p2_seepage(0.05)

    // Angle-dependent scaling (for RZZ, etc.)
    .with_p2_angle_scaling(AngleScaling::Quadratic)

    // Idle error during two-qubit gate
    .with_p2_idle_rate(0.001)

    // Custom two-qubit Pauli model
    .with_p2_pauli_model(TwoQubitPauliWeights { ... })
```

#### Preparation

```rust
CompositeNoiseModelBuilder::new()
    // Preparation error rate
    .with_p_prep(0.001)

    // Of prep errors, fraction that leak (vs bit flip)
    .with_p_prep_leak_ratio(0.1)

    // Crosstalk during preparation
    .with_p_prep_crosstalk(0.0001)
```

#### Measurement

```rust
CompositeNoiseModelBuilder::new()
    // Asymmetric measurement error
    .with_p_meas(0.02, 0.03)  // P(report 1 | true 0), P(report 0 | true 1)

    // Symmetric measurement error
    .with_p_meas_symmetric(0.02)
```

#### Leakage

```rust
CompositeNoiseModelBuilder::new()
    // Leakage and seepage rates
    .with_leakage(
        0.001,  // Probability of leaking per gate
        0.1,    // Probability of seeping back per gate
    )
```

#### Idle/Decoherence

```rust
CompositeNoiseModelBuilder::new()
    // T1/T2 style decoherence
    .with_idle_t1_t2(50e-6, 30e-6)  // T1, T2 in seconds

    // Set time scale for idle calculations
    .with_time_scale(TimeScale::Nanoseconds)
```

#### Crosstalk

```rust
CompositeNoiseModelBuilder::new()
    // Measurement-induced crosstalk
    .with_measurement_crosstalk(
        0.001,  // Global probability (affects all other qubits)
        0.01,   // Local probability (affects nearby qubits)
    )

    // Custom neighbor function
    .with_crosstalk_neighbors(|q1, q2| distance(q1, q2) < 3)

    // Crosstalk transition probabilities by outcome
    .with_crosstalk_transitions(CrosstalkTransitions {
        outcome_0: TransitionProbs { nothing: 0.9, flip: 0.05, leak: 0.05 },
        outcome_1: TransitionProbs { nothing: 0.85, flip: 0.10, leak: 0.05 },
    })
```

## Using Noise with CircuitRunner

```rust
use pecos_neo::runner::CircuitRunner;
use pecos_neo::command::CommandBuilder;
use pecos_qsim::SparseStab;

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
    .with_p1(0.001)
    .with_p2(0.01)
    .build();

// Run with noise
let mut state = SparseStab::new(2);
let mut runner = CircuitRunner::<SparseStab>::new()
    .with_noise(noise)
    .with_seed(42);

let outcomes = runner.apply_circuit(&mut state, &circuit)?;
println!("Qubit 0: {}", outcomes.get_bit(QubitId(0)).unwrap());
println!("Qubit 1: {}", outcomes.get_bit(QubitId(1)).unwrap());
```

## Custom Noise with Primitives

For advanced customization, build noise directly from primitives:

```rust
use pecos_neo::noise::composite::prelude::*;
use pecos_neo::noise::composite::{CompositeChannel, CompositeEventFilter};

// Custom single-qubit noise: skip leaked qubits, then apply depolarizing
let sq_primitive = seq(vec![
    skip_if_leaked(),
    prob(0.001, depolarize()),
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
prob(0.01, action)

// Dynamic probability based on gate
prob_fn(|gate| gate.angle().map(|a| a.to_radians().abs()).unwrap_or(0.0), action)

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
pauli_weighted(0.3, 0.3, 0.4)  // Custom weights
depolarize()               // Same as uniform pauli
dephase()                  // Z error only

// Leakage
leak()                     // Mark qubit as leaked
seep(0.1)                  // With prob 0.1, unleak and apply random Pauli

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
| `NoiseResponse::SkipGate` | Don't execute the triggering gate |
| `NoiseResponse::FlipOutcomes(qubits)` | Toggle measurement outcomes for qubits |
| `NoiseResponse::ForceOutcomes(pairs)` | Set outcomes to specific values |
| `NoiseResponse::MarkLeaked(qubits)` | Mark qubits as leaked in context |
| `NoiseResponse::Multiple(responses)` | Combine multiple responses |

## Noise Context

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
let channel = CompositeChannel::new("fast_depol", prob(0.001, depolarize()))
    .with_probability(0.001)  // Enables geometric sampling
    .with_filter(CompositeEventFilter::SingleQubitGate);
```

### Compiled Primitives

Complex primitive trees can be compiled for better performance:

```rust
use pecos_neo::noise::composite::compiled::CompiledPrimitive;

let primitive = seq(vec![...]);
let compiled = CompiledPrimitive::compile(&primitive);
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
    .with_p1(0.0001)
    .with_p2(0.001)
    .with_p1_emission_ratio(0.1)
    .with_p2_emission_ratio(0.1)
    .with_leakage(0.0001, 0.1)
    .with_p_meas(0.001, 0.01)
    .build();
```

### Superconducting Style

```rust
let noise = CompositeNoiseModelBuilder::new()
    .with_p1(0.001)
    .with_p2(0.01)
    .with_idle_t1_t2(50e-6, 30e-6)
    .with_time_scale(TimeScale::Nanoseconds)
    .with_p_meas_symmetric(0.02)
    .build();
```

### Pure Depolarizing (Testing)

```rust
let noise = CompositeNoiseModelBuilder::new()
    .with_p1(p)
    .with_p2(p)
    .build();
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
    .add_channel(SingleQubitChannel::depolarizing(0.001))
    .add_channel(TwoQubitChannel::depolarizing(0.01)
        .with_angle_scaling(AngleScaling::linear()))
    .add_channel(MeasurementChannel::asymmetric(0.02, 0.03))
    .add_channel(IdleChannel::linear(0.0001)
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
    .build()
    // Add custom channels
    .add_channel(CrosstalkChannel::new()
        .with_global_rate(0.001)
        .with_transitions(CrosstalkTransitions::symmetric_with_leakage()));
```

### Custom Channels

Implement your own noise channels by implementing the `NoiseChannel` trait:

```rust
use pecos_neo::noise::*;
use pecos_rng::PecosRng;

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
    .add_channel(IdleChannel::linear(0.001)  // 0.1% error per time unit
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
    .with_idle_t1_t2(50e-6, 30e-6);  // T1=50us, T2=30us in seconds
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
    .add_plugin(DepolarizingPlugin::new(0.01, 0.02)); // Simple depolarizing
```

## Key Types Reference

### Pauli Weights

Control error distributions:

```rust
// Uniform (default)
PauliWeights::uniform()  // 1/3 X, 1/3 Y, 1/3 Z

// Z-biased (dephasing)
PauliWeights::z_biased(0.9)  // 90% Z, 5% X, 5% Y

// Custom
PauliWeights::custom(0.1, 0.2, 0.7)  // 10% X, 20% Y, 70% Z
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
)
```
