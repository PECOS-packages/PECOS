# Noise Channels

Built-in channels for [composing noise models](noise-model.md). Each channel
is event-driven -- it declares which [events](events.md) it responds to and
produces a `NoiseResponse` when those events fire during simulation.

## Which Events Trigger Which Channels?

| Channel | Events |
|---------|--------|
| SingleQubitChannel | `AfterGate` (1Q gates) |
| TwoQubitChannel | `AfterGate` (2Q gates) |
| MeasurementChannel | `AfterMeasurement` |
| PreparationChannel | `AfterPreparation` |
| IdleChannel | `IdleTime` |
| CrosstalkChannel | `AfterGate` |
| LeakageChannel | `BeforeGate`, `AfterGate`, `BeforeMeasurement` |
| Custom channels | Any event, including `Signal` |

You don't need to wire this up yourself -- each channel self-selects which
events it cares about. Just add channels to your noise model and the event
system handles the rest.

## Common Channels

These cover the most typical noise sources. Most noise models use some
combination of these.

### SingleQubitChannel

Pauli errors after single-qubit gates:

```rust
SingleQubitChannel::depolarizing(0.001)

// With more control
SingleQubitChannel::depolarizing(0.001)
    .with_pauli_weights(PauliWeights::z_biased(0.9))   // 90% Z errors
    .with_emission(0.1, 0.05)                           // 10% emission, 5% leakage
```

Other constructors: `dephasing(p)`, `bit_flip(p)`.

### TwoQubitChannel

Pauli errors after two-qubit gates:

```rust
TwoQubitChannel::depolarizing(0.01)

// With angle scaling (for parameterized gates like RZZ)
TwoQubitChannel::depolarizing(0.01)
    .with_angle_scaling(AngleScaling::linear())
```

Angle scaling options: `constant()`, `linear()`, `quadratic()`,
`polynomial(a, b, c, d)`.

### MeasurementChannel

Readout errors -- flips measurement outcomes with given probabilities.

```rust
MeasurementChannel::symmetric(0.02)              // Same rate both directions
MeasurementChannel::asymmetric(0.02, 0.03)       // P(0->1), P(1->0)
```

### PreparationChannel

Errors during state preparation -- bit flips or leakage.

```rust
PreparationChannel::new(0.001)
    .with_leakage(0.1)    // 10% of prep errors cause leakage
```

### IdleChannel

T1/T2 decay during idle periods. Rate scales with duration.

```rust
IdleChannel::linear(0.0001)                   // Rate per time unit
    .with_linear_depolarizing()               // Uniform X/Y/Z

IdleChannel::from_t1_t2(50e-6, 30e-6)        // Physical T1/T2 times
```

## Specialized Channels

For more specific noise scenarios.

### CrosstalkChannel

Errors on bystander qubits when gates or measurements happen nearby.

```rust
CrosstalkChannel::new()
    .with_global_rate(0.001)       // Affects all other qubits
    .with_local_rate(0.01)         // Affects nearby qubits
```

### LeakageChannel

Handles the effects of leaked qubits on gates -- runs at high priority so it
processes before other channels.

```rust
LeakageChannel::new()
    .with_scale(1.0)
```

### GateDependentChannel

Different error rates for different gate types:

```rust
GateDependentChannel::new()
    .with_gate_error(GateType::H, 0.0005)
    .with_gate_error(GateType::CX, 0.005)
    .with_default(0.001)
```

### CategoryBasedChannel

Error rates by gate category (single-qubit, two-qubit, etc.):

```rust
CategoryBasedChannel::new()
    .with_category(GateCategory::SingleQubit, 0.001)
    .with_category(GateCategory::TwoQubit, 0.01)
```

## Signal-Reactive Channels

Any channel can also respond to user-defined [signals](events.md). This lets
you change noise behavior mid-circuit based on custom triggers -- for example,
adjusting error rates per QEC round or per hardware zone. A channel does this
by responding to both its normal events and `Signal` events:

```rust
fn responds_to(&self, event: &NoiseEvent<'_>) -> bool {
    event.is_signal::<ZoneTemperature>() || matches!(event, NoiseEvent::AfterGate { .. })
}
```

When a signal arrives, the channel updates its internal state. When a gate
event arrives, it uses that state to scale error rates. See the
[developer noise guide](../dev/noise.md) for the full `NoiseChannel` trait
and implementation pattern.

## Going Deeper

For implementing custom channels, `CompositeChannel` with primitive decision
trees, and geometric sampling, see the
[developer noise guide](../dev/noise.md).
