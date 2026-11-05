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
| CrosstalkChannel | `AfterPreparation` (noisy `PZ`) or `CrosstalkPayload` + `AfterHiddenMeasurement`, chosen at construction |
| LeakageChannel | `BeforeGate`; `AfterGate` only with positive partner Pauli probability |
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
SingleQubitChannel::depolarizing(0.001)?

// With more control
SingleQubitChannel::depolarizing(0.001)?
    .with_pauli_weights(PauliWeights::z_biased(0.9)?)?   // 90% Z errors
    .with_emission(0.1, 0.05)?                           // 10% emission, 5% leakage
```

Other constructors: `dephasing(p)`, `bit_flip(p)`.

### TwoQubitChannel

Pauli errors after two-qubit gates:

```rust
TwoQubitChannel::depolarizing(0.01)?

// With angle scaling (for parameterized gates like RZZ)
TwoQubitChannel::depolarizing(0.01)?
    .with_angle_scaling(AngleScaling::linear())?
```

Angle scaling options: `constant()`, `linear()`, `quadratic()`,
`polynomial(a, b, c, d)`.

### MeasurementChannel

Readout errors -- flips measurement outcomes with given probabilities.

```rust
MeasurementChannel::symmetric(0.02)?              // Same rate both directions
MeasurementChannel::asymmetric(0.02, 0.03)?       // P(0->1), P(1->0)
```

### PreparationChannel

Errors during state preparation -- bit flips or leakage.

`NoiseModelBuilder`, `GeneralNoiseModelBuilder` and `CompositeNoiseModelBuilder`
register `CorePlugin` automatically. Its `AfterPreparation` handler marks qubits
active and clears previous leakage before preparation noise runs, so a new leak
from that same preparation remains. Its `AfterMeasurement` handler marks qubits
inactive without clearing leakage. Crosstalk targets other non-leaked qubits
prepared this shot, including measured, noiselessly prepared and `QAlloc` qubits.
Custom `active()` conditions still exclude measured qubits. A bare
`ComposableNoiseModel` needs `.add_plugin(&CorePlugin)` for
these preparation and measurement state updates.

```rust
PreparationChannel::new(0.001)?
    .with_leakage(0.1)?    // 10% of prep errors cause leakage
```

### IdleChannel

T1/T2 decay during idle periods. Rate scales with duration.

```rust
IdleChannel::linear(0.0001).unwrap()                   // Rate per time unit
    .with_linear_depolarizing()               // Uniform X/Y/Z

// T1=50us and total T2=30us when one abstract time unit is 1ns
IdleChannel::from_t1_t2(50_000.0, 30_000.0).unwrap()
```

`from_t1_t2` uses the first-order Pauli twirl, requires total `T2 <= 2 * T1`, and is valid for
idle durations much shorter than both coherence times. Use
`ComposableNoiseModel::with_idle_t1_t2` with a `TimeScale` when supplying physical seconds.

## Specialized Channels

For more specific noise scenarios.

### CrosstalkChannel

Measurement crosstalk fires only on `MeasCrosstalkGlobalPayload` or
`MeasCrosstalkLocalPayload`, both `GateCategory::Metadata`. Global operands
exclude victims; local operands name potential victims. An empty global payload
targets every existing, non-leaked qubit. Measured qubits remain eligible.

The payload emits `CrosstalkPayload`. Each victim independently fires at its
kind's rate. A nonempty `MeasureHidden { gate_type, qubits }` response measures
victims directly, without user outcomes, readout noise, measurement events or
lifecycle changes. `AfterHiddenMeasurement` carries the raw outcomes; the optional
table selects stay, X, or leakage from the actual victim outcome. Without a
table, only collapse occurs. This continuation finishes before the next command.
No fired victims returns `None`. Noiseless payloads produce no hidden measurement.

Preparation crosstalk still fires only on noisy PZ and defaults to a fair Z coin.
Neither ordinary gates nor ordinary measurements trigger measurement crosstalk.
Neo rates are post-scale; engine parity is scoped to `leakage_scale = 1`.

```rust
CrosstalkChannel::measurement()
    .with_global_rate(0.001)?
    .with_local_rate(0.01)?
```

Use `CommandBuilder::meas_crosstalk_global_payload(&[...])` for exclusions or
`meas_crosstalk_local_payload(&[...])` for explicit victims. Put the payload at
the desired point in the command stream. Neighbor callbacks and the channel's
`with_leakage` setter are removed; leak transitions belong in the table.

### LeakageChannel

Suppresses gates on leaked operands. Every builder registers it, and suppression
does not depend on where it sits among the other channels.

```rust
LeakageChannel::new()
    .with_scale(1.0)?
```

### GateDependentChannel

Different error rates for different gate types:

```rust
GateDependentChannel::new()
    .with_gate_error(GateType::H, 0.0005)?
    .with_gate_error(GateType::CX, 0.005)?
    .with_default(0.001)?
```

### CategoryBasedChannel

Error rates by gate category (single-qubit, two-qubit, etc.):

```rust
CategoryBasedChannel::new()
    .with_category(GateCategory::SingleQubit, 0.001)?
    .with_category(GateCategory::TwoQubit, 0.01)?
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

Gate-event `qubits` contains executed groups only after a gate. Custom channels
that model seepage, partner noise or idle effects on suppressed groups must read
`leaked_suppressed` explicitly; `gated_qubits()` includes both sets. Composite
filters classify by `group_size` and now also fire on batched commands, with
partner conditions evaluated within each group. On noisy operations, suppressed groups run primitives
only on currently leaked members.

The seepage parameter is conditional on an emission fault: its marginal rate is
`p * emission_ratio * seepage`, where `p` includes angle scaling. Leakage scale
does not scale seepage. Two-qubit channels share the outer fault and emission
draws per pair, except through `channel_action` (#1568); composite builders
preserve the per-qubit marginal rate.

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
