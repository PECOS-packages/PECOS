# Adding Noise

## Quickest Option

One-liner on `sim_neo`:

```rust
sim_neo(circuit).auto().depolarizing(0.01)?.sampling(monte_carlo(1000)).run();
```

## Per-Channel Control

Set different rates for single-qubit, two-qubit, and measurement errors:

```rust
sim_neo(circuit).auto()
    .noise(GeneralNoiseModelBuilder::new()
        .with_p1(0.001)
        .with_p2(0.01)
        .with_p_meas_symmetric(0.005).build()?)
    .sampling(monte_carlo(1000))
    .run();
```

## How It Works

Under the hood, noise is event-driven. As the simulator runs your circuit, it
emits events -- `AfterGate`, `AfterMeasurement`, `IdleTime`, etc. Each noise
channel declares which events it responds to and injects errors when they fire.
The builders above set all of this up for you.

When you need more control -- picking individual channels, reacting to custom
signals, or writing your own -- see [Composable Noise Model](noise-model.md)
and [Noise Channels](noise-channels.md).

## Full Noise Model Builder

`CompositeNoiseModelBuilder` gives you control over leakage, emission, seepage,
crosstalk, and more. It and the presets below require the `composite-noise`
Cargo feature on `pecos-neo`:

```rust
use pecos_neo::noise::composite::CompositeNoiseModelBuilder;
use pecos_core::QubitId;

// Two-qubit device: each qubit is the other's neighbor.
fn neighbors(active: &[QubitId]) -> Vec<QubitId> {
    [QubitId(0), QubitId(1)].into_iter().filter(|q| !active.contains(q)).collect()
}

let noise = CompositeNoiseModelBuilder::new()
    // Gate errors
    .with_p1(0.001)?                     // Single-qubit error rate
    .with_p2(0.01)?                      // Two-qubit error rate
    .with_p1_emission_ratio(0.1)?        // Fraction of errors that leak (vs Pauli)

    // Measurement
    .with_p_meas(0.02, 0.03)?           // Asymmetric: P(report 1|true 0), P(report 0|true 1)

    // Preparation
    .with_p_prep(0.001)?                // Prep error rate

    // Leakage
    .with_leakage(0.001, 0.1)?         // Leak rate, seepage rate

    // Crosstalk
    .with_p_meas_crosstalk(0.001, 0.01)?  // Global, local rates
    .with_p_meas_crosstalk_local_fn(neighbors)

    .build()?;
```

## Common Presets

**Ion trap style:**
```rust
CompositeNoiseModelBuilder::new()
    .with_p1(0.0001)?.with_p2(0.001)?
    .with_p1_emission_ratio(0.1)?.with_p2_emission_ratio(0.1)?
    .with_leakage(0.0001, 0.1)?
    .with_p_meas(0.001, 0.01)?
    .build()?;
```

**Superconducting style:**
```rust
CompositeNoiseModelBuilder::new()
    .with_p1(0.001)?.with_p2(0.01)?
    .with_idle_t1_t2(50e-6, 30e-6)?
    .with_p_meas_symmetric(0.02)?
    .build()?;
```

**Pure depolarizing (testing):**
```rust
CompositeNoiseModelBuilder::new().with_p1(p)?.with_p2(p)?.build()?;
```


`NoiseModelBuilder`, `GeneralNoiseModelBuilder` and `CompositeNoiseModelBuilder`
register `CorePlugin` automatically. Its `AfterPreparation` handler marks qubits
active and clears previous leakage before preparation noise runs, so a new leak
from that same preparation remains. Its `AfterMeasurement` handler marks qubits
inactive without clearing leakage. Thus crosstalk targets other prepared,
unmeasured, non-leaked qubits, and custom `active()` conditions follow this
lifecycle. A bare `ComposableNoiseModel` needs `.add_plugin(&CorePlugin)` for
these preparation and measurement state updates.

## Going Deeper

For custom noise primitives, implementing your own `NoiseChannel`, composable
decision trees, geometric sampling, and full API reference, see the
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
