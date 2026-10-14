# Noise Model Builders

```hidden-python
from pecos import sim, Qasm, GeneralNoiseModelBuilder
```

PECOS provides builder classes for constructing quantum noise models with a fluent, method-chaining API. The `GeneralNoiseModelBuilder` is the most comprehensive builder, offering fine-grained control over various noise parameters.

## Quick Start

The simplest way to add noise to your QASM simulations is using the `GeneralNoiseModelBuilder`:

```python
from pecos import sim, Qasm, GeneralNoiseModelBuilder

# Define a circuit
qasm = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q -> c;
"""

# Set single-qubit error to 0.001 and two-qubit error to 0.01
noise = GeneralNoiseModelBuilder().with_p1(0.001).with_p2(0.01)

# Use with sim()
results = sim(Qasm(qasm)).noise(noise).seed(42).run(1000)
```

## GeneralNoiseModelBuilder

The `GeneralNoiseModelBuilder` provides methods to configure all aspects of quantum noise:

### Basic Error Probabilities

```python
noise = (
    GeneralNoiseModelBuilder()
    # Gate errors
    .with_p1(0.001)  # Single-qubit gate error
    .with_p2(0.01)  # Two-qubit gate error
    # State preparation and measurement
    .with_p_prep(0.0005)  # State preparation error
    .with_p_meas_0(0.002)  # Measurement 0→1 flip
    .with_p_meas_1(0.003)
)  # Measurement 1→0 flip
```

### Average vs Total Probabilities

The builder supports both "total" and "average" error probabilities:

```python
# Average probability (recommended for physical intuition)
noise = GeneralNoiseModelBuilder().with_average_p1(0.001).with_average_p2(0.01)  # Converted to total internally

# Total probability (used internally by the engine)
noise = GeneralNoiseModelBuilder().with_p1(0.00133).with_p2(0.0133)  # Total for single-qubit  # Total for two-qubit
```

**Note**: Average probabilities are more intuitive as they represent the actual error rate per gate. Total probabilities include a conversion factor based on the number of Pauli operators.

### Pauli Error Models

Specify custom Pauli error distributions instead of uniform depolarizing noise:

```python
noise = (
    GeneralNoiseModelBuilder()
    # Single-qubit Pauli errors
    .with_p1_pauli_model(
        {
            "X": 0.5,  # 50% X errors
            "Y": 0.3,  # 30% Y errors
            "Z": 0.2,  # 20% Z errors
        }
    )
    # Two-qubit Pauli errors
    .with_p2_pauli_model(
        {
            "IX": 0.25,  # 25% error on second qubit only
            "XI": 0.25,  # 25% error on first qubit only
            "XX": 0.5,  # 50% correlated X errors
        }
    )
)
```

### Gate-Hook Channels

`StateTransitionChannel` and `PauliLeakageChannel` add stochastic operations before or after
single- and two-qubit gate events. They use the general model's existing samplers, leakage
tracking, and internal measurement continuation. Legacy p1/p2, emission, seepage, and crosstalk
configuration retains its existing execution path.

Every channel has its own outer `probability`. It is **not multiplied by `p1`, `p2`, or emission
ratios**. As in the prototype, global `scale` and the corresponding `p1_scale` or `p2_scale`
multiply this probability; configurations with scaled probabilities outside `[0, 1]` are rejected.
Angle-dependent p2 scaling does not affect hooks.
A failed outer coin is a no-op, including no measurement. An explicit `L` event or destination
always marks the operand leaked, independently of `leakage_scale`. The legacy leakage paths
retain that setting's leakage-to-depolarization approximation.

At each gate event:

1. Before hooks execute in insertion order **across both channel families**, completing any
   internal measurements. Recovery may enable the ideal gate; new leakage may suppress it.
2. The existing ideal-gate and ordinary p1/p2 noise logic runs.
3. After hooks execute in insertion order across both families. They run even when leakage
   suppressed the ideal gate; recovery here cannot retroactively execute it.
4. Configured after-two-qubit idle noise runs last.

`add_*` appends one entry. A family-specific `with_*` list replaces that family's entries at
that hook and appends the replacement list in slice order, preserving other-family entries.
Batching two-qubit gates does not couple the pairs: leakage, recovery, selection, and suppression
are evaluated for each pair. Gates in `noiseless_gates` bypass these hooks, just as they bypass
ordinary noise. Preparation, measurement, and idle events do not trigger these gate hooks.

### State Transition Channels

`StateTransitionChannel` models stochastic transitions among `0`, `1`, and classically tracked
leakage `L`. Use `"i->j"` keys with probabilities `P(j | i)`. Each supplied source row must sum
to one separately. A nested `{source: {destination: probability}}` map is also accepted.

If either computational source row is supplied, **both `0` and `1` rows are required**.
When the channel fires on a computational qubit, it performs `MZ`, consumes that result, then
samples the destination. Even `{"0->0": 1.0, "1->1": 1.0}` collapses superpositions: explicit
identity population rows are not the identity quantum map. Internal MZ results are not program
measurement results and do not receive configured readout errors. Omitting both computational
rows, as in a leakage-only recovery channel, preserves computational coherence. An omitted
leakage row leaves leakage unchanged; omitted destinations have probability zero.

```python
from pecos import GeneralNoiseModelBuilder, StateTransitionChannel, PauliLeakageChannel

population = StateTransitionChannel(
    probability=0.02,
    transitions={"0->0": 0.8, "0->L": 0.2, "1->1": 0.9, "1->L": 0.1},
)
recovery = StateTransitionChannel.leak_recovery(probability=0.7, p_zero=0.5)
leakage = PauliLeakageChannel(0.03, {"L": 1.0})

noise = (
    GeneralNoiseModelBuilder()
    .add_p1_pauli_leakage_channel_before_gate(leakage)
    .add_p1_transition_channel_before_gate(recovery)
    .with_p1_transition_channels_after_gate([population, recovery])
    # One outer coin per operand: symmetric independent recovery.
    .add_p2_transition_channel_before_gate(recovery)
)
```

Reversing `leakage` and `recovery` reverses their execution order. The values in these examples
are illustrative, not device calibration parameters.

### Pauli and Leakage Channels

`PauliLeakageChannel(p, events)` samples a state-independent event over `{I, X, Y, Z, L}` when
its outer coin succeeds. Event weights must be finite, lie in `[0, 1]`, and sum to one **globally**.
`I` is allowed, including `II` in a joint channel. `L` means `any -> L`; Paulis are ignored on
already leaked operands. Pauli events preserve coherence according to their usual quantum map.

```python
from pecos import PauliLeakageChannel, TwoQubitPauliLeakageChannel, P2PauliLeakageStep

single = PauliLeakageChannel(0.01, {"I": 0.1, "X": 0.6, "L": 0.3})
joint = TwoQubitPauliLeakageChannel(0.02, {"IX": 0.4, "XL": 0.4, "LL": 0.2})
noise = (
    GeneralNoiseModelBuilder()
    .with_p1_pauli_leakage_channels_after_gate([single])
    .add_p2_pauli_leakage_channel_before_gate(single)
    .with_p2_pauli_leakage_steps_after_gate([P2PauliLeakageStep.joint(joint)])
)
```

### Joint Transitions and Identity Wires

`TwoQubitStateTransitionChannel` accepts pair states `00`, `01`, `0L`, ..., `LL`, with one shared
outer coin and a conditional destination distribution for each supplied source. Each supplied
leakage pattern must include all computational completions: `0L` requires `1L`, `L0` requires
`L1`, and any of `00`, `01`, `10`, or `11` requires all four. `LL` stands alone. Missing rows
are rejected with an error naming each incomplete leakage pattern and its required sources
(`C` denotes `0` or `1` in these pattern names); explicit identity rows still follow MZ.

When the channel fires, it measures a non-leaked leg only if the channel depends on that leg's
computational state and at least one source row matches the pair's known leakage pattern.
If no pattern matches, the channel is identity without measurement. Wildcard and tensor identity
legs remain unmeasured. Use `*` for a true identity wire.

```python
from pecos import TwoQubitStateTransitionChannel, P2TransitionStep

joint = TwoQubitStateTransitionChannel(0.2, {"LL->00": 0.5, "LL->11": 0.5})
recover_either = TwoQubitStateTransitionChannel(0.8, {"*L->*0": 0.5, "*L->*1": 0.5, "L*->0*": 0.5, "L*->1*": 0.5})
noise = (
    GeneralNoiseModelBuilder()
    .add_p2_transition_channel_before_gate(recover_either)
    .with_p2_transition_steps_after_gate([P2TransitionStep.joint(joint)])
)
```

The wildcard must remain in the same position in every destination. That wire is not resolved,
measured, collapsed, or changed by the rule. `*L` and `L*` may coexist: they write different
operands and both apply to `LL` under the shared outer coin, with independent conditional draws.
Computational wildcard groups such as `*0` and `*1` must be supplied together. Mixing concrete
and wildcard rows is rejected, preventing overlapping groups from writing conflicting outcomes.
Use separate ordered channels if both forms are needed.

For asymmetric independent operands, use `P2TransitionStep.independent(first, second)` or
`P2PauliLeakageStep.independent(first, second)`. The existing `first * second` shorthand has the
same independent-coin meaning. `TransitionDict` and `PauliLeakageDict` can tensor conditional
maps with `left * right` for a joint channel with one outer coin. Keep the resulting dictionary
object when constructing the channel: converting a transition tensor to a plain dictionary
loses its identity-wire metadata and can cause additional measurements.

### Rust Construction and Validation

Rust provides the same placements, ordering, independent/joint modes, and validation as Python.
Channel and dictionary constructors return `Result<_, ChannelValidationError>`. Python translates
these structured errors to `ValueError`. Weights use the existing sampler normalization tolerance
of `1e-5` for row sums; each probability and weight must independently be finite and in `[0, 1]`.
The row-sum tolerance never permits an individual weight above one. Wrong arity, malformed labels,
incomplete computational rows, and conflicting wildcard groups are rejected.

```rust
use std::collections::BTreeMap;
use pecos_engines::noise::{
    ChannelValidationError, GeneralNoiseModel, PauliLeakageChannel, PauliLeakageEvent,
    QubitTransitionState, StateTransitionChannel, TwoQubitStateTransitionChannel,
};

fn configured_noise() -> Result<GeneralNoiseModel, ChannelValidationError> {
    use QubitTransitionState::{Leakage, Zero, One};
    let recovery = StateTransitionChannel::from_rows(0.7, &BTreeMap::from([
        (Leakage, BTreeMap::from([(Zero, 0.5), (One, 0.5)])),
    ]))?;
    let faults = PauliLeakageChannel::from_events(0.02, &BTreeMap::from([
        (PauliLeakageEvent::X, 0.8), (PauliLeakageEvent::Leakage, 0.2),
    ]))?;
    // String configuration also supports joint identity-wire rules.
    let joint = TwoQubitStateTransitionChannel::from_transitions(0.8, &BTreeMap::from([
        ("*L->*0".into(), 1.0),
    ]))?;
    Ok(GeneralNoiseModel::builder()
        .add_p1_pauli_leakage_channel_before_gate(&faults)
        .add_p1_transition_channel_before_gate(&recovery)
        .add_p2_transition_channel_before_gate(&recovery)
        .add_p2_transition_channel_after_gate(&joint)
        .build())
}
```

For typed joint construction, `TwoQubitStateTransitionChannel::from_rows` uses
`[QubitTransitionState; 2]` keys and `TwoQubitPauliLeakageChannel::from_events` uses
`[PauliLeakageEvent; 2]` keys. `new` accepts nested string transition rows or string event maps;
`from_transitions` accepts flat `"i->j"` maps.

### Scope and Prototype Migration

The draft's public `TransitionChannel` (Rust `QubitTransitionChannel`) and
`TwoQubitTransitionChannel` names are now `StateTransitionChannel` and
`TwoQubitStateTransitionChannel`. Update Rust callers to handle constructor results. Replace
prototype matrix composition (`@`, `compose`, `then`) with ordered hooks. No compatibility aliases
are provided for these draft APIs.

This increment does not provide mixture/product channel objects, conditional follow-up channels,
arbitrary Kraus composition, or coherent superpositions involving leakage. The Selene wrapper
does not expose these hook lists yet. Runtime-frame profiles reject hook lists until internal
channel measurement barriers can preserve their source-operation boundaries. Ordinary `sim()`
gate execution supports these hooks. Existing gate-noise configuration is not migrated into this
channel executor; such a migration would require separate seeded and statistical parity checks.

### Leakage readout within a batch

The Rust general noise model records leakage when each measurement is emitted.
A later reset or leakage fault in the same batch cannot change that earlier
readout. `MeasureLeaked` returns 2 for a leaked qubit; ordinary measurement starts
at 1 and still applies the configured asymmetric readout fault.

`MPZ` measures and then prepares zero: its readout uses the pre-reset leakage,
and its reset clears leakage for following gates, including when `MPZ` is marked
noiseless. Its preparation half does not sample a preparation fault.

Recording leakage draws no randomness. With no injected measurement-crosstalk
victims, execution keeps the existing single-call path and gate-before-readout
sampling order. At an actual measurement-crosstalk injection site, the general
controller instead suspends the remaining gates, obtains the injected outcomes,
and emits their leakage/Pauli transitions before resuming those gates. A later
readout sees the transition; a later reset clears its leakage as expected.

This uses one controller lifecycle without restarting the noise model or applying
gate noise to generated transitions. Readout faults for the executed prefix are
sampled before resuming the suffix's gate faults. Crosstalk-enabled seeded results
therefore intentionally change from the earlier deferred implementation. Extra
simulator dispatches occur only when measurement-crosstalk victims are injected.
Preparation crosstalk is a measurement-only channel: its collapse already occurs
in operation order and needs no outcome-dependent transition or new boundary.

Legacy and metadata runtime-frame routes share this execution. Frame expansion
and continuation counts remain bounded; existing physical-event admission is not
widened. Scheduled profiles still exclude crosstalk and retain their single-call
contract. Failed crosstalk continuations stay blocked across clones until reset,
which clears pending gates and buffered readouts. Low-level callers of
`apply_noise_on_start` must process each returned command message and feed replies
to `apply_noise_on_continue_processing` until no commands remain; the initial
command message need not contain every source gate.

### Scaling and Global Parameters

```python
noise = (
    GeneralNoiseModelBuilder()
    .with_scale(1.5)  # Scale all error rates by 1.5x
    .with_leakage_scale(0.1)  # 10% of errors cause leakage
    .with_emission_scale(0.05)
)  # 5% spontaneous emission
```

### Noiseless Gates

Make specific gates ideal (no noise):

```python
noise = (
    GeneralNoiseModelBuilder()
    .with_p1(0.001)
    .with_p2(0.01)
    # Single gate
    .with_noiseless_gate("H")
    # Multiple gates
    .with_noiseless_gate("S")
    .with_noiseless_gate("T")
    .with_noiseless_gate("MEASURE")
)
```

### Idle Locations

`Idle` gates are timing markers by default. They do not silently inherit
single-qubit gate noise from `p1` or `with_p1(...)`.

Configure idle decoherence with any combination of these independent families:

- `with_p_idle_linear(rate, model)` samples one linear-rate event from a
  normalized X/Y/Z/L distribution.
- `with_p_idle_sin_squared(rate, model)` independently samples each X/Y/Z/L
  mechanism with `sin²(rate * multiplier * duration)`. Its rate is radians per
  time unit and its multipliers are intentionally unnormalized because each
  axis has its own rate.
- `with_p_idle_coherent(rate, model)` deterministically applies RX/RY/RZ with
  angle `rate * multiplier * duration`. Its rate is radians per time unit, with
  no `2*pi` or coherent-to-incoherent conversion. Its model is also
  intentionally unnormalized: the values are relative generator-rate
  multipliers, not probabilities. Omitting the Python model uses
  `{"RX": 1.0, "RY": 1.0, "RZ": 1.0}`.

The unpaired legacy idle setters have been removed. Migrate them as follows:

| Removed setter | Replacement |
|---|---|
| `with_p_idle_linear_rate(r)` | `with_p_idle_linear(r, model)`; use the symmetric `{"X": 1/3, "Y": 1/3, "Z": 1/3}` model if no model was previously set |
| `with_p_idle_linear_model(m)` | `with_p_idle_linear(r, m)`; the rate and normalized model are now configured together |
| `with_p_idle_quadratic_rate(r)` | `with_p_idle_sin_squared(r * PI, {"Z": 1.0})` |
| `with_p_idle_quadratic_coherent(true)` | `with_p_idle_coherent(rate, model)`; choose the coherent family instead of switching another law's mode |
| `with_p_idle_quadratic_coherent(false)` | `with_p_idle_sin_squared(rate, model)`; choose the stochastic family directly |
| `with_p_idle_coherent_to_incoherent_factor(f)` | No replacement; the factor only modified the removed quadratic-rate path |
| `with_average_p_idle_linear_rate(r)` / `with_average_p_idle_quadratic_rate(r)` | No replacement; a gate-channel average-error conversion is not duration independent for a rate-times-duration law |

The old quadratic rate was in cycles per time and was converted before it
reached the runtime. Family rates are in radians per time and receive no such
conversion. At the removed path's default factor of `1.0`, the exact migration
is:

```text
with_p_idle_quadratic_rate(r)  ==  with_p_idle_sin_squared(r * PI, {"Z": 1.0})
```

Copying `r` directly into the family setter changes the channel by a factor of
pi.

Coherent evolution is not sampled and consumes no RNG draws. Whether it can be
consumed depends on the downstream consumer: the standard DEM builder rejects
coherent idle noise, the EEG route in `exp/pecos-eeg` represents it with an RZ
generator, and a simulator applies it only when it has a rotation executor.
PECOS #437 tracks the case where a missing executor silently dropped rotations.

To add the same kind of idle-noise site to both qubits after every two-qubit
gate, set its duration with `with_idle_after_2q(...)`:

```python
noise = (
    GeneralNoiseModelBuilder().with_p_idle_linear(0.01, {"X": 1 / 3, "Y": 1 / 3, "Z": 1 / 3}).with_idle_after_2q(1.0)
)
```

The duration only chooses where and how long idling occurs. It is not a
standalone probability: all configured linear, sine-squared, and coherent idle
families apply at these sites just as they do at a
scheduled `Idle` gate. A duration of `0.0` disables the after-two-qubit sites.
Consequently, code that previously used `with_p2_idle(0.01)` without a linear
idle rate now produces no after-2q idle noise; the equivalent configuration is
`with_p_idle_linear(0.01, {"X": 1/3, "Y": 1/3, "Z": 1/3}).with_idle_after_2q(1.0)`.

## Common Noise Model Examples

### Basic Depolarizing Noise

Simple uniform noise on all operations:

```python
# Uniform depolarizing noise
noise = (
    GeneralNoiseModelBuilder().with_p1(0.001).with_p2(0.01).with_p_prep(0.001).with_p_meas_0(0.001).with_p_meas_1(0.001)
)
```

### Realistic Hardware Noise

Model based on typical superconducting qubit parameters:

```python
noise = (
    GeneralNoiseModelBuilder()
    # Gate errors (two-qubit gates are typically 10x worse)
    .with_average_p1(0.0001)  # 0.01% single-qubit error
    .with_average_p2(0.001)  # 0.1% two-qubit error
    # State prep and measurement (often dominant errors)
    .with_p_prep(0.001)  # 0.1% prep error
    .with_p_meas_0(0.01)  # 1% false positive
    .with_p_meas_1(0.005)
)  # 0.5% false negative
```

### Ion Trap Noise Model

Ion traps have different characteristics than superconducting qubits:

```python
noise = (
    GeneralNoiseModelBuilder()
    # Excellent single-qubit gates
    .with_average_p1(0.00001)  # 0.001% error
    # Two-qubit gates are the limiting factor
    .with_average_p2(0.003)  # 0.3% error
    # State preparation
    .with_p_prep(0.001)  # 0.1% error
    # Asymmetric measurement: dark state error
    .with_p_meas_0(0.001)
    # Bright state error (higher)
    .with_p_meas_1(0.005)
)
```

### Biased Noise Model

Model with biased errors (e.g., more phase errors than bit flips):

```python
noise = (
    GeneralNoiseModelBuilder()
    # Biased single-qubit errors
    .with_average_p1(0.001)
    .with_p1_pauli_model(
        {
            "X": 0.1,  # 10% bit flips
            "Y": 0.1,  # 10% Y errors
            "Z": 0.8,  # 80% phase errors (dominant)
        }
    )
    # Biased two-qubit errors
    .with_average_p2(0.01)
    .with_p2_pauli_model(
        {
            "IZ": 0.3,  # 30% phase on second qubit
            "ZI": 0.3,  # 30% phase on first qubit
            "ZZ": 0.2,  # 20% correlated phase
            "XX": 0.2,  # 20% other errors
        }
    )
)
```


## Complete Example

Here's a comprehensive example showing various builder features:

```python
from pecos import sim, Qasm
from pecos_rslib import GeneralNoiseModelBuilder
from collections import Counter

# QASM circuit: 3-qubit GHZ state
qasm = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q[0];
cx q[0], q[1];
cx q[1], q[2];
measure q -> c;
"""

# Build comprehensive noise model
noise = (
    GeneralNoiseModelBuilder()
    # Global scaling
    .with_scale(1.2)  # 20% higher error rates
    # Make Hadamard gates perfect
    .with_noiseless_gate("H")
    # State preparation
    .with_p_prep(0.001)
    # Single-qubit gates with biased errors
    .with_average_p1(0.0001)
    .with_p1_pauli_model(
        {
            "X": 0.2,
            "Y": 0.2,
            "Z": 0.6,  # More dephasing
        }
    )
    # Two-qubit gates
    .with_average_p2(0.001)
    # Asymmetric measurement
    .with_p_meas_0(0.002)
    .with_p_meas_1(0.005)
)

# Run simulation
results = sim(Qasm(qasm)).noise(noise).seed(42).run(1000)

# Analyze results
counts = Counter(results.to_dict()["c"])
print("GHZ state measurement results:")
for state, count in counts.most_common(5):
    binary = format(state, "03b")
    print(f"|{binary}>: {count}")
```

## Tips

1. **Use Average Probabilities**: They're more intuitive and match experimental error rates.

2. **Set Seeds for Reproducibility**: Use `.seed()` on the simulation builder. The shared Monte Carlo loop reseeds every component before each shot, overriding seeds set directly on noise models with `.with_seed()`.

3. **Start Simple**: Begin with uniform probabilities, then add complexity as needed.

4. **Match Hardware Specs**: Use error rates from device calibration data when available.

5. **Consider Error Hierarchies**: Typically: measurement > two-qubit > state prep > single-qubit.

6. **Use Noiseless Gates Sparingly**: Only for gates that are effectively perfect (e.g., virtual Z rotations).

## Comparison with Predefined Noise Models

While builders offer maximum flexibility, PECOS also provides simpler convenience functions:

```python
from pecos import depolarizing_noise, GeneralNoiseModelBuilder

# Simple depolarizing (uniform probability)
simple = depolarizing_noise().with_uniform_probability(0.001)

# Equivalent with GeneralNoiseModelBuilder
builder = (
    GeneralNoiseModelBuilder()
    .with_p1(0.001)
    .with_p2(0.001)
    .with_p_prep(0.001)
    .with_p_meas_0(0.001)
    .with_p_meas_1(0.001)
)

# Builder advantages:
# - Fine-grained control
# - Pauli error models
# - Scaling factors
# - Noiseless gates
# - Crosstalk modeling
```

## Next Steps

- For performance optimization, see [QASM Simulation Guide](qasm-simulation.md)
- For the complete API reference, see the [API Documentation](../api/api-reference.md)
