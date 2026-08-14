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

# Create noise model with builder
noise = (
    GeneralNoiseModelBuilder()
    .with_seed(42)  # Reproducible randomness
    .with_p1(0.001)  # Single-qubit gate error
    .with_p2(0.01)
)  # Two-qubit gate error

# Use with sim()
results = sim(Qasm(qasm)).noise(noise).run(1000)
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

### Reusable Pauli-Plus-Leakage Channels

Use `PauliLeakageChannel` when a gate hook needs its own outer application probability and a
relative distribution of state-independent Pauli or leakage events. The event weights form a
normalized conditional distribution. Identity is implicit when the outer coin fails, so a
single-qubit `"I"` event and the all-identity multi-qubit event are not accepted.

```python
from pecos import GeneralNoiseModelBuilder, PauliLeakageChannel

control_faults = PauliLeakageChannel(
    probability=0.001,
    events={"X": 0.40, "Y": 0.20, "Z": 0.30, "L": 0.10},
)

noise = (
    GeneralNoiseModelBuilder()
    .add_p1_pauli_leakage_channel_before_gate(control_faults)
    .add_p1_pauli_leakage_channel_after_gate(control_faults)
)
```

`L` means `any -> L`. A Pauli event is ignored on an already leaked qubit. Leakage events follow
the general model's existing `leakage_scale`: when leakage is converted to depolarization, the
sampled replacement Pauli is applied instead. Channel outer probabilities follow the global and
corresponding p1 or p2 scale factors.

Two-qubit hooks accept either independently drawn single-qubit channels or one correlated event
distribution. `I` is allowed on an individual leg of a multi-qubit event such as `IX` or `IL`:

```python
from pecos import (
    P2PauliLeakageStep,
    PauliLeakageChannel,
    PauliLeakageDict,
    TwoQubitPauliLeakageChannel,
)

first_leg = PauliLeakageChannel(0.01, {"X": 0.8, "L": 0.2})
second_leg = PauliLeakageChannel(0.02, {"Z": 0.9, "L": 0.1})

# Two independently drawn outer coins.
independent = first_leg * second_leg

# One outer coin followed by one correlated pair event.
joint = P2PauliLeakageStep.joint(
    TwoQubitPauliLeakageChannel(
        probability=0.01,
        events={"IX": 0.30, "XI": 0.30, "XX": 0.20, "IL": 0.10, "LI": 0.10},
    )
)

noise = (
    GeneralNoiseModelBuilder()
    .with_p2_pauli_leakage_steps_before_gate([independent])
    .with_p2_pauli_leakage_steps_after_gate([joint])
)

# Event dictionaries can also be tensored under one joint outer coin.
x_or_leak = PauliLeakageDict({"X": 0.8, "L": 0.2})
z_or_leak = PauliLeakageDict({"Z": 0.9, "L": 0.1})
product_events = x_or_leak * z_or_leak
```

At each hook, Pauli-plus-leakage entries execute in insertion order and then transition entries
execute in insertion order. Thus a before-gate leakage event can be recovered by a configured
before-gate transition channel. The complete ordering is: before Pauli-plus-leakage, before
transitions, ideal gate and ordinary p1/p2 noise, after Pauli-plus-leakage, then after transitions.
After-2Q idle noise remains last.

### Population and Leakage Transition Channels

Use `TransitionChannel` for stochastic population transfer among `"0"`, `"1"`, and leakage
`"L"`. The nested mapping is a conditional transition matrix:
`transitions[source][destination] = P(destination | source)`. Every supplied source row must sum
to one. An omitted source row is exact identity, so a leakage-recovery-only channel does not
measure or disturb a computational qubit.

The separate `probability` is the chance that PECOS considers the channel at that location. The
effective map is `(1 - probability) * identity + probability * transitions`; an explicit identity
entry with probability near one is therefore unnecessary.

```python
from pecos import GeneralNoiseModelBuilder, TransitionChannel

population_transfer = TransitionChannel(
    probability=0.01,
    transitions={
        "0": {"0": 0.05, "1": 0.05, "L": 0.90},
        "1": {"0": 0.10, "1": 0.80, "L": 0.10},
        "L": {"0": 0.45, "1": 0.45, "L": 0.10},
    },
)

# Multiple channels are independent and execute in insertion order.
noise = (
    GeneralNoiseModelBuilder()
    .add_p1_transition_channel_before_gate(population_transfer)
    .add_p1_transition_channel_after_gate(TransitionChannel.leak_recovery(probability=0.90, p_zero=0.50))
)
```

A supplied `"0"` or `"1"` row is a Z-basis population map: PECOS measures that computational
source internally and consumes the result. This intentionally destroys computational-basis
coherence. Pauli faults remain in `with_p1_pauli_model` and `with_p2_pauli_model`; transition
channels are not a replacement for `rho -> P rho P` noise. Use the reusable
`PauliLeakageChannel` family when those faults need their own explicit before/after hooks.

`TransitionDict` validates the conditional matrix independently of its outer application
probability and provides transition-map algebra. `left * right` is a tensor/Kronecker product.
`after @ before` is sequential composition: it applies `before` first, then `after`. The named
methods `tensor`, `compose`, and `then` are equivalent and can be clearer in library code.

```python
from pecos import TransitionChannel, TransitionDict, TwoQubitTransitionChannel

leak_zero = TransitionDict({"0": {"L": 1.0}})
recover_to_one = TransitionDict({"L": {"1": 1.0}})

# One-qubit composition: 0 -> L -> 1.
reset_zero_to_one = recover_to_one @ leak_zero
channel = TransitionChannel(probability=0.01, transitions=reset_zero_to_one)

# A joint map formed from two independent conditional matrices, under one outer coin.
recover_first = TransitionDict({"L": {"0": 1.0}})
recover_second = TransitionDict({"L": {"1": 1.0}})
joint_channel = TwoQubitTransitionChannel(
    probability=0.90,
    transitions=recover_first * recover_second,
)
```

Pass the resulting `TransitionDict` directly to the channel constructor. Besides avoiding a second
validation pass, this preserves which tensor factors have omitted identity rows, so PECOS does not
perform an unnecessary computational-basis measurement on those legs. `to_dict()` returns a plain
nested dictionary when serialization is needed.

At a two-qubit hook, pass a `TwoQubitTransitionChannel` directly for a correlated map on all nine
pair states `00`, `01`, `0L`, ..., `LL`. Use an explicit `P2TransitionStep.independent` only for
two potentially different single-qubit channels:

```python
from pecos import TransitionChannel, TwoQubitTransitionChannel

first_leg = TransitionChannel.leak_recovery(0.90, p_zero=0.75)
second_leg = TransitionChannel.leak_recovery(0.80, p_zero=0.25)
independent = first_leg * second_leg

joint = TwoQubitTransitionChannel(
    probability=0.02,
    transitions={
        "0L": {"L1": 0.20, "0L": 0.80},
        "LL": {"00": 0.50, "11": 0.50},
    },
)

noise = (
    GeneralNoiseModelBuilder()
    .with_p2_transition_steps_before_gate([independent])
    .add_p2_transition_channel_after_gate(joint)
)
```

Use `"*"` as a coherent identity wire when a two-qubit channel acts on only one gate leg. The
wildcard position must be the same in the source and every destination. PECOS neither measures nor
otherwise resolves that leg. Rules for both orientations can coexist; if both qubits are leaked,
the two matching rules act independently under the channel's one outer application coin:

```python
recover_either_leg = TwoQubitTransitionChannel(
    probability=1.0,
    transitions={
        "*L": {"*0": 0.45, "*1": 0.45, "*L": 0.10},
        "L*": {"0*": 0.45, "1*": 0.45, "L*": 0.10},
    },
)

noise = GeneralNoiseModelBuilder().add_p2_transition_channel_after_gate(recover_either_leg)
```

For example, `"*L" -> "*0"` inspects only the second leg's classical leakage flag and recovers
that leg to zero. PECOS retains the wildcard's identity metadata when compiling the channel, so it
does not introduce a computational-basis measurement on the first leg.

A channel definition uses either this factorized wildcard form or fully concrete pair-state rows;
the two forms cannot be mixed in one dictionary. Use an additional ordered channel when both forms
are needed at the same gate hook.

The two products intentionally have different outer-probability semantics:

- `TransitionDict * TransitionDict`, followed by `TwoQubitTransitionChannel(p, ...)`, makes one
  joint conditional matrix selected by one outer probability `p`.
- `TransitionChannel(p_first, ...) * TransitionChannel(p_second, ...)` makes an independent-leg
  `P2TransitionStep`; PECOS draws the two outer application coins separately. This is also
  available as `P2TransitionStep.tensor_product(first_leg, second_leg)` or
  `P2TransitionStep.independent(first_leg, second_leg)`.

Before-gate transitions finish before PECOS decides whether to execute the ideal gate, so recovery
can enable it and new leakage can suppress it. After-gate transitions run after ordinary p1/p2
noise; recovery there does not retroactively execute a suppressed gate. The convenience
`add_p2_transition_channel_before_gate(channel)` and the corresponding after-gate method accept
either channel type directly: a `TransitionChannel` is applied independently to both legs, while a
`TwoQubitTransitionChannel` is applied jointly. Construct an explicit `P2TransitionStep` only when
the two legs need different single-qubit channels or when assembling a mixed step list.

### Scaling and Global Parameters

```python
noise = (
    GeneralNoiseModelBuilder()
    .with_seed(42)  # Random seed for reproducibility
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
    .with_seed(42)
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
    .with_seed(42)
    # Excellent single-qubit gates
    .with_average_p1(0.00001)  # 0.001% error
    # Two-qubit gates are the limiting factor
    .with_average_p2(0.003)  # 0.3% error
    # State preparation
    .with_p_prep(0.001)  # 0.1% error
    # Asymmetric measurement (bright/dark state detection)
    .with_p_meas_0(0.001)  # Dark state error
    .with_p_meas_1(0.005)
)  # Bright state error (higher)
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
    # Reproducibility
    .with_seed(42)
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
results = sim(Qasm(qasm)).noise(noise).run(1000)

# Analyze results
counts = Counter(results.to_dict()["c"])
print("GHZ state measurement results:")
for state, count in counts.most_common(5):
    binary = format(state, "03b")
    print(f"|{binary}>: {count}")
```

## Tips

1. **Use Average Probabilities**: They're more intuitive and match experimental error rates.

2. **Set Seeds for Reproducibility**: Always use `.with_seed()` for reproducible results in research.

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
