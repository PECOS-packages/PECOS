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
that hook and appends the replacement list in the given order, preserving other-family entries.
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
maps with `left * right` for a joint channel with one outer coin. A tensor of two one-qubit
`TransitionDict`s exposes wildcard rows through `.transitions` and `to_dict()`, so rebuilding
from a plain dictionary preserves identity wires and measures the same legs. Higher-arity
transition tensors continue to expose expanded rows.

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

The model does not provide mixture/product channel objects, conditional follow-up channels,
arbitrary Kraus composition, or coherent superpositions involving leakage. Hooks trigger on the
single- and two-qubit gates the model processes, so every gate execution path that uses
`GeneralNoiseModel` runs them: ordinary `sim()` execution, the Selene general-noise wrapper
(which exposes these ordered hook lists through plain-data hook specs and matching fluent
setters), and runtime frames. In a runtime frame, a hook that needs an internal measurement
holds later frame records until the paused gate completes, as a measurement-crosstalk payload
does. Scheduled profiles do not accept hooks: they allow one simulator call per batch. Existing
gate-noise configuration is not migrated into this channel executor; such a migration would
require separate seeded and statistical parity checks.

### Leakage readout within a batch

The Rust general noise model records leakage when each measurement is emitted.
A later reset or leakage fault in the same batch cannot change that earlier
readout. `MeasureZLeaked` returns 2 for a leaked qubit; ordinary measurement starts
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
widened. Scheduled profiles keep their single-call contract for batches without
explicit [measurement-crosstalk exposures](scheduled-event-adapters.md#measurement-crosstalk-exposures),
including batches with preparation crosstalk. Failed crosstalk continuations stay
blocked across clones until reset, which clears pending gates and buffered readouts. Low-level callers of
`apply_noise_on_start` must process each returned command message and feed replies
to `apply_noise_on_continue_processing` until no commands remain; the initial
command message need not contain every source gate.

### Readout matrices

`GeneralNoiseModel` supports separate conditional readout matrices for four Z-measurement
flavors. `MZ` (also used by `MPZ`) reports `{0, 1}`; `MeasureZLeaked` reports `{0, 1, 2}`;
`MeasureZLost` reports `{0, 1, 3}`; and `MeasureZLeakedLost` reports `{0, 1, 2, 3}`.
Here 2 heralds leakage and 3 heralds loss. A readout matrix can also report 3 for a
present qubit; this readout misidentification does not remove the atom.

Rows describe the true state: `"0"`, `"1"`, `"leaked"`, or `"lost"`. The simulator supplies the
computational outcome; loss and leakage are captured when the measurement is emitted. Columns are
string outcome labels. For example, this JSON configures readout of present qubits:

```json
{
  "measurement": {
    "readout": {
      "MeasureZLost": {
        "0": {"0": 0.9958, "1": 0.004, "3": 0.0002},
        "1": {"1": 0.9958, "0": 0.004, "3": 0.0002}
      }
    }
  }
}
```

Python provides the same configuration through immutable builder chaining:

```python
from pecos_rslib import general_noise

noise = general_noise().with_readout(
    "MeasureZLost",
    {
        "0": {"0": 0.9958, "1": 0.004, "3": 0.0002},
        "1": {"1": 0.9958, "0": 0.004, "3": 0.0002},
    },
)
```

Rust accepts `ReadoutMatrix::new(flavor, &rows)?` followed by
`GeneralNoiseModelBuilder::with_readout(matrix)`, with nested `BTreeMap<String, ...>` rows.
Calling `with_readout` again replaces that flavor's entire matrix; other flavors accumulate.
A present JSON `measurement.readout` replaces all builder readout matrices, including when
it is `{}`; omitting the key preserves existing matrices. Flavor names use the shared,
case-insensitive gate parser (`Measure` is an alias of `MZ`). Two JSON names resolving to
one flavor are rejected. Configure `MZ` for `MPZ`; other measurement bases and non-measurement
gates have no readout matrix. The new heralds have no QASM, PHIR, QIS, Selene, or SLR source syntax.

Missing outcomes have probability zero. Each configured row must be nonempty, use outcomes
in its flavor's alphabet, and contain finite probabilities in `[0, 1]` whose sum is within
`1e-5` of one. Accepted sums differing from one by more than `1e-10` are renormalized.
Empty flavor maps and unknown rows are errors. Python row and outcome keys must be strings;
invalid matrices raise `ValueError`. A well-formed `"lost"` row can be constructed, but
Rust `try_build`, JSON `build`, and Python simulation building reject it unless qubit loss
is configured. Configure loss or remove the lost row.

Unconfigured rows keep the legacy readout exactly: a true 0 or 1 takes one asymmetric
`p_meas_0` or `p_meas_1` flip draw. A leaked qubit nominally reads 2 if its alphabet includes
2, with no default draw; otherwise it nominally reads 1 and takes the default `p_meas_1`
flip. That fallback never uses a configured `"1"` row. Consequently, without configuration,
`MeasureZLost` behaves like `MZ`, and `MeasureZLeakedLost` like `MeasureZLeaked`, including
RNG consumption.

Each configured row replaces only its flavor and true-state default, and always consumes
one categorical draw, even for a deterministic row. Configured probabilities are never
scaled; unconfigured rows keep the existing measurement and overall scaling. Noiseless
readout returns the nominal outcome with no draw: `MZ` and `MPZ` use the `MZ` noiseless key,
and each herald uses its own gate key. Gate emission still uses the gate's own noiseless
key. Transition-hook and crosstalk-injected measurements bypass readout matrices.

Configured readout is outside the simple probability subset: neo routing, runtime-frame
noise, and the Selene general-noise adapter reject it. Remove `measurement.readout` to use
the latter two consumers. Versioned runtime and scheduled wire formats also reject the
herald gates by name.

Classical storage is unchanged. QASM and PHIR-JSON store any nonzero outcome as bit 1;
the Python PHIR interpreter masks indexed writes with `& 1`. QIS retains the full `u32`.
The Selene adapter accepts leakage outcomes only up to 2. None provides source syntax for
the herald gates; use core gates or byte-message builders for those gates.

### Qubit loss

`GeneralNoiseModel` models a missing atom separately from leakage out of the computational
subspace. Loss has its own configuration, channels and state. A loss event disentangles the
backend with an injected `PZ`, marks the atom lost and clears leakage. Recovery and reload
also clear leakage; an atom cannot be both lost and leaked. Injected resets and recovery
Paulis do not run noise or crosstalk and do not change the prepared-qubit pool. Shot reset
clears loss but retains configuration. With no loss section, existing seeded streams,
outcomes and RNG consumption are unchanged.

The JSON section is optional. If present, `enabled` is required. `{"enabled": false}` disables
loss; any other key alongside `enabled: false` is an error, even if its value is zero or false.
A present section replaces previous builder loss; an absent section preserves it. Unknown
keys, wrong JSON types and invalid distributions are errors. For example:

```json
{
  "loss": {
    "enabled": true,
    "measurement": 0.0,
    "prep": 0.0,
    "idle_rate": 0.0,
    "lost_partner": {"loss_probability": 1.0, "pauli_model": {"I": 0.5, "Z": 0.5}},
    "reload_on_prep": false,
    "default_recovery": true,
    "channels": [
      {"site": "before_two_qubit", "kind": "recovery", "false_negative": 0.001, "false_positive": 0.001},
      {"site": "before_two_qubit", "kind": "events", "probability": 0.01, "events": {"LI": 0.5, "IL": 0.5}}
    ]
  }
}
```

In Python, `general_noise().with_loss(loss_dict)` converts the dict strictly to JSON, then
uses the same parser as `with_config_json(json.dumps({"loss": loss_dict}))`:

```python
from pecos_rslib import general_noise

noise = general_noise().with_loss(
    {"enabled": True, "default_recovery": False, "measurement": 0.006, "reload_on_prep": True},
)
```

Only dicts with string keys, lists, strings, booleans, i64 integers and finite floats convert
to JSON. `None`, tuples, sets, numpy scalars and integers outside i64 raise `ValueError`.
Booleans stay booleans and are rejected as probabilities. Rust uses
`QubitLoss::builder().with_measurement(p).with_default_recovery(false).build()?`, followed by
`GeneralNoiseModelBuilder::with_loss(loss)`. `without_loss()` clears it. `LossSite` names the
four sites; `add_events_channel(site, p, &weights)` and
`add_recovery_channel(site, false_negative, false_positive)` append channels. Builder errors
name the field, channel index and bad value, and suggest a correction.

Defaults with `enabled: true` are zero measurement/prep/idle loss, partner loss probability
one, no partner Pauli distribution, no reload, and default recovery enabled. If no recovery
channel was listed, default recovery adds exactly one before-two-qubit channel with
false-negative and false-positive rates both `1e-3`. This causes about `2e-3` resets per
present two-qubit gate even without a loss source. Set `default_recovery: false` to disable
this behavior; an explicit recovery anywhere also prevents adding the default.

A lost operand suppresses ordinary one-qubit operations and a two-qubit pair containing it,
including base fault, emission, seepage and hook selection draws. `QFree` followed by `QAlloc`
does not restore a lost atom. `Custom` is one operand group: before-site recovery can restore
its lost atoms, and each non-loss before-hook skips the group only if an atom is still lost
when that hook runs. After the before-hooks, any remaining lost operand suppresses the whole
operation and its p1 draws. Non-loss after-hooks skip the group if it was suppressed or any
atom is lost now; loss channels still visit its qubits individually. `TrackedPauliMeta`,
`MX` and `MeasureFree` follow the generic single-qubit operand rule. Outcomes from `MX` and
`MeasureFree` are not tracked by this noise model and fail its outcome-count check. The
leakage path filters `Custom` operands individually. `Idle` emits only noise. Hook sites
exclude `Idle`, preparation, tracked measurements and crosstalk payloads.

Only `PZ` and `MPZ` reload, and only with `reload_on_prep: true`. Without reload, lost `PZ`
operands are dropped before prep faults or crosstalk, and are not added to the prepared pool.
`MPZ` always emits its measurement, but its lost operand remains unprepared unless reloaded.
Measurements of the five tracked flavors always emit, preserving outcome counts. State is
captured at emission, before measurement loss or the preparation half of `MPZ`.

| Gate | Lost | Leaked | Present |
| --- | --- | --- | --- |
| `MZ` / `MPZ` | 0, then default `p_meas_0` flip | 1, then default `p_meas_1` flip | 0/1 plus readout noise |
| `MeasureZLeaked` | 0, then default `p_meas_0` flip | 2 | 0/1 plus readout noise |
| `MeasureZLost` | 3 | 1, then default `p_meas_1` flip | 0/1 plus readout noise |
| `MeasureZLeakedLost` | 3 | 2 | 0/1 plus readout noise |

A configured `lost` row replaces the lost fallback with one categorical draw, even for a
one-label row. The fallback never uses a configured `"0"` row. Its flip always draws, even
at zero probability; nominal 3 makes no draw. Noiseless readout returns nominal values
without a draw. `MPZ` shares `MZ`'s readout/noiseless-readout key, but uses its own `MPZ` key
to disable its loss-source draws. Other noiseless operations still drop lost operand groups,
and noiseless `PZ`/`MPZ` reload only if configured. They make no loss draws or spreading.

Loss sources and recovery:

- `measurement`: after each user measurement command, visit operands in order and draw
  loss for each non-lost atom, including leaked atoms. A hit emits its `PZ` after the
  measurement, so the current readout is unchanged. Injected measurements never draw it.
- `prep`: after a preparation actually runs and after reload, draw per prepared operand.
  `PZ` runs all ordinary prep faults, then all prep-loss draws, then prep crosstalk. A
  prep-lost operand remains in the prepared pool. `MPZ` orders measurement, measurement
  loss, its prep half/reload, then prep loss; no reload means a still-lost atom skips prep loss.
- `idle_rate`: at every idle site, including `idle_after_2q`, draw loss first with
  `clamp(rate * duration, 0, 1)`. Linear, sine-squared and coherent idle families then skip
  lost atoms. Rates may exceed one; rates and rate-duration products must be finite.
- `events`: a positive outer probability draws once; a hit draws one conditional label.
  Single-qubit labels are `X/Y/Z/L`. Joint two-qubit labels are ordered pairs of
  `I/X/Y/Z/L` except `II`. `L` loses that leg; a Pauli on a leaked leg is sampled and
  discarded. An already-lost operand or pair skips the entire entry, at either site.
- `recovery`: visit each operand (each pair's legs in order). A lost atom draws the
  false-negative rate; a present or leaked atom draws the false-positive rate. A caught
  loss or false alarm emits `PZ`, then a fair coin and optional `X`, producing the maximally
  mixed state and clearing both lost and leaked marks. Detection and reset finish for one
  leg before visiting the next. A missed loss remains lost.

At a suppressed two-qubit gate, the surviving atom is lost with
`lost_partner.loss_probability`. Otherwise, if `pauli_model` is configured, sample its
`I/X/Y/Z` distribution and emit the Pauli unless the survivor is leaked. Both-lost pairs
have no survivor. A pair lost by that command's own before-site events does **not** spread
again at this gate. A previously lost leg, including one missed by recovery, does spread.
Spreading output follows the emitted gates of the batched command. After-site loss never
retroactively suppresses a gate or spreads at it.

Before-sites run recovery entries, existing non-loss hooks, then events entries. After-sites
run loss entries in their listed order, then non-loss hooks. Listed order is preserved
within each kind. Non-loss before-hooks skip currently lost groups after recovery;
after-hooks skip groups suppressed for loss even if recovery restored them, and groups
newly lost at that site. Loss entries run even when the gate was suppressed. All state and
hook positions survive simulator round trips. After-two-qubit idle follows after-hooks.
Hooks iterate hook-major, pair-minor; splitting a batch preserves distributions, not seeded
streams or command order. Loss is excluded from all crosstalk victim draws.

Measurement, prep and idle loss scale with overall times their matching section scale.
An events outer probability scales with overall times the site's single-/two-qubit scale.
Conditional event weights, recovery detection rates, and **both** `lost_partner` entries
are unscaled. In particular, `scaling.overall` does not reach partner loss or its Pauli model.
All probabilities must be finite in `[0, 1]`, both before and after scaling; conditional
weights must sum to one within `1e-5`. Every loss probability exactly zero after scaling
makes no draw; every positive probability (including one) draws once. This includes FN/FP
and spreading. Categorical draws and the fallback lost-readout flip are exceptions: they
always draw when reached. Recovery's fair coin always draws.

Loss is outside the simple probability subset (`simple_probabilities()` returns `None`).
Neo routing, `RuntimeNoise`, and the Selene general-noise adapter reject it; remove the
`loss` section to use the latter two. Scheduled frames, pecos-qasm's flat schema,
pecos-neo's own model and the Python Selene plugin are unchanged. Chain/group loss, decoder
erasure flags, failing reload and recovery at prep/measurement/idle sites are not supported.

#### Published-model recipes

The following recipes implement models from the papers listed below, checked on
2026-11-12. **[inf] marks our inference, not a statement by the paper.**
Every published numeric instance disables default recovery; row 8 intentionally enables it.
Pauli errors for gates that execute remain separately configurable through the existing
noise sections. The JSON below is a complete `GeneralNoiseConfig` for each loss recipe;
conditional partner probabilities remain unscaled.

| Row | Model | Loss site / main limitation |
| --- | --- | --- |
| 1 | Perrin, Roger, Pupillo, arXiv:2603.24237 | Before CZ, correlated partner loss |
| 2 | Perrin, Jandura, Pupillo, arXiv:2412.07841 | Before CZ, independent atoms; marginal survivor noise [inf] |
| 3 | Liu et al., arXiv:2603.04156 | After operations; random-bit post-processing of flagged loss |
| 4 | Baranes et al., arXiv:2502.20558 | Correlated model B; model A product recipe |
| 5 | Bluvstein et al., arXiv:2506.20661 | After CZ; uniform prep rate only; no spectator loss |
| 6 | Nishio et al., arXiv:2607.29603 | Data-only idle between rounds, circuit-controlled reload |
| 7 | Coble, Ye, Delfosse, arXiv:2511.16632 | Whole-chain loss cannot be expressed |
| 8 | PECOS default | No sources, before-2q recovery at `1e-3`/`1e-3` |

**1. Perrin, Roger, Pupillo — arXiv:2603.24237**

One atom per CZ is chosen uniformly; its partner is lost with `p_c`, otherwise gets
I 3/8, Z 3/8, X 1/8, Y 1/8 (Sec. II.B, Eq. A4). Use outer `p_l`, `LL: p_c`,
`LI, IL, LZ, ZL: (1-p_c)*3/16` each and `LX, XL, LY, YL: (1-p_c)/16` each.
A gate starting beside an already-lost atom loses the survivor with probability one.
Measurements, 1q gates and idling are loss-free. Reload is stated for data atoms via the
loss detection unit; ancilla reload is [inf]. Read loss detection units with `MeasureZLost`.
Example: `p_l = 0.01`, `p_c = 0.2`.

```json
{
  "loss": {
    "enabled": true,
    "default_recovery": false,
    "channels": [
      {
        "site": "before_two_qubit",
        "kind": "events",
        "probability": 0.01,
        "events": {
          "LL": 0.2,
          "LI": 0.15000000000000002,
          "IL": 0.15000000000000002,
          "LZ": 0.15000000000000002,
          "ZL": 0.15000000000000002,
          "LX": 0.05,
          "XL": 0.05,
          "LY": 0.05,
          "YL": 0.05
        }
      }
    ],
    "lost_partner": {
      "loss_probability": 1
    },
    "reload_on_prep": true
  }
}
```

**2a. Perrin, Jandura, Pupillo — arXiv:2412.07841**

Each CZ atom is independently lost with `p_l` (joint `p_l^2`); a survivor next to
an already-lost atom can still be lost with `p_l`. The two-qubit depolarizing channel is
kept on removed gates (Sec. 2.1, 7.2). Let `r = 4*p_d/15`, `w = (1-p_l)/(2-p_l)`;
outer probability is `p_l*(2-p_l)`, `LL: p_l/(2-p_l)`, single-loss weights per lost side
`I: w*(1-3r)`, `X, Y, Z: w*r`, with mirror labels. Partner Pauli weights are
`I: 1-3r`, `X, Y, Z: r`. The `r` terms [inf] emulate the kept channel's survivor marginal;
without them Eq. 24's loss detection unit non-detection rate is not reproduced. Ancilla
reload is [inf]. Example: `p_l = 0.01`, `p_d = 0.003`.

```json
{
  "loss": {
    "enabled": true,
    "default_recovery": false,
    "channels": [
      {
        "site": "before_two_qubit",
        "kind": "events",
        "probability": 0.0199,
        "events": {
          "LL": 0.005025125628140704,
          "LI": 0.49629346733668345,
          "IL": 0.49629346733668345,
          "LX": 0.00039798994974874375,
          "XL": 0.00039798994974874375,
          "LY": 0.00039798994974874375,
          "YL": 0.00039798994974874375,
          "LZ": 0.00039798994974874375,
          "ZL": 0.00039798994974874375
        }
      }
    ],
    "lost_partner": {
      "loss_probability": 0.01,
      "pauli_model": {
        "I": 0.9976,
        "X": 0.0008,
        "Y": 0.0008,
        "Z": 0.0008
      }
    },
    "reload_on_prep": true
  }
}
```

**2b. Refined variant**

Refined variant (Sec. 8): survivor Z with probability 1/2 at the loss gate. Replace
the preceding single-loss survivor weights by `I, Z: w*(1-2r)/2`, `X, Y: w*r`.
The outer probability, `LL` and partner Pauli distribution remain unchanged. The same
example parameters and [inf] qualifications apply.

```json
{
  "loss": {
    "enabled": true,
    "default_recovery": false,
    "channels": [
      {
        "site": "before_two_qubit",
        "kind": "events",
        "probability": 0.0199,
        "events": {
          "LL": 0.005025125628140704,
          "LI": 0.24834572864321605,
          "IL": 0.24834572864321605,
          "LX": 0.00039798994974874375,
          "XL": 0.00039798994974874375,
          "LY": 0.00039798994974874375,
          "YL": 0.00039798994974874375,
          "LZ": 0.24834572864321605,
          "ZL": 0.24834572864321605
        }
      }
    ],
    "lost_partner": {
      "loss_probability": 0.01,
      "pauli_model": {
        "I": 0.9976,
        "X": 0.0008,
        "Y": 0.0008,
        "Z": 0.0008
      }
    },
    "reload_on_prep": true
  }
}
```

**3. Liu et al. — arXiv:2603.04156**

Loss is AFTER each operation, so the gate executes and keeps its noise. Each 2q leg
is independently lost with `q = p_loss/2`: outer `1-(1-q)^2`, `LI, IL: (1-q)/(2-q)`,
`LL: q/(2-q)`. Later gates on lost atoms are removed; the survivor gets an idling error.
Partner loss zero is [inf; unstated]; interpreting the idling error as one-qubit
depolarizing is [inf]: `I: 1-p_pauli`, `X, Y, Z: p_pauli/3`. Lost atoms are replaced by
fresh |0> atoms (Sec. II, III.A, VII.A, App. M.1). Readout has rows
`0: {0: 1-p_readout/2, 3: p_readout/2}`, `1: {1: 1-p_readout/2, 3: p_readout/2}`,
`lost: {3: 1-p_readout/2, 0: p_readout/4, 1: p_readout/4}`. Assign a random bit to
outcome 3 in post-processing. Loss after 1q operations would use an after-single-qubit
`{L: 1}` entry; its rate is unstated and omitted. Example:
`p_loss = p_readout = 0.002` (tied by the paper), `p_pauli = 0.001`.

```json
{
  "loss": {
    "enabled": true,
    "default_recovery": false,
    "channels": [
      {
        "site": "after_two_qubit",
        "kind": "events",
        "probability": 0.001998999999999973,
        "events": {
          "LI": 0.4997498749374687,
          "IL": 0.4997498749374687,
          "LL": 0.0005002501250625312
        }
      }
    ],
    "lost_partner": {
      "loss_probability": 0,
      "pauli_model": {
        "I": 0.999,
        "X": 0.0003333333333333333,
        "Y": 0.0003333333333333333,
        "Z": 0.0003333333333333333
      }
    },
    "reload_on_prep": true
  },
  "measurement": {
    "readout": {
      "MeasureZLost": {
        "0": {
          "0": 0.999,
          "3": 0.001
        },
        "1": {
          "1": 0.999,
          "3": 0.001
        },
        "lost": {
          "3": 0.999,
          "0": 0.0005,
          "1": 0.0005
        }
      }
    }
  }
}
```

**4. Baranes et al. — arXiv:2502.20558**

Correlated model B (Table I channel 2) uses `p′ = 1-sqrt(1-p_CZ)`, outer `2*p′`,
and `LI, IL, LZ, ZL: 1/4` each. Before-two-qubit placement is [inf; timing unstated];
partner loss zero is [inf; unstated]. Default model A (App. F) instead uses the product
of two per-leg distributions: for each label `ab`, conditional weight is
`P(a)*P(b)/(1-P(I)^2)`, excluding `II`, and outer probability `1-P(I)^2`. If the outer
probability is zero, omit the entry. Movement errors are Pauli idle errors, not loss;
there is no `idle_rate`. Read `MeasureZLost`, assigning a random value to 3 afterwards.
Reload depends on syndrome extraction; this instance chooses reload on prep.
Example model B: `p_CZ = 0.01`.

```json
{
  "loss": {
    "enabled": true,
    "default_recovery": false,
    "channels": [
      {
        "site": "before_two_qubit",
        "kind": "events",
        "probability": 0.01002512578676007,
        "events": {
          "LI": 0.25,
          "IL": 0.25,
          "LZ": 0.25,
          "ZL": 0.25
        }
      }
    ],
    "lost_partner": {
      "loss_probability": 0
    },
    "reload_on_prep": true
  }
}
```

**5. Bluvstein et al. — arXiv:2506.20661**

Use the stated numbers: each qubit lost independently with `q = 0.00183` after CZ
and its two-qubit Pauli channel; outer `1-(1-q)^2`, `LI, IL: (1-q)/(2-q)`,
`LL: q/(2-q)`. Readout loss is `0.006`; reset loss is `0.006` for data and `0.011` for
ancillas, but only one uniform rate is expressible (this instance uses data's `0.006`).
Idle loss is `0.09e-6` per time unit. The linear `rate*duration` matches the paper's
`1-(1-rate)^duration` only to first order. Methods turns off a gate with a lost atom
while still applying gate error. Partner loss `0.00183` is [inf, from the entangling-zone
spectator rule]; partner Pauli `I: 0.9984, X: 5e-5, Y: 5e-5, Z: 0.0015` is its 2q
channel's one-leg marginal. Ancillas are a fresh block each round; data is prepared once,
so enable reload on prep. The paper reads a third outcome and its decoder assigns zero:
use `MZ` (dark 0), or `MeasureZLost` and map 3 to 0. Loss of spectator atoms in the
entangling zone cannot be expressed.

```json
{
  "loss": {
    "enabled": true,
    "default_recovery": false,
    "channels": [
      {
        "site": "after_two_qubit",
        "kind": "events",
        "probability": 0.003656651100000019,
        "events": {
          "LI": 0.4995420810041188,
          "IL": 0.4995420810041188,
          "LL": 0.0009158379917624627
        }
      }
    ],
    "measurement": 0.006,
    "prep": 0.006,
    "idle_rate": 9e-08,
    "lost_partner": {
      "loss_probability": 0.00183,
      "pauli_model": {
        "I": 0.9984,
        "X": 5e-05,
        "Y": 5e-05,
        "Z": 0.0015
      }
    },
    "reload_on_prep": true
  }
}
```

**6. Nishio et al. — arXiv:2607.29603**

Data atoms are lost i.i.d. between rounds; ancillas never are. Place `Idle` only on
data between rounds with `idle_rate*duration = p_loss`. At a gate with lost data the
survivor gets a generic Pauli channel `p_P`; put the user's distribution in
`lost_partner.pauli_model`. Ideal Rydberg CZ uses identity, so the instance omits that
map. Partner loss probability is zero. Reload is protocol-driven: the user's circuit
decides; this instance has `reload_on_prep: false`. Example: `p_loss = 0.01`,
`idle_rate = 0.01`, `Idle` duration 1.

```json
{
  "loss": {
    "enabled": true,
    "default_recovery": false,
    "idle_rate": 0.01,
    "lost_partner": {
      "loss_probability": 0
    },
    "reload_on_prep": false
  }
}
```

**7. Coble, Ye, Delfosse — arXiv:2511.16632**

Trapped-ion whole-chain loss cannot be expressed, so there is no numeric JSON instance.
Its beacon detection can be represented: prepare beacons in |1>, read with plain `MZ`,
and lost beacons read zero (before any configured flip). Independent loss channels do
not model the chain-loss source.

**8. PECOS default**

One before-two-qubit recovery at `1e-3`/`1e-3`, no loss sources, no reload. This is the
only row exempt from disabling default recovery.

```json
{"loss": {"enabled": true}}
```

Ion chain loss (row 7) remains out of scope. The effective-Pauli treatment of
arXiv:2509.18294 is an ordinary Pauli channel, not a loss model.

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
    .with_noiseless_gate("MEASUREZ")
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
