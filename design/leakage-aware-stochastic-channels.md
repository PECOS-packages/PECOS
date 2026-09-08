# RFC: Leakage-Aware Stochastic Channels

Status: draft for discussion.

Scope: leakage-aware channel semantics, compatibility convenience methods in `GeneralNoiseModel`,
Python bindings, and downstream wrappers.

Prototype: [PECOS #518](https://github.com/PECOS-packages/PECOS/pull/518).

Placement architecture: [Event-Driven Noise Modeling](event-driven-noise-model.md).

## Summary

This RFC proposes additive, reusable noise channels for the effective qutrit state space
`{|0>, |1>, |L>}` used by PECOS leakage simulations. It separates two physically different
families:

1. **Population-transition channels**, described by conditional probabilities
   `P(destination | source)` over `0`, `1`, and `L`.
2. **Pauli-plus-leakage channels**, described by an overall application probability and relative
   weights over stochastic Pauli and leakage events.

Both families can be attached in ordered stacks before or after gate and user-defined trigger
events. Gates are automatically observable event anchors; explicit triggers can be inserted into
the command stream without pretending to be physical gates. The proposal is additive: existing
`GeneralNoiseModel` fields, p1/p2 Pauli models, leakage ratios, and gate-replacement labels retain
their current behavior during migration.

Leakage is part of PECOS's backend-independent effective state model. PECOS classically tracks
whether each qubit is in `|L>` and lifts that behavior above the computational-state simulator, so
native simulator support for a third level is not required. Preserving this classically tracked
leakage semantics is the default. Replacing leakage with computational-subspace noise is an
explicit model approximation, not an implicit backend fallback.

The RFC intentionally specifies semantics before implementation. PR #518 is an exploratory
prototype, not the proposed compatibility baseline.

## Motivation

The existing general noise model handles common Pauli, emission, preparation, measurement, idle,
and crosstalk errors well. More detailed leakage studies need reusable channels that can express:

- leakage conditioned on a computational population, such as `0 -> L`;
- recovery such as `L -> 0`, `L -> 1`, or recovery to a configurable mixture;
- a 90% recovery attempt before or after every two-qubit gate;
- leakage on one leg of a two-qubit gate while the other leg is an identity wire;
- correlated pair-state transitions such as `0L -> L1` or `LL -> 00`;
- explicitly placed stochastic Pauli-plus-leakage events; and
- several independently sampled channels at the same event phase.

Trying to approximate recovery by increasing the existing p2 seepage parameter is insufficient.
It does not clearly distinguish incoming leakage from leakage produced at the gate, does not expose
placement, and cannot represent general conditional population transfer.

## Goals

- Give physicists a recognizable conditional-transition representation.
- Preserve the familiar QEC convention of an overall error probability plus relative event
  weights for Pauli-like channels.
- Make identity behavior implicit instead of requiring entries such as `"I": 0.999`.
- Support one- and two-qubit channels with explicit before/after placement.
- Permit multiple channels per event phase with deterministic insertion ordering.
- Preserve an untouched two-qubit leg without measuring or resolving it.
- Keep Rust and Python semantics aligned.
- Allow a fast outer-probability check for the low error rates typical of QEC simulation.
- Preserve classically tracked leakage on every PECOS simulator backend by default.
- Make every requested approximation explicit and report how the declared model was lowered.
- Fail loudly when a DEM builder or wrapper cannot preserve or explicitly lower the requested
  semantics.

## Non-goals

- An arbitrary Kraus-operator or density-matrix channel interface.
- Coherence between the computational and leakage sectors.
- General `n`-qutrit correlated transition matrices in the first version.
- Changing existing p1/p2 Pauli models, crosstalk configuration, seepage, or emission behavior.
- Changing existing gate-replacement label syntax.
- Automatically approximating unsupported transition channels as Pauli noise.
- Defining correlated two-qubit Pauli channels beyond the events already representable by Pauli
  strings and leakage markers.
- Defining the event-dispatch, trigger-transport, or general-noise migration architecture; those
  contracts live in the related event-driven RFC.

## Effective State Model

PECOS represents each qubit as either:

- an active computational qubit in the span of `|0>` and `|1>`, or
- a classically tracked leaked state `|L>`.

This representation is lifted from the underlying simulator. The simulator continues to evolve
the active computational state, while the PECOS execution layer tracks leakage, suppresses or
modifies operations involving leaked qubits, defines leaked-qubit measurement behavior, and
handles recovery or re-preparation. Consequently, every PECOS simulator backend can support this
effective leakage model without natively representing `|L>`.

The proposed transition interface treats this as an effective qutrit space for channel
specification. PECOS does not preserve coherent superpositions between `|L>` and the computational
subspace.

Each proposed channel is a fixed convex mixture of trace-preserving component maps on this
effective space. Including `|L>` makes leakage and recovery trace preserving; the API does not
expose arbitrary Kraus operators even though the components can be described that way
mathematically.

A transition conditioned on computational population can require a hidden Z-basis measurement.
For example, a channel with separate `0` and `1` rows is a population channel and can dephase a
computational superposition. The hidden outcome is consumed by the noise model and is never
returned as a program measurement.

A channel containing only an `L` row does not inspect or measure computational states. PECOS knows
whether a qubit is leaked from its classical leakage tracker.

## Semantic Specification, Lowering, and Approximation

Noise configuration should describe the physical or effective model the user intends to study,
not whichever primitive operations happen to be convenient for one simulator. The architecture
should distinguish:

1. `NoiseParameters`, which contain backend-independent channel parameters and physical intent;
2. `ApproximationPolicies`, which explicitly authorize particular semantic transformations;
3. gate placement and ordered composition in a noise-model specification;
4. a compiled noise model containing the operations executed by PECOS and its selected backend;
   and
5. a compilation report identifying preserved, approximated, and unsupported features.

The default approximation policy is strict in the sense that it authorizes no transformations.
It does **not** reject leakage on a backend without native qutrit support: classically tracked
leakage is already PECOS's standard execution semantics and is preserved by default.

An illustrative Python shape is:

```python
parameters = NoiseParameters(
    # Population-transition and Pauli-plus-leakage channel parameters.
)

policies = ApproximationPolicies.strict()

spec = NoiseModelSpec(
    parameters=parameters,
    approximation_policies=policies,
)

target = NoiseCompilationTarget(
    runtime=runtime.capabilities(),
    simulator=simulator.capabilities(),
)

compiled_noise, report = spec.compile_for(target)
```

The related event-driven RFC defines `NoiseCompilationTarget` and the runtime, simulator, and DEM
capabilities represented by `target`.

An opt-in policy may replace some or all leakage-creation events with a completely depolarizing
channel on the computational subspace:

```python
policies = ApproximationPolicies.strict().replace_leakage_with_completely_depolarizing(
    replacement_probability=1.0,
)
```

The final API name is subject to review, but `replacement_probability` must mean the probability
of replacing a selected leakage-creation event. It must not be ambiguously described as both a
leakage-retention probability and a replacement probability. Intermediate values must remain
numeric rather than being coerced to booleans.

Replacing an `x -> L` creation event is distinct from specifying what happens when an
already-leaked qubit participates in a later gate. For example, skipping that gate and
depolarizing its non-leaked partner is an interaction rule in the classically tracked leakage
model; it is not the same transformation as preventing the original leakage. Policy names,
serialization, and reports must keep these operations separate.

The compilation report should record at least the policy used, the affected channel or parameter,
the scope of the transformation, and its numerical probability. No backend, DEM builder, or
wrapper may silently replace leakage with depolarizing noise.

This separation also provides the intended direction for other noise families. For example, a
coherent idle specification should describe its Hamiltonian or rotation rates in
`NoiseParameters`; Pauli twirling belongs in an explicit approximation policy rather than in a
second physical parameter named after the approximation. The complete coherent-idle API is
outside the scope of this RFC.

## Placement Architecture

The channels in this RFC attach through the event and rule system specified by
[Event-Driven Noise Modeling](event-driven-noise-model.md). That RFC owns typed events, phases,
trigger transport, handler ordering, batch semantics, gate disposition, operation origin, the
general-noise compatibility facade, and Selene differential conformance.

This RFC owns the mathematical and operational semantics of transition and Pauli-plus-leakage
channels. Compatibility methods such as `add_p2_transition_channel_after_gate` compile to event
rules; they do not create a separate hook execution path.

## Common Outer-Probability Convention

Every proposed channel has an outer application probability `p`, with `0 <= p <= 1`. For a
component channel `C`, the effective channel is

```text
C_effective = (1 - p) Identity + p C.
```

PECOS should sample this outer coin before doing state resolution or more expensive conditional
sampling. When the coin fails, the channel is exact identity and must not perform hidden
measurements.

The meaning of the inner configuration differs between the two channel families and must not be
conflated:

- Transition rows are independently normalized conditional distributions `P(y | x)`.
- Pauli-plus-leakage event weights form one normalized distribution conditioned on the outer coin
  succeeding.

## Population-Transition Channels

### One-qubit definition

A `TransitionChannel` contains:

```text
TransitionChannel(
    probability: float,
    transitions: dict[str, dict[str, float]],
)
```

The nested dictionary means

```text
transitions[source][destination] = P(destination | source, outer coin succeeded).
```

For example:

```python
population_transfer = TransitionChannel(
    probability=0.01,
    transitions={
        "0": {"0": 0.05, "1": 0.05, "L": 0.90},
        "1": {"0": 0.10, "1": 0.80, "L": 0.10},
        "L": {"0": 0.45, "1": 0.45, "L": 0.10},
    },
)
```

Each supplied source row must be nonempty, nonnegative, finite, and sum to one within numerical
tolerance. A destination omitted from a supplied row has probability zero. A source row omitted
from the dictionary is exact identity for that source.

The row weights are probabilities, not arbitrary relative weights. Normalizing each row in the
implementation may be convenient, but accepting substantially unnormalized rows would hide user
errors and is not proposed.

### Recovery helper

The common leakage-recovery case should have a named constructor:

```python
recovery = TransitionChannel.leak_recovery(
    probability=0.90,
    p_zero=0.50,
)
```

This means:

```text
with probability 0.90:
    L -> 0 with probability 0.50
    L -> 1 with probability 0.50
otherwise:
    identity
```

It therefore recovers 90% of leaked inputs at each configured site. The equivalent representation
with one always-selected channel is:

```python
TransitionChannel(
    probability=1.0,
    transitions={"L": {"0": 0.45, "1": 0.45, "L": 0.10}},
)
```

By contrast, putting the latter row inside a channel with `probability=0.01` gives only
`0.01 * 0.90 = 0.009` total recovery probability per configured site.

To model a two-qubit gate that independently recovers 90% of the leakage present on either leg
after ordinary gate noise, attach the one-qubit helper directly to the p2 after-gate hook:

```python
noise = GeneralNoiseModelBuilder().add_p2_transition_channel_after_gate(recovery)
```

After-gate placement includes leakage produced by that gate's ordinary noise. Before-gate
placement sees only leakage already present when the gate site begins.

### Component-map interpretation

The transition matrix denotes a fixed stochastic, trace-preserving population map on the
effective qutrit space. It is not a stochastic unitary channel:

- `0 -> 1` and `1 -> 0` are population-conditioned transitions;
- `L -> 0` and `L -> 1` recover a leaked qubit;
- `0 -> L` and `1 -> L` are state-selective leakage;
- omitted rows are identity without state resolution.

This distinction is why Pauli errors are specified by a separate channel family.

## Two-Qubit Transition Channels

`TwoQubitTransitionChannel` represents a channel attached to a two-qubit site. It has two forms:

1. a concrete joint conditional matrix with one shared outer coin; or
2. a product of one-qubit transition channels, retaining their independent outer coins.

No public `P2TransitionStep` type is proposed. An implementation may use an internal enum, but the
public abstraction remains a channel.

### Concrete joint matrix

A concrete joint map uses all pair states in `{0, 1, L}^2`:

```python
joint = TwoQubitTransitionChannel(
    probability=0.02,
    transitions={
        "0L": {"L1": 0.20, "0L": 0.80},
        "LL": {"00": 0.50, "11": 0.50},
    },
)
```

Every supplied pair-state row is independently normalized. Missing pair-state rows are identity.
The one outer coin is shared by the entire joint map.

Concrete rows can require resolving either computational input. For example, selecting between a
`0L` row and a `1L` row requires a Z-basis population measurement of the first leg.

### Identity-wire form

When a two-qubit channel acts on only one leg, `*` denotes an identity wire:

```python
recover_either_leg = TwoQubitTransitionChannel(
    probability=1.0,
    transitions={
        "*L": {"*0": 0.45, "*1": 0.45, "*L": 0.10},
        "L*": {"0*": 0.45, "1*": 0.45, "L*": 0.10},
    },
)
```

The `*` contract is stronger than ordinary wildcard matching:

- In a source, it accepts any state on that leg without resolving the computational state.
- In a destination, it carries the same subsystem through with the identity operation.
- It must occur in the same position in the source and every destination of that row.
- The identity leg must not acquire a hidden measurement dependency.
- It does not promise that an entangled partner is unaffected by a physical measurement performed
  on the acted-on leg.

Thus `"*L" -> "*0"` examines only the second leg's classical leakage flag, recovers the second leg
to zero, and performs no operation or measurement on the first leg.

Rules for `*x` and `x*` may coexist in the same channel. If both match, they act on their respective
legs. For an `LL` input in the example above, both recovery rows are sampled independently after
the channel's one shared outer coin succeeds.

For the first version, a dictionary is either entirely concrete or entirely in identity-wire
form. Mixing concrete pair rows with identity-wire rows in one dictionary is rejected because
overlap and precedence would otherwise be ambiguous. Users can attach a second ordered channel
when both behaviors are required.

`**` is not proposed. Each identity-wire row must act on exactly one leg.

### Product of one-qubit channels

Two single-qubit channels can form an independent-leg two-qubit channel:

```python
first = TransitionChannel.leak_recovery(0.90, p_zero=0.75)
second = TransitionChannel.leak_recovery(0.80, p_zero=0.25)
independent_pair = first * second
```

The product has two independent outer coins, one for each leg. This differs from constructing a
joint conditional dictionary and then supplying one outer probability.

Named forms should also be available for clarity:

```python
independent_pair = TwoQubitTransitionChannel.independent(first, second)
same_on_both = TwoQubitTransitionChannel.same_on_each(recovery)
```

### Transition-dictionary algebra

`TransitionDict` validates transition matrices independently from channel application
probabilities. It supports:

- `left * right` or `left.tensor(right)` for a Kronecker product;
- `after @ before` or `after.compose(before)` for sequential matrix composition; and
- `first.then(second)` as a readable application-order alias.

Multiplying dictionaries and multiplying channels intentionally have different outer-coin
semantics:

- `TransitionDict * TransitionDict`, wrapped in `TwoQubitTransitionChannel(p, result)`, has one
  shared outer coin `p`.
- `TransitionChannel(p1, ...) * TransitionChannel(p2, ...)` retains two independent outer coins.

The identity-wire form is a compact representation of a factorized dictionary and must retain
identity metadata through compilation. It must not be naively expanded in a way that introduces
computational-state measurements on identity legs.

## Pauli-Plus-Leakage Channels

Population transitions and stochastic Pauli operations should not share one event dictionary.
Paulis implement `rho -> P rho P` in the computational subspace and do not require the
population-measurement semantics of conditional transition rows.

### One-qubit definition

```python
faults = PauliLeakageChannel(
    probability=0.001,
    events={
        "X": 0.40,
        "Y": 0.20,
        "Z": 0.30,
        "L": 0.10,
    },
)
```

Here the event values are nonnegative relative weights. They need not sum to one and are
normalized as a single distribution conditioned on the outer coin succeeding:

```text
P(event i) = p * weight_i / sum(weights).
```

`L` means `any -> L`. A Pauli selected on an already leaked qubit is a no-op under the existing
classical-leakage model.

Identity is represented by failure of the outer coin. A one-qubit `I` event and an all-identity
multi-qubit event are rejected, avoiding confusing configurations such as `{"I": 0.999, ...}`.

### Two-qubit definition

A joint two-qubit event distribution uses strings over `{I, X, Y, Z, L}`:

```python
joint_faults = TwoQubitPauliLeakageChannel(
    probability=0.002,
    events={
        "IX": 0.30,
        "XL": 0.20,
        "LL": 0.05,
        "ZZ": 0.45,
    },
)
```

As with transition channels, multiplying two one-qubit channels retains independent outer coins,
while constructing a joint two-qubit channel uses one shared outer coin.

`PauliLeakageDict` may provide validated mapping and tensor-product ergonomics parallel to
`TransitionDict`, while retaining its own relative-weight normalization rules.

The final names should parallel the transition family. No public "step" type is needed.

## Compatibility Gate Methods and Ordered Composition

For compatibility and discoverability, the general noise builder initially exposes four ordered
views of event rules:

- before single-qubit gate sites;
- after single-qubit gate sites;
- before two-qubit gate sites; and
- after two-qubit gate sites.

Each view contains typed channel variants in exact insertion order. Transition and
Pauli-plus-leakage channels must share the same underlying event-phase sequence so users can
control cross-family ordering. Separate storage that always runs one family before another is not
proposed.

Typed builder methods append to the common stack:

```python
noise = (
    GeneralNoiseModelBuilder()
    .add_p2_pauli_leakage_channel_before_gate(leak_fault)
    .add_p2_transition_channel_before_gate(recovery)
    .add_p2_transition_channel_after_gate(joint_recovery)
)
```

In this example the before-gate leakage channel runs first and the recovery channel observes its
result. Every channel has its own outer sampling decision unless its definition explicitly shares
one coin across legs.

Bulk replacement methods may accept lists, but they must preserve list order and replace the whole
heterogeneous event-phase sequence rather than creating family-specific ordering domains. A
generic advanced form can be added if needed:

```python
builder.with_p2_channels_before_gate([leak_fault, recovery])
```

Typed `add_*` methods remain useful for discoverability and static typing.

These methods compile to event rules selecting all gates of the corresponding arity. More specific
gate selectors and generic triggers use the event-driven rule API. The convenience methods do not
create a second placement implementation.

## Placement Semantics

The related event-driven RFC defines the general-noise compatibility ordering, including composite
p1/p2 sampling, emission replacement, leaked-input suppression, and after-two-qubit idle effects.
For the channels defined here, that lifecycle has these consequences:

- Before-gate recovery can allow an incoming leaked qubit to participate in the ideal gate.
- Before-gate leakage can suppress the ideal gate under the existing leaked-input policy.
- After-gate recovery can act on incoming leakage or leakage produced by ordinary gate noise.
- After-gate recovery cannot retroactively execute a gate that was already suppressed.
- A later channel observes state changes made by every earlier channel at the same event phase.

## Proposed Python API

```python
from pecos import (
    GeneralNoiseModelBuilder,
    PauliLeakageChannel,
    PauliLeakageDict,
    TransitionChannel,
    TransitionDict,
    TwoQubitPauliLeakageChannel,
    TwoQubitTransitionChannel,
)

recovery = TransitionChannel.leak_recovery(0.90, p_zero=0.50)

joint_recovery = TwoQubitTransitionChannel(
    probability=1.0,
    transitions={
        "*L": {"*0": 0.45, "*1": 0.45, "*L": 0.10},
        "L*": {"0*": 0.45, "1*": 0.45, "L*": 0.10},
    },
)

noise = (
    GeneralNoiseModelBuilder()
    # A one-qubit channel supplied at a p2 hook is sampled independently on each leg.
    .add_p2_transition_channel_before_gate(recovery)
    # A two-qubit channel is applied according to its joint or independent definition.
    .add_p2_transition_channel_after_gate(joint_recovery)
)
```

Proposed signatures:

```python
class TransitionChannel:
    def __init__(
        self,
        probability: float,
        transitions: TransitionDict | dict[str, dict[str, float]],
    ) -> None: ...

    @staticmethod
    def leak_recovery(
        probability: float,
        p_zero: float = 0.5,
    ) -> TransitionChannel: ...

    def __mul__(
        self,
        other: TransitionChannel,
    ) -> TwoQubitTransitionChannel: ...


class TwoQubitTransitionChannel:
    def __init__(
        self,
        probability: float,
        transitions: TransitionDict | dict[str, dict[str, float]],
    ) -> None: ...

    @staticmethod
    def independent(
        first: TransitionChannel,
        second: TransitionChannel,
    ) -> TwoQubitTransitionChannel: ...

    @staticmethod
    def same_on_each(
        channel: TransitionChannel,
    ) -> TwoQubitTransitionChannel: ...


class GeneralNoiseModelBuilder:
    def add_p1_transition_channel_before_gate(
        self,
        channel: TransitionChannel,
    ) -> GeneralNoiseModelBuilder: ...

    def add_p1_transition_channel_after_gate(
        self,
        channel: TransitionChannel,
    ) -> GeneralNoiseModelBuilder: ...

    def add_p2_transition_channel_before_gate(
        self,
        channel: TransitionChannel | TwoQubitTransitionChannel,
    ) -> GeneralNoiseModelBuilder: ...

    def add_p2_transition_channel_after_gate(
        self,
        channel: TransitionChannel | TwoQubitTransitionChannel,
    ) -> GeneralNoiseModelBuilder: ...
```

The Pauli-plus-leakage family follows the same arity and product conventions.

## Proposed Rust API Shape

Rust should expose the same concepts without exposing Python-driven names or an implementation-only
stack entry type:

```rust
pub struct TransitionDict { /* validated representation */ }

pub struct TransitionChannel { /* one-qutrit channel */ }

pub enum TwoQubitTransitionChannel {
    Independent {
        first: TransitionChannel,
        second: TransitionChannel,
    },
    Joint {
        probability: f64,
        transitions: TransitionDict,
    },
}

pub enum OneQubitHookChannel {
    Transition(TransitionChannel),
    PauliLeakage(PauliLeakageChannel),
}

pub enum TwoQubitHookChannel {
    Transition(TwoQubitTransitionChannel),
    PauliLeakage(TwoQubitPauliLeakageChannel),
}

```

Exact enum names are open to normal Rust API review. The required property is that the public API
speaks in terms of channels. The event-driven RFC defines the heterogeneous executable action and
rule representation. Arity should be validated before execution rather than discovered through a
failed cast in the hot path.

`From`, `Into`, and `Mul` implementations may provide concise construction without making a
public `Step` wrapper necessary.

## Validation Rules

Construction should fail immediately when:

- an outer probability is nonfinite or outside `[0, 1]`;
- a transition dictionary is empty;
- a label has the wrong arity or contains an unsupported symbol;
- a supplied conditional row is empty, negative, nonfinite, or not normalized;
- a Pauli-plus-leakage event map is empty or has no positive total weight;
- a one-qubit identity event or all-identity multi-qubit event is supplied;
- an identity-wire row does not contain exactly one `*`;
- the destination moves or removes the `*` identity position;
- concrete and identity-wire rows are mixed in one two-qubit transition dictionary; or
- an operation combines channels with incompatible arity.

Error messages should identify the channel, source row, destination label, and violated rule.
Python construction errors should be ordinary `ValueError` or `TypeError`, not uncaught Rust panic
exceptions.

## Scaling

Scaling must preserve conditional normalization. Therefore scale factors can modify outer channel
probabilities but must not multiply individual transition-row probabilities or relative event
weights.

Proposed default:

```text
p_effective = clamp(p * global_scale * site_scale, 0, 1)
```

where `site_scale` is the applicable p1 or p2 scale. Conditional rows and relative event weights
remain unchanged.

Whether the existing `leakage_scale` should additionally modify explicit `L` events is an open
question. This RFC recommends **no implicit leakage scaling for explicitly configured channels**:
their dictionaries already specify the intended leakage fraction. A separately named opt-in scale
could be introduced if experiments need to scan only explicit leakage branches.

## Leakage Replacement and Legacy Configuration

PECOS's default is to preserve leakage events using its classical leakage tracker. A
leakage-to-depolarizing setting changes the declared channel: a selected transition into `L` is
instead replaced by a completely depolarizing channel and no leaked state remains for a later
`L -> x` recovery channel to observe.

Long term, this transformation should be represented by an explicit approximation policy with a
clearly named replacement probability and an entry in the compilation report. During migration,
legacy settings such as `leakage_scale` or `leak2depolar` may remain supported, but their adapters
must translate their documented numeric convention into that policy. In particular, an API using
a leakage-retention probability must convert it to a replacement probability rather than merely
renaming the field.

Python and Selene wrappers must preserve intermediate numeric values. They must not narrow a
probability to a boolean where only the two endpoints remain representable. Serialization should
use one unambiguous convention, even if compatibility constructors accept legacy aliases.

An explicit transition channel containing `x -> L` is subject to the configured replacement
policy at its own ordered location. If the event is replaced, later channels see a computational
qubit; if it is preserved, later channels see `L`. This ordering must be visible and tested.

## Execution and DEM Requirements

The PECOS execution layer consuming these channels must:

- provide classically tracked leakage independently of the selected computational-state
  simulator;
- preserve ordered channel application;
- make the outer-probability no-op path measurement-free;
- avoid resolving omitted transition rows;
- avoid resolving an identity-wire leg;
- apply population measurements only to legs whose source rows distinguish `0` from `1`;
- preserve deterministic sampling for a fixed seed and configuration; and
- report unsupported operations rather than silently dropping them.

The underlying computational-state simulator is not required to represent `|L>` or expose native
qutrit operations. The execution layer must prevent an active-state operation from being applied
where the classical leakage state says it should be suppressed or replaced.

A DEM builder must not silently replace arbitrary transition channels with depolarizing noise.
Exact support may require hidden-measurement branch replay similar to measurement crosstalk. Until
supported, the DEM builder should identify the unsupported channel and fail loudly or report it in
an explicit coverage result. An explicitly requested leakage-replacement approximation may be
used during DEM lowering, but it must be included in that result.

## Serialization and Wrapper Requirements

The plain nested dictionary is the canonical human-facing transition representation. Rust may
compile it into a denser or factorized form, but `to_dict()` should preserve the user's validated
definition, including identity-wire labels.

Wrappers should expose:

- the same outer probability as the core channel;
- the same conditional-row validation;
- ordered before/after event placement;
- both one- and two-qubit channel types;
- numeric rather than boolean-only leakage-conversion settings; and
- clear rejection for capabilities the wrapper cannot forward.

The Selene PECOS wrapper should forward channel objects or a stable serialized schema rather than
reinterpreting transition probabilities as legacy seepage fields. Generic event and trigger
transport requirements live in the related event-driven RFC.

## Relationship to Crosstalk

PECOS crosstalk models already use labels such as `0 -> L`, `1 -> L`, and population flips. Their
conditional semantics should eventually share the same validated transition representation and
component-map definitions proposed here. Reusing one parser and sampler vocabulary would prevent
the gate-channel and crosstalk paths from assigning different meanings to the same label.

This RFC does not replace crosstalk placement or victim selection. Crosstalk remains responsible
for deciding which non-gate qubits are affected; a reusable transition channel describes what
happens after a victim is selected.

## Performance Expectations

- Sample the outer coin first.
- Precompile labels and conditional samplers when the noise model is built.
- Do not allocate transition dictionaries in the per-gate hot path.
- Preserve sparse omitted-row identity behavior.
- Compile identity-wire rules into factorized leg-local samplers or equivalent metadata; do not
  infer the identity leg's computational state.
- Keep deterministic iteration and sampling order independent of Python dictionary hashing.

## Backward Compatibility

When no new channels are configured, generated noise operations and RNG consumption must remain
unchanged for a fixed seed.

This RFC does not reserve `*` globally. Its identity-wire meaning is scoped to transition-state
labels. Existing uses of `*` in other configuration namespaces, including any gate-replacement
labels, remain unchanged.

The new types and builder methods are additive. Existing p1/p2 Pauli models remain the preferred
interface for ordinary gate-local Pauli noise.

Compatibility convenience methods such as `add_p2_transition_channel_after_gate` compile into
event rules. Their public behavior does not depend on exposing event-dispatch internals.

## Alternatives Considered

### Increase p2 seepage

Rejected as a general solution. It cannot express placement, input-conditioned output mixtures,
or recovery of leakage generated at a particular point in gate processing.

### One dictionary containing transitions and Paulis

Rejected for the initial API. Conditional population rows and stochastic unitary events have
different normalization and coherence semantics. Combining them makes both harder to explain and
validate.

### One generic `ChannelDict` for every channel family

Rejected for the initial API. Typed `TransitionDict` and Pauli event dictionaries have different
normalization rules and support different algebra. They can share low-level mapping ergonomics,
but a common untyped container would permit configurations whose meaning depends on which
constructor consumes them.

### Arbitrary Kraus operators

Deferred. They are mathematically general but do not map naturally onto PECOS's classically lifted
leakage representation or every DEM builder. The proposed channels cover the immediate QEC use
cases with explicit operational semantics.

### Expand `*L` into `0L`, `1L`, and `LL`

Rejected as a semantic implementation strategy. Selecting among those concrete rows can require
resolving the first qubit and would violate the identity-wire contract.

### Public two-qubit "step" objects

Rejected. Independent-leg and joint maps are both channels. A step is an internal representation
of one ordered stack entry, not a concept users should need to construct.

### Fixed ordering between channel families

Rejected. Separate family-specific vectors create surprising behavior when users call append
methods in a different order. All explicit channels at one event phase should share insertion
ordering.

## Minimum Test Matrix

### Validation

- Valid and invalid outer probabilities.
- Normalized, unnormalized, empty, negative, and nonfinite rows.
- Missing rows remain identity.
- Invalid labels and arity mismatches.
- Identity-wire position mismatch and `**` rejection.
- Concrete/wildcard mixing rejection.

### Semantics

- `L -> 0`, `L -> 1`, `0 -> L`, `1 -> L`, and population flips.
- A leakage-only row does not measure a computational input.
- A computationally conditioned row performs only the required hidden measurement.
- `*L` never measures or resolves the first leg.
- `L*` never measures or resolves the second leg.
- `*L` and `L*` both apply to `LL` with the specified shared-coin semantics.
- Concrete joint pair transitions cover all nine source states.
- Independent products retain separate outer coins.

### Placement and composition

- Before recovery can enable ideal-gate execution.
- Before leakage can suppress ideal-gate execution.
- After recovery observes leakage produced by ordinary gate noise.
- Multiple channels run in insertion order across channel families.
- Existing after-two-qubit idle noise remains last.

### Compatibility

- Empty channel stacks preserve existing bytes and RNG streams.
- Rust and Python configurations produce identical seeded behavior.
- Every simulator backend preserves the same classically tracked leakage semantics.
- Strict/default policy preserves leakage rather than rejecting or replacing it.
- Explicit leakage replacement uses the requested numeric probability and is reported.
- Leakage-creation replacement and leaked-partner interaction remain distinguishable.
- Serialization round-trips identity-wire definitions.
- Unsupported DEM and wrapper paths fail loudly.

## Implementation Sequence After RFC Approval

1. Add validated channel value types and pure sampler tests.
2. Define the semantic specification, approximation-policy, and compilation-report boundaries.
3. Integrate the channel families with the executable action and rule interfaces accepted by the
   event-driven RFC.
4. Implement transition execution and measurement-dependency tracking.
5. Add Python bindings, type stubs, and compatibility convenience methods.
6. Add Selene wrapper forwarding without narrowing numeric leakage replacement probabilities.
7. Add user documentation, end-to-end seeded tests, and channel performance benchmarks.
8. Add explicit DEM coverage and approximation reporting before attempting branch replay.

The prototype in PR #518 may supply test cases and implementation ideas, but code should be
reworked against the accepted public model rather than merged by default.

## Open Questions

1. Should explicit `L` events ignore `leakage_scale`, as recommended here, or have a separately
   named scan multiplier?
2. What public names and serialized form should replace the ambiguous legacy
   `leakage_scale`/`leak2depolar` conventions?
3. Should bulk `with_*_channels_*` methods accept a heterogeneous list, or should the first release
   provide only ordered `add_*` methods?
4. Should identity-wire syntax remain limited to two-qubit channels in the first release?
5. What numerical tolerance should transition-row normalization use across Rust and Python?
6. Which transition subset can the existing DEM machinery support exactly in its first release?

## Decision Checklist

Before implementation resumes, reviewers should explicitly agree on:

- [ ] the two-family split;
- [ ] outer probability and inner normalization semantics;
- [ ] hidden-measurement behavior for computational rows;
- [ ] the `*` identity-wire contract;
- [ ] overlap semantics for `*x` and `x*`;
- [ ] no public step abstraction;
- [ ] one ordered heterogeneous sequence per selected event phase;
- [ ] classically tracked leakage as the backend-independent default;
- [ ] explicit and reported leakage-replacement policy semantics;
- [ ] scaling and legacy `leakage_scale`/`leak2depolar` migration;
- [ ] additive backward compatibility; and
- [ ] initial simulator, runtime-wrapper, and DEM support boundaries.
