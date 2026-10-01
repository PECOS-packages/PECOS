# RFC: Event-Driven Noise Modeling

Status: draft for discussion.

Scope: event-driven noise placement in `pecos-engines`, compatibility with
`GeneralNoiseModel`, Python bindings, PECOS and Selene runtime transport, and
differential conformance.

Related channel specification:
[Leakage-Aware Stochastic Channels](leakage-aware-stochastic-channels.md).

Implementation experience: the experimental `pecos-neo` `ComposableNoiseModel`,
`NoiseEvent`, typed signals, and `GeneralNoiseModelBuilder`.

Conformance foundation: [PECOS #545](https://github.com/PECOS-packages/PECOS/pull/545).

## Summary

This RFC proposes an event-driven execution layer for PECOS noise models. Ideal
gates and other runtime operations become typed event anchors. Users may also
insert generic, serializable trigger anchors into the command stream. Ordered
noise rules attach channels or gate actions to supported phases of selected
events.

The event-driven model is a new implementation boundary, not an immediate
replacement for the production `GeneralNoiseModel`. A thin compatibility facade
will compile existing general-noise parameters into event rules and delegate all
execution to the compiled event-driven model. The implementation becomes the
production default only after exact seeded compatibility, semantic conformance,
performance, and runtime-integration requirements pass a separate review.

## Motivation

The existing general noise model provides a useful collection of physical and
effective error parameters, but placement and execution are encoded in one
specialized implementation. Adding fixed before/after vectors for every channel
family, arity, gate family, and runtime marker would create separate ordering and
composition rules for each new feature.

An event-driven model separates:

- what a channel does;
- which runtime occurrence selects it;
- when it runs relative to that occurrence;
- whether an ideal gate executes, is suppressed, or is replaced; and
- how a semantic noise specification is lowered for a runtime and simulator.

This supports the leakage-aware channels in the related RFC, a compatibility
implementation of general noise, and later extensions such as transport markers,
calibration boundaries, measurement groups, and explicitly timed operations.

## Goals

- Treat program and runtime gates as typed event anchors.
- Support generic serializable triggers without representing them as fake gates.
- Provide typed preparation, measurement, reset, idle, batch, and circuit events
  required by general-noise semantics.
- Attach heterogeneous ordered noise actions before, at, or after supported
  events.
- Resolve execute, suppress, and replace gate behavior explicitly.
- Prevent accidental recursive noise on generated operations.
- Give overlapping selectors a deterministic total order.
- Preserve simultaneous runtime-batch information needed by crosstalk and idle
  handling.
- Compile human-readable event names to efficient runtime identifiers.
- Reimplement general noise through a thin compatibility facade rather than
  maintaining two long-term definitions of its semantics.
- Validate compatibility using independent oracles and legacy/event-driven
  Selene plugins.

## Non-goals

- Switching existing users to the new implementation when this RFC is accepted.
- Encoding arbitrary simulator-native Kraus or density-matrix operations.
- Treating transport/runtime source as the generation origin of an operation.
- Carrying arbitrary type-erased Rust values across Python, process, or Selene
  plugin boundaries.
- Making every custom Selene operation a PECOS trigger.
- Defining recursive noise on generated gates in the first implementation.

## Architecture

The design separates five layers:

1. `NoiseParameters` describe physical or effective noise intent.
2. `NoiseRule` binds a channel or event action to a selector and phase.
3. `ApproximationPolicies` explicitly authorize semantic transformations.
4. `CompiledEventNoiseModel` contains indexed executable handlers.
5. `NoiseCompilationReport` records preserved, approximated, and unsupported
   behavior.

`GeneralNoiseParameters` are a compatibility-oriented specialization of
`NoiseParameters`. The general-noise facade compiles them using a fixed
compatibility profile.

Compilation targets execution capabilities, not only the quantum-state
simulator:

```python
target = NoiseCompilationTarget(
    runtime=runtime.capabilities(),
    simulator=simulator.capabilities(),
    dem=None,
)

compiled_noise, report = specification.compile_for(target)
```

PECOS classically tracks leakage above the computational-state simulator, so a
simulator without native qutrit support does not require a leakage approximation.
Runtime trigger transport, injected-gate support, measurement dependencies, and
DEM lowering are separate capabilities reported by the target.

## Typed Event Algebra

Built-in event semantics must be represented with typed payloads. Opaque bytes
are reserved for versioned generic-trigger metadata and must not carry ordinary
gate angles, measurement outcomes, idle durations, or timing.

The conceptual event envelope is:

```rust
pub struct EventEnvelope<'a> {
    pub key: EventKey,
    pub phase: EventPhase,
    pub targets: &'a [QubitId],
    pub timing: Option<EventTiming>,
    pub origin: OperationOrigin,
    pub payload: EventPayload<'a>,
}

pub enum EventKey {
    Gate(GateId),
    Preparation,
    Measurement,
    Reset,
    Idle,
    Batch,
    Circuit,
    Trigger(CompiledTriggerId),
}

pub enum EventPhase {
    Before,
    At,
    After,
}

pub enum EventPayload<'a> {
    Gate {
        gate_type: GateType,
        angles: &'a [Angle64],
    },
    Preparation,
    Measurement {
        outcomes: Option<&'a [MeasurementOutcome]>,
    },
    Reset,
    Idle {
        duration: TimeUnits,
    },
    Batch {
        operations: &'a [OperationDescriptor<'a>],
    },
    Circuit {
        num_qubits: usize,
    },
    Trigger {
        stable_id: &'a TriggerId,
        metadata_version: u32,
        metadata: &'a [u8],
    },
}

pub enum OperationOrigin {
    Program,
    Noise,
    Replacement,
}
```

The exact Rust representation may use specialized borrowed event types rather
than one large enum. The semantic requirements are:

- phase is explicit;
- built-in payloads are typed;
- target qubits and timing are available without parsing metadata;
- measurement outcomes exist only when the selected phase makes them available;
- idle duration remains a typed time quantity; and
- generation origin is distinct from the runtime transport that delivered a
  program operation.

For example, a gate originating in a Guppy program and delivered through Selene
has `OperationOrigin::Program`. Selene is transport context, not a competing
origin value.

Each event kind declares its valid phases and payload invariants. The initial
matrix is:

| Event | Supported phases | Required information |
| --- | --- | --- |
| Gate | before, after | ID, type, targets, angles, timing |
| Preparation | after | prepared targets |
| Measurement | before, after | targets; outcomes after only |
| Reset | after | reset targets |
| Idle | at | targets and duration |
| Batch | before, after | ordered operation descriptors and timing |
| Circuit | before, after | qubit count |
| Trigger | before, after | stable ID, targets, versioned metadata |

Construction or compilation rejects a rule selecting an unsupported phase.

## Stable and Compiled Identifiers

`GateId` and `TriggerId` occupy separate namespaces. A trigger cannot collide
with a physical gate even when their human-readable names are equal.

Python may construct a trigger from a namespaced string:

```python
trigger = TriggerId("helios.transport_complete")
```

The stable string or an equivalent stable UUID is serialized. A compact
`CompiledTriggerId`, such as a process-local integer, is created while compiling
the noise model and is never used as the persistent identity of the trigger.

Compiled dispatch tables are keyed by event kind, compiled identifier, phase,
and arity. Event dispatch must not scan every configured channel or repeatedly
compare strings.

## Rules, Actions, and Total Ordering

Public construction APIs remain typed, while compiled rules use one heterogeneous
action representation:

```rust
pub struct NoiseRule {
    pub selector: EventSelector,
    pub phase: EventPhase,
    pub action: NoiseAction,
    registration_index: u64,
}

pub enum NoiseAction {
    OneQubitChannel(OneQubitNoiseChannel),
    TwoQubitChannel(TwoQubitNoiseChannel),
    GateAction(GateAction),
    PreparationAction(PreparationAction),
    MeasurementAction(MeasurementAction),
    ResetAction(ResetAction),
    IdleAction(IdleAction),
    BatchAction(BatchAction),
    CircuitAction(CircuitAction),
}
```

Trait-erased executable actions are also acceptable internally. A generic
`NoiseRule<C>` alone is insufficient because it cannot form the required
heterogeneous ordered sequence.

Every rule receives a monotonically increasing `registration_index` when added
to the specification. For one event phase, all matching broad and specific
selectors are merged and executed in ascending registration order. Selector
specificity does not implicitly change precedence.

For example, if a two-qubit catch-all rule is registered before an RZZ-specific
rule, it executes first for RZZ even though the second selector is more specific.
Compilation may pre-merge common selector combinations, but it must preserve the
same total order.

Compatibility-generated rules receive explicit registration positions from the
compatibility profile. User-added compatibility methods append at the documented
locations. Iteration order must not depend on Python dictionary hashing.

Public gate selectors match `OperationOrigin::Program` by default. The first
implementation does not expose selectors that recursively apply ordinary gate
noise to `Noise` or `Replacement` origins.

Illustrative Python construction is:

```python
noise = EventDrivenNoiseModelBuilder()

noise.on_gate("RZZ").before(incoming_recovery).after(outgoing_leakage)
noise.on_trigger("helios.transport_complete").after(transport_recovery)
```

A declarative rule-list API is equally acceptable if it makes ownership and
ordering clearer.

## Gate Lifecycle and Disposition

The gate lifecycle is:

1. Dispatch matching before-gate rules in total registration order.
2. Resolve the gate disposition and any scheduled effects.
3. Execute the original gate, suppress it, or execute its replacement body.
4. Dispatch matching after-gate rules in total registration order.

The conceptual disposition is:

```rust
pub enum GateDisposition {
    Execute,
    Suppress,
    Replace(ReplacementBody),
}
```

Multiple suppression requests are idempotent. The initial model rejects at
compile time any configuration in which more than one replacement-producing
rule can match the same gate event. It also rejects overlap between a possible
replacement and a separately configured suppression rule unless one composite
action defines their precedence explicitly. This conservative rule can be
relaxed later with a reviewed composition algebra.

Noise- and replacement-originated gates do not recursively emit program-gate
noise events. Recursive processing is outside the first implementation. This
provenance rule prevents emission replacements and injected Paulis from
accidentally receiving another copy of gate noise.

After-gate rules run after the resolved body, including when the original body
was suppressed or replaced. They observe state changes from before rules and the
resolved body but cannot retroactively execute a suppressed ideal gate.

## Generic Trigger Lifecycle

A generic trigger has no ideal quantum body:

```text
before-trigger rules -> inert trigger anchor -> after-trigger rules
```

Triggers are neither suppressible nor replaceable. They may target zero, one,
two, or more qubits, but a channel attached to a trigger must accept that arity.
Arity compatibility is checked during compilation.

The trigger is consumed by the event layer and is never forwarded to the
computational-state simulator.

## Runtime Batches and Simultaneous Operations

Runtime adapters must preserve batch boundaries rather than immediately flattening
every batch into an unmarked sequential gate stream. The normalized lifecycle is:

1. Dispatch the before-batch event with the complete ordered operation descriptor
   list and timing.
2. Build per-operation anchors in stable source order.
3. Apply per-operation before-phase effects and resolve dispositions in stable
   source order using the compatibility profile's documented ordering.
4. Submit the resolved ideal/replacement operation batch to the simulator.
5. Dispatch per-operation after rules in stable source order.
6. Dispatch the after-batch event.

Batch-level channels handle behavior that depends on the full simultaneous set,
including measurement crosstalk victim selection. Per-operation ordering is a
deterministic sampling convention; it does not assert that disjoint ideal gates
occur at different physical times.

The first implementation rejects overlapping ideal operations on the same qubit
within one simultaneous batch unless the runtime defines a supported meaning.
Compatibility lowering must reproduce the production model's crosstalk victim
ordering, idle insertion, and RNG consumption for Selene batches.

An explicit generic trigger occupies its own ordered command position or runtime
batch/barrier in the first cross-runtime format. A trigger mixed into a batch of
simultaneous gates would otherwise have no unambiguous before/after relationship
with its peers.

## General-Noise Compatibility Facade

The long-term public relationship is:

```text
GeneralNoiseParameters
          |
          v
GeneralNoiseCompatibilityCompiler
          |
          v
CompiledEventNoiseModel
```

The compatibility-facing model is deliberately thin:

```rust
pub struct GeneralNoiseModel {
    parameters: GeneralNoiseParameters,
    compiled: CompiledEventNoiseModel,
    report: NoiseCompilationReport,
}
```

It contains no independent sampling, gate-processing, leakage, or crosstalk
logic. It validates and retains the original parameters, invokes the
compatibility compiler, delegates seeding, reset, message handling, and execution
to the compiled model, and exposes the compilation report and parameters for
inspection.

During migration, explicit legacy and event-driven build paths remain available:

```text
parameters.build_legacy()
parameters.build_event_driven()
```

The event-driven path compiles the same defaults, validation, scaling, weighted
samplers, composite outer coins, leakage behavior, gate replacement, crosstalk,
idle placement, and noiseless-gate controls. The compatibility compiler must not
split one legacy composite sampling decision into independently sampled event
rules.

Exact seeded output and RNG-stream parity are requirements for the compatibility
facade. A difference requires an explicitly reviewed compatibility change; it is
not waived merely because sampled distributions are close. Independently
constructed event-driven models promise semantic/distributional correctness but
do not promise the legacy model's random-number consumption order.

The production `GeneralNoiseModel` remains the default until a separate review
confirms:

- exact compatibility tests pass;
- independent semantic tests pass;
- performance and allocation targets pass;
- runtime and wrapper support is ready;
- downcast/introspection compatibility is addressed; and
- release and deprecation plans are documented.

## Compatibility Gate Convenience Methods

Arity-specific methods remain useful for discovery:

```python
noise = (
    GeneralNoiseModelBuilder()
    .add_p2_transition_channel_before_gate(recovery)
    .add_p2_transition_channel_after_gate(joint_recovery)
)
```

They compile to rules selecting all gates of the corresponding arity and append
at the compatibility profile's documented location. They do not create a second
hook implementation. Gate-specific and trigger-specific placement uses the
general event rule API.

## PECOS and Selene Trigger Transport

PECOS should add a serializable trigger command containing:

- stable `TriggerId`;
- ordered target qubits;
- metadata schema version; and
- serialized metadata bytes.

Rust typed signals may remain an in-process extension mechanism, but their
`TypeId` and type-erased payload are not a stable cross-language trigger format.

Selene already carries tagged custom operations. A prototype may reserve a
coordinated, namespaced custom tag whose versioned payload contains the same
trigger fields. Other custom operations continue to be rejected. The reserved
tag and payload schema must be shared rather than privately chosen by each
plugin. If generic triggers become common, a first-class Selene trigger operation
should replace the provisional custom encoding.

## Leakage and Simulator Capabilities

Classically tracked leakage is part of PECOS's execution layer and is available
with every computational-state simulator backend. Strict/default compilation
preserves that model. Replacing leakage with computational-subspace depolarizing
noise requires an explicit approximation policy and compilation-report entry.

Trigger support, injected gates, outcome-dependent measurement effects, and DEM
coverage are independent capabilities and must not be conflated with native
simulator leakage support.

## Differential Conformance

The device-neutral Selene general-noise plugin and conformance suite in
[PECOS #545](https://github.com/PECOS-packages/PECOS/pull/545) provide the
acceptance harness. Two plugin implementations accept the same immutable
`GeneralNoiseParameters` and share runtime-operation parsing and translation:

```text
GeneralNoiseParameters
        +-- legacy GeneralNoiseModel plugin
        +-- event-driven general-noise plugin
```

Only model construction differs. This isolates noise-model behavior from adapter
differences.

Conformance has three levels:

1. Run the independent analytic basis-state and qutrit-reference cases against
   both plugins. These prevent a shared PECOS defect from being accepted as
   parity.
2. Compare output distributions over multiple seeds for every noise family and
   representative combined configurations.
3. Run exact seeded differential traces comparing simulator operations, Boolean
   and leakage-valued outcomes, leakage state, suppression, replacement, idle
   insertion, crosstalk ordering, and subsequent RNG behavior.

Level 3 is mandatory for the compatibility facade. Level 2 is the appropriate
equivalence guarantee for independently composed models that do not claim legacy
RNG ordering.

The PR #545 mutation audit should be repeated against the event-driven plugin.
Additional mutations should cover:

- broad/specific handler reordering;
- lost targets or timing;
- flattened batch boundaries;
- recursive generated-gate processing;
- trigger forwarding;
- incorrect origin tagging; and
- conflicting gate dispositions.

Generic triggers have no legacy equivalent and therefore use focused semantic,
serialization, runtime, and performance tests rather than legacy parity tests.

## Performance Requirements

- Compile stable names to compact runtime IDs.
- Index handlers by event kind, identifier, phase, and arity.
- Pre-merge common broad and specific selector lists while retaining registration
  order.
- Avoid per-event parsing of built-in payloads or trigger identifiers.
- Avoid allocations for no-op events and common one- and two-qubit actions.
- Preserve fast outer-probability checks inside channels.
- Preserve deterministic iteration independent of hash-map implementation.
- Benchmark the event-driven compatibility facade against the production general
  noise model on representative QEC workloads before changing defaults.

## Serialization and Introspection

Serialization preserves:

- semantic noise parameters;
- approximation policies;
- event selectors and phases;
- rule registration order;
- stable gate and trigger identifiers;
- trigger targets and versioned metadata; and
- the selected compatibility profile version.

Process-local compiled IDs, cached samplers, and dispatch-table layout are not
serialized as semantic identities.

The compatibility facade exposes its original parameters and compilation report.
Any existing code that downcasts to the concrete production `GeneralNoiseModel`
must be inventoried before the default implementation changes.

## Alternatives Considered

### Continue adding fixed fields and hook vectors

Rejected as the long-term architecture. Convenience methods may remain, but they
compile into event rules rather than creating bespoke execution paths.

### Maintain two independent general-noise implementations

Rejected. Separate implementations are useful during migration, but the
parameter schema and compatibility behavior have one specification. After
qualification, the public facade delegates to the event-driven implementation.

### Represent triggers as fake gates

Rejected. Triggers have no ideal unitary body and must not participate in gate-set
validation, decomposition, replacement, or simulator execution.

### Match event names as strings during shots

Rejected. Human-readable names are construction and serialization identifiers;
compiled dispatch uses compact typed IDs.

### Use selector specificity as implicit precedence

Rejected. A global registration order is easier to compose and does not silently
reorder a later broad rule ahead of an earlier specific rule or vice versa.

### Require only statistical parity for the compatibility facade

Rejected. It would silently break seeded reproducibility for existing users.
Distributional parity remains sufficient for models that do not claim legacy
compatibility.

## Minimum Test Matrix

### Event validation

- Supported and unsupported phase/event combinations.
- Gate and trigger namespace separation.
- Stable trigger serialization versus process-local compiled IDs.
- Trigger metadata versions and malformed payload rejection.
- Rule/channel arity compatibility.
- Overlapping replacement-rule rejection.
- Replacement/suppression overlap rejection.

### Dispatch and lifecycle

- Broad and specific rules merge by registration order.
- Before rules observe earlier before-rule state changes.
- After rules run after execute, suppress, and replace dispositions.
- Generated operations do not recursively receive program-gate noise.
- Measurement outcomes are available only after measurement.
- Idle handlers receive typed durations.
- Trigger anchors remain inert and are not forwarded.

### Batch behavior

- Batch boundaries and timing survive runtime normalization.
- Per-operation anchors use stable source order.
- Batch-level measurement crosstalk sees the complete measured set.
- Overlapping target operations are rejected when unsupported.
- Trigger barriers cannot be ambiguously mixed with simultaneous gates.

### General-noise compatibility

- Defaults and validation match the production builder.
- Fixed seeds produce identical operations, outcomes, leakage state, and subsequent
  RNG behavior.
- Every scale, weighted sampler, emission/replacement path, seepage path, idle
  family, crosstalk mode, and noiseless-gate control is covered.
- Empty/noiseless models retain the production fast path.
- Rust, Python, PECOS runtime, and Selene runtime configurations agree.

## Implementation Sequence

1. Finalize typed event, phase, timing, origin, batch, and stable-ID contracts.
2. Implement indexed heterogeneous dispatch with global registration ordering.
3. Implement gate disposition validation and non-recursive generated-operation
   provenance.
4. Add a serializable PECOS trigger command and focused runtime tests.
5. Implement the event-driven model behind an explicit non-default build path.
6. Compile `GeneralNoiseParameters` through the compatibility profile without
   changing composite sampling decisions.
7. Add the event-driven Selene plugin and run PR #545's independent,
   distributional, seeded, and mutation conformance suites.
8. Add the provisional coordinated Selene custom-trigger transport.
9. Add Python bindings, serialization, introspection, and compilation reporting.
10. Benchmark representative QEC workloads and optimize compiled dispatch.
11. Review a production-default change separately after every acceptance criterion
    passes.

## Open Questions

1. Should the provisional Selene trigger transport use a reserved `Custom`
   operation, or should a first-class Selene trigger operation be proposed first?
2. Which versioned metadata format should generic cross-language triggers use?
3. Should event-driven support for circuit and batch selectors be public in the
   first release or initially reserved for compatibility channels?
4. Which event and trigger constructs can the DEM builder represent exactly in its
   first release?

## Decision Checklist

Before implementation begins, reviewers should explicitly agree on:

- [ ] the typed built-in event algebra and payload invariants;
- [ ] stable versus process-local event identifiers;
- [ ] valid phases for each event kind;
- [ ] one heterogeneous action representation;
- [ ] global registration ordering across overlapping selectors;
- [ ] batch preservation and deterministic per-operation ordering;
- [ ] conservative gate-disposition conflict rejection;
- [ ] generation origin and non-recursive generated operations;
- [ ] generic trigger arity, serialization, and barrier semantics;
- [ ] the thin general-noise compatibility facade;
- [ ] exact seeded/RNG-stream compatibility requirements;
- [ ] the legacy/event-driven Selene comparison strategy;
- [ ] classically tracked leakage as the backend-independent default;
- [ ] performance acceptance criteria; and
- [ ] a separate review before changing the production default.
