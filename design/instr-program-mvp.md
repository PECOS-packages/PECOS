# InstrProgram MVP: Rust-first surface-code vertical slice

Status: proposed implementation subset.

Companion architecture: [`surface-logical-circuit-guppy.md`](surface-logical-circuit-guppy.md).

## Purpose

Implement the smallest useful version of the typed instruction model end to
end. A user must be able to construct one two-patch surface-code experiment in
Rust or Python, select implementations explicitly, lower it to a portable
protocol-level physical plan and then to a normalized `TickCircuit`, optionally
lower the same protocol plan through Guppy/QIS, and build a DEM.

This document is normative for the MVP. The companion design explains how the
model later grows to modules, PHIR, code switching, structured control,
space-time composition, visualization, and HUGR import.

## Demonstration program

The MVP is complete when this logical experiment works:

```python
surface = SurfaceInstrSet()
program = InstrProgram(instruction_sets=[surface])
graph = program.main()

control = graph.add_block("control", SurfacePatch.rotated(3))
target = graph.add_block("target", SurfacePatch.rotated(3))

control = graph.apply(surface.prepare(basis="Z"), patch=control)
target = graph.apply(surface.prepare(basis="X"), patch=target)
control = graph.apply(
    surface.syn_extract(rounds=3).using("szz"),
    patch=control,
)
target = graph.apply(
    surface.syn_extract(rounds=3).using("szz"),
    patch=target,
)
control, target = graph.apply(
    surface.cx.using("transversal"),
    control=control,
    target=target,
)
control = graph.apply(
    surface.syn_extract(rounds=3).using("szz"),
    patch=control,
)
target = graph.apply(
    surface.syn_extract(rounds=3).using("szz"),
    patch=target,
)
control_result = graph.apply(surface.measure(basis="Z"), patch=control)
target_result = graph.apply(surface.measure(basis="X"), patch=target)

resolved = program.resolve()
protocol = resolved.to_protocol_physical_plan()
direct = protocol.to_tick_program(reference_schedule)
dem = direct.build_dem(noise_model)
guppy = protocol.to_guppy_program()
trace = guppy.trace_qis()
```

The Rust API must construct the same serialized program and resolved plan. Rust
may use fluent port setters where Python uses keyword arguments.

## Required instruction substrate

The MVP generic crate/module contains only:

- `InstrProgram`: one entry `InstrGraph`, imported `InstrSet` identities, and
  serialization metadata;
- `InstrGraph`: declarations, straight-line calls, SSA value versions, and
  exported results;
- `InstrDef`: stable qualified ID, named input/output ports, parameter schema,
  and a dialect semantic-interface reference;
- `InstrCall`: definition ID, named input `ValueId`s, canonical parameters,
  optional explicit implementation ID, output `ValueId`s, and source ID;
- `ValueType`: registered opaque type ID plus canonical dialect-owned payload;
- `UsePolicy`: the small, orthogonal resource-use rule `SingleUse` or
  `Reusable`;
- `InstrSet`: definitions, implementation candidates, and optional explicitly
  configured defaults;
- `InstrImpl`: stable ID, structured support assessment, declared
  requirements, and plan construction;
- `ResolvedInstrProgram`: every implementation choice, concrete value type,
  selected plan, measurement identity, and selection source fixed.

`InstrGraph` does not switch on instruction names. Dialect traits validate and
instantiate their own type payloads and semantic interfaces.

The first schema is versioned. IDs are deterministic Rust newtypes and are not
Python object identities or display names.

## Required identity model

The MVP distinguishes persistent semantic identity from transient value
versions:

| Identity | Meaning | Stability |
|---|---|---|
| `InstrDefId` / `ImplDefId` | reusable instruction and implementation definitions | stable across programs for the same versioned definition |
| `CallId` | one instruction application | stable through resolution and lowering |
| `CodeBlockInstanceId` | one logical/code-block instance | stable from declaration through preparation, syndrome extraction, gates, and measurement |
| `ValueId` | one SSA state version of a block or classical value | replaced by a consuming call; never used as persistent block identity |
| `CodeElementId` | a stable data site, check, or other element of a code specification | stable while target bindings may change |
| `ProtocolWireId` | an implementation-local physical role, including a temporary ancilla role | stable within the selected protocol-plan instance |
| `MeasurementId` | one semantic measurement occurrence | stable through both backend routes and bound to any runtime result identity |

Two patches with identical geometry have distinct `CodeBlockInstanceId`s.
Syndrome extraction returns a new `ValueId` for the same block instance.
Target physical-resource identities are deliberately absent from the authored
and protocol artifacts; a mapped backend may bind them later without changing
these PECOS identities.

## Required type checking

Construction and final validation enforce:

- named-port and arity correctness;
- parameter presence and schema correctness;
- input type compatibility;
- concrete output type instantiation by the owning dialect;
- definition before use;
- at most one consumption of every `SingleUse` value version;
- no reuse of a consumed `SingleUse` value;
- exported values being live and well typed.

The builder infers returned value types and advances optional block cursors.
Users do not manually write ownership types or type expressions for ordinary
calls. Whether a live block must eventually be measured, exported, or
explicitly discarded is a QEC lifecycle rule, not a generic linear type rule.

Implementation-specific restrictions are not forced into the generic type
system. For example, logical CX has two patch inputs and two patch outputs;
matching layouts required by the transversal implementation are checked by
that implementation's structured support assessment during resolution.

## Required resolution behavior

Each candidate returns a structured `SupportAssessment`, not only a Boolean.
For the MVP it records `Supported` or `Unsupported`, machine-readable reason
codes, human-readable diagnostics, required capabilities, and resource
quantities whose precision is `Exact`, bounded, estimated, or unknown.

For each call, resolution uses this order:

1. an explicit `.using(impl_id)` constraint;
2. an explicitly configured `InstrSet` default;
3. the sole supported candidate;
4. otherwise, an unsupported or ambiguity error.

There is no silent fallback from an explicit choice and no process-global
registry. Diagnostics list the instruction, operand types, candidates, support
failures, and the action needed to resolve ambiguity.

The resolved artifact records `Explicit`, `ConfiguredDefault`, or
`SoleCandidate` as the selection source.

Instruction sets are explicit package dependencies, never process-global
registrations. A serialized program and resolved program record the package's
stable ID, semantic version, serialization version, implementation/profile
fingerprint, canonical parameter bindings, and content digest. A test-only
instruction set defined in a separate Rust crate or package must be importable,
serializable, and resolvable without dynamic plugin loading.

## Required QEC and surface subset

The QEC layer defines a `SingleUse` `QecBlockType` carrying:

- a canonical surface-patch structural key/specification;
- lifecycle state: `Declared` or `Active`;
- encoded logical-interface identity needed by the supported operations.

`CodeBlockInstanceId` is value/provenance identity, not part of
`QecBlockType`: two same-geometry patches are type-compatible while remaining
different block instances. Each block-valued `ValueId` records which persistent
instance or explicitly declared merge/split/create relation it represents.

The MVP must not assume that every future instruction preserves block type or
arity, even though the initial active-patch operations do.

Required semantic instructions are:

| Instruction | Inputs | Outputs | Declared logical effect |
|---|---|---|---|
| `surface.prepare(basis=X/Z)` | one declared patch | one active patch | logical preparation |
| `surface.syn_extract(rounds>0)` | one active patch | replacement active patch | identity |
| `surface.cx` | two active patches | two replacement active patches | logical CX |
| `surface.measure(basis=X/Z)` | one active patch | one logical result | destructive measurement |

Required implementations are:

- existing compatible X/Z preparation and measurement plans;
- SZZ syndrome extraction;
- transversal CX.

Preparation and measurement may resolve as sole candidates. Syndrome
extraction and CX are explicit in the demonstration program. No hidden
conventional profile is required for the MVP.

The surface layer uses one canonical Rust patch representation or a documented
lossless adapter from the existing `SurfacePatch`. Rust and Python must not
independently calculate geometry, checks, logical supports, schedules, or
implementation support.

## Shared protocol physical plan

Resolution composes selected `QecInstrPlan`s into one Rust
`ProtocolPhysicalPlan`. This is portable physical intent, not a final target
mapping or total schedule. It contains:

- portable physical operations and a dependency DAG;
- persistent code-element identities, implementation-local
  `ProtocolWireId`s, temporary resource roles, and lifetimes;
- atomic or tightly ordered stages, permitted concurrency, and quiescence
  boundaries;
- typed resource/service requirements and locality, connectivity, workspace,
  feedback, and latency constraints without preassigned target resources;
- instruction and syndrome-round boundaries;
- measurement IDs and result tags;
- detector and observable definitions;
- links back to calls, blocks, checks, and logical results.

The direct TickCircuit and Guppy backends consume this plan. They must not each
reconstruct syndrome schedules, ancillas, measurement order, or detector
boundaries from the logical graph.

A scheduling backend refines this artifact into a `ScheduledPhysicalPlan` with
concrete resource bindings, a legal operation order, and an explicit schedule
origin. An observed `ExecutionTrace` may additionally bind physical result IDs,
actual branch outcomes, and authoritative runtime timing. Neither refinement
may replace PECOS semantic identities with target identities.

## Required outputs

### Direct Rust route

`ProtocolPhysicalPlan::to_tick_program(reference_schedule)` returns a
`GeneratedTickProgram` containing:

- the reference `ScheduledPhysicalPlan` used for this route;
- a normalized `TickCircuit`;
- measurement, detector, observable, and result metadata;
- selected implementation and schedule provenance;
- `build_dem(noise_model)` using the PECOS-native DEM builder.

This route imports neither Python nor Guppy.

### Guppy route

`ProtocolPhysicalPlan::to_guppy_program()` returns deterministic Guppy source
plus the same semantic sidecars. The thin Python bridge compiles and executes
it through HUGR/QIS and returns the traced normalized `TickCircuit` plus its
schedule/result bindings.

The initial reference scheduling context must allow the direct and traced
routes to be compared operation-by-operation. In general, conformance compares
the protocol dependency order, logical action, resource lifetimes, measurement
ledger, detectors, and observables. Exact tick equality is required only when
both routes declare the same scheduling policy.

### Optional exports

The direct or traced normalized TickCircuit may be converted to `DagCircuit` or
Stim. Native DEM construction does not require Stim.

## Rust and Python boundary

All program state, validation, resolution, planning, serialization, direct
TickCircuit lowering, and deterministic Guppy source generation live in Rust.

PyO3 exposes bound Rust objects. Python provides keyword-friendly calls and
Guppy compilation/runtime orchestration only. It does not retain callbacks,
shadow SSA state, a second implementation registry, or a second renderer.

## Explicitly deferred

The MVP does not implement:

- reusable `InstrModule` bodies, cross-file linking, or HUGR import;
- generic parameterized module elaboration;
- PHIR emission;
- conditional regions, loops, byproduct values, or Pauli-frame manipulation;
- H, S/S-dagger, injection, teleportation, merge/split, surgery, deformation,
  code switching, or heterogeneous codes;
- general `QecTypeExpr` algebra beyond the concrete rules needed above;
- `SpaceTimeProgram`, shape/volume planning, visualization, or Bevy;
- target mapping, routing, calibrated timing, or adaptive runtime control;
- execution-trace import and target-specific physical-resource identities;
- third-party implementation plugins;
- replacing or removing existing factories.

Deferred features must not be represented by placeholder public APIs that
silently constrain their later design.

## Implementation order

1. Land the generic Rust IDs, definitions, signatures, graph, values,
   validation, serialization, resolution, and structured diagnostics.
2. Prove the generic boundary with one tiny non-QEC instruction set using a
   `SingleUse` value and two competing implementations.
3. Add the minimal Rust QEC block type and surface instruction definitions.
4. Adapt/reuse existing patch geometry and surface planning for prepare,
   SZZ syndrome extraction, transversal CX, and measurement.
5. Build `ProtocolPhysicalPlan`, the reference `ScheduledPhysicalPlan`, and
   direct `GeneratedTickProgram` lowering.
6. Build the direct native DEM and match existing surface-code references.
7. Generate Guppy from the same plan, trace QIS, and compare the normalized
   TickCircuit and metadata.
8. Add thin PyO3/Python authoring wrappers and demonstrate byte-equivalent Rust
   and Python artifacts.

## Acceptance criteria

The MVP is complete only when:

- the demonstration program works in Rust and Python;
- Rust and Python serialize byte-equivalent authored and resolved artifacts;
- invalid arity, type, lifecycle, single-use reuse, parameters, and implementation
  choices fail with structured actionable diagnostics;
- direct lowering and DEM construction run in a Rust-only test;
- the direct normalized TickCircuit, measurement ledger, detectors,
  observables, and representative noisy DEM match an existing PECOS reference;
- generated Guppy compiles and its QIS trace matches the direct route under the
  equivalence scheduling context;
- two same-geometry patches retain distinct instance identities;
- syndrome extraction preserves each `CodeBlockInstanceId` while producing a
  new `ValueId`;
- a temporary ancilla has a `ProtocolWireId` and bounded lifetime but is not a
  persistent patch value or preassigned target resource;
- one candidate's structured support failure reports a machine-readable reason
  and actionable explanation;
- serialization round-trips definition, implementation, call, block, value,
  code-element, protocol-wire, and measurement identities;
- a backend may legally reorder independent operations only when it preserves
  the protocol partial order and semantic ledgers;
- the PRs implementing the MVP do not add any deferred public abstraction.

After these criteria pass, the companion architecture determines which feature
is added next. A reasonable first extension is reusable `InstrModule`
composition, followed by either PHIR emission or additional surface
instructions.
