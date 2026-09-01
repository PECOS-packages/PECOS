# InstrProgram MVP: Rust-first surface-code vertical slice

Status: proposed implementation subset.

Companion architecture: [`surface-logical-circuit-guppy.md`](surface-logical-circuit-guppy.md).

## Purpose

Implement the smallest useful version of the typed instruction model end to
end. A user must be able to construct one two-patch surface-code experiment in
Rust or Python, select implementations explicitly, lower it directly to a
normalized `TickCircuit`, optionally lower the same physical plan through
Guppy/QIS, and build a DEM.

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
direct = resolved.to_tick_program()
dem = direct.build_dem(noise_model)
guppy = resolved.to_guppy_program()
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
- `ValueType`: registered opaque type ID plus canonical dialect-owned payload
  and `Linear` or `Copyable` usage;
- `InstrSet`: definitions, implementation candidates, and optional explicitly
  configured defaults;
- `InstrImpl`: stable ID, `supports` check, and plan construction;
- `ResolvedInstrProgram`: every implementation choice, concrete value type,
  selected plan, measurement identity, and selection source fixed.

`InstrGraph` does not switch on instruction names. Dialect traits validate and
instantiate their own type payloads and semantic interfaces.

The first schema is versioned. IDs are deterministic Rust newtypes and are not
Python object identities.

## Required type checking

Construction and final validation enforce:

- named-port and arity correctness;
- parameter presence and schema correctness;
- input type compatibility;
- concrete output type instantiation by the owning dialect;
- definition before use;
- exactly one consumption of every linear value version;
- no use of a consumed value;
- exported values being live and well typed.

The builder infers returned value types. Users do not manually write type
expressions for ordinary calls.

Implementation-specific restrictions are not forced into the generic type
system. For example, logical CX has two patch inputs and two patch outputs;
matching layouts required by the transversal implementation are checked by
that implementation's `supports` method during resolution.

## Required resolution behavior

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

## Required QEC and surface subset

The QEC layer defines a linear `QecBlockType` carrying:

- a canonical surface-patch structural key/specification;
- lifecycle state: `Declared` or `Active`;
- encoded logical-interface identity needed by the supported operations.

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

## Shared physical plan

Resolution composes selected `QecInstrPlan`s into one Rust
`PhysicalCircuitPlan`. It contains:

- physical operations and ordering constraints;
- stable physical/data/ancilla identities and lifetimes;
- instruction and syndrome-round boundaries;
- measurement IDs and result tags;
- detector and observable definitions;
- links back to calls, blocks, checks, and logical results.

The direct TickCircuit and Guppy backends consume this plan. They must not each
reconstruct syndrome schedules, ancillas, measurement order, or detector
boundaries from the logical graph.

## Required outputs

### Direct Rust route

`ResolvedInstrProgram::to_tick_program()` returns a
`GeneratedTickProgram` containing:

- a normalized `TickCircuit`;
- measurement, detector, observable, and result metadata;
- selected implementation and schedule provenance;
- `build_dem(noise_model)` using the PECOS-native DEM builder.

This route imports neither Python nor Guppy.

### Guppy route

`ResolvedInstrProgram::to_guppy_program()` returns deterministic Guppy source
plus the same semantic sidecars. The thin Python bridge compiles and executes
it through HUGR/QIS and returns the traced normalized `TickCircuit`.

The initial scheduling context must allow the direct and traced routes to be
compared operation-by-operation. Later target-specific Guppy schedules may
differ while retaining provenance and semantic equivalence.

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
- third-party implementation plugins;
- replacing or removing existing factories.

Deferred features must not be represented by placeholder public APIs that
silently constrain their later design.

## Implementation order

1. Land the generic Rust IDs, definitions, signatures, graph, values,
   validation, serialization, resolution, and structured diagnostics.
2. Prove the generic boundary with one tiny non-QEC instruction set using a
   linear value and two competing implementations.
3. Add the minimal Rust QEC block type and surface instruction definitions.
4. Adapt/reuse existing patch geometry and surface planning for prepare,
   SZZ syndrome extraction, transversal CX, and measurement.
5. Build `PhysicalCircuitPlan` and direct `GeneratedTickProgram` lowering.
6. Build the direct native DEM and match existing surface-code references.
7. Generate Guppy from the same plan, trace QIS, and compare the normalized
   TickCircuit and metadata.
8. Add thin PyO3/Python authoring wrappers and demonstrate byte-equivalent Rust
   and Python artifacts.

## Acceptance criteria

The MVP is complete only when:

- the demonstration program works in Rust and Python;
- Rust and Python serialize byte-equivalent authored and resolved artifacts;
- invalid arity, type, lifecycle, linear use, parameters, and implementation
  choices fail with structured actionable diagnostics;
- direct lowering and DEM construction run in a Rust-only test;
- the direct normalized TickCircuit, measurement ledger, detectors,
  observables, and representative noisy DEM match an existing PECOS reference;
- generated Guppy compiles and its QIS trace matches the direct route under the
  equivalence scheduling context;
- two same-geometry patches retain distinct instance identities;
- the PRs implementing the MVP do not add any deferred public abstraction.

After these criteria pass, the companion architecture determines which feature
is added next. A reasonable first extension is reusable `InstrModule`
composition, followed by either PHIR emission or additional surface
instructions.
