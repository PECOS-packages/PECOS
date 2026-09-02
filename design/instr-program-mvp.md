# InstrProgram MVP: Rust-first surface-memory vertical slice

Status: proposed normative implementation subset.

Companion architecture: [`instr-program-architecture.md`](instr-program-architecture.md).

## Purpose

Implement the smallest independently testable version of the instruction
program model against PECOS behavior that exists today.

The MVP constructs one distance-three, Z-basis, CX-syndrome surface-memory
experiment, resolves statically linked Rust providers, builds a portable
`ProtocolProgram`, and lowers it directly in Rust to a normalized
`TickCircuit`.

The MVP does **not** generate Guppy. It does not include SZZ, transversal CX,
PHIR, reusable modules, dynamic control, target mapping, or visualization.

The existing Python surface stack is the migration oracle. The exact reference
fixture is:

```python
patch = SurfacePatch.create(distance=3)
reference = generate_tick_circuit_from_patch(
    patch,
    num_rounds=3,
    basis="Z",
    interaction_basis="cx",
)
```

This is current PECOS code, not proposed API. The MVP comparison predicate and
noise fixture are defined below.

## Prerequisites

Implementation starts only after these Stage 0 artifacts are reviewed:

1. A short retrospective covering `pecos.qeccs`, legacy `LogicalCircuit`,
   `_check_plan`, `SurfaceCircuitStep`/`CircuitRenderer`, and
   `LogicalCircuitBuilder`.
2. A dependency decision establishing standalone `pecos-instr` on
   `pecos-core`, with QEC and PHIR bridges pointing in one direction.
3. A versioned serialization note defining canonical ordering, deterministic
   ID allocation, schema versioning, and provider fingerprint matching.
4. A measurement note defining
   `SemanticMeasId -> pecos_core::MeasId -> record offset`.
5. A canonical Rust `PatchSpec` schema and parity fixtures for the subset used
   by this MVP.

These are design prerequisites, not permission to implement the deferred
features themselves.

## Proposed authoring API

This example is proposed API and becomes an executable documentation test when
the MVP lands:

```python
surface = SurfaceInstrSet.providers()
program = InstrProgram()

data = program.qec_block(
    "data",
    SurfacePatch.create(distance=3),
)
data.prepare(basis=Basis.Z)
data.syn_extract(
    rounds=3,
    using=surface.impls.syndrome_cx,
)
result = data.measure(basis=Basis.Z)
program.export("result", result)

resolved = program.resolve(
    providers=surface,
    context=SurfaceReferenceContext(),
)
protocol = resolved.to_protocol_program()
generated = protocol.to_tick_program(SurfaceReferenceSchedule())
```

DEM construction is deliberately outside this authoring/lowering API. A
separate integration test may pass `generated` and a fixed physical noise model
to the existing native DEM compiler.

The cursor API advances a Rust-owned current `ValueId`. Rust exposes equivalent
typed builders. Tooling may use the lower-level graph API, where every call
returns explicit replacement values; normative examples do not mix cursor and
SSA-rebinding styles.

`surface.impls.syndrome_cx` is a typed `ImplDefRef`, not a string or a universal
language enum. It is scoped to the `surface.syn_extract` instruction.

## Required generic substrate

The MVP generic Rust crate/module contains only:

- `InstrProgram`: one entry graph, provider/descriptor requirements, exports,
  and serialization metadata;
- `InstrGraph`: declarations, straight-line calls, value edges, and exports;
- `InstrDef`: stable qualified ID, named ports, canonical parameter schema,
  use policies, and a dialect semantic-interface reference;
- `BoundInstrCall`: definition reference, named input `ValueId`s, canonical
  parameters, optional typed `ImplDefRef`, output `ValueId`s, and `CallId`;
- `ValueType`: registered opaque type ID plus canonical dialect-owned payload;
- `UsePolicy`: `SingleUse` or `Reusable`;
- `InstrImplDescriptor`: serializable instruction-scoped implementation ID,
  version, provider identity, and fingerprint;
- `InstrImplProvider`: executable support assessment and implementation-body
  construction;
- `ResolutionContext`: explicit, serializable facts needed by providers; and
- `ResolvedInstrProgram`: every call selected with its descriptor, selection
  source, provider fingerprint, and implementation body.

There are no reusable modules, regions, loops, space-time quantities, service
leases, package manager, or dynamic plugins in this subset.

## Identity model

The following Rust newtypes are distinct:

| Identity | MVP meaning |
|---|---|
| `InstrDefId` / `ImplDefId` | reusable versioned descriptors |
| `CallId` | one call in the program |
| `CodeBlockInstanceId` | one persistent patch instance |
| `ValueId` | one state version of that patch or a result |
| `CodeElementId` | stable data site or check from `PatchSpec` |
| `ProtocolWireId` | physical role local to the selected protocol program |
| `SemanticMeasId` | semantic measurement occurrence |
| `MeasId` | existing identity in the generated `TickCircuit` |

Prepare, syndrome extraction, and measure preserve the same
`CodeBlockInstanceId` while producing fresh `ValueId`s. Destructive measurement
ends the active block lifetime. Two same-geometry patches are type-compatible
but have different block instance IDs.

Generic validation rejects a second use of a consumed `SingleUse` value. QEC
validation separately checks preparation, active lifetime, measurement, and
export obligations.

## Descriptor/provider resolution

Serialized descriptors do not contain Rust trait behavior. Resolution is:

```rust
let resolved = program.resolve(&providers, &context)?;
```

The provider receives the entire canonical `BoundInstrCall`, including basis
and round parameters:

```rust
fn assess_support(
    &self,
    call: &BoundInstrCall,
    context: &ResolutionContext,
) -> SupportAssessment;
```

The MVP assessment is `Supported` or `Unsupported` with a stable reason code
and human-readable explanation. Deferred feasibility is reserved for the
architecture but not implemented here.

Resolution order is:

1. explicit typed `ImplDefRef`;
2. explicitly configured instruction-set choice;
3. sole supported provider;
4. otherwise structured unsupported or ambiguity error.

Explicit choices do not fall back. Candidate diagnostics are sorted by stable
qualified ID. Missing providers, version/fingerprint mismatches, and an
implementation reference scoped to the wrong instruction are distinct errors.

The resolved selection source is `Explicit`, `Configured`, or `SoleCandidate`.

## Required QEC subset

The QEC layer defines a `SingleUse` `QecBlockType` carrying:

- canonical `PatchSpec` structural identity;
- lifecycle class: `Declared` or `Active`; and
- encoded logical interface.

Instance-local `QecBlockState` contains the block instance ID, current value ID,
lifecycle state, logical Pauli frame, code-element Clifford/check frame, and a
reference to program-level QEC analysis state. The MVP frames remain identity,
but their fields and transfer checks prevent later H/S/SZZ support from needing
a second block-state model.

Required instructions are:

| Instruction | Input | Output | Ideal logical effect |
|---|---|---|---|
| `surface.prepare(basis=Z)` | declared patch | active patch | prepare logical zero |
| `surface.syn_extract(rounds>0)` | active patch | replacement active patch | decoded logical identity under the protocol's success contract |
| `surface.measure(basis=Z)` | active patch | logical result | destructive logical-Z measurement |

Required providers are:

- current compatible Z preparation;
- CX-based syndrome extraction; and
- current compatible Z measurement.

Preparation and measurement can be sole providers because each provider sees
the canonical basis parameter. The MVP does not claim support for X/Y basis,
SZZ, H/S, multi-patch gates, injection, surgery, or code switching.

## Canonical patch subset and explicit Rust port

The MVP canonical `PatchSpec` supports exactly the reviewed subset needed by
the reference fixture:

- rotated distance-three square patch;
- orientation used by the current reference;
- stable data and X/Z-check IDs;
- exact Python check supports, ordering, coordinates, and logical supports; and
- separation of data geometry from protocol ancillas.

The parity fixture is serialized from the existing Python `SurfacePatch`, then
loaded and validated by Rust. Rust must not independently regenerate check
order and call that parity.

The surface-planning work is explicitly part of the MVP:

1. import/represent the parity fixture;
2. port CX preparation and measurement planning;
3. port the CX check plan, touch order, ancilla assignment, and round structure;
4. emit semantic measurement events and protocol operations; and
5. compose the reference detectors and logical observable.

Broader patch parity and other interaction bases are separate projects.

## Program-level QEC composition owner

Per-call implementation bodies do not finalize detectors or observables. The
MVP includes a small program-level `SurfaceMemoryAnalysis` pass over the
resolved calls and their semantic measurement events. It owns:

- preparation-boundary detectors;
- comparisons between consecutive syndrome rounds;
- terminal measurement detectors;
- the logical-Z observable;
- stabilizer/check epoch consistency; and
- `SemanticMeasId` allocation and mapping.

This pass is intentionally sufficient only for one straight-line, single-patch
memory experiment. It establishes the ownership seam that later multi-call and
cross-patch stabilizer-flow analysis will generalize.

## Protocol program

Selected call implementation bodies and `SurfaceMemoryAnalysis` compose into
one Rust `ProtocolProgram` containing:

- portable preparation, one-/two-qubit, measurement, and tick-barrier intent;
- an operation dependency DAG and named syndrome rounds;
- persistent `CodeElementId`s and temporary `ProtocolWireId`s;
- ancilla lifetimes and cleanup;
- instruction and round boundaries;
- semantic measurements and their eventual `MeasId` bindings;
- detectors and logical observable; and
- provenance back to calls, block versions, checks, and providers.

The MVP program does not contain target addresses, routing, calibrated time,
adaptive control, resource estimates, service requirements, or execution
traces.

For this straight-line slice, each Rust provider builds a typed lower-level
`InstrGraph` fragment containing portable preparation, quantum operation,
measurement, resource-lifetime, and dependency instructions. The composed
`ProtocolProgram` is therefore an inspectable graph, not an opaque callback
that happens to produce a `TickCircuit`. Structured control is deferred, but
the lower-level graph boundary is the one that will host it later.

## Reference scheduling and outputs

`SurfaceReferenceSchedule` is a versioned deterministic policy reproducing the
existing CX memory fixture's qubit numbering, check ordering, touch ordering,
ancilla assignment, round barriers, and measurement order.

It returns `GeneratedTickProgram` containing:

- normalized `TickCircuit`;
- `SemanticMeasId -> MeasId -> record ordinal` maps;
- detectors and logical observable;
- block/call/check/provider provenance; and
- reference schedule ID and version.

## Separate DEM integration fixture

The core MVP ends at `GeneratedTickProgram`. A separate downstream integration
test passes that artifact to `TickDemCompiler` with a fixed physical noise
fixture:

```text
REFERENCE_NOISE_V1
    one_qubit_depolarizing = 0.001
    two_qubit_depolarizing = 0.002
    preparation_flip = 0.003
    measurement_flip = 0.004
    idle = 0.0
```

If the current native builder uses a richer schema, the fixture explicitly
sets every additional field to zero. The checked-in canonical fixture records
that full schema and version.

This test proves that the lowered circuit preserves the measurement, detector,
and observable information required by PECOS analysis. It does not make DEM
construction a method of `InstrProgram`, `ProtocolProgram`, or
`GeneratedTickProgram`. Future DEM consumers may accept higher-level artifacts
only with an explicit noise-model dialect and control/scheduling semantics.

## Rust/Python boundary and serialization

Rust owns every authoritative object and serializer. Python calls the Rust
builders; therefore “Rust and Python serialize byte-equivalently” is not an
independent semantic oracle.

Instead, the MVP uses:

- canonical golden fixtures for authored and resolved programs;
- deterministic IDs from documented traversal/allocation rules;
- ordered maps or explicitly sorted key/value sequences;
- no floating-point fields in the authored/resolved MVP schema;
- schema/version and provider fingerprint fields;
- Rust round-trip tests; and
- Python binding tests that prove each ergonomic operation invokes the expected
  Rust transition and returns the same Rust-owned artifact.

The exact wire format is decided in the Stage 0 serialization note.

## Explicitly deferred

- Guppy generation or direct-versus-Guppy equivalence;
- Guppy/HUGR import;
- PHIR, QIR, or MLIR lowering;
- reusable `InstrModule` definitions or higher-order constructs;
- `ParallelRegion`, generic scheduling regions, branches, loops, or adaptive
  control;
- X/Y memory, SZZ, transversal gates, code-element Clifford evolution beyond
  identity, injection, teleportation, surgery, and code switching;
- general detector/observable stabilizer-flow algebra;
- asymmetric, standard, repetition, or distance-1 patch parity;
- target mapping, routing, native rebasing, calibrated timing, and traces;
- `SpaceTimeProgram`, resource quantities, services, visualization, or Bevy;
- package management and dynamic providers; and
- replacing or removing existing factories.

## Implementation order

1. Complete the retrospective, crate-boundary, serialization, and measurement
   identity notes.
2. Land the tiny generic Rust graph, IDs, typed implementation references,
   provider resolution, use checks, and golden serialization fixture.
3. Land distance-three `PatchSpec` parity from the Python reference fixture.
4. Port the exact CX memory planning subset into QEC implementation bodies.
5. Implement `SurfaceMemoryAnalysis` and its detector/observable ledger.
6. Build `ProtocolProgram` and `SurfaceReferenceSchedule` lowering.
7. Match the existing ideal TickCircuit, measurement maps, detectors, and
   observable.
8. Add a separate adapter integration test matching the native DEM under
   `REFERENCE_NOISE_V1`.
9. Add thin PyO3/Python cursor bindings and executable documentation tests.

Each step is independently reviewable and testable. No step silently includes
the rest of the Python surface stack.

## Acceptance criteria

The MVP is complete only when:

- the proposed program works in Rust and Python through the same Rust-owned
  artifacts;
- every normative API example is executed by repository tests;
- `SurfacePatch.create(distance=3)` parity preserves exact data/check IDs,
  supports, ordering, coordinates, and logical supports;
- invalid ports, parameters, lifecycle, reuse, provider availability,
  fingerprint, implementation scope, and ambiguity fail with structured errors;
- prepare/syndrome/measure preserve one `CodeBlockInstanceId` and produce fresh
  `ValueId`s;
- the protocol program exposes bounded temporary ancilla lifetimes and no target
  resource IDs;
- the generated TickCircuit matches the reference's ideal operations, qubit
  numbering, round/tick boundaries, and measurement order;
- `SemanticMeasId -> MeasId -> record` maps round-trip and drive detector and
  observable construction;
- detector definitions and logical observable match the existing reference;
- the separate DEM consumer integration matches native DEM output under the
  complete `REFERENCE_NOISE_V1` schema without adding DEM behavior to the core
  HDL artifacts;
- Rust tests do not import Python or Guppy; and
- Python tests add no shadow graph, resolver, planner, or serializer.

After this gate passes, the architecture roadmap determines the next slice.
The likely next work is X-basis memory and then resolving the existing SZZ
reference limitations before registering SZZ as a real alternative provider.
