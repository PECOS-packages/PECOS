# `InstrProgram` rationale and implementation notes

Status: supporting, non-normative design notes.

See [`instr-program.md`](instr-program.md) for the reader-oriented design and
[`instr-program-mvp.md`](instr-program-mvp.md) for the normative first
implementation. This document preserves historical context, detailed Rust
boundaries, alternatives, and work beyond the MVP.

## Existing PECOS ideas to reuse

PECOS already contains several generations of related abstractions. A short
retrospective should be completed before public APIs are frozen.

### `pecos.qeccs`

The shipped package models code-agnostic logical instructions, instruction
caching, and selectable code implementations across surface and color-code
families. It remains load-bearing in threshold tooling. The retrospective
should identify useful interfaces, limitations of string symbols and parameter
bags, why `pecos.qec.surface` developed separately, and whether adapters or
shared traits are appropriate.

### Earlier circuit containers

The original `QuantumCircuit` usefully stored instruction symbols separately
from their later interpretation. The live `LogicalCircuit` also shows a failure
mode: logical information held in a list parallel to an underlying physical
circuit can drift from the circuit it describes.

`InstrProgram` keeps late binding but requires one authoritative artifact and
provenance-preserving transforms rather than parallel mutable models.

### Surface planning and renderer prototypes

The current surface stack contains working seams worth extracting:

- `_check_plan.py` and `_clifford_deformation.py` canonicalize choices and
  attach semantic hashes;
- `SurfaceCircuitStep` is a portable surface-operation stream;
- the `CircuitRenderer` family renders related intent to Tick, DAG, Stim, and
  Guppy forms; and
- `LogicalCircuitBuilder` owns cross-call detectors, stabilizer relabeling,
  observables, and multi-patch gate boundaries.

It also contains limitations the new model should not preserve: duplicated
state, incomplete basis support, position-dependent lifecycle behavior, and
renderer-specific semantic side passes.

SLR provides the main HDL analogy and useful lower/middle circuit experience.
The new implementation should carry that experience into typed, Rust-owned,
serializable graphs without depending on SLR or reproducing its original
transpiler goal.

## Detailed Rust model

Names in this section are provisional until implementation proves them.

### Structural artifacts

- `InstrProgram` is a versioned linkage unit with entry graphs, imported
  descriptors, provider requirements, exports, and serialization metadata.
- `InstrGraph` contains typed declarations, calls, values, dependencies, and
  later first-order structured control.
- `InstrDef` declares named ports, canonical parameters, generic effects, use
  policy, and a dialect semantic interface.
- `BoundInstrCall` binds values and parameters to a definition and may constrain
  the implementation through a typed handle.
- `ResolvedInstrProgram` records selected implementations, selection sources,
  implementation bodies, provider fingerprints, and QEC composition products.

The graph never dispatches on display names or surface-code operations. The
owning dialect interprets its opaque value payloads and semantic interfaces.

### Why descriptors and providers are separate

Serialized data cannot contain Rust trait behavior. An implementation therefore
has two sides:

| Side | Responsibility |
|---|---|
| Descriptor | Stable instruction-scoped identity, semantic/schema versions, provider identity, content fingerprint, configuration schema, requirements, and provenance |
| Provider | Executable Rust support assessment and implementation-body construction |

Resolution receives providers and context explicitly:

```rust
let resolved = program.resolve(&providers, &context)?;
```

Loading a serialized program never silently discovers process-global behavior.
Dynamic plugin discovery and packages are later concerns.

Support depends on the complete bound call:

```rust
fn assess_support(
    &self,
    call: &BoundInstrCall,
    context: &ResolutionContext,
) -> SupportAssessment;
```

The architecture may eventually use `Supported`, `Unsupported`, and `Deferred`
assessments with machine reasons, human diagnostics, and missing-context
requirements. The MVP needs only supported/unsupported.

Resolution is deterministic: explicit typed choice, configured choice, sole
supported candidate, or error. Explicit choices do not fall back. Candidate
diagnostics sort by stable qualified identity. Composite expansion must be
acyclic.

### Identity and ownership

Persistent semantic identity is different from one dataflow state version:

| Identity | Meaning |
|---|---|
| `InstrDefId` / `ImplDefId` | Versioned reusable definitions |
| `CallId` | One instruction application |
| `CodeBlockInstanceId` | One persistent logical/code-block instance |
| `ValueId` | One dataflow state version |
| `CodeElementId` | Stable data site, check, or code feature |
| `ProtocolWireId` | Portable implementation-local resource role |
| `SemanticMeasId` | Semantic measurement before circuit lowering |
| `MeasId` | Existing PECOS circuit measurement identity |

`UsePolicy` remains the small orthogonal rule `SingleUse | Reusable`, not a
general linear or affine type system. The generic verifier rejects reuse of a
consumed value. A dialect separately decides whether a live resource must be
measured, exported, or explicitly discarded.

The measurement identity ladder is:

```text
SemanticMeasId
    -> MeasId
    -> record ordinal or negative offset
```

Record positions are boundary encodings for formats such as DEM or Stim; they
are not semantic identity. `SemanticMeasId` should not reuse the existing
propagator `MeasurementId { tick, qubit, basis }` name.

### QEC block state and composition

`QecBlockType` carries structural compatibility such as code family, patch
specification, logical interface, and lifecycle class. Instance-local state
contains the persistent block ID, current value ID, lifecycle, logical Pauli
frame, code-element Clifford/check frame, and a QEC composition-state reference.

The code-element frame is required before target mapping. It records check
relabeling and pending local Clifford effects over stable code elements or
portable protocol roles. A mapped physical Pauli frame belongs to a later
artifact over target bindings and layout epochs.

Detectors and observables can span calls, blocks, and rounds. A QEC-wide pass
therefore owns stabilizer/check flow, detector construction, cross-block
relations, observable propagation, preparation and measurement boundaries,
frame evolution, and epoch alignment. Gadget bodies emit events and transfer
relations rather than finalizing these products independently.

Existing `LogicalCircuitBuilder` behavior is an oracle until a general
stabilizer-flow algebra replaces its hand-written boundary rules.

## Protocol and control boundaries

A selected implementation normally produces a lower-level `InstrGraph` body in
a portable protocol/physical dialect. Bodies may be authored as data, built by
Rust, imported from a supported HUGR subset, or represented as explicitly
opaque externals. Registration always supplies the same QEC contract.

Composed bodies form `ProtocolProgram`, which may contain portable quantum and
classical operations, resource acquisition and release, dependencies, atomic
stages, semantic measurements, QEC annotations, and provenance. It does not
contain final target addresses, target-native routing, inserted idles,
calibrated durations, or authoritative device time.

Dataflow supplies concurrency: calls with no dependency path may overlap.
Typed before/after, alignment, exclusion, and relative space-time constraints
can restrict scheduling. A generic `parallel` region is unnecessary initially.

Later `if` and bounded-repeat nodes must use explicit arguments, yields, and
carried values. Branch joins must reconcile frame state, not merely value
types. The representation must distinguish planned alternatives, externally
selected realizations, and runtime outcomes.

`TickCircuit` remains straight-line. A future control-capable scheduled IR
might use regions or blocks with Tick-like leaf sequences, or control may lower
through PHIR or another established IR. Unsupported adaptive lowering returns
a capability error rather than erasing control.

Reusable cells remain first-order definitions with typed ports and static
parameters. They cannot accept functions as values, capture live host values,
mutate hidden global state, or require general lexical scope.

## Compiler and analysis boundaries

### PHIR and other compiler outputs

PHIR is the preferred inspectable output after its existing QEC dialect and
dependency boundary are audited. It should preserve high-level QEC operations
and selected implementation identity, with progressive lowering to portable
physical operations and provenance maps.

QIR or quantum MLIR dialects may consume PHIR or `ProtocolProgram`. They should
not independently rediscover gadget semantics, measurements, frames, or
resource lifetimes.

### DEM construction

Lowering and analysis are separate extension axes. A consumer declares its
accepted artifact level, dialect, capabilities, configuration, and output.

The concrete physical route is conceptually:

```text
generated Tick artifact + physical NoiseModel
    -> Tick DEM compiler
    -> DEM + provenance and diagnostics
```

The compiler owns noise interpretation, fault propagation, and detector-error
construction. The Tick artifact owns circuit semantics and measurement,
detector, observable, and provenance mappings.

A higher-level DEM compiler could consume resolved gadgets or
`ProtocolProgram`, but only with a noise dialect that defines hierarchy,
symbolic scheduling, branches, loops, correlations, and timing. A single DEM
may be insufficient for runtime branching; a consumer may need specialization,
a conditional/family artifact, or a capability error.

### Guppy/HUGR import

PECOS does not generate Guppy. Import uses compiled HUGR as the boundary:

```text
Guppy -> HUGR -> structural graph, opaque external, or rejection
```

Structural import requires an explicitly supported HUGR subset and must reject
unknown semantics rather than silently skip nodes or infer control from names.
An imported body becomes a QEC implementation only with an explicit QEC
contract, port/resource map, lifecycle and frame transfer, measurement roles,
requirements, and conformance evidence.

## Canonical surface-code port

The Python `SurfacePatch` and current Rust `SurfaceCode` are not losslessly
interchangeable. The canonical Rust `PatchSpec` must represent or explicitly
exclude rotated and standard geometry, independent `dx`/`dz`, orientation,
stable data/check identities, exact supports and ordering, coordinates, logical
supports, and separation of code geometry from protocol ancillas.

Migration starts from Python-generated parity fixtures. Rust imports
authoritative supports, identities, coordinates, and ordering instead of
regenerating them with a different traversal.

The named Rust port covers patch parity, CX preparation/check
scheduling/ancillas/measurement, program-wide detector and observable
composition, `ProtocolProgram`, reference Tick lowering, and thin Python
bindings. SZZ, multi-patch CX, Clifford deformation, and other protocols follow
as separately verified slices.

## Crate direction

The provisional dependency graph is:

```text
pecos-core
    ^
    |
pecos-instr
    generic graph, identities, use checks,
    descriptors/providers, serialization
    ^
    |
pecos-qec
    block state, QEC contracts and implementations,
    composition analysis, ProtocolProgram, Tick lowering

pecos-phir
    existing general compiler IR

one-way instr/QEC-to-PHIR bridge
```

The generic instruction layer should not live inside `pecos-phir` if that
creates an execution-engine dependency or a `pecos-qec`/`pecos-phir` cycle.
The bridge must audit the existing PHIR `qec` namespace before adding dialect
operations or types.

Rust owns semantics and serialization. PyO3 exposes Rust artifacts; Python may
provide cursors and fluent conveniences but no parallel graph, resolver,
planner, or serializer.

## Alternatives not selected

| Alternative | Reason not selected |
|---|---|
| Lower to Guppy | Guppy is a source language, not an appropriate compiler target; its ownership and type constraints would burden scheduling and lowering. |
| Store implementations as strings | Strings are unscoped and typo-prone; instruction sets should provide typed discoverable handles. |
| Universal implementation enum | Third-party and code-specific implementations cannot be known by the generic language. |
| General high-level language | Variables, lexical scope, higher-order values, and unrestricted CFGs are unnecessary for the initial QEC dataflow problem. |
| Explicit `parallel` blocks | Dependencies already expose available concurrency; typed scheduling constraints are more precise. |
| Put control into `TickCircuit` | It would weaken the meaning of an existing straight-line scheduled artifact. |
| Make DEM a method on lowered programs | DEM construction is one configurable analysis consumer, not intrinsic HDL semantics. |
| Require a linear type system | A small use policy plus dialect lifecycle checks provides the needed safety with less authoring complexity. |
| Let Python own semantics | Parallel Rust/Python models would drift and make Rust a second-class route. |
| Use flat Tick circuits as all gadget bodies | They lose portable resources, hierarchy, semantic measurements, and structured control. |

## Space-time and visualization

Resolved gadgets may expose coarse shapes and detailed portable realizations.
A renderer-independent `SpaceTimeView` can link calls, blocks, code elements,
protocol resources, measurements, detectors, observables, and operations. It
may show patch slices, 2+1D occupancy, hierarchy expansion, lifetimes,
constraints, and symbolic versus mapped coordinates.

Interactive rendering is deferred until semantic artifacts stabilize. Bevy is
one possible optional viewer, not part of the core representation.

## Roadmap after the MVP

| Stage | Outcome |
|---|---|
| Foundation | PECOS retrospective, crate/serialization/measurement decisions, generic graph fixture |
| Surface memory | Normative MVP: distance-3 Z memory, CX syndrome, protocol program, Tick parity |
| More checked implementations | X memory, resolved SZZ limitations, real alternative-provider diagnostics, PHIR bridge |
| Multi-block Clifford composition | Alignment, code-element frame evolution, verified H/CX, cross-block detectors and observables |
| First-order hierarchy and import | Reusable acyclic modules, supported HUGR import, one explicitly contracted imported implementation |
| Structured control | Pauli byproducts, frame-only corrections, branches, bounded repetition, lower-level control prerequisites |
| Space-time tools | Placement constraints, stable view serialization, headless and optional interactive rendering |

Existing factories remain until semantic parity tests prove a migration path.

## Open questions

1. Which `pecos.qeccs` interfaces should be adapted, retained, or deprecated?
2. Should initial `PatchSpec` reject unsupported Python patch variants or cover
   them all?
3. What algebra should replace hand-written detector and observable boundaries?
4. Which existing PHIR QEC operations and types are reusable?
5. Which HUGR subset is safe for structural import without silent loss?
6. Should Rust ergonomics use typed builders, macros, or both?
7. Should controlled schedules use Tick-like leaf regions or lower directly
   through PHIR or another established IR?
8. Do any higher-level noise dialects justify DEM construction before concrete
   Tick scheduling?
