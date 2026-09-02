# Typed instruction programs for QEC

Status: proposed architecture.

Initial implementation companion: [`instr-program-mvp.md`](instr-program-mvp.md).
The companion is normative for the first implementation. This document records
the longer-term architecture and the boundaries the MVP must preserve.

## Purpose

PECOS should provide an HDL-like, first-order dataflow representation for
describing, composing, inspecting, and lowering QEC experiments at several
levels of abstraction.

The same program should let:

- an experiment author work with logical code blocks and QEC gadgets;
- a QEC implementer define alternative realizations of those gadgets;
- a compiler inspect resource dependencies, protocol structure, and portable
  physical intent;
- a mapper or target backend bind resources and choose a legal schedule; and
- analysis and visualization tools relate logical calls to physical operations,
  measurements, detectors, observables, and space-time occupancy.

As in an HDL, a component can be treated as a black box at one level and opened
at a more detailed level when needed. The representation is strongly inspired
by SLR and earlier PECOS circuit models, but it is not a transpiler to a
high-level host language. In particular, **PECOS will not lower
`InstrProgram` to Guppy**.

The principal lowering and consumption paths are:

```text
InstrProgram
    |
    v
resolved QEC gadgets and program-level QEC analysis
    |
    +--------------------------+
    |                          |
    v                          v
PHIR                    ProtocolProgram
(inspectable IR)        (portable Rust mid-level program)
                               |
                    +----------+----------+
                    |                     |
                    v                     v
          PECOS reference schedule   external target backend
                    |                     |
                    v                     v
          normalized TickCircuit   ScheduledPhysicalPlan
                    |                     |
                    v                     v
               DagCircuit          trace/import when supported

Analysis/compiler consumers may attach at the level whose semantics they
understand. In particular:

    normalized TickCircuit + physical noise model
        -> TickDemCompiler
        -> DEM
```

QIR or appropriate MLIR quantum dialects may become additional lower-level
exports after their semantic and dependency boundaries are audited. They are
not MVP requirements.

Guppy remains relevant only as an **input language**. A Guppy function can be
compiled to HUGR and imported structurally or opaquely. With an explicit QEC
contract, such an imported module may serve as one implementation of a gadget.
PECOS must not infer logical QEC semantics from an unannotated gate pattern.

## Scope decisions

This revision makes the following decisions explicit:

1. `InstrProgram` is a first-order dataflow and hierarchy model, not a general
   high-level programming language.
2. Data dependencies define required ordering. Independent calls are unordered
   and therefore potentially concurrent.
3. There is no generic `parallel` block in the initial model. Explicit manual
   scheduling uses typed space-time or scheduling constraints.
4. Branching and repetition may be added as structured control nodes, but are
   not required by the MVP and must not turn `InstrProgram` into an unrestricted
   CFG or host language.
5. Implementation choices use typed, instruction-scoped references, never
   unscoped strings such as `.using("transversal")`.
6. The raw graph API is a tooling substrate. Human authors use cursors, fluent
   methods, generated builders, Rust macros, or a future textual syntax.
7. Rust owns the authoritative graph, validation, resolution, QEC analysis,
   protocol expansion, serialization, and direct circuit lowering. Python is a
   thin ergonomic wrapper.
8. Portable QEC physical intent is separate from target mapping, routing,
   native scheduling, and authoritative device time.
9. DEM construction is a downstream analysis/compiler concern. It is not an
   intrinsic method of `InstrProgram`, a gadget, or `ProtocolProgram`.

## Required retrospective before implementation

PECOS already contains several generations of related abstractions. Before
freezing new public APIs, Stage 0 must document why each was not extended and
which parts should be reused.

### `pecos.qeccs`

The shipped `pecos.qeccs` package already models code-agnostic logical
instructions, instruction caching, and selectable code implementations across
surface and color-code families. It remains load-bearing in threshold tooling.
The retrospective must identify:

- which interfaces are semantically useful;
- where string symbols, parameter bags, caching, or ownership became limiting;
- why `pecos.qec.surface` developed separately; and
- whether migration adapters or shared traits are appropriate.

### Legacy `QuantumCircuit` and `LogicalCircuit`

The original `QuantumCircuit` usefully separated stored instruction symbols
from later interpretation. The live `LogicalCircuit` also demonstrates a
failure mode: logical information stored in a list parallel to an underlying
physical circuit can drift from the circuit it describes. The new design keeps
late binding but requires one authoritative artifact and provenance-preserving
lowering rather than parallel mutable models.

### Surface planning and renderer prototypes

The current surface stack already contains important prototypes:

- `_check_plan.py` and `_clifford_deformation.py` canonicalize named options,
  produce resolved artifacts, and attach semantic hashes;
- `SurfaceCircuitStep` represents a portable surface operation stream;
- the `CircuitRenderer` family renders related operation intent to Tick, DAG,
  Stim, and Guppy forms;
- `LogicalCircuitBuilder` owns cross-call detector construction, stabilizer
  relabeling, observable propagation, and multi-patch gate boundaries.

The new resolver and `ProtocolProgram` should generalize or extract these
working seams rather than introduce parallel implementations. Their limitations
must also be recorded, including duplicated state, incomplete basis support,
position-dependent lifecycle behavior, and renderer-specific semantic side
passes.

## Crate and dependency boundary

Stage 0 must settle the dependency direction before implementation. The
provisional split is:

```text
pecos-core
    ^
    |
pecos-instr
    generic IDs, first-order dataflow graph, use checks,
    descriptors, provider interfaces, serialization
    ^
    |
pecos-qec
    code-block state, QEC gadget contracts and providers,
    composition analysis, ProtocolProgram, direct Tick lowering

pecos-phir
    existing general PHIR representation

pecos-instr-phir or pecos-qec-phir
    one-way bridge depending on the required source crates and pecos-phir
```

The generic instruction layer must not live inside `pecos-phir` if doing so
would pull execution-engine dependencies upward or create a `pecos-qec` ↔
`pecos-phir` cycle. The bridge must also audit the existing PHIR `qec` dialect
namespace before adding new QEC operations or types.

## Core first-order model

### Artifacts

- `InstrProgram` is one versioned linkage/compilation unit with entry graphs,
  imported definition descriptors, provider requirements, and exports.
- `InstrGraph` is a typed dataflow graph of declarations and `InstrCall`s.
- `InstrDef` is a stable instruction interface: named ports, parameters,
  generic effects, and dialect semantic interfaces.
- `InstrCall` binds named input values and canonical parameters to an
  `InstrDefRef`, optionally constrains its implementation, and produces output
  values.
- `InstrImplDescriptor` is serializable identity and metadata for one
  implementation.
- `InstrImplProvider` is executable Rust behavior that assesses calls and
  constructs implementation bodies matching a descriptor.
- `ResolvedInstrProgram` records one selected implementation and selection
  source for every call plus implementation bodies and QEC
  composition-analysis products.

The graph never dispatches on display names or surface-code operation names.
Dialect implementations interpret their own types and semantics.

### Descriptor/provider split

Serialized data cannot contain Rust trait behavior. Resolution therefore takes
providers explicitly:

```rust
let resolved = program.resolve(&providers, &context)?;
```

An `InstrImplDescriptor` contains:

- instruction-scoped `ImplDefId`;
- semantic and schema versions;
- provider/package identity and content fingerprint;
- canonical configuration schema; and
- declared static requirements and provenance.

At runtime, `InstrImplProvider` objects are supplied explicitly. Resolution
matches descriptor IDs, versions, and fingerprints to providers and reports a
deterministic missing-provider error. Loading a serialized program never
silently discovers process-global implementations.

Dynamic plugin discovery and package management are later concerns. The MVP
uses statically linked providers.

### Typed implementation references

`.using(...)` accepts an `ImplDefRef` belonging to the called instruction:

```python
surface.syn_extract(
    patch=data,
    rounds=3,
    using=surface.impls.syndrome_cx,
)
```

The instruction set exports discoverable typed handles such as
`surface.impls.syndrome_cx`. The language does not know a universal enum of all
possible implementations, and authors do not type raw IDs. Serialized data
stores the stable qualified `ImplDefId` behind the handle. Passing a CX
implementation to a preparation instruction is rejected before resolution.

### Bound calls and support assessment

Support depends on the entire bound call, including canonical parameters, not
only its operand types:

```rust
fn assess_support(
    &self,
    call: &BoundInstrCall,
    context: &ResolutionContext,
) -> SupportAssessment;
```

`SupportAssessment` returns structured `Supported`, `Unsupported`, or
`Deferred` status with machine-readable reasons, human diagnostics, and any
requirements needed by a later feasibility context.

Resolution order is deterministic:

1. a call's explicit typed implementation reference;
2. an explicitly configured instruction-set choice;
3. the sole supported candidate after the required context is available;
4. otherwise an unsupported, deferred-context, or ambiguity error.

Explicit choices never silently fall back. Candidate ordering is by stable
qualified ID, never map iteration order. Composite implementations form an
acyclic expansion graph; recursive self-selection or cycles are errors.

### Identity and use checks

Persistent identity is distinct from SSA state versions:

| Identity | Meaning |
|---|---|
| `InstrDefId` / `ImplDefId` | versioned reusable definitions |
| `CallId` | one instruction application |
| `CodeBlockInstanceId` | one persistent logical/code-block instance |
| `ValueId` | one dataflow state version |
| `CodeElementId` | stable data site, check, or code feature |
| `ProtocolWireId` | implementation-local portable physical role |
| `SemanticMeasId` | semantic measurement occurrence before circuit lowering |
| `MeasId` | existing PECOS circuit measurement identity |

A separate `UsePolicy` is limited to `SingleUse | Reusable`. It is not a
general linear or affine type system. Generic validation rejects a second use
of a consumed `SingleUse` value. The QEC dialect separately verifies whether a
live block must be measured, exported, or deliberately discarded.

Two blocks with identical geometry have different `CodeBlockInstanceId`s but
compatible block types. Syndrome extraction returns a new `ValueId` for the
same block instance.

## QEC gadget model

`QecInstr` is the semantic interface of a QEC gadget. It declares:

- named code-block and classical input/output ports;
- canonical parameters;
- lifecycle effects;
- an ideal/noiseless encoded logical action or channel;
- logical-frame transfer; and
- semantic measurement/result roles.

The ideal logical action is the intended decoded behavior under the
instruction's stated success conditions. It does not claim that noisy physical
execution or an undecoded syndrome is literally identity.

`QecInstrImpl` is one selectable realization. Resolving it binds the
implementation to a specific call, code-block states, parameters, and context,
then produces an inspectable implementation body or an explicitly opaque
external body.

### Implementation bodies as lower-level instruction graphs

An implementation is not normally stored as an already scheduled
`TickCircuit`. Its inspectable form is a lower-level Rust `InstrGraph` body in a
protocol/physical dialect. The same generic graph substrate can therefore be
used at different abstraction levels without teaching it surface-code names:

```text
logical/QEC InstrGraph
    QEC block values and gadget calls
        |
        | select and expand QecInstrImpl bodies
        v
protocol/physical InstrGraph fragments
    portable operations, resource lifetimes, measurements,
    dependencies, and structured control
        |
        | compose and verify
        v
ProtocolProgram
```

The lower dialect is similar in role to the useful lower and middle circuit
layers in SLR, but is Rust-owned, typed, serializable, and provenance linked.
It may contain portable quantum and classical operations, resource
acquire/release, barriers or dependency edges, semantic measurements, and—once
the structured-control prerequisite exists—`if` and bounded repetition with
explicit arguments and yields.

An implementation may be authored directly as this graph, built by Rust code,
imported from a supported HUGR subset, or retained as an opaque external body.
Regardless of authoring route, registration supplies the same QEC contract and
conformance obligations. Rust builder code that immediately emits a flat
`TickCircuit` is permitted only for a deliberately straight-line leaf adapter;
it cannot represent the general gadget-body contract.

The model naturally covers:

- preparation: no active input state to one active block;
- syndrome extraction: one active block to a replacement state version with
  declared logical identity;
- logical gates: replacement versions with a declared logical transform;
- code switching or deformation: input and output block types may differ;
- merge/split: arity may change; and
- destructive measurement: a block is consumed and results remain.

### Block state and frames

`QecBlockType` contains structural compatibility facts such as code family,
patch specification, logical interface, and lifecycle class. Instance-local
state is carried separately on each block-valued version:

```text
QecBlockState
    CodeBlockInstanceId
    ValueId
    PatchSpec / code binding
    lifecycle state
    logical Pauli frame
    code-element local Clifford/check frame
    composition-analysis state reference
```

The code-element Clifford frame is required before target mapping. It captures
X/Z check relabeling after logical Cliffords and pending local Clifford effects
that a later physical operation must host. It is expressed over
`CodeElementId` or `ProtocolWireId`, not mapped physical-qubit IDs.

A mapped physical Pauli frame is a later artifact over target resource bindings
and layout epochs. Branch joins must explicitly merge compatible logical and
code-element frame states; matching value types alone is insufficient.

### Program-level QEC composition analysis

Detectors and logical observables are not generally owned by one call. They can
span calls, code blocks, rounds, and gate boundaries. `ResolvedInstrProgram`
therefore runs a program-level QEC composition pass over the selected
implementation bodies.

The pass owns:

- stabilizer/check flow across instruction boundaries;
- detector construction from semantic measurements;
- cross-block detector relations at multi-block operations;
- logical observable propagation and determinism/existence;
- preparation and terminal-measurement boundaries;
- logical and local-Clifford frame evolution; and
- required round or epoch alignment.

Per-call bodies emit semantic events and transfer relations. They do not finalize
detectors or observables in isolation. Existing `LogicalCircuitBuilder`
behavior remains a test oracle until an algebraic stabilizer-flow implementation
replaces its hand-written boundary rules.

## Dataflow, scheduling, and structured control

### Implicit concurrency

Source order is not physical time. SSA/resource dependencies and explicit
completion edges impose ordering. Calls with no dependency path are eligible
for concurrent scheduling.

The initial graph has no `ParallelRegion`. Manual control is expressed through
typed constraints attached to calls or groups, for example:

- align named protocol rounds or epochs;
- same-start or same-finish constraints;
- before/after constraints not implied by dataflow;
- mutual exclusion;
- relative space-time placement; and
- pinned coordinates or intervals when a user deliberately explores a layout.

These constraints restrict legal schedules; they do not themselves assign
target resources or authoritative time. The protocol scheduler diagnoses
inconsistent constraints.

### Branching and repetition

Later structured control may add `IfNode`, static `RepeatNode`, and bounded
dynamic repetition with explicit arguments, yields, and carried values. It
must distinguish:

- planned alternatives;
- externally selected or sampled realizations; and
- observed runtime outcomes.

Backends lacking adaptive control return a capability error without erasing the
source semantics. `TickCircuit` currently has no implemented classical-control
model, so it remains a straight-line scheduled projection. A `ProtocolProgram`
with control can lower to Tick only after compile-time specialization,
unrolling, or a proved frame-only rewrite. Otherwise it requires PHIR, an
appropriate quantum MLIR/QIR form, or a control-capable target backend.

A future scheduled control-flow artifact may use regions or basic blocks whose
straight-line bodies are TickCircuit-like sequences. That is conceptually
“TickCircuit plus control flow,” but it should be a separate typed IR rather
than weakening `TickCircuit` semantics.

### First-order hierarchy

Reusable cells remain useful, but they are first-order definitions, not
higher-order host-language functions. A future `InstrModule` may have typed
ports and canonical static parameters, instantiate other definitions, and be
called many times. It cannot capture live host values, accept functions as
values, mutate hidden global state, or require general lexical scope.

The MVP defers reusable modules. Ordinary program entry graphs and implementation
bodies are sufficient to validate the initial shape.

## Ergonomic authoring

The core operations `declare`, `apply`, and explicit SSA results remain
available for generators and compiler tooling. They are not the expected
everyday experiment syntax.

A thin cursor/fluent layer can look like:

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
```

The cursor holds a Rust-owned reference to the current `ValueId` and advances
after a successful call. There is one authoring rule: cursor methods mutate the
cursor's current version; core graph methods return explicit replacement
values. Examples must not mix those styles ambiguously.

Rust should offer typed builders and may add macros after the data structure
stabilizes. A textual HDL is a possible later frontend, not an MVP deliverable.

## Portable protocol program

Selected lower-level implementation bodies compose into `ProtocolProgram`. It
is a portable Rust mid-level program containing:

- portable physical operations with named operand roles;
- an operation dependency DAG;
- persistent code elements and temporary `ProtocolWireId` roles;
- resource acquisition/release and cleanup scopes;
- atomic or tightly ordered stages and quiescence boundaries;
- permitted concurrency;
- locality, connectivity, workspace, feedback, and service requirements;
- semantic measurements and program-level detectors/observables; and
- provenance back to calls, block versions, code elements, and implementations.

It does **not** contain final target resource addresses, target-native routing,
inserted idles, calibrated duration, or authoritative device time.

A PECOS reference scheduler can refine a supported straight-line
`ProtocolProgram` into a normalized `TickCircuit`. An external target backend
can instead produce a `ScheduledPhysicalPlan` with target bindings and legal
operation order, plus an optional execution trace. A control-capable backend
may retain structured control. A backend may reject an infeasible program but
cannot change its QEC semantics or silently select another implementation.

Backend conformance compares the protocol dependency relation, logical action,
resource lifetimes, frames, measurements, detectors, observables, and
provenance. Exact tick order is required only when two routes claim the same
scheduling policy.

## Downstream analyses and DEM construction

Lowering and analysis are separate extension axes. A consumer declares the
artifact level and dialect it accepts, required capabilities, configuration,
and output type. It can operate on the authored graph, resolved QEC program,
PHIR, `ProtocolProgram`, `ScheduledPhysicalPlan`, or a concrete circuit without
being built into those artifacts.

DEM construction is one such consumer. The existing precise route is shown
below; `TickDemCompiler` is a provisional interface name rather than a required
crate or public type name.

```text
GeneratedTickProgram
    normalized TickCircuit
    measurement/detector/observable bindings
    lowering provenance
        + physical NoiseModel
        |
        v
TickDemCompiler
        |
        v
DEM + provenance/diagnostics
```

Conceptually the API is `dem_compiler.compile(&generated, &noise_model)`, not
`generated.build_dem(...)`. The compiler owns noise insertion/interpretation,
fault propagation, detector-error construction, and capability diagnostics.
The generated circuit owns only circuit semantics and the identity maps the
compiler consumes.

Other DEM compilers may work at a higher HDL level if their noise models are
defined on that level. For example, a gadget-level stochastic model could
consume resolved gadget calls, while a protocol-operation noise model could
consume `ProtocolProgram`. Such a compiler must declare how it treats
hierarchy, symbolic scheduling, branches, loops, correlations, and target
timing; it must reject semantics it cannot faithfully model. A single ordinary
DEM may be insufficient for runtime branching, in which case a consumer may
produce a conditional/family artifact or report that specialization is
required.

Detector and observable composition remains QEC program analysis rather than
DEM construction. The QEC pass defines semantic detector/observable relations;
lowering binds their measurement identities; a DEM compiler evaluates faults
against those definitions under a noise model.

## Measurement identity

The identity ladder is explicit:

```text
SemanticMeasId
    program/QEC meaning
        |
        v
MeasId
    existing PECOS circuit identity
        |
        v
record ordinal or negative offset
    boundary encoding for DEM/Stim/runtime formats
```

`SemanticMeasId` must not reuse the existing propagator
`MeasurementId { tick, qubit, basis }` name. Lowering returns explicit maps
between every rung. Record positions are boundary encodings, not semantic IDs,
even though the current DEM representation stores offsets.

## PHIR and lower-level compiler outputs

PHIR is the primary inspectable compiler output from the resolved program. The
bridge should preserve high-level QEC operations and selected implementation
references, with progressive lowering to portable physical operations when
requested. Every transform returns provenance maps.

The design does not require PHIR for the MVP because Stage 0 must first settle
crate dependencies and the existing PHIR QEC dialect namespace. Once that
boundary is proven by a consumer, PHIR becomes the preferred general compiler
route rather than growing `InstrGraph` into a general IR.

QIR or quantum MLIR dialect exports should consume PHIR or `ProtocolProgram`,
not reimplement QEC implementation expansion. Their exact role depends on
whether they can preserve hierarchy, measurement identity, control, and
scheduling constraints required by a selected backend.

## Guppy/HUGR import

PECOS does not generate Guppy from an instruction program. Guppy import uses
compiled HUGR as the stable boundary:

```text
Guppy function
    -> Guppy compiler
    -> HUGR package/function
    -> HugrInstrImporter
    -> structural InstrGraph fragment, opaque external module, or rejection
```

The importer must be built against an explicitly supported HUGR subset. It
must not inherit converters that silently skip unknown nodes or use name
heuristics for control flow.

Import outcomes are:

1. **Structural import:** every operation, type, region, and ownership edge is
   supported and becomes inspectable instruction structure.
2. **Opaque external module:** the typed HUGR function and digest are retained;
   only backends that understand it may link or execute it.
3. **Rejected import:** unsupported or ambiguous semantics produce a diagnostic
   identifying the offending HUGR node/type.

An imported physical function does not automatically become a QEC gadget
implementation. Registration as a `QecInstrImpl` additionally requires:

- an explicit `QecInstr` contract;
- a mapping from gadget ports and code elements to HUGR ports/resources;
- lifecycle and logical/frame-transfer semantics;
- measurement/result role annotations;
- implementation requirements; and
- semantic verification or a declared conformance obligation.

These may come from registered PECOS annotations/sidecars or a user-authored
wrapper. Unannotated gate-pattern recognition is not sufficient.

## Canonical surface representation and porting work

The current Python `SurfacePatch` and Rust `SurfaceCode` are not losslessly
interconvertible. The Rust type currently omits or regenerates information that
Python treats as authoritative, including asymmetric/repetition geometries,
coordinates, check ordering, schedule touches, and ancilla-related metadata.

The new canonical Rust `PatchSpec` must represent or explicitly exclude:

- rotated and standard geometry;
- independent `dx` and `dz`, including distance-1/repetition cases;
- orientation and stable data-site/check IDs;
- exact check supports and ordering;
- coordinates and logical supports; and
- separation of intrinsic code geometry from protocol ancilla layout.

Migration begins with parity fixtures generated from the current Python
representation. The adapter transfers authoritative supports, indices,
coordinates, and schedule metadata; it must not regenerate them through a
different traversal and call the result lossless.

The Rust-first surface work is a named implementation project, not one hidden
MVP bullet. It includes:

1. canonical `PatchSpec` and parity fixtures;
2. CX-based preparation, check scheduling, ancilla assignment, and measurement;
3. program-level detector/observable composition for the supported memory
   experiment;
4. `ProtocolProgram` construction;
5. reference Tick lowering and a separate Tick-to-DEM integration adapter; and
6. PyO3 cursors and typed provider handles.

SZZ, multi-patch CX, Clifford deformation, and other protocols follow in
separate slices after their existing reference limitations are resolved.

## Space-time and visualization

Each resolved gadget may expose a coarse `SpaceTimeShape` and a detailed
portable realization. Shapes carry typed input/output faces, resource
occupancy, partial-order stages, and exact/bounded/estimated/unknown quantities.

Visualization is a read-only projection, `SpaceTimeView`, with stable links to
calls, block instances, code elements, protocol wires, measurements,
detectors, observables, and physical operations. It can show:

- patch geometry at a selected abstract round or scheduled time;
- 2+1D occupancy and interaction trajectories;
- hierarchy and black-box expansion;
- resource lifetimes and alignment constraints; and
- symbolic versus mapped coordinates and time.

Interactive rendering is deferred until the core protocol program and factory
migration work. Bevy remains one possible optional viewer, but dependency-version
alignment elsewhere in PECOS is not itself a justification. The stable product
boundary is renderer-independent `SpaceTimeView` data.

## Roadmap

### Stage 0: retrospective and boundary audit

- Write the `pecos.qeccs`, legacy circuit, `_check_plan`,
  `SurfaceCircuitStep`, renderer, and `LogicalCircuitBuilder` retrospective.
- Settle the standalone `pecos-instr` and PHIR bridge dependency graph.
- Specify canonical serialization, deterministic ID allocation, provider
  matching, and instruction-scoped implementation IDs.
- Define the `SemanticMeasId -> MeasId -> record offset` mapping.
- Produce one tiny generic dataflow fixture before QEC-specific implementation.

### Stage 1: canonical surface memory foundation

- Implement Rust `PatchSpec` and Python parity fixtures.
- Implement the generic graph subset, explicit providers, and use validation.
- Add surface prepare-Z, CX-based syndrome extraction, and measure-Z.
- Add the minimal block state and program-level stabilizer/detector/observable
  composition required by one memory experiment.
- Produce `ProtocolProgram` and a reference TickCircuit.
- Exercise the existing native DEM builder through a separate consumer adapter,
  without making DEM construction part of the program/lowering API.
- Add thin Python cursor bindings only after the Rust path passes.

### Stage 2: additional checked implementations

- Add X-basis memory.
- Resolve the existing SZZ reference limitations, then add SZZ as an alternative
  syndrome implementation.
- Add typed requirements and support diagnostics exercised by real candidates.
- Add PHIR lowering after the bridge boundary is implemented and consumed.

### Stage 3: multi-block Clifford composition

- Add round/epoch alignment constraints.
- Implement the code-element Clifford frame.
- Add transversal H only with correct subsequent check relabeling.
- Add transversal CX only for compatible block frame/layout states.
- Implement cross-block detector and observable propagation.

### Stage 4: first-order hierarchy and import

- Add acyclic, parameterized `InstrModule` definitions if repeated program and
  implementation bodies justify them.
- Add supported HUGR structural import and opaque external modules.
- Demonstrate one explicitly contracted imported HUGR body as a gadget
  implementation.

### Stage 5: structured control and adaptive prerequisites

- Add typed Pauli byproducts and frame-only corrections.
- Add structured branches and bounded repetition.
- Scope or implement the lower-level classical-control representation required
  by direct, PHIR, QIR, or target backends.
- Reject unsupported adaptive lowering without branch specialization.

### Stage 6: space-time tooling and visualization

- Add typed placement/alignment constraints and resource projections.
- Define deterministic `SpaceTimeView` serialization and a headless renderer.
- Evaluate optional interactive renderers only after the semantic artifacts are
  stable.

## Validation and conformance

Each phase owns a verifier:

- authored graph: ports, parameters, use policy, definition/use, and exports;
- resolution: provider availability, implementation support, deterministic
  selection, and acyclic expansion;
- QEC composition: lifecycle, frames, stabilizer flow, detectors, observables,
  and alignment;
- protocol program: resource roles, operation dependencies, cleanup, control,
  semantic measurements, and provenance;
- scheduled plan: target bindings and schedule satisfy the protocol program;
- circuit products: measurement maps, detectors, observables, and declared
  logical behavior are preserved;
- analysis consumers: accepted input level, noise/configuration semantics,
  capabilities, provenance, and rejection behavior are explicit.

Normative examples must either execute in repository tests after implementation
or be explicitly labeled proposed pseudocode. Existing constructors and
references named by an acceptance criterion must be verified against the code.

## Non-goals for the first implementation

- lowering `InstrProgram` to Guppy;
- a general-purpose programming language or unrestricted CFG;
- reusable modules or higher-order functions;
- dynamic control or direct classical-condition lowering;
- SZZ or multi-patch logical gates;
- PHIR, QIR, or MLIR output before their dependency bridge is proven;
- target mapping, routing, calibrated timing, or execution-trace import;
- package management or dynamic plugins;
- space-time editing, automatic architecture synthesis, or an interactive GUI;
- replacing existing factories before semantic parity tests pass.

## Open questions

1. Which parts of `pecos.qeccs` should be adapted, deprecated, or retained?
2. Should canonical `PatchSpec` initially cover every Python patch variant or
   reject some with explicit migration diagnostics?
3. What algebra should replace hand-written detector-boundary and observable
   rules after the memory MVP?
4. Which PHIR QEC operations/types are reusable, and should the existing
   dialect namespace be revised?
5. Which HUGR subset is safe to import structurally without silent loss?
6. Which ergonomic Rust form—typed builders, macros, or both—best overlays the
   core data structure?
7. Should the first control-capable scheduled IR use structured regions whose
   leaves are TickCircuit-like sequences, or lower directly through PHIR?
8. Which higher-level noise-model dialects, if any, justify DEM construction
   before concrete Tick scheduling?
