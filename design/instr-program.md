# Instruction programs for QEC

Status: proposed design.

This is the reader-oriented overview. The
[`InstrProgram` MVP](instr-program-mvp.md) is the normative first
implementation, and [`InstrProgram` rationale](instr-program-rationale.md)
contains detailed Rust, migration, and roadmap notes.

## Why this exists

PECOS should let users describe a QEC experiment as logical operations on code
blocks, choose among compatible implementations, and inspect or lower the
result at several levels. An experiment author should not need to start from a
purpose-built circuit factory, while a QEC implementer should not need to bake
surface-code names into a generic graph container.

The design is HDL-like: an instruction is a cell with an interface and may be
treated as a black box or expanded into a body made from other instructions.
Calls can be recorded before their definitions or implementations are present,
then linked and type-checked later. This combines the open instruction
vocabulary of earlier PECOS circuit containers with SLR's composition of
larger QEC gadgets from smaller primitives. It is not a general programming
language and does not transpile programs to Guppy.

## A surface-memory example

The following is proposed Python API. Rust owns the program and exposes an
equivalent typed builder; Python is a thin ergonomic layer.

```python
surface = SurfaceInstrSet.builtin()
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

linked = program.link(definitions=surface.definitions)
resolved = linked.resolve(
    providers=surface.providers,
    context=SurfaceReferenceContext(),
)
protocol = resolved.to_protocol_program()
generated = protocol.to_tick_program(SurfaceReferenceSchedule())
```

The patch handle advances through state versions:

```text
declared patch
    -> prepare
active patch v1
    -> syndrome extraction
active patch v2
    -> destructive measure
logical result
```

Reusing an older consumed version is an error. The persistent patch identity is
unchanged across these versions, so provenance and visualization can still
recognize one logical block.

`surface.impls.syndrome_cx` is a typed handle supplied by the surface
instruction set. It is not a string and is not part of a universal language
enum. If the user does not choose an implementation, resolution may use an
explicitly configured choice or the only compatible candidate; ambiguity is an
error.

## The core concepts

| Concept | Meaning |
|---|---|
| `InstrProgram` | The generic authored dataflow program. It stores instruction references and calls but does not interpret their names. |
| Instruction declaration | A linkable identity plus its input, output, parameter, and use shape. It is enough to connect a call before its definition is available. |
| QEC instruction definition | The declaration plus its lifecycle, ideal logical effect, frame transfer, and semantic result contract. |
| Implementation | One selectable realization whose body is another `InstrGraph`, possibly at a lower dialect level. |
| Linked program | The program after references have definitions and all call signatures have been checked. |
| Resolved program | The program after every call has a selected implementation and QEC-wide analysis has run. |
| `ProtocolProgram` | A validated portable-dialect view of the same graph model, containing physical/protocol operations, resources, measurements, dependencies, and later structured control. |
| Consumer or backend | A scheduler, compiler, target backend, visualizer, or analysis that accepts a declared program level. |

`InstrProgram`, implementation bodies, and `ProtocolProgram` use the same graph
representation at different linkage and dialect phases. The first can carry
open logical/QEC calls; a selected implementation recursively replaces a call
with a graph of simpler calls; and `ProtocolProgram` certifies that the
remaining graph is in the portable protocol/physical dialect. It is a phase
boundary, not a second incompatible programming model.

## What happens to the example

```text
authored InstrProgram
    declared or unresolved calls and patch state versions
        |
        | link definitions and check call signatures
        v
linked InstrProgram
    typed QEC calls and explicit contracts
        |
        | select implementations
        v
resolved program
    selected gadget bodies, frame transfer,
    semantic measurements, detectors, observables
        |
        | recursively expand and compose graph bodies
        v
ProtocolProgram
    same graph model, certified portable dialect,
    resource lifetimes,
    dependencies, provenance
        |
        +----------------------+----------------------+
        |                      |                      |
        v                      v                      v
      PHIR             reference scheduler     target backend
                               |                      |
                               v                      v
                         TickCircuit       scheduled target artifact
```

Analysis tools attach to the level they understand. For example, the existing
physical DEM path consumes a generated `TickCircuit`, its measurement and
detector bindings, and a physical noise model. DEM construction is not a method
of `InstrProgram`, a gadget, or `ProtocolProgram`.

## Linking instructions and expanding implementations

Authoring is open-world. A call may refer to an instruction whose definition or
implementation will be supplied later, as in an HDL module declaration or the
older PECOS `QuantumCircuit`. The call still carries a declared port and
parameter shape so values can be connected. Linking binds the stable reference
to a definition and rejects missing definitions or signature mismatches before
implementation selection. Typed instruction-set builders provide declarations
up front and therefore catch most mistakes while authoring; the graph itself
does not dispatch on display names.

A QEC instruction declares named code-block and classical ports, canonical
parameters, lifecycle effects, its ideal decoded logical action, logical-frame
transfer, and semantic result roles. Preparation, syndrome extraction, gates,
code switching, merge/split, and destructive measurement all fit this model;
their input and output block types and arities may differ.

An implementation is checked against the complete bound call, including its
parameters and block types. Its normal body is an ordinary `InstrGraph`, with
formal ports bound to the call's values. That body may itself call other
instructions, so the same HDL builds physical primitives into syndrome rounds,
rounds into QEC gadgets, and gadgets into experiments. Expansion continues
until every remaining call is accepted as a primitive by the selected consumer
or is an explicitly supported opaque external. Recursive definition cycles are
an error unless represented by a future structured repetition construct.

The standard PECOS `GateType` operations should be exposed as a built-in typed
instruction library rather than hard-coded into the generic graph. They supply
the first physical leaves for portable implementations. A normal inspectable
body is not an already scheduled `TickCircuit`: it preserves resource
lifetimes, dependencies, semantic measurements, hierarchy, and, later,
structured control. A flat Tick-producing adapter is acceptable for a
deliberately straight-line leaf, but it cannot define the general
implementation contract.

## Instructions as space-time cells

An instruction call is also a cell instance that can acquire a space-time
realization. Typed resource and value wires enter through its input ports and
replacement resources or results leave through its output ports. Its selected
implementation places child cells inside the volume and connects their ports.
A port's use policy states whether an input is consumed or reusable; a
resource-transforming call returns a successor wire version rather than
mutating the incoming one.
A code block is an extended spatial cross-section carried through time, while
physical qubits and classical values appear as finer world-lines when the cell
is expanded.

![A coarse QEC experiment volume containing preparation, three syndrome-round
cells, and measurement, with one round expanded into lower-level child
cells.](instr-program-spacetime.svg)

The semantic instruction does not have one universal fixed box. Parameters,
implementation choice, scheduling constraints, and target mapping can produce
different realizations of the same call. A realization therefore progresses
from unplaced, through constrained, to placed. Parent/child identities and
local coordinates preserve hierarchy so a tool can show only the envelope or
peek through to the physical circuit. Frame-only operations may have a
degenerate or zero-volume realization.

## Dataflow, concurrency, and control

Data and resource dependencies define required ordering. Independent calls are
unordered and therefore eligible for concurrent scheduling. The initial model
does not need a generic `parallel` region.

Users who want manual control may add typed constraints such as before/after,
round alignment, mutual exclusion, or relative space-time placement. These
restrict legal schedules without pretending to assign authoritative device
time or target resource addresses.

Structured branches and bounded repetition are later extensions. They must
have explicit arguments, yielded values, and frame-state joins. `TickCircuit`
remains a straight-line scheduled representation. A controlled
`ProtocolProgram` can lower to Tick only after specialization, unrolling, or a
proved frame-only rewrite; otherwise it needs a control-capable IR or backend.

## QEC-wide state and analysis

Some semantics do not belong to an individual call. Detectors and observables
can span preparation, syndrome rounds, logical gates, multiple blocks, and
terminal measurement. The resolved program therefore runs a QEC composition
pass over all selected implementation bodies.

That pass owns stabilizer/check flow, detector construction, observable
propagation, preparation and measurement boundaries, frame evolution, and
round or epoch alignment. Each gadget body emits semantic measurements and
transfer relations; it does not finalize cross-call detectors by itself.

Each active code block also carries a logical Pauli frame and a code-element
Clifford/check frame. Target-mapped physical frames belong to a later artifact.
This keeps logical and code-level state meaningful before physical resources
are assigned.

## Outputs and consumers

PHIR is the preferred inspectable compiler output once its QEC dialect and
crate boundary are settled. Appropriate QIR or quantum MLIR dialect exports may
follow. They should consume the resolved program, PHIR, or `ProtocolProgram`
rather than reimplementing QEC gadget expansion.

A PECOS reference scheduler lowers supported straight-line protocol programs to
a normalized `TickCircuit`. A target backend may instead bind resources,
route, rebase, and schedule the portable program. It may reject an infeasible
program but must not silently change its selected QEC implementation or logical
semantics.

An analysis consumer declares its accepted input level and configuration. A
physical DEM compiler can consume a generated Tick artifact plus a physical
noise model. A future gadget- or protocol-level DEM compiler is possible only
if it defines how its noise model treats hierarchy, symbolic schedules,
branches, correlations, and timing. Unsupported semantics must be rejected
rather than guessed.

Guppy is input-only. A Guppy function may be compiled to HUGR and imported as a
supported structural graph or an opaque external body. It becomes a QEC gadget
implementation only when paired with an explicit QEC contract, port/resource
mapping, lifecycle and frame semantics, measurement roles, and a conformance
obligation. PECOS does not infer logical QEC meaning from unannotated gate
patterns.

## Decided

- Rust owns authoritative graphs, validation, resolution, QEC composition,
  serialization, and reference lowering; Python remains thin.
- The graph is first-order dataflow, not a general-purpose language or
  unrestricted control-flow graph.
- Authoring may precede definition; linking makes every call typed before
  implementation selection or lowering.
- Instruction identities are generic; dialects provide names and semantics.
- Implementation choices are typed and instruction-scoped.
- Normal implementation bodies use the same `InstrGraph` representation and
  may recursively call lower-level instructions.
- Standard PECOS gates form a built-in instruction library, not generic graph
  syntax.
- Consumed resource versions cannot be reused, without exposing a general
  linear-type system to authors.
- Concurrency follows from dataflow; explicit constraints refine scheduling.
- `ProtocolProgram` is portable and target-independent.
- `TickCircuit` stays straight-line.
- DEM generation is a separate analysis/compiler concern.
- Guppy is not a lowering target.

## Deferred or open

The first implementation deliberately defers general user packages, structured
control, PHIR integration, additional patch geometries and logical gates,
target mapping, generalized stabilizer-flow algebra, placed space-time tooling,
interactive visualization, and dynamic providers. It preserves graph
hierarchy and cell identity so those tools do not require a later IR rewrite.

Open architectural questions include the exact PHIR QEC dialect, the first
control-capable scheduled representation, whether any higher-level noise
dialects justify pre-Tick DEM construction, and which ergonomic Rust layers
should complement the core builders.

The normative first slice and exact acceptance tests are in
[`instr-program-mvp.md`](instr-program-mvp.md). Historical context, detailed
Rust identities, crate boundaries, alternatives, and the longer roadmap are in
[`instr-program-rationale.md`](instr-program-rationale.md).
