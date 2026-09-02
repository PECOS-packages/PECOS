# Typed instruction programs for QEC

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

The design is HDL-like: a gadget has a typed interface and may be treated as a
black box or expanded into a lower-level implementation. It builds on lessons
from SLR and earlier PECOS circuit models, but it is not a general programming
language and does not transpile programs to Guppy.

## A surface-memory example

The following is proposed Python API. Rust owns the program and exposes an
equivalent typed builder; Python is a thin ergonomic layer.

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

## The six core concepts

| Concept | Meaning |
|---|---|
| `InstrProgram` | The authored typed dataflow program. It does not know QEC instruction names. |
| QEC instruction | A typed gadget contract: inputs, outputs, parameters, lifecycle, and ideal logical effect. |
| Implementation | One selectable realization of a QEC instruction. |
| Resolved program | The program after every call has a selected implementation and QEC-wide analysis has run. |
| `ProtocolProgram` | A portable lower-level Rust program of physical/protocol operations, resources, measurements, dependencies, and later structured control. |
| Consumer or backend | A scheduler, compiler, target backend, visualizer, or analysis that accepts a declared program level. |

`InstrProgram` and `ProtocolProgram` use the same general graph idea at
different dialect levels. The first carries logical/QEC calls; the second
carries lower-level protocol and physical operations. This is how the design
keeps the container generic without flattening every gadget immediately.

## What happens to the example

```text
authored InstrProgram
    typed QEC calls and patch state versions
        |
        | resolve implementation choices
        v
resolved program
    selected gadget bodies, frame transfer,
    semantic measurements, detectors, observables
        |
        | expand and compose lower-level bodies
        v
ProtocolProgram
    portable operations, resource lifetimes,
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

## Instructions and implementations

A QEC instruction declares named code-block and classical ports, canonical
parameters, lifecycle effects, its ideal decoded logical action, logical-frame
transfer, and semantic result roles. Preparation, syndrome extraction, gates,
code switching, merge/split, and destructive measurement all fit this model;
their input and output block types and arities may differ.

An implementation is checked against the complete bound call, including its
parameters and block types. Resolving an implementation produces an
inspectable lower-level graph body or an explicitly opaque external body. A
normal inspectable body is not an already scheduled `TickCircuit`: it uses a
portable protocol/physical dialect that can represent resource lifetimes,
dependencies, semantic measurements, and, later, structured control.

Rust code may build such a body programmatically. A flat Tick-producing adapter
is acceptable for a deliberately straight-line leaf, but it cannot define the
general implementation contract.

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
- Instruction identities are generic; dialects provide names and semantics.
- Implementation choices are typed and instruction-scoped.
- Consumed resource versions cannot be reused, without exposing a general
  linear-type system to authors.
- Concurrency follows from dataflow; explicit constraints refine scheduling.
- `ProtocolProgram` is portable and target-independent.
- `TickCircuit` stays straight-line.
- DEM generation is a separate analysis/compiler concern.
- Guppy is not a lowering target.

## Deferred or open

The first implementation deliberately defers reusable modules, structured
control, PHIR integration, additional patch geometries and logical gates,
target mapping, generalized stabilizer-flow algebra, space-time tooling,
visualization, package management, and dynamic providers.

Open architectural questions include the exact PHIR QEC dialect, the first
control-capable scheduled representation, whether any higher-level noise
dialects justify pre-Tick DEM construction, and which ergonomic Rust layers
should complement the core builders.

The normative first slice and exact acceptance tests are in
[`instr-program-mvp.md`](instr-program-mvp.md). Historical context, detailed
Rust identities, crate boundaries, alternatives, and the longer roadmap are in
[`instr-program-rationale.md`](instr-program-rationale.md).
