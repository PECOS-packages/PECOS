# Typed instruction programs for logical QEC and physical lowering

Status: proposed.

## Summary

PECOS should support constructing programs from typed resources and unresolved
instruction applications, then selecting implementations in an explicit
lowering environment. Logical QEC is the first motivating dialect: users create
abstract surface-code patches and logical operations, then lower the program to
Guppy, `TickCircuit`, `DagCircuit`, Stim, and detector error models.

The generic authoring model is an `InstrProgram` containing reusable
`InstrModule` definitions and appendable `InstrGraph` bodies. It does not have
built-in knowledge of instruction names. `InstrDef` objects declare typed
ports, parameters, effects, and domain semantic interfaces; `InstrImpl` objects
describe selectable realizations. QEC adds code-block types, logical
transformations, Pauli-frame transfer, protocol planning, and surface-specific
implementation support without changing the generic graph container.

A surface-code convenience API should live under
`pecos.qec.surface` and use `SurfacePatch` as its only source of surface-code
geometry. SLR is useful precedent for separating authoring from rendering, but
the new API should not depend on SLR. The existing SLR surface gate library is
incomplete, and its generic representation is lower-level than users need for
logical QEC experiments.

The generic graph must not become a second general-purpose PECOS IR. PHIR
already has modules, regions, blocks, SSA operands/results, custom dialects,
and extensible types. `InstrProgram`/`InstrGraph` should be a deliberately
smaller high-level authoring, elaboration, and implementation-resolution model
that can generate PHIR after its domain choices are resolved. It may reuse PHIR
types and structural utilities where that keeps the boundary clean, but it
should not duplicate PHIR's general compiler responsibilities. QEC definitions,
semantics, physical planning, direct TickCircuit lowering, and Guppy generation
remain in `crates/pecos-qec`.
PyO3 bindings expose the Rust-owned artifacts, while `quantum-pecos` supplies
only Pythonic construction conveniences and the bridge into Guppy's Python
compiler.

The core direction is:

```text
Typed resources + opaque instruction applications
                         |
                         v
      InstrProgram / InstrModule / InstrGraph
        (compact high-level structure)
                         |
                         v
   parameter/type/hierarchy/dialect elaboration
                         |
                         v
             ElaboratedInstrProgram
                         |
                         v
       explicit InstrSet resolution
       + QEC semantic verification
                         |
                         v
            resolved instruction program
                         |
             +-------------+
             |             |
             v             v
           PHIR    PhysicalCircuitPlan
     (MLIR-like target)      |
                       +-----+-----+
                       |           |
                       v           v
                  direct Rust    Guppy
                    lowering       |
                       |           v
                       |    HUGR / QIS lowering
                       |           |
                       |           v
                       |    runtime QIS trace
                       |           |
                       +-----+-----+
                             |
                             v
                   normalized TickCircuit
                             |
                             v
                     DagCircuit / DEM

              optional exports:
              TickCircuit -> Stim circuit
              DEM -> Stim-compatible DEM text

              optional import:
              Guppy -> HUGR -> InstrModule
                (supported structural subset or opaque external module)
```

Purpose-specific functions such as `make_surface_code` remain convenient
shortcuts, but should eventually be implemented by constructing this IR rather
than owning separate experiment-generation logic.

## Motivation

Today PECOS contains most of the required pieces, but they are split across
layers:

- `pecos.qec.surface.SurfacePatch` is the canonical surface-code geometry. It
  owns stabilizer supports, logical supports, layout, orientation, and schedule
  metadata.
- `pecos.qec.surface.LogicalCircuitBuilder` composes memory segments,
  transversal H, transversal S/S-dagger, transversal CX, and injection
  protocols. Its source of truth is currently a `TickCircuit`; it can also
  produce DAG, Stim, DEMs, and decoder inputs.
- `pecos.guppy_gen.surface` generates reusable Guppy patch primitives and
  specific memory experiments.
- `pecos.guppy_gen.transversal` separately generates specific two-patch
  transversal-CX experiments.
- SLR can render generic physical programs to Guppy, but its surface logical
  operations are largely placeholders.

Consequently, a user can express a logical experiment or obtain a Guppy
program, but cannot use one supported abstraction to do both. Adding another
specific experiment factory makes this split worse.

There is older PECOS precedent for the generic shape. The original
`QuantumCircuit` stored gate symbols, locations, and parameters without owning
their executable meaning. A simulator's `gate_dict` resolved each symbol to a
callable when the circuit ran. The modern `GateRegistry` similarly stores
custom gate signatures and decompositions for later expansion. The proposed
model retains that useful late binding while replacing string dispatch with
stable definition identities, typed SSA values, explicit outputs, structured
control, hierarchy, semantic contracts, and deterministic target-aware
resolution.

### Current documentation and the missing layer

The existing documentation does show how to obtain Guppy for surface-code
experiments, but only through specific generators. In particular,
`docs/user-guide/qec-guppy.md` documents `make_surface_code` for memory
experiments and `make_css_transversal_cnot` for a fixed transversal experiment.
`docs/user-guide/qec-geometry.md` points from QEC geometry to those factories,
or to manual low-level SLR construction. `docs/development/slr-qeclib.md`
documents SLR's multi-backend block representation.

What is not documented—because PECOS does not yet provide it as one supported
abstraction—is the desired path from `SurfacePatch` instances through an
arbitrary composition of abstract logical/QEC operations to generated Guppy.
This proposal defines that missing layer rather than documenting another
factory-specific recipe.

## Goals

1. Let users declare one or more `SurfacePatch` instances and compose supported
   logical operations on them.
2. Keep instruction names and domain semantics out of `InstrGraph` itself.
3. Separate a QEC instruction's semantic contract from the QEC protocol or
   physical implementation that realizes it.
4. Make every instruction declare its typed QEC-block inputs and outputs and
   its intended logical transformation, including explicit identity actions.
5. Represent preparation, syndrome extraction, logical gates, destructive
   measurement, injection, and surgery uniformly as QEC instruction applications.
6. Generate executable Guppy while retaining patch structure and linear qubit
   ownership.
7. Reach PECOS-native circuit and analysis forms directly in Rust as well as
   through the executed Guppy-to-QIS trace path.
8. Preserve measurement identity, detector definitions, observables, result
   tags, and code-block ownership across every lowering.
9. Reuse the canonical geometry, check plans, Clifford-deformation rules, and
   ancilla scheduling already under `pecos.qec.surface`.
10. Keep existing factory APIs and `LogicalCircuitBuilder` programs working.
11. Leave room for color codes, small block codes, and heterogeneous protocols
   without forcing surface-specific concepts into the generic IR.
12. Make the generic instruction substrate usable by physical quantum,
   classical, pulse, calibration, and other PECOS dialects without requiring
   them to depend on QEC.
13. Reuse or strengthen PHIR's structural IR rather than creating a parallel
   module/region/SSA representation.
14. Import supported compiled Guppy/HUGR functions as typed `InstrModule`
    definitions or opaque external modules without claiming to infer missing
    high-level QEC semantics.

## Non-goals for the first version

- Lowering every form of adaptive measurement-dependent control in the first
  version; the generic structure may represent and validate more than every
  initial backend supports.
- A new general compiler IR replacing PHIR, HUGR, or SLR.
- Automatic lattice-surgery synthesis.
- Supporting operations whose detector-boundary semantics are not defined.
- Converting an arbitrary `TickCircuit` back into structured Guppy.
- Losslessly translating every arbitrary Guppy/HUGR program into the deliberately
  smaller `InstrGraph` model, or recovering logical QEC instructions from
  unannotated physical gate patterns.
- Removing existing purpose-specific factories.

## Rust-first ownership and thin Python API

PECOS should have one implementation of this model, in Rust. Python must not
maintain a parallel instruction graph, type checker, implementation resolver,
surface planner, or serialization format. A small Rust instruction-program
layer owns generic authoring and resolution; QEC semantics live in
`crates/pecos-qec`; a Rust lowering generates `pecos-phir`; PyO3 bindings live
in `python/pecos-rslib`; and the `pecos` Python package re-exports or lightly
wraps the bound types.

### Rust core

The conceptual ownership should be along these lines:

```text
pecos_instr (crate/module name provisional)
    InstrProgram, InstrModule, InstrGraph, InstrCall
    InstrDef, InstrSet, InstrImpl, generic resolver interfaces
    InstrDefId, ImplDefId, ModuleDefId, ModuleInstanceId
    ValueId, CallId, RegionId
    symbol tables, hierarchy, typed SSA/linearity validation
    dialect schemas, annotations, provenance, serialization

pecos_phir::instr_lowering
    ResolvedInstrProgram -> PHIR Module/Function/Region/Instruction
    typed dialect operation and type emission
    source/provenance maps back to instruction instances and values

pecos_qec::instr
    QecInstr, QecInstrSet, QecInstrImpl, QecInstrPlan
    QecInstrSemantics, QecTypeExpr, QecLogicalTransform
    QecFrameTransfer, QEC-specific support constraints
    type variables, substitution, constraints, and unification

pecos_qec::surface
    SurfacePatch / surface geometry
    built-in surface instructions and implementations
    SZZ and CX syndrome-extraction plans
    transversal logical implementations

pecos_qec::spacetime
    SpaceTimeProgram, volume constraints, SpaceTimePlan

pecos_qec::physical
    PhysicalCircuitPlan, scheduling context, GeneratedTickProgram
    direct normalized TickCircuit and native DEM handoff

pecos_qec::guppy
    deterministic Guppy source and semantic sidecar generation
```

The exact crate/module split can evolve after a focused PHIR integration audit.
The important boundary is semantic: `InstrProgram` contains only the
definition/instance/value/region structure needed for high-level composition
and resolution, while PHIR owns the general compiler IR generated from it.
Stable IDs should be Rust newtypes rather than Python object identity. Core artifacts and
diagnostics should use typed Rust enums/structs and have versioned `serde`
representations so Rust, Python, a future HDL, and stored experiment artifacts
all share the same schema.

PHIR's current `CustomOp`, `CustomType`, and dialect registration provide the
right lowering targets but not the complete high-level contract needed here.
`InstrProgram` must add stable definition references, named typed ports,
parameter schemas, value linearity/copyability, implementation candidates,
support diagnostics, and deterministic resolution. PHIR emission must use
registered typed dialect operations/types and preserve source IDs; core QEC
facts must not be smuggled through unvalidated string attributes.

`QecInstrImpl` should be a Rust trait implemented by built-in strategies. An
explicit `QecInstrSet` owns the implementations available to one compilation;
there is no process-global plugin registry. Python callbacks must not participate
in built-in resolution or planning, because that would make the Rust API a
second-class path and complicate determinism, threading, and serialization.
Future third-party implementations should cross a deliberate serialized or
plugin boundary rather than being arbitrary objects hidden inside the IR.

### Canonical surface geometry

Rust-first also applies to `SurfacePatch`. The existing Python
`pecos.qec.surface.SurfacePatch` currently contains geometry and scheduling
behavior not all represented by Rust's `pecos_qec::SurfaceCode`. The new work
should not copy more logic into both places. It should establish a canonical
Rust patch/specification type with the geometry, orientation, stabilizers,
logical supports, and schedule metadata required by instruction planning, then
make the Python `SurfacePatch` a thin bound wrapper or compatibility facade.

A temporary conversion from the existing Python patch descriptor into the Rust
type is acceptable during migration, but it must be lossless, validated, and
removed once parity is reached. Rust and Python must not independently compute
stabilizers or implementation support decisions.

The canonical type should distinguish the code's intrinsic geometry from any
particular extraction circuit. A useful layered model is:

1. `CodeSpec` describes the algebraic code: physical data-qubit identities,
   stabilizers or gauges, logical operators, encoded interface, and code
   parameters.
2. `CodeGeometry` optionally embeds those data qubits and code features in a
   geometric space. A planar topological code can provide a two-dimensional
   cellulation, embedded graph, or integer-grid view; a code with no useful
   spatial embedding may omit this interface.
3. `PatchSpec` selects a finite code region, boundaries, orientation, defects,
   and logical-interface faces. `SurfacePatch` is the surface-code realization
   of this layer and exposes a stable two-dimensional data-qubit view.
4. `ProtocolLayout` adds the implementation resources needed around the code,
   such as measurement ancillas, flag qubits, buses, routing workspace, and
   classical-control endpoints. It may remain relative or be refined into a
   target-specific `PhysicalLayout`.
5. `SpaceTimeRealization` describes the selected implementation through time
   with preparation, interaction, measurement, reset, idle, and feedback
   events. It can be inspected as an abstract `SpaceTimeShape` or as a detailed
   physical circuit embedded in space and time.

The first three layers describe what code block exists; the latter two describe
how a selected QEC instruction is realized. In particular, syndrome ancillas
must not become part of `SurfacePatch` merely because one extraction protocol
uses them. Different SZZ, CX-based, flagged, or hardware-specific extraction
implementations may refine the same data-qubit patch into different physical
layouts and space-time shapes.

`CodeGeometry` should not require every code to be a rectangular grid. Its
portable core should be stable element identities, incidence/adjacency, optional
coordinates, dimensionality, and named boundary or logical-interface features.
Grid coordinates are an ergonomic surface-code view over that core. Coordinates
may be exact lattice coordinates, symbolic/relative coordinates, or absent;
target placement is a later mapping. This avoids conflating an abstract planar
code drawing with calibrated device coordinates.

### PyO3 and Python responsibilities

Bindings should be added to `python/pecos-rslib` under `pecos_rslib.qec`, using
the same wrapper pattern as existing `StabilizerCode`, circuit, and fault-
tolerance bindings. Bound objects own or reference their Rust counterparts;
Python exceptions are translations of structured Rust error variants.

The public `pecos.instr`, `pecos.qec`, and `pecos.qec.surface` modules provide:

- generic `InstrProgram`, `InstrModule`, and `InstrGraph` builders;
- imports and Python type annotations for the bound Rust types;
- keyword-friendly constructors and instruction lookup;
- `LogicalBlock` cursor objects for the ergonomic imperative API;
- `parallel()` and other context managers that submit structural regions to
  the Rust builder;
- small convenience profiles such as `SurfaceInstrSet.szz_transversal()`;
- Guppy module loading/compilation and existing runtime orchestration.

A Python `LogicalBlock` cursor contains a Rust graph handle plus a `BlockId`;
it does not own a separate current-value model. `append`, `apply`, validation,
default resolution, and introspection delegate immediately to Rust. The exact
same Rust program can therefore be authored from Rust directly, from Python,
or later from an HDL.

### Guppy boundary

Guppy is a Python language/toolchain, so a completely Python-free final step is
not necessary. Rust should nevertheless own all semantic decisions and produce
a `GeneratedGuppySource` artifact containing deterministic source plus the
measurement, detector, observable, instruction-call, and provenance sidecars.
The thin Python layer compiles/loads that source with Guppy and passes the
result into the existing HUGR/QIS runtime path.

No Python renderer should independently walk the instruction graph and decide how
an instruction is implemented. If a short-term migration reuses existing
Python Guppy emission, it must consume a fully resolved serialized Rust plan and
be treated as a replaceable bridge, not an alternate source of semantics.

### Rust and Python API parity

Both public APIs should follow the same sequence: obtain a typed instruction-call
builder from the instruction set, bind its named ports and parameters, optionally
choose `.using(...)`, and append the bound call. Rust uses fluent typed setters
where Python can use keyword arguments.

Rust:

```rust
let surface = SurfaceInstrSet::szz_transversal();
let mut program = InstrProgram::new([surface.instr_set()]);
let graph = program.main_mut();

let data = graph.block("data", SurfacePatch::rotated(3)?)?;
let ancilla = graph.block("ancilla", SurfacePatch::rotated(3)?)?;

graph.parallel(|region| {
    region.append(surface.prepare().patch(data).basis(Basis::Z))?;
    region.append(surface.prepare().patch(ancilla).basis(Basis::X))?;
    Ok(())
})?;

graph.append(surface.syn_extract().patch(data).rounds(3))?;
graph.append(surface.h().patch(data))?;
graph.append(surface.cx().control(data).target(ancilla))?;

let data_x = graph.append(surface.measure().patch(data).basis(Basis::X))?;
graph.export("data_x", data_x)?;
```

Python:

```python
surface = SurfaceInstrSet.szz_transversal()
program = InstrProgram(instruction_sets=[surface])
graph = program.main()

data = graph.block("data", SurfacePatch.rotated(3))
ancilla = graph.block("ancilla", SurfacePatch.rotated(3))

with graph.parallel():
    graph.append(surface.prepare(patch=data, basis="Z"))
    graph.append(surface.prepare(patch=ancilla, basis="X"))

graph.append(surface.syn_extract(patch=data, rounds=3))
graph.append(surface.h(patch=data))
graph.append(surface.cx(control=data, target=ancilla))

data_x = graph.append(surface.measure(patch=data, basis="X"))
graph.export("data_x", data_x)
```

The spelling is intentionally parallel: `patch`, `basis`, `rounds`, `control`,
and `target` are the same ports/parameters backed by the same Rust definitions.
Python must not rename concepts merely to appear more idiomatic. Conversely,
Rust should provide generated or handwritten typed call builders rather than
forcing users to construct string maps for common instructions.

Both APIs also retain a generic escape hatch for tooling: Rust can append a
`BoundInstrCall`, while Python can call `apply(instr, **ports)`. Every public
operation delegates to Rust and must produce byte-for-byte-equivalent
serialized artifacts and equivalent structured errors.

## Minimal generic instruction model

The generic layer should remain deliberately small:

- `InstrProgram` owns imported instruction sets, reusable module definitions,
  entry points, exports, and program-level metadata.
- `InstrModule` owns a named typed interface, parameters, requirements, and one
  or more `InstrGraph` regions.
- `InstrGraph` owns typed `ValueRef` edges, `InstrCall` nodes, structural
  control regions, and exports. `append` validates a bound call and returns its
  typed result values.
- `InstrDef` owns a stable identity, named input/output ports, parameters,
  generic traits/effects, and typed dialect semantic interfaces.
- `InstrSet` owns definitions, candidate `InstrImpl` objects, explicit
  defaults/preferences, and version identity.
- `InstrImpl` answers whether it supports a bound call in a lowering context
  and produces a serializable resolved plan or dialect lowering.

The graph understands generic value properties such as type, linearity,
copyability, definition/use, region ownership, and source identity. It does not
understand QEC patches, qubits, pulses, decoder meanings, or operation names.
Those belong to registered dialect interfaces. A program may import multiple
dialects, allowing a later phase to compose QEC, physical quantum, and
classical instructions without putting all of their concepts in one enum.

This is intentionally less abstract than a general IR. The first version does
not need arbitrary memory models, exceptions, object systems, unrestricted
CFG construction, or a generic optimization framework. When resolved domain
instructions need those facilities, lowering generates PHIR operations,
regions, types, functions, and modules. PHIR then owns general analyses and
compiler transformations.

The generic resolution order mirrors the QEC-specific policy developed below:
an explicit call constraint wins, then an explicit program/profile preference,
then a sole supported candidate; zero or multiple remaining candidates are
errors. A dialect can refine `supports`, plan contents, and semantic
verification, but cannot change that deterministic selection contract.

The relationship to existing artifacts is therefore:

```text
Original QuantumCircuit       symbol + locations + params
Current GateRegistry          gate signature + one decomposition
InstrProgram / InstrGraph     typed definition calls + selectable implementations
ResolvedInstrProgram          every implementation and support decision fixed
PhysicalCircuitPlan           shared structured physical realization + metadata
PHIR                          general MLIR-like compiler representation
TickCircuit / DagCircuit      concrete scheduled quantum circuit representations
```

## QEC instructions and implementations

A QEC instruction describes **what** should happen and gives the operation
a typed input/output contract. A protocol implementation describes **how** it
happens for particular operand codes and a particular lowering environment.
The spelling “QEC instruction” is deliberate: it avoids collision with Python's
`typing.Protocol` while preserving “QEC protocol” for an actual fault-tolerant
realization.

For example, these are one instruction with different implementations:

```text
LogicalCX(control, target)
    - surface.transversal_cx
    - surface.lattice_surgery_cx
    - color.transversal_cx
    - heterogeneous.code_switch_cx
```

Preparation, syndrome extraction, measurement, logical gates, injection, and
lattice surgery are therefore all QEC instructions. None is privileged circuit
syntax. A surface-code syndrome round is one protocol implementation of a more
general instruction for preserving/error-correcting a block.

Definitions and implementations belong to explicit `QecInstrSet` objects. They
do not become methods on `SurfacePatch`: a patch is an immutable code
specification, while an instruction set is a named collection of operation
contracts and lowering strategies. This allows two experiments using the same
patch geometry to choose different QEC protocols or implementation strategies.

Conceptually:

```python
@dataclass(frozen=True)
class InstrDef:
    definition_id: InstrDefId
    name: QualifiedInstrName
    inputs: tuple[InstrInput, ...]
    outputs: tuple[InstrOutput, ...]
    parameters: tuple[InstrParameter, ...] = ()
    traits: tuple[InstrTrait, ...] = ()
    semantic_interfaces: tuple[DialectSemantics, ...] = ()


@dataclass(frozen=True)
class QecInstrSemantics(DialectSemantics):
    logical_transform: QecLogicalTransform
    logical_frame_transfer: QecFrameTransfer


@dataclass(frozen=True)
class QecInstr:
    definition: InstrDef
    semantics: QecInstrSemantics


class QecInstrImpl(InstrImpl):
    implementation_id: str
    instruction: QecInstr

    def supports(
        self,
        inputs: tuple[BoundInstrInput, ...],
        context: LoweringContext,
    ) -> SupportResult: ...

    def plan(
        self,
        call: InstrCall,
        operands: tuple[ValueRef, ...],
        context: LoweringContext,
    ) -> QecInstrPlan: ...
```

The minimal generic definition contract is **stable identity, name, typed
inputs, typed outputs, parameters, traits, and dialect semantic interfaces**.
The QEC interface requires an intended logical transformation and
logical-frame transfer. Frame transfer may be canonically derived for standard
identity, preparation, measurement, and Clifford transforms, but it is still
explicit in the elaborated artifact. Parameters describe non-resource
configuration such as round count or basis; they do not replace typed input
ports. Input and output ports are named so diagnostics and generated metadata
can say `control`, `target`, `syndrome`, or `measured_patch` rather than only
operand 0 or result 1. Physical-frame effects belong to
`QecInstrImpl`/`QecInstrPlan`, because they depend on the selected code
realization and mapping.

The name belongs to the QEC instruction and its owning set. It may have a
stable qualified representation such as `pecos.surface/cx@1` for serialization,
but the graph stores it as opaque identity and never parses or dispatches on
the name.

### Code-block types and logical transformations

The primary QEC resource flowing through an instruction graph is a typed QEC code
block. Conceptually, a block type identifies at least its code specification,
number and organization of encoded logical qubits, and instruction-owned
lifecycle state:

```python
QecBlockType(code=P, logical=OneQubit, state=Active)
```

`LogicalPatch[P]` in the surface examples is a geometry-bearing refinement of
such a block type. It is not merely a Python annotation on an arbitrary value.
The instruction signature defines which block types are consumed and which new
block versions are produced. It may also expose non-block ports, such as a
logical measurement result, syndrome information, a classical decision, or a
decoder handle.

The port-type transformation and logical transformation are related but
distinct:

- syndrome extraction or memory can preserve the block type and declare the
  identity logical transformation;
- logical H preserves the block type but declares a non-identity logical H;
- logical CX preserves two block types while declaring a joint two-block
  transformation;
- code deformation can change the patch geometry while declaring identity on
  the encoded logical information, including the chosen logical-basis map;
- preparation consumes a declared/uninitialized block and returns an active
  block with a specified logical state;
- destructive measurement consumes an active block and returns a typed logical
  measurement result;
- merge, split, code switching, and gauge fixing may change block arity or code
  type and declare an isometry, channel, measurement, or more general logical
  relation between their input and output interfaces.

Identity must be explicit. An absent transform means “semantics unspecified”
and should not be accepted for an instruction that participates in semantic
verification. In particular, syndrome extraction is not assumed to be logical
identity just because its input and output block types happen to match.

`QecLogicalTransform` should be an extensible semantic contract rather than a
gate-name enumeration. Initial forms should cover identity, preparation,
measurement, Clifford/symplectic maps, composition, conditional transforms,
and Pauli byproducts. Later forms can express general logical channels and
block-interface isometries.
The generic graph need not dispatch on these forms to append a call; semantic
analysis and verification consume them through a separate interface.

Every `QecInstrImpl` has a proof obligation: its generated
`QecInstrPlan` must realize the instruction's declared block-boundary and
logical transformation for every input accepted by `supports`. This obligation
is stronger than producing a type-correct physical circuit and gives #513 a
precise property to verify.

## QEC instruction signatures and patch constraints

QEC instruction port types may be concrete or parameterized. A type variable binds
properties from an input and can be referenced by later inputs and outputs.
This supports three useful levels of specificity without changing the generic
instruction model.

### Generic over patch parameters

```python
P = TypeVar("P", bound=SurfacePatchSpec)

surface.h = QecInstr(
    name="surface/h",
    inputs=(Input("patch", LogicalPatch[P]),),
    outputs=(Output("patch", LogicalPatch[P]),),
    logical_transform=LogicalClifford.h(on="patch"),
)
```

The semantic instruction accepts a surface patch of any parameters and preserves
its code specification. A transversal-H implementation can refine support to
square rotated patches, while a different implementation may accept a broader
set. Those restrictions belong to `implementation.supports`, not to the
graph.

Syndrome extraction can have the same block boundary while making its intended
identity action explicit:

```python
surface.syn_extract = QecInstr(
    name="surface/syn_extract",
    inputs=(Input("patch", LogicalPatch[P, Active]),),
    outputs=(Output("patch", LogicalPatch[P, Active]),),
    parameters=(Parameter("rounds", PositiveInt),),
    logical_transform=LogicalIdentity(input="patch", output="patch"),
)
```

The selected implementation may emit internal syndrome and decoder sidecar
information, but it must preserve the encoded logical state according to this
declared map. Any such value intended for composition by later instructions
must instead be declared as a typed output port.

The name `syn_extract` deliberately does not claim that decoding or correction
has occurred. Decoder invocation, classical feedback, and Pauli-frame updates
can be separate typed instructions or explicit implementation-plan effects. A
higher-level convenience may compose them into an error-correction routine
without changing the primitive instruction's meaning.

### Relations between multiple inputs

```python
C = TypeVar("C", bound=SurfacePatchSpec)
T = TypeVar("T", bound=SurfacePatchSpec)

surface.cx = QecInstr(
    name="surface/cx",
    inputs=(
        Input("control", LogicalPatch[C]),
        Input("target", LogicalPatch[T]),
    ),
    outputs=(
        Output("control", LogicalPatch[C]),
        Output("target", LogicalPatch[T]),
    ),
    logical_transform=LogicalClifford.cx(control="control", target="target"),
)
```

The instruction says that CX consumes and returns both logical patches without
changing their specifications. Its transversal implementation may require
`C.data_layout == T.data_layout`; a lattice-surgery implementation may instead
require compatible boundaries and allocate workspace. Both implement the same
visible contract.

### An instruction restricted by definition

Some instructions are intrinsically tied to a code family or geometry rather than
merely having a restricted implementation:

```python
surface.d3_factory_calibration = QecInstr(
    name="surface/d3_factory_calibration",
    inputs=(Input("patch", LogicalPatch[RotatedSurfacePatch[3, 3]]),),
    outputs=(Output("patch", LogicalPatch[RotatedSurfacePatch[3, 3]]),),
)
```

The distinction is semantic: if other patch choices would still mean the same
operation, keep the instruction generic and constrain the implementation. If the
procedure's meaning itself depends on the exact patch choice, make the instruction
signature specific.

### Output-dependent code specifications

Outputs need not always preserve their input specification. Code switching,
deformation, merge, and split instructions may describe output types derived from
their inputs:

```python
surface.grow = QecInstr(
    name="surface/grow",
    inputs=(Input("patch", LogicalPatch[P]),),
    outputs=(Output("patch", LogicalPatch[Grow(P, dx=2)]),),
    logical_transform=LogicalIdentity(input="patch", output="patch"),
)
```

The initial implementation need only support specification-preserving
instructions, but the signature model must not assume every input patch is returned
unchanged.

Code switching requires a genuinely different input/output type relation. For
example, a generic instruction may bind its destination code from an explicit
type parameter while requiring the same encoded logical interface:

```python
From = PatchTypeVar("From")
To = PatchTypeVar("To")
L = LogicalInterfaceVar("L")

qec.code_switch = QecInstr(
    name="qec/code_switch",
    inputs=(Input("block", QecBlockType(code=From, logical=L, state=Active)),),
    outputs=(Output("block", QecBlockType(code=To, logical=L, state=Active)),),
    parameters=(
        CodeSpecParameter("target_code", binds=To),
        Parameter("logical_basis_map", LogicalBasisMap),
    ),
    constraints=(CompatibleLogicalInterface(From, To, logical=L),),
    logical_transform=LogicalIsometry(
        input="block",
        output="block",
        basis_map=ParameterRef("logical_basis_map"),
    ),
)
```

An application could then consume a surface block and return a color-code
block:

```python
color_data = graph.append(
    qec.code_switch(
        block=surface_data,
        target_code=TriangularColorPatch(distance=5),
        logical_basis_map="preserve_xz",
    ).using("surface_to_color_teleportation")
)
```

The source block is consumed. The returned block has a different concrete code
type but the logical interface promised by `L`. If the physical procedure also
requires an independently allocated destination patch, magic state, gauge
block, or measured ancilla, those are resource input/output ports—not ordinary
parameters—and must appear in the instruction signature.

The logical map is not automatically identity merely because the encoded state
is preserved. A code switch may exchange logical X and Z, introduce a known
Clifford frame, change the ordering of multiple logical qubits, or implement a
teleportation channel with classical byproducts. Its `QecLogicalTransform`
must state that map explicitly.

### Serializable type expressions

Parameterized block types must be represented by a runtime type-expression
algebra in Rust, not only by Rust generics or Python type annotations. The
initial `QecTypeExpr` vocabulary should include:

- `Exact(spec_id)` for one concrete code or patch specification;
- `Var(type_var, bounds)` for a code, patch, logical-interface, or lifecycle
  variable;
- `Parameter(param_id)` for a type supplied explicitly at a call site;
- `SameAs(input_port)` for a result that preserves an input type;
- `Apply(constructor_id, arguments)` for derived types such as `Grow(P, dx=2)`;
- structural constraints such as code-family membership, equal logical shape,
  compatible boundaries, or a required lifecycle state.

When a call is appended, the Rust validator unifies concrete input block types
with the input expressions, binds a substitution environment, validates
constraints known at authoring time, and instantiates concrete output types.
Implementation resolution may add narrower support checks, but it cannot alter
those instantiated public output types.

These expressions and their substitutions must be versioned and serializable.
The Python wrapper may expose friendly generic-looking representations, but
the Rust expression and unification result remain authoritative. Arbitrary
Python predicates or unserializable Rust closures are not valid type
constraints.

An implementation may narrow the substitutions it supports, but it cannot
silently change the instruction's port names, arity, ownership, or visible output
contract. If two procedures expose different inputs or outputs, they are
different instructions even if users informally give them the same gate name.

`supports` returns a useful diagnostic, not only a Boolean. A transversal CX
implementation can therefore explain that its operands have different data
layouts; a transversal H can reject a rectangular patch; and a lattice-surgery
implementation can require compatible boundary orientations and workspace.

Instruction-set resolution must be deterministic and local to the program or
lowering call. There is no process-global registry whose installed plugins
silently change compilation. A QEC instruction application may carry an implementation
constraint:

```python
graph.apply(surface.cx.using("transversal"), control, target)
```

An unconfigured `SurfaceInstrSet()` has no hidden defaults. Resolution follows
this order:

1. Use the call's `.using(...)` constraint, if present.
2. Otherwise use an explicitly configured instruction-set default.
3. Otherwise select the implementation if exactly one candidate supports the
   bound inputs and lowering context.
4. Report an unsupported-instruction error if no candidate supports the call.
5. Report an ambiguous-implementation error if multiple candidates support it.

The recommended conventional surface profile is SZZ syndrome extraction plus
transversal logical operations, but selecting it must be visible in source. The
fully explicit form is:

```python
surface = SurfaceInstrSet().with_defaults(
    syn_extract="szz",
    cx="transversal",
)
```

A named `SurfaceInstrSet.szz_transversal()` profile may provide exactly that
configuration as discoverable convenience; it is not the behavior of the bare
constructor. `surface.defaults()` returns the configured mapping using canonical
instruction and implementation IDs. `surface.implementations(surface.cx)` and
`surface.explain_resolution(...)` expose the candidates and the reason for a
selection.

For example, an unresolved CX with both transversal and lattice-surgery
implementations should produce a diagnostic resembling:

```text
Ambiguous implementation for instruction pecos.surface/cx@1.
Supported candidates:
  - pecos.surface/transversal_cx@1
  - pecos.surface/lattice_surgery_cx@1
Select one with surface.cx.using("...") or configure
SurfaceInstrSet.with_defaults(cx="...").
```

A configured default that does not support the bound patches is an error; the
resolver must not silently fall back to another implementation. The resolved
instruction, QEC protocol, implementation IDs, selection source
(`call_constraint`, `configured_default`, or `sole_candidate`), and options are
recorded in the generated artifact for reproducibility.

## Generic instruction programs and dialect-owned authoring

`InstrProgram` is not surface-code-specific and does not know instruction
names. It owns definitions/imports, reusable modules, entry points, and
exports. Each `InstrModule` contains an `InstrGraph` whose core concepts are
typed values, opaque instruction applications, attributes, results, regions,
and data dependencies. Neither container branches on strings or an enum such
as `H`, `CX`, `PREPARE`, or `MEASURE`.

An `InstrSet` owns user-facing definitions and implementations. `QecInstrSet`
refines that generic contract with QEC semantic interfaces. A QEC instruction
may carry a stable ID for diagnostics and serialization, but `InstrGraph`
treats the definition reference as opaque identity. Type checking is driven by
the definition's supplied signature, traits, and dialect interfaces, not by
recognizing its name.

The abstraction should stop there. `InstrProgram` is not intended to model
arbitrary host-language objects, general memory, operating systems, or every
compiler concern. It is a compact typed instruction-composition layer with
HDL-like definitions, instances, modules, elaboration, and implementation
selection. PHIR remains the broader compiler representation.

### HDL and netlist analogy

This model is intentionally HDL-like: instructions are typed cells that
transform resources, and an `InstrGraph` is a dataflow/control graph of
instances. QEC specializes those generic ideas:

| HDL concept | Generic instruction model | QEC specialization |
|---|---|---|
| Cell/module interface | `InstrDef` with named typed ports and parameters | `QecInstr` semantic interfaces |
| Cell instance | Bound `InstrCall` | QEC block and classical operands |
| Typed net | A versioned `ValueRef` | Usually a `QecBlockType` or frame/result value |
| Cell library | Explicit `InstrSet` | `QecInstrSet` / `SurfaceInstrSet` |
| Alternative cell implementation | `InstrImpl` | `QecInstrImpl` / QEC protocol |
| Elaboration | Parameter binding, type unification, and hierarchical expansion | `QecTypeExpr` and block-interface instantiation |
| Technology mapping | Deterministic `InstrImpl` resolution | `QecInstrPlan` generation |
| Implementation realization | Resolved implementation body | `QecInstrPlan` / `SpaceTimeRealization` |
| Abstract occupancy view | Dialect-owned resource projection | `SpaceTimeShape` |
| Placement/timing constraints | Dialect-owned constraint views | `SpaceTimeProgram` / `SpaceTimePlan` |
| Executable lower-level form | Generated PHIR | QEC PHIR and Guppy followed by HUGR/QIS |

The analogy is useful for Rust APIs, serialization, an eventual textual HDL,
and visual tooling. Hierarchical instructions can elaborate into lower-level
instruction graphs in the same way that a module can elaborate into cell
instances, while preserving a source/provenance map to the original call.

There are important differences from an ordinary combinational netlist:

- QEC block values are linear owned resources, not freely reusable wires; a
  consumed value cannot fan out or be used again;
- a cell may change resource type or arity, as in code switching, merge, split,
  preparation, and destructive measurement;
- every cell declares an encoded logical transformation or effect, not merely
  a bit-level input/output function;
- measurement identities, classical results, decoder interactions, and
  adaptive regions may coexist with quantum resource ports;
- spatial adjacency, duration, and reserved workspace can be part of a
  space-time composition without becoming physical target placement yet.

“Cell” is therefore a helpful mental model, but `InstrDef`, `InstrCall`, and
`InstrGraph` should remain the generic API terms. `QecInstr` remains the
QEC-domain term. This is an authoring/elaboration layer, not a replacement for
PHIR/HUGR or the eventual mapped and timed physical graph.

A “space-time cell” is consequently a useful view of one resolved instruction
call, not another instruction definition or port-signature mechanism. Its input
and output faces come directly from the `QecInstr` signature, and its contents
come from the selected `QecInstrImpl`/`QecInstrPlan`. A logical frame update or
compile-time annotation can have a zero-volume realization. Syndrome extraction
naturally has a time-extruded patch shape; transversal CX has a joint two-patch
shape; lattice-surgery CX may elaborate to merge, hold, and split regions; and
code switching can have different code types and geometries on its existing
typed input and output faces.

### HDL lessons incorporated into the design

Several established HDL/IR ideas should become explicit design requirements,
not only analogies.

#### Definition, instance, and symbol identity

CIRCT's HW dialect separates module definitions from instances, gives instances
named input/output ports and parameter bindings, and maintains an instance
graph. MLIR separately defines scoped symbols and SSA value dominance. PECOS
should mirror those distinctions:

- `InstrDef` and `InstrImpl` are reusable definitions with stable qualified
  `InstrDefId` and `ImplDefId` identities; QEC attaches its typed semantics to
  those definitions;
- `InstrCall` is an instance with a program-local `CallId`, optional display
  label, bound parameters, and named value edges;
- definition names, instance labels, and generated Guppy identifiers are
  different fields and must never be conflated;
- hierarchical instance paths are stable ID paths, not strings reconstructed
  from generated names.

This follows the structure of [CIRCT's HW module and instance
operations](https://circt.llvm.org/docs/Dialects/HW/) and [MLIR symbol/SSA
scoping](https://mlir.llvm.org/docs/LangRef/).

#### Explicit elaboration

Parameterized HDL distinguishes a reusable definition from its elaborated
instances. PECOS should add an `ElaboratedInstrProgram` artifact between the
authored `InstrProgram` and implementation resolution. Elaboration:

- canonicalizes every supplied parameter value;
- expands named convenience profiles into explicit instruction-to-
  implementation preferences;
- unifies input types and instantiates concrete output `QecBlockType` values;
- resolves definition symbols and validates port bindings;
- specializes reusable hierarchical definitions while retaining instance
  boundaries and source paths;
- rejects unresolved required parameters rather than silently inventing values.

The elaborated artifact is immutable and serializable. It contains no
unrecorded builder defaults or Python callbacks. CIRCT's HW rationale is useful
precedent here: canonical parameter expressions make inter-module analysis
simpler and avoid treating defaults as a special case after elaboration.
[CIRCT documents that parameter model and canonicalization
explicitly](https://circt.llvm.org/docs/Dialects/HW/RationaleHW/).

Implementation resolution happens after this semantic elaboration. It produces
a `ResolvedInstrProgram` in which every implementation choice and selection
source is explicit. A backend may request a flattened plan, but hierarchy must
remain available for diagnostics, reuse, visualization, and provenance.

#### Reusable logical modules and composite implementations

A compound instruction composition should use a hierarchical definition, not a Python
helper that eagerly copies calls into its caller. `InstrModule` is a reusable,
parameterized subcircuit whose body is the same typed instruction/control graph
as an `InstrProgram` entry graph. Its interface declares named typed block and
classical ports, parameters, outputs, instruction-set imports, and required
services. It cannot implicitly capture a live block, decoder, allocator, or
host-language value.

Calling a module creates a `InstrModuleCall` with a stable `ModuleInstanceId`
and hierarchy path. Each call consumes its own linear inputs and returns fresh
output versions; reusing a module definition never means reusing the same patch
instance. Elaboration specializes parameters and block types, validates the
module boundary, and may inline the body for a backend while retaining a
provenance map to the definition and instance. Recursion should be rejected in
the first version.

A module and a QEC instruction serve different purposes:

- `InstrModule` names and reuses a fixed composition. It introduces no new
  implementation-selection point; recursive resolution selects implementations
  for the `InstrCall` nodes inside it.
- `QecInstr` declares a semantic operation contract. A
  `CompositeQecInstrImpl` may use a `InstrModule` as one selectable
  implementation of that contract.

For example, an experiment may call a user-defined teleportation module
directly. If `surface.teleport` is intended to support alternative transversal,
lattice-surgery, or heterogeneous implementations, its public contract should
instead be a `QecInstr`, with this same module registered as one
`CompositeQecInstrImpl`.

Python authoring could look like:

```python
P = TypeVar("P", bound=SurfacePatchSpec)

teleport = InstrModule(
    name="example/teleport",
    inputs={
        "source": LogicalPatch[P, Active],
        "bell": LogicalPatch[P, Active],
        "destination": LogicalPatch[P, Active],
    },
    outputs={"destination": LogicalPatch[P, Active]},
    parameters={"rounds": PositiveInt},
    instruction_sets=[surface],
)

with teleport.body() as body:
    source = body.input("source")
    bell = body.input("bell")
    destination = body.input("destination")

    body.append(surface.h(patch=bell).using("transversal"))
    body.append(surface.cx(control=bell, target=destination).using("transversal"))
    body.append(surface.cx(control=source, target=bell).using("transversal"))
    body.append(surface.h(patch=source).using("transversal"))

    z_bit = body.append(surface.measure(patch=source, basis="Z"))
    x_bit = body.append(surface.measure(patch=bell, basis="Z"))
    body.append(surface.x(patch=destination).using("frame_update").when(x_bit))
    body.append(surface.z(patch=destination).using("frame_update").when(z_bit))
    body.append(
        surface.syn_extract(
            patch=destination,
            rounds=body.parameter("rounds"),
        ).using("szz")
    )
    body.output("destination", destination)

teleported = graph.call(
    teleport,
    source=source_a,
    bell=bell_a,
    destination=destination_a,
    rounds=3,
)
```

The Python context manager is definition-time builder sugar over a Rust-owned
module body; no Python callback is retained for elaboration or resolution.
Rust should expose the same operation through `InstrModuleBuilder`, typed
port handles, fluent call binding, and `finish()`.

Module definitions should be independently serializable and hashable. A
compiler may cache their verified/elaborated bodies by definition ID, canonical
parameter substitution, imported instruction-set versions, and implementation
profile. Caching must not merge instance-local block IDs, measurement IDs,
frame expressions, annotations, or result provenance.

#### Structure versus control

Calyx explicitly separates instantiated cells and connections from the control
program that schedules groups. CIRCT's DC rationale similarly argues for
separating data and control where useful. PECOS should therefore distinguish:

- the typed resource/dataflow graph;
- authored control constraints such as `seq`, `parallel`, `repeat`, conditional
  regions, and completion dependencies;
- resolved space-time and target schedules.

Source order is not automatically a total schedule. SSA/resource dependencies
impose required order; explicit control regions add constraints; independent
calls remain unordered and may be scheduled concurrently. This avoids having
Python statement order accidentally become physical time. See the [Calyx
structure/control split](https://docs.calyxir.org/tutorial/language-tut.html)
and [CIRCT DC rationale](https://circt.llvm.org/docs/Dialects/DC/RationaleDC/).

#### Classical values, conditional regions, and logical byproducts

Measurement-dependent feed-forward must be representable in the logical IR.
This is required for teleportation, lattice-surgery outcomes, magic-state
injection, decoder decisions, and repeat-until-success protocols. SLR's
`If(condition).Then(...).Else(...)` demonstrates that conditional authoring can
lower to Guppy, but the new representation must additionally account for
linear QEC blocks at branch boundaries.

The generic graph should have typed classical SSA values such as
`Bit`, `PauliByproduct[OneQubit]`, and instruction-defined decoder results.
Value types declare whether they are copyable. Ordinary measurement bits may
be copied; a QEC block remains linear; a byproduct token should normally be
linear so it cannot accidentally be discharged twice.

A conditional is a structural region, not a surface-code instruction and not a
special kind of named gate. It has:

- a typed, shot-time or compile-time Boolean predicate;
- explicit region inputs, including any linear blocks used by either branch;
- `then` and `otherwise` regions with block arguments rather than implicit
  capture of linear resources;
- explicit yields from both regions;
- a join requiring the same output arity and compatible output types.

Conceptually:

```text
(%data1) = if %m (%data0) {
  then(%data):
    %corrected = call %surface.x(%data)
    yield %corrected
  otherwise(%data):
    yield %data
}
```

The join result is a new SSA value; neither branch-local value can escape. This
is the quantum analogue of a block-argument/phi join and lets the verifier
reject patch duplication, a missing pass-through branch, or branches that
return incompatible patch types. Both branch bodies are present in the
authored artifact even though only one executes in a shot.

Predicating a single instruction is common enough to deserve generic sugar:

```python
data = graph.append(
    surface.x(patch=data).using("frame_update").when(m_x)
)
```

`when` belongs to the bound-call/graph authoring API, not to `surface.x`.
It desugars to an `if` whose false branch passes through the call's inputs.
The shorthand is legal only when the instruction declares an unambiguous
input-to-output pass-through with identical types; otherwise the diagnostic
asks for an explicit `if_else` with region yields. There should be no implicit
truth-value conversion or implicit capture of a patch.

#### Structured loops and branch joins

Loops should use the same structured-region model rather than introduce an
unrestricted control-flow graph into `InstrGraph`. The initial forms should be:

- `repeat(count)`, where `count` is a compile-time or elaboration-time positive
  integer and the body is a reusable region;
- `while_loop(condition, carried=...)`, where a shot-time Boolean controls
  entry to another iteration;
- `repeat_until(success, carried=...)`, useful sugar for protocols such as
  state preparation, decoding, or repeat-until-success injection where the
  condition is produced by the body.

Every loop has explicit loop-carried arguments and yields. A linear QEC block
enters one iteration exactly once, is consumed and replaced inside the body,
and the replacement becomes either the next iteration's argument or the loop's
result. The verifier requires compatible carried types at the back edge and at
loop exit; it rejects implicit capture, dropping, duplicating, or returning an
iteration-local QEC value. Copyable classical constants may be explicit
invariant arguments.

Conceptually:

```text
(%data_out, %last_syndrome) = repeat_until (%data_in) {
  body(%data):
    (%next, %syndrome) = call %surface.syn_extract(%data)
    %done = call %decoder.converged(%syndrome)
    continue_if_not %done yielding %next, %syndrome
}
```

This syntax is illustrative. The serialized representation needs typed region
arguments, yielded carried values, a typed condition, and an explicit result
signature. It must also distinguish compile/elaboration-time control from
shot-time control. Static repetition can remain compact for analysis and then
be unrolled or lowered to a backend loop. Dynamic loops require runtime
feedback support and may require a declared maximum-iteration bound, timeout
result, or explicitly unbounded resource estimate before a target will accept
them.

Branches follow the same ownership rule. Both arms receive their own region
arguments representing mutually exclusive uses of the incoming values, and the
join produces one new value version. This is not quantum cloning: only the
selected arm executes in a shot. Nested `if_else`, `repeat`, and dynamic loops
are legal when their region boundaries and backend capabilities compose.

Logical Pauli feed-forward needs a still higher-level representation. A raw
teleportation or surgery instruction may return both a block and a typed
`PauliByproduct` computed from its measurement outputs. A separate instruction
can consume the block and byproduct and promise a corrected logical output:

```python
teleported, byproduct = graph.append(
    surface.teleport_raw(source=data, destination=destination)
)
teleported = graph.append(
    surface.apply_byproduct(
        patch=teleported,
        byproduct=byproduct,
    ).using("frame_update")
)
```

This is preferable inside reusable composite protocols because it states the
semantic intent without baking in two nested conditionals or choosing whether
the correction is physical. Implementations of `apply_byproduct` may update a
tracked Pauli frame, emit conditional logical Pauli operations, or discharge
the correction through a later measurement-basis reinterpretation. The
resolved plan records which occurred. The public instruction has the same
corrected logical transformation in all cases.

Frame state is semantic state associated with a logical value version, not a
new surface-patch geometry and not a hidden mutable global. Clifford
instructions propagate it symbolically. A lowering must materialize or
otherwise discharge it at a boundary that cannot consume the frame, and final
measurement/observable metadata must incorporate it. Exporting a live logical
block exports its frame state with it; exporting a measured classical result
requires the frame to have been incorporated into that result's interpretation.
A standalone byproduct token may not be silently dropped.

##### First-class logical and physical Pauli frames

PECOS should model Pauli frames as first-class, versioned state while keeping
them out of the way in ordinary logical-circuit authoring. Every active
`QecBlockValue` conceptually owns a `FrameStateRef`. Appending an instruction
consumes the block version and produces a new block version with a transformed
frame reference, just as it transforms lifecycle and logical state. The cursor
API advances this state automatically.

The model must distinguish two layers:

- `LogicalPauliFrame` is defined over the encoded logical interface. For a
  block containing `k` logical qubits it records an element of the logical
  Pauli group, normally as `2k` X/Z components whose coefficients may be
  symbolic Boolean expressions over measurement and decoder results. It also
  records the logical basis/order convention that gives those components
  meaning.
- `PhysicalPauliFrame` is defined over mapped physical-qubit identities and a
  particular code realization/layout epoch. It records deferred physical
  Paulis, together with the stabilizer/gauge convention needed to interpret
  equivalent representatives. It cannot generally exist in the authored
  code-independent program and first becomes concrete in a resolved or mapped
  plan.

These are related but not interchangeable. A physical frame can project to a
logical component plus a stabilizer/gauge component; a logical frame may have
many physical representatives. Syndrome decoding can update a physical frame
without changing the intended logical transformation. Teleportation usually
updates the logical frame from known measurement byproducts. A logical
correction may later be materialized by choosing a supported physical
representative.

“Per block” is the usual ownership rule, not a claim that frames transform
independently. Every `QecInstr` declares a frame-transfer contract across its
complete interface:

- identity and syndrome extraction preserve the logical frame, although a
  selected syndrome/decoder implementation may update the physical frame;
- logical Clifford instructions conjugate logical frame components;
- CX propagates components between its control and target frames;
- merge, split, deformation, gauge fixing, and code switching map frames across
  changing block boundaries and conventions;
- preparation initializes a known frame convention;
- measurement consumes the relevant frame component and adjusts the logical
  result/observable interpretation;
- non-Clifford boundaries declare which frame components they can propagate and
  which must first be materialized or converted into adaptive control.

Two per-block frame components may share the same symbolic predicate DAG, so
classical correlations created by a multi-block instruction are retained
without pretending that the classical frame itself is a quantum-entangled
resource. Temporary protocol ancillas may carry implementation-owned frame
state until they are returned, measured, or discarded under a verified rule.

Most users should see only ordinary block operations. Teleportation and expert
workflows may update or inspect frame state explicitly through instruction-set
operations and artifact introspection, for example:

```python
destination = graph.append(
    surface.update_logical_frame(
        patch=destination,
        delta=LogicalPauli(
            x=m_z,
            z=m_x,
        ),
    )
)

frame_expr = graph.frame_of(destination)  # symbolic, read-only view
```

`update_logical_frame` is a QEC instruction supplied by an instruction set;
`InstrGraph` does not recognize its name. `frame_of` is generic graph
introspection and cannot mutate the frame behind a live block. The update is
therefore explicit in serialized dataflow and consumes/returns the block like
any other instruction. `apply_byproduct(...).using("frame_update")` is the
typed convenience when the delta is already packaged as a linear
`PauliByproduct` token.

The Rust artifacts should make frame evolution inspectable at three points:

1. the authored/elaborated program contains logical frame expressions and
   instruction frame-transfer contracts;
2. the resolved plan records frame policy, symbolic rewrites, materialization
   points, and decoder/service dependencies;
3. the mapped/traced artifact may contain physical frame versions tied to
   physical qubit IDs, measurement IDs, and layout epochs.

Frame policy is an explicit implementation choice when more than one supported
choice exists. Introspection should report whether a correction was propagated,
absorbed into measurement interpretation, materialized physically, or lowered
as shot-time conditional control. Serialization and provenance must preserve
frame expression IDs and the reason for every materialization.

This yields three deliberately separate constructs:

1. `if_else` for general structured classical control;
2. `.when(predicate)` for a type-preserving predicated instruction;
3. `apply_byproduct` for correction semantics whose implementation may be a
   virtual frame update rather than conditional quantum execution.

The distinction matters for analysis. Pauli-frame feed-forward normally leaves
the physical TickCircuit and DEM fault structure unchanged while changing
frame/observable interpretation. A truly adaptive non-Pauli operation, such as
a conditional logical S in an injection protocol, changes the executed
program and may require branch-aware tracing, scheduling, and DEM support.
Backends that cannot represent such shot-time adaptation must reject it with a
diagnostic identifying the predicate and call; they must not flatten it into an
unconditional operation or silently assume one branch.

#### Typed channels and declared services

CIRCT ESI distinguishes typed application channels from their later physical
signaling and models required services explicitly. PECOS should borrow this
selectively:

- ordinary QEC block transitions remain linear SSA values, not streaming
  channels;
- measurement streams, decoder request/response paths, and adaptive classical
  feedback may use typed channel values when latency or repeated communication
  is semantically relevant;
- shared capabilities such as decoder access, ancilla allocation, calibration,
  or classical feedback must be declared as typed implementation requirements,
  never discovered through global state;
- the lowering context binds those requirements to concrete providers and
  records the binding in provenance.

The logical type should describe the message/service contract without fixing a
wire protocol or target latency, following [ESI's separation of typed channels
from signaling](https://circt.llvm.org/docs/Dialects/ESI/RationaleESI/). The
first version need not implement general channels or services; it must avoid IR
choices that prevent adding them.

#### Typed metadata with rewrite rules

HDL toolchains demonstrate both the value and danger of annotations. FIRRTL
annotations can target definitions, ports, or instance paths, and compiler
passes must keep those targets synchronized. PECOS should not use an arbitrary
attribute dictionary for semantic information.

Core facts—logical transforms, detector identities, implementation choices,
space-time constraints, and service requirements—are typed fields. Extensible
metadata uses a versioned `QecAnnotation` envelope with an explicit target ID,
retention policy, and rewrite behavior for clone, inline, specialize, flatten,
and delete transformations. Unknown semantic annotations are errors; explicitly
advisory annotations may be preserved opaquely. This is informed by [FIRRTL's
annotation and target model](https://circt.llvm.org/docs/Dialects/FIRRTL/FIRRTLAnnotations/).

Every transformation returns a provenance map from old definition/instance/
value IDs to new IDs. Guppy result tags and QIS measurement identities are
consumers of that map, not substitutes for it.

#### Verification after every phase

Each explicit artifact has its own verifier:

- authored program: symbol, port, linearity, and region well-formedness;
- elaborated program: all parameters concrete, output types instantiated, and
  hierarchy valid;
- resolved program: one supported implementation per call and all service/
  capability requirements bound;
- instruction plan: resource/control constraints and declared logical semantics
  satisfied;
- generated/traced artifacts: source identities, measurements, detectors, and
  observables preserved.

Transforms should fail at the phase that owns the violated invariant rather
than allowing a malformed artifact to reach Guppy.

### HDL ideas not adopted directly

PECOS should not inherit HDL mechanisms that do not match this abstraction:

- no implicit nets, implicit port connections, or “last connect wins” rules;
- no unrestricted fanout of linear QEC resources;
- no four-state bit semantics, clocks, or resets in the logical instruction IR;
- no assumption that every value is a ready/valid channel, as in a fully
  elastic dataflow circuit;
- no host-language callbacks during elaboration or implementation resolution;
- no requirement to flatten hierarchy before analysis;
- no physical cycle timing until mapped/timed lowering.

These exclusions keep the model HDL-influenced without turning it into a
classical RTL language.

For example:

```python
from pecos.instr import InstrProgram
from pecos.qec.surface import SurfacePatch, SurfaceInstrSet

surface = SurfaceInstrSet.szz_transversal()
program = InstrProgram(instruction_sets=[surface])
graph = program.main()

data = graph.add_block("data", SurfacePatch.rotated(3))
target = graph.add_block("target", SurfacePatch.rotated(3))

data = graph.apply(surface.prepare(basis="Z"), patch=data)
target = graph.apply(surface.prepare(basis="X"), patch=target)
data = graph.apply(surface.syn_extract(rounds=3), patch=data)
target = graph.apply(surface.syn_extract(rounds=3), patch=target)
data = graph.apply(surface.h, patch=data)
data, target = graph.apply(
    surface.cx,
    control=data,
    target=target,
)
data = graph.apply(surface.syn_extract(rounds=3), patch=data)
target = graph.apply(surface.syn_extract(rounds=3), patch=target)
data_result = graph.apply(surface.measure(basis="X"), patch=data)
target_result = graph.apply(surface.measure(basis="Z"), patch=target)
```

Here `apply` is the only operation understood by `InstrGraph`. The names
`prepare`, `syn_extract`, `h`, `cx`, and `measure` are defined entirely by
`SurfaceInstrSet`. Another instruction set may expose different names and
contracts without modifying the graph class.

`CodeSpec` is an intentionally small interface implemented or adapted by
`SurfacePatch`, color-code specifications, and block-code specifications. It
provides stable structural identity and logical-qubit count; code-family details
remain on the concrete specification and are inspected by matching instruction
implementations.

For discoverability and a narrow first implementation, PECOS may expose a
surface authoring facade, but that facade should belong to the instruction set
rather than subclassing the graph. For example, convenience functions can
build calls on an ordinary `InstrGraph`:

```python
from pecos.instr import InstrProgram
from pecos.qec.surface import SurfacePatch, SurfaceInstrSet

surface = SurfaceInstrSet.szz_transversal()
program = InstrProgram(instruction_sets=[surface])
graph = program.main()
data = graph.add_block("data", SurfacePatch.rotated(3))
data = surface.prepare_z(graph, data)  # sugar for graph.apply(...)
```

The facade must not define a second program/graph type or IR. Surface-only options
such as check plans are QEC protocol implementation options supplied through the
surface instruction set.

This split avoids both extremes:

- a surface-only circuit that must later be replaced to support another code;
- a prematurely universal QEC IR that encodes surface stabilizers, boundaries,
  and lattice surgery as generic concepts.

Code-specific behavior remains expressible through extensible QEC instruction
definitions. For example, `MergeBoundary` may be provided by the surface
instruction set without becoming a method every `CodeSpec` or `InstrGraph`
must support.

## Two authoring styles over one instruction graph

The API should support both a circuit-like view and a space-time composition
view without creating two semantic IRs. Both authoring styles produce the same
typed graph of `InstrCall` nodes and `ValueRef` edges. A
space-time program adds placement, adjacency, concurrency, and ordering
constraints to that graph; it does not redefine what an instruction means.

This common graph is important for interoperability. Python builders, a future
QASM-like syntax, an SLR/Zlup front end, and graphical tools can all construct
the same artifact. Instruction-set resolution, implementation planning, Guppy
generation, source mapping, and verification then operate on one model.

### Circuit-like sequential authoring

The primary Python API should feel similar to building a physical circuit:
declare named logical blocks, then append instructions in reading order. Named
arguments correspond directly to the instruction's named input ports, which
makes multi-patch operations readable and produces useful diagnostics.

```python
surface = SurfaceInstrSet.szz_transversal()
program = InstrProgram(instruction_sets=[surface])
graph = program.main()

data = graph.add_block("data", SurfacePatch.rotated(3))
target = graph.add_block("target", SurfacePatch.rotated(3))

data = graph.apply(surface.prepare(basis="Z"), patch=data)
target = graph.apply(surface.prepare(basis="X"), patch=target)
data = graph.apply(surface.h, patch=data)
data, target = graph.apply(
    surface.cx,
    control=data,
    target=target,
)
data = graph.apply(surface.syn_extract(rounds=3), patch=data)
target = graph.apply(surface.syn_extract(rounds=3), patch=target)
result = graph.apply(surface.measure(basis="Z"), patch=data)
```

`apply` is generic: it binds arguments using the supplied instruction
signature, emits one opaque call, consumes linear input versions, and returns
the typed output versions. It does not recognize `h`, `cx`, or any other name.
A future QASM-like textual front end should be equally direct:

```text
patch data: surface.rotated(distance=3);
patch target: surface.rotated(distance=3);
surface.prepare(basis="Z") patch=data;
surface.prepare(basis="X") patch=target;
surface.h patch=data;
surface.cx using="transversal" control=data, target=target;
surface.syn_extract(rounds=3) patch=data;
surface.syn_extract(rounds=3) patch=target;
result = surface.measure(basis="Z") patch=data;
```

That syntax is illustrative, not a language proposal. Its useful property is
that instruction names and ports resolve through an explicitly imported
instruction set rather than a fixed grammar-level gate enumeration.

For notebooks and long experiment descriptions, an optional imperative facade
may hold mutable `LogicalBlock` cursors so users do not have to rebind every SSA
value. Those handles are only builder conveniences: internally every
instruction still consumes one value version and produces another. The
functional `apply` form remains the precise core API and serialization model.

Parallel and repeated regions can later be structural circuit constructs, for
example `graph.parallel()` and `graph.repeat(count)`. They constrain the
dataflow graph but do not introduce named QEC operations into `InstrGraph`.

### Space-time-volume composition

Some QEC experiments are more naturally described by arranging patch histories
and interaction volumes than by listing operations. A `SpaceTimeProgram` should
therefore be a constrained view over the same instruction graph:

```python
space_time = SpaceTimeProgram(instruction_set=surface)
data = space_time.add_patch("data", d3, at=(0, 0))
target = space_time.add_patch("target", d3, at=(0, 4))

data_prepared = space_time.place(surface.prepare(basis="Z"), patch=data)
target_prepared = space_time.place(surface.prepare(basis="X"), patch=target)
corrected = space_time.after(
    data_prepared,
    surface.syn_extract(rounds=3),
    patch=data,
)
entangled = space_time.after(
    (corrected, target_prepared),
    surface.cx.using("lattice_surgery"),
    control=data,
    target=target,
)
space_time.after(
    entangled,
    surface.measure(basis="X"),
    patch=data,
)

program = space_time.to_instr_program()
```

The exact fluent spelling is provisional. The stable concepts should be:

- instruction applications with the same named input and output ports as the
  circuit-like API;
- logical or relative spatial placement, boundary compatibility, adjacency,
  and reserved workspace;
- temporal precedence, concurrency, and duration constraints;
- input and output faces for patch, measurement, and classical values;
- composition operations such as `after`, `parallel`, `connect`, and `repeat`;
- source identities that survive conversion to the common instruction graph.

#### Space-time realizations and shape views

The instruction model already defines the cell-like interface: a `QecInstr`
takes typed QEC block and classical inputs and returns typed QEC block and
classical outputs. The space-time model must reuse that interface rather than
introduce a parallel `SpaceTimeCellDef` type system.

After an implementation is selected, its `QecInstrPlan` describes a
`SpaceTimeRealization` for that particular instruction call. The realization
may be parameterized until patch geometry, placement, or timing is known, but
it is still the implementation of the existing typed call. A
`SpaceTimeShape` is an abstraction or projection of that realization; the
space-time physical circuit is the detailed view of the same realization.

```text
QecInstr(input types -> output types)       semantic request
    -> QecInstrImpl                         selected protocol
        -> QecInstrPlan / SpaceTimeRealization
              +-> SpaceTimeShape            abstract occupancy/projection
              +-> space-time physical circuit
                    gates and resources embedded in 2 space + 1 time
```

For a planar topological code, a realization begins with each input patch's
abstract two-dimensional data-qubit geometry. The implementation may introduce
ancilla qubits and workspace, impose adjacency or interaction constraints, and
describe how all of those resources persist or change along the time axis. The
physical-circuit view locates its gates and resources in two spatial dimensions
plus one ordered time dimension. Projecting or summarizing that circuit gives
an operation shape. Its boundary faces are induced by the instruction's typed
input/output patches and classical or measurement ports, rather than being a
second declaration of them or merely a box with width, height, and duration.

The shape can intentionally discard circuit detail at several useful levels:

- an envelope records only the outer occupied region and boundary faces;
- an occupancy shape distinguishes persistent data, temporary ancilla, and
  reserved workspace regions;
- an annotated shape adds interaction corridors, measurement/classical ports,
  rounds, latency, or fault/noise summaries;
- the fully detailed view contains the physical operations and resource
  trajectories from which those projections can be checked or derived.

A coarse shape may be produced before the full physical circuit, for example by
a parameterized implementation planner. It is then a claimed abstraction that
the eventual detailed circuit must satisfy, not a separate semantic
implementation.

Structured control has corresponding shape projections:

- a conditional realization is a `ChoiceShape` with a shared typed entry and
  join plus one mutually exclusive shape per branch;
- a static `repeat(count)` is a `RepeatShape` whose body shape can be retained
  symbolically or expanded along time;
- a dynamic loop is a loop-body shape plus its carried-resource interface,
  feedback edge, termination condition, and any iteration bound or resource
  distribution known to the planner.

These are compact descriptions, not literal cyclic physical time. A concrete
execution trace unfolds whichever branch and iterations actually execute into
an acyclic 2+1D history. Placement may reserve the union of conditional branch
occupancy, reuse resources known to be mutually exclusive, or use a target-
specific dynamic policy; the choice must be recorded. Duration and volume
queries must state whether they report a per-branch value, minimum, maximum,
expected estimate, bounded range, or `Unknown`. An unbounded dynamic loop cannot
claim a finite exact total space-time volume.

This model must also work before exact mapping. A realization can contain symbolic
coordinates, relative placement constraints, alternative orientations, and
resource bounds. Only a mapped realization names target physical qubits and
calibrated durations. Time may initially be rounds or a partial order, then
later refine to ticks and hardware time; the model should not assume that every
abstract shape is an integer rectangular prism.

The black-box shape or realization summary should make at least the following
information inspectable when known:

- typed QEC block input/output faces and their `CodeGeometry` views;
- the logical transform/channel and logical/physical frame-transfer contract;
- persistent data-qubit occupancy and allowed deformation through time;
- temporary ancillas, routing/workspace, and their allocation/lifecycle rules;
- duration, latency, and concurrency bounds;
- adjacency, connectivity, orientation, and boundary compatibility constraints;
- measurement, decoder, and classical-feedback ports and latency requirements;
- detector/observable boundaries and live resources crossing the cell boundary;
- fault/noise summaries or hooks when an implementation can provide them.

Quantitative facts should carry their precision instead of presenting estimates
as exact values. A common wrapper such as
`ResourceQuantity<T> = Exact(T) | LowerBound(T) | UpperBound(T) | Estimate(T) |
Unknown` can describe qubit count, area, duration, or volume at different
refinement stages.

Tools should be able to inspect the cheapest sufficient projection without
forcing physical-circuit generation:

1. the instruction signature plus shape summary for type, logical-effect, and
   resource-bound analysis;
2. a protocol/composite view showing constituent realizations and partial order;
3. a mapped view with concrete physical resources and scheduled intervals;
4. the physical circuit with gates, measurements, resets, and feedback;
5. the executed QIS trace, normalized TickCircuit, and DEM.

#### Visualization boundary

Both the coarse shape and detailed space-time circuit should be straightforward
to visualize because they share stable resource, instruction-call, geometry,
time, hierarchy, measurement, and provenance identities. Visualization should
consume a Rust-produced, serializable `SpaceTimeView` rather than teach a GUI
how to resolve instructions or reconstruct physical circuits.

`SpaceTimeView` is a read-only presentation artifact, not another semantic IR.
It contains selected geometry and time coordinates, drawable primitives or
resource trajectories, hierarchy, annotations, and links back to the source
`InstrCall`, `QecInstrPlan`, code element, physical operation, measurement,
detector, observable, or frame effect. It may be regenerated at any requested
level of detail from the authoritative realization.

Useful presentations include:

- an `x-y` patch/layout view at a selected time or round;
- `x-time` and `y-time` projections showing resource motion and occupancy;
- a rotatable 2+1D volume view of data, ancilla, workspace, and interactions;
- resource-lane or Gantt-like views for durations, concurrency, measurement,
  decoding, and feedback;
- hierarchical expansion from instruction shape to composite protocol to
  physical gates, with matching source/provenance selection;
- overlays for boundaries, logical supports, physical-qubit IDs, instruction
  calls, measurement IDs, detector/observable regions, Pauli-frame effects,
  and fault/noise information when available.

Conditional and repeated control should remain navigable rather than being
flattened prematurely. A viewer can show a `ChoiceShape` as selectable branch
tabs or juxtaposed alternatives and a `RepeatShape` as a collapsed body with an
iteration count/range. It may optionally unfold a chosen branch, a fixed number
of iterations, or an observed execution trace. Mutually exclusive branches
must not be visually presented as simultaneously executed occupancy unless the
view is explicitly showing reserved-union resources.

The view model must label whether positions and times are symbolic, relative,
scheduled, or target-calibrated, and whether resource quantities are exact,
bounded, estimated, or unknown. Large realizations need hierarchy and
level-of-detail queries so opening a factory or long syndrome history does not
require materializing every gate primitive at once.

The initial renderer can be modest: deterministic scene JSON plus a static
`x-y`/time-slice or SVG-like debugging view from Rust, wrapped by a thin Python
notebook display. An interactive web or desktop viewer can later consume the
same scene artifact. Rendering choices must never feed back into implementation
selection, mapping, or semantic verification.

##### Bevy as an interactive viewer

Bevy is a strong candidate for the interactive Rust viewer, but it should be an
optional consumer rather than a dependency of `InstrProgram`, `pecos-qec`,
PHIR, Guppy generation, or `SpaceTimeView` construction. A separate crate or
binary such as `pecos-spacetime-viewer` can deserialize or directly receive a
`SpaceTimeView` and map its stable IDs to Bevy entities/components.

The ECS model is a natural fit for selectable data qubits, ancillas,
interaction volumes, instruction instances, hierarchy nodes, and overlay
components. Bevy provides native 2D/3D rendering, gizmos, cameras, UI, and mesh
picking suitable for time slicing, orbiting a 2+1D realization, selecting a
resource, and following its provenance. Its WebAssembly rendering options also
leave open a browser or notebook-hosted viewer. At the time of this design,
PECOS and Bevy 0.19 both use `wgpu` 29, which makes an initial integration spike
particularly reasonable. See Bevy's [0.19 release
notes](https://bevy.org/news/bevy-0-19/), [picking
example](https://bevy.org/examples/picking/simple-picking/), and [rendering
feature documentation](https://docs.rs/crate/bevy/latest/source/docs/cargo_features.md).

The separation remains important:

- core and headless PECOS builds must not compile Bevy or require a display/GPU;
- the stable interchange is versioned `SpaceTimeView` data, not Bevy ECS world
  serialization or Bevy entity IDs;
- stable PECOS IDs are components on viewer entities and remain authoritative
  for selection, annotations, cross-highlighting, and provenance;
- deterministic static snapshots remain the CI/reference renderer because an
  interactive GPU view is a poor golden-test boundary;
- Bevy features should be selected narrowly to control compile time, binary
  size, platform dependencies, and web payload;
- another viewer must be able to consume the same scene without reproducing
  QEC resolution or mapping logic.

The spike should demonstrate one synchronized `x-y` time slice and 2+1D view,
time/round scrubbing, visibility toggles for data/ancilla/workspace, picking
back to an `InstrCall` and physical operation, and hierarchy expansion from a
coarse instruction shape to gates. That is enough to validate the architecture
before investing in editing, layout manipulation, or a full IDE.

Each refinement carries a verification obligation:

```text
QecInstrPlan realizes the QecInstr semantic contract
mapped physical circuit realizes the QecInstrPlan
physical-circuit occupancy satisfies its claimed SpaceTimeShape projections
```

Black-boxing must therefore preserve all externally observable outputs and
resource lifetimes. A live ancilla, measurement identity, decoder dependency,
Pauli-frame effect, or detector boundary cannot disappear merely because an
analysis chooses not to expand the cell internals.

Implementation selection and placement may need to cooperate. Resolution can
retain several supported shape/realization alternatives until the
space-time planner proves one feasible; it must still record the final selected
implementation and selection source. An early explicit `.using(...)` remains a
hard constraint and should produce an actionable placement error rather than
silently falling back to another protocol.

The authoring artifact should store constraints rather than pretend it already
contains a target schedule. Before implementation selection, an instruction
may expose only an abstract volume contract: required patch faces, possible
adjacencies, workspace bounds, and ordering relations. Resolution selects QEC
protocol implementations (or preserves explicit feasible alternatives for
cooperative planning) and produces a `SpaceTimePlan` whose shapes and detailed
realizations reference the corresponding `QecInstrPlan` objects. Device placement,
native timing, idle insertion, and feedback latency still belong to #514's
mapped/timed execution layer.

Consequently, these are two views of one pipeline:

```text
InstrGraph/QASM authoring -----------------+
                                            +-> typed instruction graph
space-time authoring -> spatial constraints +            |
                                                         v
                                  instruction resolution / SpaceTimePlan
                                                         |
                                                         v
                                                       Guppy
```

A user may freely inspect a space-time projection of a sequential graph, or
linearize a space-time composition into an `InstrProgram`, provided the
partial order is preserved. Backends must never need separate implementations
of the QEC instruction semantics for the two styles.

## Proposed surface instruction API

Surface authoring should be introduced through `SurfaceInstrSet`, while the
program remains an ordinary `InstrProgram`. This avoids silently changing the
current position-dependent semantics of `LogicalCircuitBuilder.add_memory`.
The old builder can become a compatibility front end that emits QEC instruction applications
after backend equivalence is established.

### Complete circuit-like experiment

The recommended everyday form uses program-scoped `LogicalBlock` cursors. An
`append` consumes the cursor's current SSA value and advances it to the matching
typed output port. Multi-block instructions advance every participating cursor.
This keeps the source close to a physical-circuit or QASM listing without
weakening the underlying linear representation:

```python
from pecos.instr import InstrProgram
from pecos.qec.surface import SurfaceInstrSet, SurfacePatch

surface = SurfaceInstrSet.szz_transversal()
# Equivalent to:
# SurfaceInstrSet().with_defaults(syn_extract="szz", cx="transversal")
assert surface.defaults() == {
    "pecos.surface/syn_extract@1": "pecos.surface/syndrome_szz@1",
    "pecos.surface/cx@1": "pecos.surface/transversal_cx@1",
}
program = InstrProgram(instruction_sets=[surface])
graph = program.main()

data = graph.block("data", SurfacePatch.rotated(3))
ancilla = graph.block("ancilla", SurfacePatch.rotated(3))

# DeclaredBlock[d3] -> ActiveBlock[d3], with |0_L> and |+_L> semantics.
with graph.parallel():
    graph.append(surface.prepare(patch=data, basis="Z"))
    graph.append(surface.prepare(patch=ancilla, basis="X"))

# Each call is logically identity, implemented by three syndrome rounds.
with graph.parallel():
    graph.append(surface.syn_extract(patch=data, rounds=3))
    graph.append(surface.syn_extract(patch=ancilla, rounds=3))

graph.append(surface.h(patch=data))
graph.append(surface.cx(control=data, target=ancilla))

with graph.parallel():
    graph.append(surface.syn_extract(patch=data, rounds=2))
    graph.append(surface.syn_extract(patch=ancilla, rounds=2))

graph.append(surface.cx(control=ancilla, target=data))

with graph.parallel():
    graph.append(surface.syn_extract(patch=data, rounds=1))
    graph.append(surface.syn_extract(patch=ancilla, rounds=1))

# Active blocks are consumed and typed logical measurement values are returned.
with graph.parallel():
    data_x = graph.append(surface.measure(patch=data, basis="X"))
    ancilla_z = graph.append(surface.measure(patch=ancilla, basis="Z"))

graph.export("data_x", data_x)
graph.export("ancilla_z", ancilla_z)

artifact = surface.compile_to_guppy(program)
trace = artifact.trace_qis(runtime=...)
tick_circuit = trace.tick_circuit
dem = artifact.build_dem(runtime=..., noise=...)
```

The surface instruction set supplies every name, port signature, logical
transformation, and implementation choice in this example. `InstrGraph`
only understands declarations, opaque calls, typed dataflow, parallel regions,
and exported values. In particular, it has no special cases for preparation,
H, CX, syndrome extraction, or measurement.

The cursor form above desugars to the precise functional API. For example:

```python
data_v1 = graph.apply(surface.prepare(basis="Z"), patch=data_v0)
data_v2 = graph.apply(surface.syn_extract(rounds=3), patch=data_v1)
data_v3, ancilla_v2 = graph.apply(
    surface.cx,
    control=data_v2,
    target=ancilla_v1,
)
```

The imperative and functional forms therefore serialize to identical
`InstrCall` nodes. The former is intended for experiment authors; the latter
is useful to IR tooling, transforms, and tests.

```python
from pecos.instr import InstrProgram
from pecos.qec.surface import SurfacePatch, SurfaceInstrSet

d3 = SurfacePatch.rotated(3)

surface = SurfaceInstrSet.szz_transversal()
program = InstrProgram(instruction_sets=[surface])
graph = program.main()
data = graph.add_block("data", d3)
ancilla = graph.add_block("ancilla", d3)

data = graph.apply(surface.prepare(basis="Z"), patch=data)
ancilla = graph.apply(surface.prepare(basis="X"), patch=ancilla)
data = graph.apply(surface.syn_extract(rounds=3), patch=data)
ancilla = graph.apply(surface.syn_extract(rounds=3), patch=ancilla)
data = graph.apply(surface.h, patch=data)
data, ancilla = graph.apply(
    surface.cx,
    control=data,
    target=ancilla,
)
data = graph.apply(surface.syn_extract(rounds=3), patch=data)
ancilla = graph.apply(surface.syn_extract(rounds=3), patch=ancilla)
data_result = graph.apply(surface.measure(basis="X"), patch=data)
ancilla_result = graph.apply(surface.measure(basis="Z"), patch=ancilla)

artifact = surface.compile_to_guppy(program)
source = artifact.source
entry_point = artifact.entry_point

trace = artifact.trace_qis(runtime=...)
tick_circuit = trace.tick_circuit
dag_circuit = tick_circuit.to_dag_circuit()
dem = artifact.build_dem(runtime=..., noise=...)
```

`add_block` returns a program-scoped logical value, not the `SurfacePatch`
itself.
This distinguishes two logical patches with identical geometry and prevents a
patch from accidentally being used in two programs. Each QEC instruction application consumes
its input value versions and returns new ones, following linear SSA-style
ownership. A measurement instruction returns a logical-result handle that can name
the final observable without relying on record position.

Convenience instruction-set methods may expand into primitive IR operations:

```python
data = surface.inject_s(graph, data, magic, rounds_before=3, rounds_after=3)
data = surface.inject_t(graph, data, magic, rounds_before=3, rounds_after=3)
```

Their expansion should happen before implementation planning so every
downstream stage sees the same instruction expansion.

### Conditional teleportation examples

An experiment author may write the familiar measurement-controlled Pauli
corrections directly. This example uses explicit implementations so the source
states that both corrections are tracked in the Pauli frame:

```python
surface = SurfaceInstrSet()  # no hidden implementation defaults
program = InstrProgram(instruction_sets=[surface])
graph = program.main()

source = graph.block("source", SurfacePatch.rotated(3))
destination = graph.block("destination", SurfacePatch.rotated(3))
bell = graph.block("bell", SurfacePatch.rotated(3))

graph.append(surface.prepare(patch=source, basis="X"))
graph.append(surface.prepare(patch=destination, basis="Z"))
graph.append(surface.prepare(patch=bell, basis="Z"))

# The exact primitive sequence is illustrative; a surface instruction set owns
# its ports and supported implementations.
graph.append(surface.h(patch=bell).using("transversal"))
graph.append(
    surface.cx(control=bell, target=destination).using("transversal")
)
graph.append(
    surface.cx(control=source, target=bell).using("transversal")
)
m_x = graph.append(surface.measure(patch=source, basis="X"))
m_z = graph.append(surface.measure(patch=bell, basis="Z"))

destination = graph.append(
    surface.x(patch=destination).using("frame_update").when(m_z)
)
destination = graph.append(
    surface.z(patch=destination).using("frame_update").when(m_x)
)
result = graph.append(surface.measure(patch=destination, basis="Z"))
graph.export("result", result)
```

The two `.when(...)` expressions are structural conditional regions after
desugaring. They are not alternate `QecInstr` names. Selecting
`"physical_conditional"` instead of `"frame_update"`, when supported, would
retain the same authored logical semantics but produce a different resolved
plan.

A reusable teleportation protocol should generally expose its byproduct more
directly:

```python
destination, correction = graph.append(
    surface.teleport_raw(
        source=source,
        destination=destination,
    ).using("bell_measurement")
)

destination = graph.append(
    surface.apply_byproduct(
        patch=destination,
        byproduct=correction,
    ).using("frame_update")
)
```

Here `teleport_raw` declares an uncorrected teleportation channel plus the
relation between its measurement outcomes and `correction`.
`apply_byproduct` consumes that token and returns a logically corrected block.
A convenience `surface.teleport(...)` may be a hierarchical composite that
performs both calls and exposes only the corrected destination. Keeping the raw
form available is useful for experiments that deliberately defer, combine, or
route corrections.

For a genuinely branch-dependent protocol, the functional core API makes the
resource join explicit:

```python
(data,) = graph.if_else(
    condition=needs_s,
    inputs=(data,),
    then=lambda region, data: (
        region.apply(surface.s.using("injection"), patch=data),
    ),
    otherwise=lambda region, data: (data,),
)
```

The provisional Python spelling can change, but the serialized Rust IR must
retain the condition, both regions, their block arguments, and their yielded
values. An omitted `otherwise` may be inferred only for the restricted
type-preserving `.when(...)` shorthand.

## Program state and validation

QEC block values use linear SSA-style lifetimes. `InstrGraph` understands only
that a linear input version may be consumed once and that a QEC instruction application
produces zero or more typed output versions:

```text
%block0 = declare CodeBlock[SurfacePatch, declared]
%block1 = call %surface.prepare_z(%block0)
%block2 = call %surface.h(%block1)
%result = call %surface.measure_z(%block2)
```

Names and state meanings remain owned by the instruction set. The graph does not
know that `prepare_z` changes a block from "declared" to "active" or that
`measure_z` is destructive. It merely checks the opaque input and output types
declared by their instruction signatures and enforces linear use. This maps
directly to Guppy's ownership rules and permits instructions with different
lifecycles, including merge/split or code switching.

Validation occurs when an operation is appended, with a full validation pass
before lowering. Generic program/graph validation checks:

- unique block labels;
- instruction input arity and opaque signature types;
- single consumption of linear values;
- result arity and use-before-definition;
- condition type, region argument binding, and matching branch yields;
- single consumption and replacement of linear values across every branch;
- legal copying of classical values according to their declared value kind;
- attribute conformance to the instruction's declared schema;
- well-typed composition of block interfaces and declared logical
  transformations.

Instruction-set resolution checks:

- availability of an implementation for every QEC instruction application;
- implementation-specific constraints such as matching geometry for
  transversal CX, square-patch requirements, or compatible boundaries;
- supported bases and positive round counts where those concepts apply;
- legal parallel groups and resource requirements;
- availability of detector-boundary rules for each resolved implementation
  sequence;
- support for every shot-time conditional effect required by the selected
  implementation and target;
- complete discharge or deliberate export of Pauli byproducts and tracked
  frames;
- a semantic-verification strategy for the selected plan against the
  instruction's declared logical transformation.

Validation therefore has two phases. Code-independent structural and linearity
errors are reported while authoring. Implementation compatibility is checked during
deterministic instruction-set resolution, because the graph does not interpret
the instruction identity and the same contract can be legal under one
implementation and illegal under another.

This is stricter than the current `add_memory` model, where the first and last
memory operations implicitly decide when preparation and final measurement
occur.

## `InstrGraph` IR

The core semantic operation has one named-instruction form:

```python
@dataclass(frozen=True)
class InstrCall:
    definition: InstrDefRef
    inputs: tuple[PortBinding[ValueRef], ...]
    parameters: tuple[ParamBinding, ...]
    implementation_constraint: ImplConstraint | None
    outputs: tuple[PortBinding[ValueRef], ...]
```

The graph may additionally contain declarations, constants, regions, and
result exports as structural nodes, but it has no subclasses for particular
domain operations. Preparation basis, syndrome-extraction rounds, physical
qubit indices, stabilizer schedules, and words such as "transversal" are typed
parameters, dialect semantics, or implementation constraints defined by the
owning instruction set.

A call records an optional implementation constraint but not the resolved
implementation itself. This permits the same authored program to be lowered
with two explicitly selected instruction sets while retaining reproducibility in
each resolved artifact.

Structured control is represented independently of named instructions. At
minimum the Rust IR needs `SeqRegion`, `ParallelRegion`, `RepeatRegion`, and
`IfRegion` nodes with explicit region arguments and yields. An `IfRegion`
contains a typed predicate plus two regions; it does not store a surface gate
name. `.when(...)` is builder sugar and is absent after elaboration. Classical
expressions used as predicates are typed SSA operations or explicitly declared
classical instruction calls, rather than arbitrary Python callbacks.

The elaborated IR also records whether a predicate is compile-time or
shot-time. Compile-time branches may be specialized while retaining provenance.
Shot-time branches survive into Guppy/HUGR control flow unless semantic
lowering proves that they are only Pauli-frame bookkeeping. That proof and its
rewrite—from a conditional logical Pauli to a frame expression—must be
recorded in the resolved plan.

The QEC dialect retains logical intent even when a backend could be produced by first
flattening to physical gates. In particular, Guppy generation must not accept
an arbitrary `TickCircuit` as its input: that would discard patch ownership,
structured arrays, logical gate boundaries, and qubit lifetime information.

## Resolved instruction program and physical backend routes

Rendering each instruction independently into Guppy and TickCircuit would
recreate the semantic drift that already exists between the memory and
transversal generators. Both backends should instead consume a shared,
Rust-owned physical realization assembled from the selected instruction plans:

```text
InstrProgram
    -> parameter/type/hierarchy/dialect elaboration
    -> ElaboratedInstrProgram
    -> InstrSet resolution + QEC semantic verification
    -> ResolvedInstrProgram
        +-> PHIR + optional SpaceTimePlan/shape projections
        |
        +-> PhysicalCircuitPlan
              +-> direct Rust TickCircuit lowering
              |     -> normalized TickCircuit
              |     -> DagCircuit / DEM / Stim-compatible exports
              |
              +-> generated Guppy
                    -> HUGR / QIS lowering
                    -> runtime QIS trace
                    -> normalized TickCircuit
                    -> DagCircuit / DEM / Stim-compatible exports
```

Elaboration first produces an `ElaboratedInstrProgram` with canonical
parameters, concrete output value types, resolved definition symbols, explicit
profile preferences, and retained hierarchy/source paths. Instruction-set
resolution then produces a `ResolvedInstrProgram` containing:

- code-block instances and linear value versions;
- typed classical values, structured regions, and predicate timing class;
- selected QEC protocol implementation IDs and options;
- inspectable `QecInstrPlan` artifacts for the selected calls;
- `SpaceTimeRealization` artifacts, abstract `SpaceTimeShape` projections,
  feasible alternatives, and refinement/provenance links when implementations
  expose them;
- instruction-application and syndrome-extraction-round boundaries;
- detector and observable definitions in terms of instruction measurement
  identities;
- result-tag associations;
- resolved check plans, logical/physical Pauli-frame versions, and ancilla
  policies;
- conditional-effect lowering decisions, outstanding symbolic frame
  expressions, and materialization points;
- the information needed to emit typed PHIR and construct a shared physical
  circuit plan for direct TickCircuit and Guppy lowering.

This resolved program is not a second public authoring IR. Each implementation
plan describes hardware-independent realization details such as ancilla
lifecycle, interaction partial order, measurement semantics, QEC protocol phases,
and decoder dependencies. A Rust `PhysicalCircuitPlan` composes those selected
plans into physical operations, resource lifetimes, scheduling constraints,
structured control, and the measurement/detector/observable/provenance ledger.
It retains logical-block and instruction boundaries even when a backend later
flattens them.

`PhysicalCircuitPlan` is the common physical-lowering seam, not an arbitrary
already-flattened `TickCircuit`. It contains enough structure for Guppy's linear
patch values and control flow while also supporting deterministic Rust
scheduling and direct TickCircuit emission. This prevents a Python Guppy
renderer and a Rust Tick renderer from independently rediscovering ancillas,
check order, frame effects, measurements, or detector boundaries.

It is also the executable detailed form of the previously described
`SpaceTimeRealization`, not a competing representation. Before scheduling, it
may retain partial orders and relative geometry; after a scheduling context is
applied, its physical operations have concrete ticks and locations. Those same
scheduled operations project both to the detailed 2+1D circuit view and to the
normalized TickCircuit.

The PHIR lowering is a first-class Rust output of the same resolved artifact,
not a Python reconstruction. It should emit registered dialect operations and
types with source maps to program/module/call/value IDs. PHIR, direct Tick, and
Guppy are sibling outputs with explicit capability sets; none needs Python in
order for the Rust program to produce a usable physical circuit and DEM.

The direct backend deterministically schedules the physical plan under an
explicit direct-lowering context and returns a `GeneratedTickProgram` containing
the normalized `TickCircuit`, measurement ledger, detector/observable metadata,
selected implementation IDs, schedule provenance, and links to logical calls.
Its TickCircuit is authoritative for that direct route and may be passed to the
native DEM builder without compiling or executing Guppy.

The Guppy backend emits the same physical plan through structured Guppy. Guppy
compilation, QIS lowering, and a selected target/runtime may legally reschedule
operations. For that route, the executed QIS trace is authoritative for physical
gate and measurement order, and DEM construction uses the normalized
TickCircuit recovered from the trace following `build_dem_from_guppy`.

Both routes must make their schedule origin visible—for example,
`generated_tick.tick_circuit` with `schedule_source="direct"` versus
`trace.tick_circuit` with `schedule_source="qis_trace"`. Equivalence testing is
required while implementations migrate and remains an important regression
oracle, but a verified direct TickCircuit is not permanently subordinate to or
dependent on Guppy.

Some resolved programs exceed TickCircuit's control model. Static loops can be
unrolled, compile-time branches specialized, and virtual Pauli corrections
absorbed before direct emission. A genuinely dynamic branch or loop must either
be lowered under an explicit supported convention or produce a capability
diagnostic identifying the unsupported control region. The availability of a
Guppy backend does not permit the direct backend to silently erase adaptive
semantics.

### Current PECOS Stim paths

PECOS does support converting an annotated TickCircuit to a Stim circuit with
`tick_circuit_to_stim`. That converter can consume a TickCircuit obtained from
the QIS trace after Clifford normalization, and the slow traced-QIS integration
tests exercise that route to build a Stim/PyMatching reference decoder.

It is not the universal or usual intermediate for Guppy DEM construction:

- `GuppyDemBuilder.build` traces Guppy/QIS, normalizes the TickCircuit, resolves
  result identities and detector metadata, and sends that circuit to the
  PECOS-native DEM builder without first generating a Stim circuit;
- `generate_stim_from_patch` and `build_stim_circuit_from_patch` render Stim
  directly from surface geometry/schedules;
- `LogicalCircuitBuilder.to_stim` converts its directly generated reference
  TickCircuit, not a Guppy/QIS trace;
- SLR also has an independent AST-to-Stim path.

The proposed instruction pipeline supports native DEM construction from either
a directly generated normalized TickCircuit or the normalized TickCircuit
recovered from a Guppy/QIS trace. Each uses the same resolved
measurement/detector/observable identities and records its schedule source.
Stim-circuit export from either TickCircuit is a supported optional export and
an important equivalence oracle, while Stim-compatible DEM text remains an
export of the native DEM itself.

## Measurement identity

Detector and observable definitions must follow the repository's measurement
identity design rather than record positions. Each instruction-emitted
measurement creates a logical measurement reference. The Guppy/QIS trace path
resolves it to runtime identity:

- resolved instruction program: a logical measurement reference;
- Guppy: a stable scalar result tag and occurrence association;
- QIS trace and normalized Tick/DAG: `MeasId` plus runtime order;
- Stim: a record offset computed only at export;
- DEM: the resolved measurement identity expected by the analysis pipeline.

Suggested Guppy tags include the patch label, preparation epoch, operation,
round, stabilizer kind/index, and occurrence. The exact spelling is an export
format and must not become the identity itself. Repeated human-readable tags
map to ordered measurement identities as described in
`design/measurement-id-annotations.md`.

Aggregate user-facing results such as per-round `synx` and `synz` arrays may be
emitted in addition to scalar identity tags. Detector construction uses the
identity tags, never the aggregate-array ordering.

## PHIR lowering

`InstrProgram` is a high-level semantic/elaboration artifact; it should
generate PHIR rather than grow into PHIR. The Rust lowering maps:

- `InstrProgram` entry points and `InstrModule` definitions to PHIR modules and
  functions;
- `InstrGraph` regions, arguments, calls, and yields to PHIR regions, blocks,
  instructions, operands, results, and terminators;
- generic value types and dialect types to registered PHIR types;
- resolved QEC calls and plans to typed QEC dialect operations, optionally
  followed by progressive lowering to physical quantum/classical operations;
- instruction source identities to a separate typed provenance map consumed by
  diagnostics, Guppy/QIS identity tracking, and visualization.

The PHIR output must not encode essential semantics only as arbitrary strings.
The existing PHIR QEC dialect is useful scaffolding, but its generic
`logical_gate` operation and flexible attributes are not enough for code-block
type transformations, logical maps, frame transfer, selected implementation
identity, measurement identity, or protocol-plan references. Those need
registered operation/type definitions or typed interfaces with verifier hooks.

Progressive lowering may retain hierarchy and high-level resolved QEC
operations for inspection, then expand selected implementation plans into
lower-level QEC, quantum, and classical operations. Flattening is an optional
backend transformation, not the definition of PHIR emission. Each stage must
remain independently serializable and verifiable.

Python only requests and wraps the Rust-produced PHIR artifact:

```python
resolved = program.resolve(context=surface.lowering_context())
phir_module = resolved.to_phir()
```

The corresponding Rust operations should be equally direct:

```rust
let resolved = program.resolve(&surface.lowering_context())?;
let phir_module = resolved.to_phir()?;
```

PHIR generation is an additional authoritative semantic output. Direct
TickCircuit lowering and executed Guppy/QIS tracing each produce a concrete
physical schedule with an explicit origin; equivalence tests compare them and
decide when future PHIR-based backends may take over another production route.

## Direct TickCircuit lowering

The Rust API should make the Guppy-independent route ordinary rather than
special-purpose:

```rust
let resolved = program.resolve(&surface.lowering_context())?;
let physical = resolved.lower_physical(&direct_context)?;
let generated = physical.to_tick_program()?;
let tick: &TickCircuit = generated.tick_circuit();
let dem = generated.build_dem(&noise_model)?;
```

Python should be a thin wrapper over those same artifacts:

```python
resolved = program.resolve(context=surface.lowering_context())
physical = resolved.lower_physical(context=direct_context)
generated = physical.to_tick_program()
tick = generated.tick_circuit
dem = generated.build_dem(noise_model)
```

Convenience methods such as `resolved.to_tick_circuit(...)` and
`resolved.build_dem(..., via="direct")` may compose these steps, but the
inspectable `PhysicalCircuitPlan` and `GeneratedTickProgram` artifacts must
remain available. Direct lowering must run entirely in Rust and must not import
Guppy or Python.

## Guppy lowering

The Guppy backend generates a single Python module with:

1. one struct and helper family per distinct patch geometry and lowering
   policy;
2. preparation functions returning owned patch values;
3. syndrome-extraction functions returning the owned patch and syndrome when
   required by Guppy's linear type rules;
4. logical gate helpers;
5. destructive measurement helpers;
6. one experiment entry point composing the helpers in a deterministic
   topological order consistent with dataflow and explicit control constraints;
7. detector, observable, and measurement-layout sidecar metadata.

Patch labels are instance names, while generated type names derive from a
stable structural geometry key. Two patches with identical geometry share a
type and helper definitions but have distinct values. Different orientations,
layouts, dimensions, check plans, or ancilla policies must not collide.

Syndrome loops may initially be source-unrolled, matching current Guppy
generation. A later structured-loop representation can reduce source and
compile size, but must preserve one fresh measurement identity per dynamic
measurement.

`to_guppy_source()` returns deterministic source and `to_guppy()` may retain
the existing convention of returning an entry point. The new
`compile_to_guppy()` API returns a `GeneratedGuppyProgram` artifact containing
source, entry point, resolved QEC protocol implementations, measurement layout, and
detector/observable metadata. Its `trace_qis()` and `build_dem()` methods use
the same generated program and make the traced analysis route explicit. Guppy
emission consumes the shared physical plan plus retained structured
block/control information; it is not generated by converting the direct
TickCircuit back into Guppy.

## Guppy/HUGR import into instruction modules

PECOS should also support importing a Guppy function for composition as an
`InstrModule`, but the stable compiler boundary should be compiled HUGR rather
than Guppy's Python AST or Python function identity:

```text
Guppy function
    -> Guppy compilation
    -> HUGR package/function
    -> Rust HugrInstrImporter + explicit import registry
    -> imported InstrModule or opaque external module
```

The Python convenience can compile the function and pass serialized HUGR plus
optional metadata into Rust:

```python
module = InstrModule.from_guppy(
    bell_subroutine,
    imports=physical_quantum_imports,
)
out = graph.call(module, q0=q0, q1=q1)
```

The Rust API starts at the portable boundary:

```rust
let module = InstrModule::from_hugr(
    &hugr_package,
    &physical_quantum_imports,
)?;
```

`HugrInstrImportRegistry` maps stable HUGR operation/type identities to
`InstrDefId` and `ValueTypeId` values. The importer must not dispatch on a
display name such as `"CX"` without a registered extension-qualified mapping.
HUGR linearity, function signatures, regions, dataflow, conditionals, and
supported loops map naturally to module ports, SSA values, and structured
regions. Source HUGR node/function identities are retained in provenance.

Three import outcomes should be explicit:

1. **Structural import:** every required type, operation, and control construct
   is supported, so the function body becomes an inspectable `InstrModule`.
2. **Opaque external module:** the typed function signature and embedded HUGR
   artifact are retained, but the body is not represented as `InstrGraph`.
   Backends that understand HUGR may call or link it; other backends reject it
   with a capability diagnostic.
3. **Rejected import:** the signature cannot be represented safely, a linearity
   or ownership fact would be lost, or policy forbids unknown operations. No
   partial module should be returned as if it were complete.

This distinction prevents `InstrProgram` from growing into a duplicate of
HUGR/PHIR merely to accept every Guppy construct. General memory, arbitrary
functions, unsupported control flow, or extension operations may remain as an
opaque HUGR module or be imported into PHIR instead of `InstrGraph`.

Most importantly, structural import recovers program structure, not QEC intent.
A Guppy function over individual qubits imports naturally at a physical quantum
dialect level. It does not become `surface.cx`, `surface.syn_extract`, or a
typed surface-code patch transformation merely because its gates resemble a
known circuit. Turning an imported module into a `QecInstrImpl` requires an
explicit QEC instruction contract, code-block/port mapping, support predicate,
frame-transfer behavior, detector/measurement semantics, and semantic
verification evidence.

There are two reliable ways to recover higher-level QEC structure:

- Guppy authored or generated with registered PECOS annotations that identify
  QEC block types, instruction boundaries, implementation IDs, logical/frame
  effects, and measurement provenance;
- a PECOS-generated Guppy artifact accompanied by its serialized
  `InstrProgram`/resolved-plan sidecar, in which case PECOS should reload the
  original artifact and verify it against the HUGR digest rather than attempt
  heuristic decompilation.

For unannotated third-party Guppy, users can still wrap the imported module in a
new typed instruction definition and supply the semantic contract explicitly.
Pattern-based lifting may later be offered as a fallible analysis that produces
candidates and proof obligations; it must never silently assert recovered QEC
semantics.

Existing PECOS work demonstrates feasibility but should be consolidated rather
than copied: `guppy_to_hugr` already obtains HUGR bytes, Rust's
`hugr_to_dag_circuit` imports a restricted straight-line quantum subset, and
the Python HUGR-to-SLR-AST path recognizes some structured conditionals and
loops. The proposed importer moves the reusable typed subset and diagnostics
into Rust and targets the new module/region/value contracts.

## Relationship to existing code

### `SurfacePatch`

Remains the geometry authority. No stabilizer or logical-support calculation
is copied into the new program, instruction set, or renderer. A surface QEC protocol
implementation receives the concrete `SurfacePatch` specifications associated
with its operand logical values and validates them through `supports`.

### `SurfaceInstrSet`

This is the new home for surface instruction and QEC protocol selection. Its
bare constructor has no hidden defaults. The explicitly selected conventional
profile uses `surface.syndrome_szz` for `surface.syn_extract` and
`surface.transversal_cx` for `surface.cx`; callers can use `.using(...)` to
override a configured choice for one call. Initial implementation IDs are
expected to include:

```text
surface.prepare_x / surface.prepare_y / surface.prepare_z
surface.syndrome_cx / surface.syndrome_szz
surface.logical_pauli
surface.transversal_h
surface.fold_transversal_s / surface.fold_transversal_sdg
surface.transversal_cx
surface.measure_x / surface.measure_y / surface.measure_z
```

An implementation may expose typed options such as check plan, ancilla budget,
Clifford-frame policy, or boundary orientation. The instruction set owns any
explicitly configured defaults and selection policy; the `SurfacePatch` does
not. These settings must be introspectable and serialized with compilation
provenance.

### `LogicalCircuitBuilder`

Its operation vocabulary and detector-boundary implementation are the nearest
existing prototype. Migration should extract reusable planning logic from its
private `_CircuitGenerator`, rather than write a second implementation.

Its current API remains supported. Once the new traced Guppy lowering is
equivalent, the compatibility mapping is approximately:

```text
first add_memory  -> prepare + syndrome_rounds
middle add_memory -> syndrome_rounds
last add_memory   -> syndrome_rounds + measure
```

This mapping is only a compatibility rule; new code should use explicit
lifecycle operations.

### `pecos.guppy_gen.surface`

Its patch helper emission, check-plan lowering, ancilla batching, result tags,
and module-loading utilities should be reused or extracted. The new renderer
must not fork stabilizer scheduling.

### Specific Guppy factories

`make_surface_code` and surface transversal factories remain shortcuts. After
parity tests pass, they should build an `InstrProgram` using a
`SurfaceInstrSet` and delegate to its Guppy backend. Their current signatures
and result keys should remain stable.

### SLR

No dependency is proposed. SLR demonstrates the value of a renderer boundary
and may later gain adapters to or from `InstrProgram`. Its
`If(condition).Then(...).Else(...)` form and Steane injection/feed-forward
examples are useful authoring precedent. The new Rust IR should not copy SLR's
block model directly: it needs typed classical SSA values, explicit linear
region arguments and yields, and a semantic distinction between physical
conditional execution and Pauli-frame updates. Completing SLR's surface gate
placeholders is independent work.

## Relationship to issues #508-#516

This proposal is not a parallel replacement for the layered-QEC work. It is a
focused design for the logical-program, QEC-instruction, protocol,
implementation-plan, and executable-lowering seams in that work.

### [#508: layered abstractions and HDL-style QEC](https://github.com/PECOS-packages/PECOS/issues/508)

The proposed types map onto the RFC ladder as follows:

| #508 level | This proposal |
|---|---|
| `AbstractCode` | Referenced by logical value types; not redefined here |
| `EncodedProgram` / logical ISA | `InstrProgram` whose graph calls logical-level `QecInstr` interfaces |
| `QecProtocol` | A selected `QecInstrImpl`, potentially expressed hierarchically as more QEC instruction applications |
| `ImplementationPlan` | Concrete `QecInstrPlan` produced by the selected implementation |
| Circuit/hybrid program | `PhysicalCircuitPlan`, generated Guppy, and compiled HUGR/QIS |
| Mapped/timed execution | Direct Rust schedule or platform/runtime lowering observed through QIS trace |
| Analysis product | Normalized TickCircuit, DagCircuit, and DEM with recorded schedule source |

This design adopts #508's explicit-artifact rule: the authored
`InstrProgram`, elaborated program, instruction definition, selected QEC
protocol, implementation plan, generated PHIR/Guppy program, QIS trace, and DEM are distinct
typed artifacts with provenance links. They must not become one object with
optional fields for every level.

`QecInstrSet` is a library and resolution environment, not another rung in the
ladder. It may expose logical interfaces and candidate QEC implementations, but
resolution must still produce distinct selected-protocol and
implementation-plan artifacts. The convenient word "protocol" must not
collapse an ideal logical request, its fault-tolerant strategy, and its
ancilla/gate plan into one mutable object.

Hierarchical QEC instruction applications provide the bridge: an implementation of a
logical-level interface may expand into an `InstrModule` of lower-level
QEC instruction applications before those calls are themselves planned. The graph remains
oblivious to their names and abstraction levels; the owning instruction sets and
typed artifacts define the refinement boundary.

### [#509: implementation umbrella](https://github.com/PECOS-packages/PECOS/issues/509)

The recommended single-patch vertical slice is a candidate child workstream of
#509:

```text
SurfacePatch-backed logical value
    -> InstrProgram / InstrGraph QEC instruction applications
    -> ElaboratedInstrProgram
    -> selected surface implementation plans
    -> PHIR and shared PhysicalCircuitPlan
       +-> direct normalized TickCircuit
       +-> Guppy -> QIS trace -> normalized TickCircuit
    -> DEM
```

It satisfies the umbrella's requirement for multiple implementations,
provenance, and a complete path into the existing execution stack, while
deliberately not claiming to complete mapping or timed target execution.

### [#510: abstract code model](https://github.com/PECOS-packages/PECOS/issues/510)

This proposal depends on #510 for mathematical code identity, parameterized
families, validation, and provenance. The provisional `CodeSpec` terminology in
the examples must be reconciled with the accepted `AbstractCode`,
`StabilizerCode`, and `StabilizerCodeSpec` responsibilities.

`SurfacePatch` is geometry-bearing and therefore is not interchangeable with a
Level-1 abstract code. A logical value may refer to either an abstract code or
a refined patch/geometry artifact according to its instruction signature. A
surface instruction requiring boundaries or coordinates must accept the latter;
it must not infer geometry from a bare abstract stabilizer code.

### [#511: SLR/Zlup and a Rust-native QEC HDL](https://github.com/PECOS-packages/PECOS/issues/511)

`InstrDef`, opaque `InstrCall`, `InstrModule`, and `InstrGraph` are the common
structural model that an HDL, Python builder, Rust builder, or serialized schema
can target. QEC attaches `QecInstrSemantics(logical_transform,
frame_transfer, ...)` to definitions rather than changing the graph. The
examples in this document are Python-shaped but are not intended to make Python
objects the interchange format.

The HDL exploration should test whether parameterized instruction signatures,
multiple implementations, implementation constraints, linear values, and
derived output types can be expressed naturally. SLR/Zlup may become one front
end or implementation-plan notation; `InstrProgram` should not depend on its
surface syntax.

The cell/netlist analogy above provides the likely HDL structure: instruction
definitions are cell interfaces, bound calls are instances, linear QEC values
are ownership-carrying nets, and resolution is technology mapping. An HDL front
end should serialize to the same Rust `InstrProgram` rather than introduce a
parallel semantic model.

### [#512: protocol and syndrome-extraction plans](https://github.com/PECOS-packages/PECOS/issues/512)

This is the issue with the strongest direct overlap. The division proposed
here is:

- `QecInstr`: named, typed code-block interface and intended logical
  transformation at a declared abstraction seam;
- `QecInstrImpl`: a selectable QEC strategy satisfying that
  interface, optionally expressed through hierarchical QEC instruction applications;
- `QecInstrPlan`: inspectable ancilla strategy, interaction
  order, phases, measurement semantics, decoder dependencies, and adaptive
  structure;
- `QecInstrSet`: definitions, candidate implementations, explicit selection
  policy, and introspectable configured defaults;
- `ResolvedInstrProgram`: selected plans composed according to the opaque
  calls and linear values in an `InstrProgram`.

The first implementation should be developed as part of #512 rather than
introducing separate competing protocol types under `pecos.qec.surface`.

### [#513: synthesis and semantic verification](https://github.com/PECOS-packages/PECOS/issues/513)

The boundary for #513 is:

```text
ResolvedInstrProgram / QecInstrPlan
    -> shared PhysicalCircuitPlan + semantic source map
       +-> direct normalized TickCircuit
       +-> generated Guppy -> HUGR/QIS -> QIS trace
    -> resolved measurement identity map for either route
```

The generated artifact must preserve source identities for QEC instruction applications,
checks, logical operations, measurements, detectors, observables, and decoder
interactions. Verification results—intended-check measurement, logical action,
ancilla cleanup, detector correspondence, or fault propagation—attach as
certificates or analysis products at this boundary. Successful Guppy
compilation alone is insufficient.

In particular, verification compares the selected implementation plan's
induced logical map with the `QecLogicalTransform` declared by the instruction.
For syndrome extraction this is an explicit identity obligation; for H or CX it is
the corresponding Clifford action; and for preparation or measurement it is
the declared state or observable semantics.

### [#514: mapping and timed scheduling](https://github.com/PECOS-packages/PECOS/issues/514)

QEC protocol implementation plans remain hardware-independent. Target topology,
native decomposition, placement, timing, idles, and feedback latency enter
through the existing HUGR/QIS platform and runtime lowering selected by the
`LoweringContext`.

For a target/runtime route, the QIS trace is authoritative because it reflects
that route's chosen mapped schedule. For direct lowering, the
`GeneratedTickProgram` is authoritative for its explicit scheduling context.
Both artifacts retain links through the shared physical plan to QEC instruction
applications and code entities. Different direct policies, runtimes, or target
choices produce distinct scheduled artifacts and potentially distinct DEMs,
even from the same logical graph.

### [#515: qecdb importer](https://github.com/PECOS-packages/PECOS/issues/515)

qecdb feeds only the Level-1 abstract-code side of #510. An imported code can be
an input to an instruction whose signature accepts a generic abstract stabilizer
code, but it cannot be passed to a `SurfacePatch` instruction merely because its
parameters resemble a surface code. Geometry, extraction strategy, decoder,
and implementation choice require explicit refinement after import.

No qecdb transport, catalog metadata, or networking belongs in
`InstrProgram`, `QecInstr`, or `QecInstrPlan`.
Provenance follows the abstract code by reference.

### [#516: architecture documentation](https://github.com/PECOS-packages/PECOS/issues/516)

This design can supply the logical-instruction/protocol/Guppy-trace portion of the eventual
architecture documentation, but user-guide material should wait for accepted
names and implemented APIs. The first documented executable example should show
the complete provenance chain from patch/check identities through protocol
calls, implementation selection, Guppy result tags, QIS measurement identities,
TickCircuit detectors, and the resulting DEM.

Noise attachment follows #508/#516: phenomenological noise belongs at the
QEC-protocol/round seam, circuit-level noise attaches after QIS tracing, and
target-calibrated noise attaches to the mapped/timed runtime artifact. These are
experiment overlays, not different instruction, protocol, or code types.

### Recommended issue sequencing

1. Reconcile the logical value's code/patch type references with #510's
   Level-1 contract; use a narrow adapter if #510 is not yet complete.
2. Audit PHIR's module/region/SSA/dialect facilities and land the minimal
   Rust-owned `InstrProgram`, `InstrGraph`, definition, call, and deterministic
   resolution contracts without duplicating PHIR's general compiler model.
3. Add QEC semantic/type/implementation interfaces in `crates/pecos-qec` as a
   first slice of #512; add bindings only after the Rust API and serialization
   tests pass.
4. Implement `ResolvedInstrProgram` -> typed PHIR emission with source maps.
5. Implement the single-patch shared physical plan with both direct
   TickCircuit -> DEM and Guppy -> QIS trace -> TickCircuit -> DEM paths and a
   common semantic source map under #513.
6. Exercise two surface implementations of one instruction so alternate
   implementation support is demonstrated rather than only modeled.
7. Carry the source/provenance map through one target/runtime path as an initial
   integration with #514.
8. Feed the resulting semantic model and examples into #511; do not make the
   executable slice wait for a dedicated HDL syntax.
9. Treat #515 as parallel Level-1 input work and #516 as the documentation
   consolidation after names and boundaries settle.

## Alternatives considered

### Put gates directly on `SurfacePatch`

Rejected. It conflates immutable code geometry with protocol choice, makes
multiple implementations awkward, and encourages patch methods to accumulate
backend policy.

### Make the entire circuit surface-specific

Useful as a facade, but rejected as the only IR. Preparation, logical Clifford
operations, measurement, block lifetimes, and implementation selection are not
surface-specific. Surface geometry and operations such as boundary merging
remain extensions rather than generic concepts.

### Make one universal code-agnostic instruction set

Rejected. Support is a relation between an operation, all operand code specs,
and a lowering context. A nominal universal instruction set would either hide this
dispatch or become a global registry. Explicit composable instruction sets keep
the selection inspectable and reproducible.

### Lower the existing TickCircuit to Guppy

Rejected as the architectural seam. It can reproduce physical gates but has
already lost logical-block ownership, chosen protocol implementations, structured
patch values, and some lifetime intent needed by Guppy's linear types.

### Implement the semantic model in Python and port it later

Rejected. It would create two periods of semantic churn, encourage Python-only
extension points, and make Rust consumers depend on a later translation of the
IR and diagnostics. The Rust API, serialization, and tests should land first;
PyO3 and Python ergonomics follow as a thin view over those established types.

## Implementation stages

### Stage 0: PHIR boundary audit

- Inventory PHIR's module/function/region/block/instruction, SSA, custom type,
  custom operation, dialect, serialization, and source-location contracts.
- Decide the narrow Rust crate/module boundary for `InstrProgram` and a
  one-way lowering interface into PHIR without introducing circular
  dependencies.
- Specify which structural types/IDs can be reused directly and which
  high-level definition/resolution concepts remain intentionally outside PHIR.
- Add a design test fixture showing the expected PHIR produced from a tiny
  resolved generic program before adding QEC-specific lowering.

### Stage 1a: Rust `InstrProgram` and surface instruction-set skeleton

- Implement Rust-owned `InstrProgram`, `InstrGraph`, typed IDs and values,
  linear-use validation, and the opaque `InstrCall` in the generic layer.
- Implement `InstrModule`, `InstrModuleBuilder`, typed module ports,
  `InstrModuleCall`, stable instance paths, and a first-version acyclic
  instance-graph verifier.
- Include copyability traits for classical values plus serializable structured
  region interfaces; implement `IfRegion` verification even if adaptive Guppy
  lowering lands in a later stage.
- Define versioned `LogicalPauliFrame`, symbolic Boolean frame expressions,
  `FrameStateRef`, and instruction-level logical-frame transfer contracts in
  the QEC dialect. Reserve mapped `PhysicalPauliFrame` as a distinct artifact type.
- Implement `QecInstrSet`, `QecInstr`, the Rust `QecInstrImpl` trait,
  `QecInstrPlan`, structured diagnostics, and deterministic resolution there.
- Implement the serializable `QecTypeExpr` algebra, substitution environment,
  constraint checking, and concrete output-type instantiation in Rust.
- Establish the canonical Rust `SurfacePatch` representation or a lossless
  migration adapter before surface implementations depend on geometry.
- Add Rust `SurfaceInstrSet` definitions, SZZ and CX extraction candidates, and
  transversal logical implementations.
- Add versioned `serde` contracts for `InstrProgram`,
  `ElaboratedInstrProgram`, `ResolvedInstrProgram`, plans, annotations,
  diagnostics, and provenance.
- Provide ergonomic typed Rust call builders and a complete Rust-only surface
  experiment example.
- Do not begin the public Python builder until Rust unit, serialization, and
  API tests for this layer pass.

### Stage 1b: Rust PHIR emission

- Lower `ResolvedInstrProgram` into a PHIR module entirely in Rust.
- Emit generic structure through PHIR module/function/region/block/SSA forms
  and domain semantics through registered typed dialect operations/types.
- Preserve program, module definition/instance, call, value, frame,
  measurement, annotation, and source identities in an explicit provenance map.
- Test serialization and verification of the emitted PHIR independently of
  Python and Guppy.

### Stage 1c: PyO3 bindings and Python parity

- Add bound wrappers in `python/pecos-rslib` under `pecos_rslib.instr` and
  `pecos_rslib.qec` for the Rust artifacts, IDs, builders, diagnostics, and
  introspection APIs.
- Re-export them through `pecos.instr` and `pecos.qec`; keep Python
  `LogicalBlock`, `parallel()`, `if_else()`, `InstrModule` definition, and
  keyword binding as thin ergonomic facades over Rust calls.
- Map structured Rust errors to stable Python exception classes without
  reimplementing resolution logic in Python.
- Require paired Rust/Python examples and byte-equivalent serialized programs,
  resolved programs, PHIR artifacts, defaults, and provenance.

### Stage 2: shared physical plan, direct TickCircuit, Guppy, and DEM

- Port or extract reusable patch-level planning into a Rust
  `PhysicalCircuitPlan` in `crates/pecos-qec` rather than either renderer.
- Implement Rust-only `PhysicalCircuitPlan` -> `GeneratedTickProgram` lowering,
  including the normalized TickCircuit, measurement ledger,
  detector/observable definitions, provenance, and native DEM construction.
- Implement deterministic Guppy source emission from that same physical plan
  and retained block/control structure.
- Keep only Guppy source loading/compilation and runtime orchestration in the
  Python bridge.
- Lower explicit prepare, syndrome rounds, logical Pauli, and measurement.
- Support both `build_dem(via="direct")` and the existing Guppy -> QIS trace ->
  normalized TickCircuit route.
- Match `make_surface_code` result identities, TickCircuits, detector metadata,
  and DEMs; compare the direct circuit with the traced Guppy schedule under a
  context where they are expected to agree.

### Stage 3: multi-patch Clifford gates

- Add transversal H, S/S-dagger, and CX to the shared physical plan and both
  direct TickCircuit and Guppy lowerings.
- Support parallel syndrome rounds over compatible patches.
- Match the existing logical builder's Tick traces and detector boundaries.

### Stage 4: space-time constraint view

- Add `SpaceTimeProgram` as instruction calls plus placement, adjacency,
  workspace, and partial-order constraints—not as a second semantic IR.
- Define `CodeGeometry` as an optional code interface with stable data-qubit and
  feature identities, incidence/adjacency, optional coordinates, dimensionality,
  and named boundaries; expose a two-dimensional grid/cellulation view for
  planar `SurfacePatch` values.
- Define `SpaceTimeRealization` artifacts and their `SpaceTimeShape` projections,
  resource quantities with explicit precision, mapped refinements, and resolved
  `SpaceTimePlan` artifacts that reference `QecInstrPlan` objects.
- Demonstrate one syndrome-extraction cell that starts with a surface patch's
  data-qubit geometry, adds protocol ancillas, and refines from rounds/partial
  order into a concrete 2+1D physical circuit schedule.
- Verify instruction plan -> mapped realization -> physical-circuit refinement while
  retaining live resources, measurement identities, frame effects, and
  detector/observable boundaries.
- Define a deterministic, serializable `SpaceTimeView` projection with stable
  provenance links and implement one headless coarse-shape/time-slice renderer;
  keep interactive notebook/web display as a thin consumer of that artifact.
- Spike an optional `pecos-spacetime-viewer` using Bevy over the same
  `SpaceTimeView`: synchronized time slice and 2+1D views, time scrubbing,
  resource-layer toggles, stable-ID picking/provenance, and hierarchy expansion.
- Keep Bevy out of the core dependency graph and evaluate native plus WASM
  packaging, feature selection, compile time, binary/web size, and notebook
  integration before selecting it as the supported interactive frontend.
- Prove that circuit-like and space-time authoring of the same experiment
  produce equivalent instruction graphs, Guppy traces, and source identities.
- Keep device placement and target timing out of this layer and hand the
  resolved constraints to the #514 mapping/timing work.

### Stage 5: compound QEC protocols and factory migration

- Expand S/T injection protocols into primitive operations.
- Add typed Pauli-byproduct values, `apply_byproduct`, observable/frame
  metadata, and concrete logical/physical frame propagation required at
  decision points.
- Lower predicated logical Paulis through both explicit physical-conditional
  and virtual frame-update implementations, with the selected policy recorded
  in the resolved plan.
- Add raw and corrected teleportation composites with explicit branch and
  byproduct semantics.
- Reimplement existing surface Guppy factories as compatibility wrappers where
  doing so preserves behavior.

### Stage 6: structured control and repetition

- Add typed `repeat`, `while_loop`, and `repeat_until` regions with explicit
  loop-carried linear values, conditions, yields, and result signatures.
- Evaluate a structured syndrome-round loop aligned with
  `design/measurement-id-system.md`, including stable dynamic measurement
  identities across iterations.
- Add `ChoiceShape` and `RepeatShape` projections, with explicit mutually
  exclusive occupancy policy and bounded/estimated/unknown resource totals.
- Lower genuinely adaptive non-Pauli regions only for runtimes, schedulers, and
  analysis paths that explicitly advertise support; otherwise reject them.
- Keep unrolled lowering as a compatibility/debug option.

### Stage 7: Guppy/HUGR module import

- Add a Rust `HugrInstrImporter` and explicit extension-qualified operation/type
  registry, with structural, opaque-external, and rejected outcomes.
- Expose `InstrModule::from_hugr(...)` in Rust and
  `InstrModule.from_guppy(...)` as thin Python compile-and-import convenience.
- Import one straight-line physical quantum function, one conditional, and one
  loop with linear qubit values and provenance-preserving HUGR node identities.
- Reuse or replace the supported logic in `hugr_to_dag_circuit` and the Python
  HUGR-to-SLR-AST converter rather than creating a third unrelated HUGR walker.
- Define registered PECOS HUGR annotations/sidecars for exact round trips of
  generated modules and require explicit semantic contracts before an imported
  physical module can implement a `QecInstr`.
- Keep unsupported general HUGR in PHIR or an opaque external module rather than
  expanding `InstrGraph` into a general-purpose compiler IR.

## Testing and acceptance criteria

Each supported operation sequence must be validated through the Rust-owned
physical plan and every backend that claims to support it. Direct TickCircuit
lowering is a production Rust path; generated Guppy and its QIS trace are a
separate production execution path. Cross-route equivalence is required where
their declared scheduling contexts and capabilities should produce the same
circuit, but neither backend is implemented as an independent QEC planner.

Required tests include:

- Rust-only construction, validation, resolution, planning, introspection, and
  serialization without importing or embedding Python;
- the generic `InstrProgram` crate/module compiling and testing without a
  dependency on `pecos-qec`;
- one small non-QEC instruction set exercising definitions, typed calls,
  modules, deterministic implementation resolution, and PHIR emission so the
  generic boundary is real rather than nominal;
- Guppy -> HUGR -> structurally imported `InstrModule` preserving function
  signature, linear value flow, supported conditionals/loops, hierarchy, and
  HUGR node/function provenance;
- unknown HUGR operations and types being resolved only through an explicit
  extension-qualified import registry, with deterministic opaque-or-error
  policy and no display-name guessing;
- opaque imported modules preserving their typed signature and exact HUGR
  payload while unsupported backends reject calls with a capability diagnostic;
- imported physical quantum modules not acquiring QEC block types, logical
  transforms, frame semantics, or detector meaning without explicit annotations
  or a user-supplied verified QEC contract;
- PECOS-generated Guppy/HUGR plus its sidecar reloading the original instruction
  module IDs and provenance after digest verification rather than heuristic
  circuit recognition;
- elaboration canonicalizing parameters, instantiating output types, expanding
  profile preferences, and rejecting unresolved required parameters;
- stable definition IDs, instance IDs, and hierarchy paths across
  serialization and deterministic elaboration;
- hierarchical composite implementations matching their external instruction
  contracts before and after flattening;
- two calls to one `InstrModule` producing distinct instance, block,
  measurement, frame-expression, and provenance IDs while sharing the same
  definition ID;
- direct module calls and provenance-preserving inlining producing equivalent
  typed instruction graphs and traced behavior;
- module construction rejecting implicit linear captures, recursive instance
  graphs, incompatible output types, and unresolved required parameters;
- specialization/cache keys including canonical type/parameter substitutions,
  instruction-set versions, and implementation profiles without merging
  instance-local identities;
- independent calls remaining unordered unless resource dependencies or
  explicit control regions constrain them;
- conditional regions rejecting non-Boolean predicates, implicit linear
  captures, missing yields, incompatible branch result types, and double use
  of a branch input;
- loop regions rejecting incompatible carried types, implicit linear captures,
  dropped or duplicated QEC blocks, non-Boolean dynamic conditions, and
  iteration-local values escaping their scope;
- static repeats producing equivalent unrolled and structured Guppy/QIS traces,
  measurement identities, detector definitions, and shape projections;
- dynamic loop lowering being rejected with an actionable capability diagnostic
  when the runtime, mapper, trace collector, or DEM path lacks required feedback
  support;
- conditional shapes treating branches as mutually exclusive, preserving their
  shared typed join, and reporting whether resource values are per-branch,
  reserved-union, minimum, maximum, expected, bounded, or unknown;
- dynamic loop shapes never claiming a finite exact total volume without a
  proven iteration bound;
- coarse and detailed `SpaceTimeView` artifacts preserving stable resource,
  instruction, hierarchy, measurement, detector/observable, and provenance IDs;
- deterministic headless visualization snapshots for a surface patch, syndrome
  extraction, transversal CX, a conditional choice, and a repeated region;
- the optional Bevy viewer and headless renderer consuming the same versioned
  `SpaceTimeView`, with stable PECOS IDs rather than viewer entity IDs surviving
  serialization and driving cross-selection;
- core Rust construction, resolution, PHIR/Guppy generation, and headless tests
  building without Bevy, a window system, or a graphics adapter;
- visualization labeling symbolic versus mapped coordinates, abstract rounds
  versus scheduled/calibrated time, and exact versus bounded/estimated/unknown
  resource quantities;
- branch visualization distinguishing alternative execution from reserved-union
  occupancy and loop visualization distinguishing symbolic from unfolded traces;
- `.when(predicate)` serializing identically to its explicit type-preserving
  `if_else` expansion and rejecting instructions without an unambiguous
  pass-through;
- compile-time condition specialization retaining provenance while shot-time
  conditions survive elaboration;
- typed annotations either following clone/specialize/inline/flatten rewrites
  through provenance maps or failing according to their declared retention
  policy;
- paired Rust and Python authoring examples producing byte-equivalent
  instruction programs, resolved plans, defaults, selection sources, and
  provenance;
- parity between Rust fluent call builders and Python keyword-bound calls;
- Python cursor and parallel-region wrappers delegating state transitions and
  validation to Rust rather than maintaining shadow state;
- stable translation of each structured Rust diagnostic into the corresponding
  Python exception and fields;
- canonical `SurfacePatch` geometry, stabilizers, logical supports, schedules,
  and structural keys agreeing across Rust and the compatibility Python API;
- generic instruction signatures preserving concrete patch parameters in their
  outputs;
- concrete instruction signatures rejecting other patch choices during type
  checking;
- parameterized code-switch signatures binding distinct input and output code
  types while preserving or explicitly mapping the logical interface;
- Rust unification and output instantiation for `Exact`, `Var`, `Parameter`,
  `SameAs`, and derived `Apply` type expressions;
- rejection of incompatible logical-qubit counts, logical ordering, lifecycle
  states, or basis maps during code switching;
- merge/split and code-switch calls consuming their input block versions and
  returning correctly typed replacement blocks;
- implementation constraints narrowing a generic instruction without changing
  its visible inputs or outputs;
- type-correct composition of instructions whose output QEC block types feed
  later input block types;
- explicit identity semantics for syndrome extraction, memory, and
  specification-preserving deformation;
- verification that each implementation plan realizes the instruction's
  declared logical transformation;
- logical-frame transfer for identity, H, S, CX, measurement, merge/split, and
  code switching agreeing with each instruction's declared logical transform;
- correlated symbolic frame predicates propagating across CX without losing
  their shared classical expression identities;
- physical-frame versions being tied to stable physical-qubit IDs and layout
  epochs, and projecting to the expected logical frame modulo stabilizer/gauge
  representatives;
- frame materialization, absorption into a measurement, and decoder-driven
  physical-frame updates remaining distinguishable in plan provenance;
- deterministic Rust-only `ResolvedInstrProgram` -> `PhysicalCircuitPlan` ->
  `GeneratedTickProgram` lowering without importing Python or Guppy;
- direct TickCircuit measurement identities, detector/observable metadata,
  result tags, instruction boundaries, and provenance agreeing with the shared
  physical plan;
- native DEM construction succeeding directly from `GeneratedTickProgram` and
  preserving the same detector/observable identities;
- direct lowering rejecting unsupported adaptive branches/loops without
  silently specializing, flattening, or erasing their semantics;
- deterministic Rust-only `ResolvedInstrProgram` -> PHIR generation;
- PHIR module/function/region/SSA verification and serialization succeeding
  without importing Python;
- PHIR operation/type identities and source maps preserving every relevant
  program, module, call, value, frame, and measurement ID;
- the single-patch PHIR lowering, direct Tick program, and Guppy/QIS trace
  agreeing on declared logical semantics and measurement identities before any
  PHIR physical backend is treated as production-authoritative;
- automatic resolution of a sole supported implementation;
- an actionable ambiguity error when multiple implementations support an
  instruction and neither `.using(...)` nor a configured default selects one;
- no fallback when an explicitly constrained or configured-default
  implementation is unsupported;
- introspection of candidates, configured defaults, resolved implementation,
  and selection source;
- equivalence of `SurfaceInstrSet.szz_transversal()` and explicit SZZ plus
  transversal default configuration;
- useful support diagnostics for wrong code family, geometry, orientation, or
  operand arity;
- explicit implementation hints and ambiguity rejection;
- one generic instruction program lowered under two explicitly chosen instruction
  sets;
- equivalent circuit-like and space-time-volume authoring producing the same
  instruction graph and traced behavior;
- surface-code data-qubit geometry remaining identical across instruction
  implementations while SZZ, CX-based, or target-specific protocols may add
  different ancilla/workspace layouts;
- abstract grid/cellulation coordinates remaining distinct from mapped physical
  device coordinates, with stable identities preserved through mapping;
- space-time realization input/output faces being derived from the instruction's
  parameterized code-block types, including geometry-changing and code-switching
  operations, rather than redeclared in a second signature;
- black-box shape analysis agreeing with expanded composite-realization analysis
  whenever the contract claims exact counts or duration;
- mapped realizations satisfying abstract occupancy, adjacency, duration, live-resource,
  frame-transfer, measurement, and detector/observable boundary contracts;
- zero-volume semantic instructions such as virtual frame updates not acquiring
  artificial data/ancilla occupancy;
- infeasible shape/realization alternatives being rejected during cooperative resolution and
  placement without violating explicit `.using(...)` choices or silently
  changing implementations;
- rejection of unsatisfiable adjacency, workspace, and temporal constraints;
- single-patch X- and Z-memory equivalence with `make_surface_code`;
- preparation and measurement in every supported basis;
- H followed by syndrome extraction with swapped stabilizer interpretation;
- S and S-dagger boundary behavior;
- two-patch transversal CX and observable propagation;
- logical teleportation with explicit X/Z feed-forward producing the same
  logical result under physical-conditional and frame-update implementations;
- a raw teleportation byproduct being consumed exactly once, propagated
  through Clifford operations, and reflected in final observable metadata;
- exported live blocks retaining their frame state, measured results absorbing
  the relevant frame interpretation, and standalone byproduct tokens being
  rejected when silently dropped;
- branch-aware diagnostics when a conditional non-Pauli operation cannot be
  represented by the selected Guppy/runtime/DEM path;
- multiple identical geometries with distinct patch identities;
- asymmetric and non-rotated geometry where the selected gates permit it;
- ancilla-budget and check-plan preservation;
- deterministic Guppy source generation;
- Guppy compilation and execution;
- direct generated Tick operations versus traced Guppy operations when both use
  an equivalence-constrained scheduling context;
- direct/traced detector, observable, and representative noisy DEM equivalence
  where their concrete schedules agree;
- non-positional measurement IDs and repeated result-tag occurrences;
- loud rejection of invalid lifecycle and unsupported operations.

The acceptance gate for migrating an existing factory is equivalence of its
ideal operation trace, measurement identity ledger, detector definitions,
observables, and representative noisy DEM—not merely successful Guppy
compilation.

## Open questions

1. How rich should the initial generic type interface be? At minimum it needs
   linear/copyable value kinds and instruction-supplied input/output signatures;
   QEC lifecycle states can remain QEC-dialect type tokens.
2. Should instruction-set resolution happen eagerly when calls are appended or
   only when a target and lowering context are known? The proposed compromise
   is eager resolution when fully specified, followed by mandatory resolution
   during lowering.
3. Should an instruction set be passed only to `lower`, or may an authoring helper
   attach its owning set as an explicit program dependency? The graph must
   not interpret it either way, and the resolved artifact must record the
   actual choice.
4. Should `compile_to_guppy()` be the artifact-returning name, or should the new
   API use `to_guppy()` and provide an explicit compatibility helper for callers
   expecting only the entry point?
5. Which current injection helpers are semantically complete enough to expose
   in the first public version?
6. Should parallel syndrome-extraction groups require identical round counts and
   implementations, or may the planner interleave heterogeneous schedules?
7. Which abstract volume properties belong on a `QecInstr` contract, and
   which can only be known after selecting a `QecInstrImpl`?
   The proposed default is a conservative interface-level constraint followed
   by a more concrete resolved `SpaceTimePlan`.
8. Should the generic Rust layer be a small `pecos-instr` crate or a narrowly
   scoped module owned by `pecos-phir`? Stage 0 should decide this from
   dependency direction and type/ID reuse, not naming preference.
9. Should initial PHIR emission preserve resolved high-level QEC operations,
   immediately expand implementation plans, or provide both verified lowering
   levels? The proposal favors both, with explicit artifact names.

## Recommended first implementation slice

Complete the Stage 0 PHIR boundary audit, then implement Stage 1a entirely in
Rust before adding public bindings. First prove the generic boundary with one
small non-QEC `InstrSet`. Then include at least two Rust `QecInstrImpl`
implementations for one QEC instruction so selection and support diagnostics
are real rather than interfaces with only one path. Freeze and test the
serialized artifacts, lower one resolved fixture to PHIR in Rust, then add the
Stage 1c PyO3 wrappers and prove Rust/Python construction parity.

Only then implement one complete single-patch vertical slice: Rust
`InstrProgram` QEC applications -> Rust-resolved plan -> Rust-generated PHIR
and shared `PhysicalCircuitPlan`, then both (a) Rust-direct normalized
TickCircuit -> DEM and (b) Guppy source/metadata -> thin Python compilation ->
QIS trace -> normalized TickCircuit -> DEM. Compare both routes' measurement
ledgers, detectors, observables, representative DEMs, and PHIR semantics against
`make_surface_code`, and compare their concrete schedules where the selected
contexts promise equivalence. This proves a Python-free PECOS-native route and
the Guppy execution route before adding more instructions.
