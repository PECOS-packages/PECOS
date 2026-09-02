# `InstrProgram` MVP: Rust-first surface memory

Status: proposed normative first implementation.

See [`instr-program.md`](instr-program.md) for the design overview and
[`instr-program-rationale.md`](instr-program-rationale.md) for non-normative
background and later extensions.

## 1. Goal and reference

The MVP constructs one distance-three, Z-basis, CX-syndrome surface-memory
experiment, resolves statically linked Rust implementations, composes a
portable `ProtocolProgram`, and lowers it to a normalized `TickCircuit`.

The existing Python surface stack is the migration oracle:

```python
patch = SurfacePatch.create(distance=3)
reference = generate_tick_circuit_from_patch(
    patch,
    num_rounds=3,
    basis="Z",
    interaction_basis="cx",
)
```

The core MVP ends at a generated Tick artifact. A separate integration test
passes that artifact and a fixed physical noise model to the existing native
DEM builder.

## 2. Proposed authoring API

This proposed Python API must become an executable documentation test. Rust
must offer equivalent typed builders over the same Rust-owned objects.

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

The cursor advances its current Rust-owned value after every successful call.
The lower-level graph API instead returns explicit replacement values. Examples
must not mix the two styles ambiguously.

`surface.impls.syndrome_cx` is a typed, instruction-scoped implementation
handle. Raw strings and a universal implementation enum are not accepted.

## 3. Scope

| Included | Deferred |
|---|---|
| One rotated distance-3 patch | Other distances, standard/asymmetric/repetition patches |
| Prepare logical Z | X/Y preparation |
| Three CX-based syndrome rounds | SZZ and other syndrome implementations |
| Destructive logical-Z measurement | Logical gates, injection, surgery, teleportation, code switching |
| Straight-line dataflow | Modules, branches, loops, adaptive control |
| Static Rust providers | Packages and dynamic providers |
| Portable protocol graph | Target mapping, routing, calibrated timing, traces |
| Reference Tick lowering | PHIR, QIR, MLIR, Guppy output/import |
| Separate native DEM integration test | Higher-level noise dialects and DEM compilers |

Space-time quantities, visualization, and replacing existing factories are also
deferred.

The three QEC instructions have these contracts:

| Instruction | Input | Output | Ideal decoded effect |
|---|---|---|---|
| Prepare Z | Declared patch | Active patch | Prepare logical zero |
| Syndrome extraction, three CX rounds | Active patch | Replacement active patch | Logical identity under the protocol's success contract |
| Measure Z | Active patch | Logical result | Destructive logical-Z measurement |

## 4. Required pipeline

### Author and resolve

The generic Rust substrate contains only a straight-line `InstrProgram` and
`InstrGraph`, typed definitions and bound calls, opaque registered value types,
`SingleUse | Reusable` use policy, serializable implementation descriptors,
executable Rust providers, explicit resolution context, and a resolved program.

A serialized descriptor contains stable instruction-scoped identity, versions,
provider identity, and fingerprint; executable Rust behavior remains in its
provider. A provider assesses the complete bound call, including the basis and
round parameters, before constructing an implementation body.

Resolution order is deterministic:

1. the call's explicit typed implementation;
2. an explicitly configured instruction-set choice;
3. the sole supported provider;
4. otherwise a structured unsupported or ambiguity error.

Explicit choices never fall back. Missing providers, fingerprint/version
mismatches, wrong-instruction implementation handles, unsupported parameters,
and ambiguity are distinct diagnostics sorted by stable identity.
The resolved artifact records whether each choice was explicit, configured, or
the sole candidate.

### Preserve identities and use rules

The implementation distinguishes reusable definition identity, call identity,
persistent code-block identity, dataflow value versions, code elements,
protocol-local resource roles, semantic measurements, and existing circuit
measurement identity.

Preparation, syndrome extraction, and measurement preserve one persistent
code-block identity while producing fresh value versions. Destructive
measurement ends the block lifetime. Generic validation rejects a second use
of a consumed value; QEC validation handles preparation, active lifetime,
measurement, and export obligations.

The measurement mapping is explicit:

```text
SemanticMeasId -> pecos_core::MeasId -> record ordinal
```

### Match the canonical patch

The Rust `PatchSpec` subset must preserve the current Python distance-3 patch's
data and X/Z-check identities, exact supports and ordering, coordinates,
logical supports, orientation, and separation of data geometry from protocol
ancillas.

The fixture is serialized from the existing Python `SurfacePatch` and loaded by
Rust. Rust must not independently regenerate ordering and call that parity.

The surface port includes CX preparation and measurement, the CX check plan and
touch order, ancilla assignment and lifetime, round structure, semantic
measurement events, detectors, and the logical-Z observable.

### Compose QEC semantics

Each selected provider builds a typed lower-level `InstrGraph` body containing
portable preparation, quantum operation, measurement, resource-lifetime, and
dependency instructions. It does not emit an opaque Tick circuit.

A surface-memory composition pass—type name provisional—runs over the resolved
calls and their events. It owns preparation-boundary detectors, comparisons
between consecutive syndrome rounds, terminal detectors, the logical-Z
observable, stabilizer/check epoch consistency, and semantic measurement
allocation.

Block state includes logical Pauli and code-element Clifford/check frames. They
remain identity in this MVP, but their presence and transfer checks reserve the
correct ownership boundary for later logical Cliffords.

### Build and lower the protocol program

The implementation bodies and QEC composition results form one inspectable
`ProtocolProgram` containing portable operations, an operation dependency DAG,
named syndrome rounds, code elements, temporary protocol resources, ancilla
lifetimes and cleanup, semantic measurements, detector/observable definitions,
and provenance to calls, block versions, checks, and providers.

It contains no target addresses, routing, calibrated time, adaptive control,
resource estimates, service requirements, or execution traces.

The versioned `SurfaceReferenceSchedule` reproduces the Python oracle's qubit
numbering, check and touch order, ancilla assignment, tick boundaries, and
measurement order. It returns a generated Tick artifact containing the
normalized `TickCircuit`, measurement maps, detectors, logical observable,
provenance, and schedule identity/version.

### Keep DEM construction separate

The DEM integration test uses a provisional adapter interface, conceptually:

```text
TickDemCompiler.compile(generated_tick_artifact, REFERENCE_NOISE_V1)
```

The noise fixture is:

```text
REFERENCE_NOISE_V1
    one_qubit_depolarizing = 0.001
    two_qubit_depolarizing = 0.002
    preparation_flip = 0.003
    measurement_flip = 0.004
    idle = 0.0
```

Every additional field in the current native schema is explicitly zero and the
fixture records that schema version. This test adds no DEM behavior to
`InstrProgram`, `ProtocolProgram`, or the generated Tick artifact.

### Keep Rust authoritative

Rust owns every authoritative graph, transition, resolver, protocol body, and
serializer. Python adds no shadow model or callbacks to built-in resolution.

Tests use canonical authored/resolved fixtures, deterministic documented ID
allocation, ordered or explicitly sorted data, schema and provider fingerprint
fields, no floating-point values in the authored/resolved MVP schema, Rust
round trips, and Python binding tests against the same Rust-owned artifacts.
The wire format is settled before public API stabilization.

## 5. Implementation slices

1. Record the PECOS retrospective, crate direction, serialization rules,
   measurement mapping, and canonical patch schema.
2. Implement the generic Rust graph, typed implementation handles, provider
   resolution, use checks, structured errors, and golden serialization.
3. Land distance-3 `PatchSpec` parity from the Python fixture.
4. Port the exact CX memory implementation bodies and QEC composition pass.
5. Build `ProtocolProgram` and reference Tick lowering.
6. Match ideal Tick operations, identities, detectors, and observable.
7. Add the separate native DEM adapter integration test.
8. Add thin PyO3/Python cursors and executable documentation tests.

Each slice must be independently reviewable and must not silently include the
rest of the Python surface stack.

## 6. Acceptance tests

The MVP is complete only when:

- the proposed program works from Rust and Python through the same Rust-owned
  artifacts;
- every normative API example executes in repository tests;
- patch parity preserves exact data/check identities, supports, ordering,
  coordinates, orientation, and logical supports;
- invalid ports, parameters, lifecycle, reuse, provider availability,
  fingerprint, implementation scope, unsupported choices, and ambiguity return
  structured errors;
- prepare, syndrome extraction, and measurement preserve one block identity and
  produce fresh value versions;
- the protocol program has bounded temporary ancilla lifetimes and no target
  resource identities;
- Tick output matches the oracle's ideal operations, qubit numbering, tick
  boundaries, measurement order, detectors, and logical observable;
- semantic measurement, circuit measurement, and record mappings round-trip;
- the separate DEM consumer matches native output under the complete versioned
  noise fixture without adding analysis behavior to the HDL artifacts;
- Rust tests import neither Python nor Guppy; and
- Python tests contain no shadow graph, resolver, planner, or serializer.
