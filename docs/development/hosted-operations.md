# Hosted Operations

This note sketches a generic PECOS abstraction for operations that must stay
attached to a later quantum host operation during lowering, scheduling, tracing,
and DEM construction.

## Motivation

Some source circuits contain local operations whose physical meaning is only
complete when paired with a later host operation. A common example is a
single-qubit Clifford basis change that prepares a data qubit for a two-qubit
interaction. If a compiler or runtime legally moves that local pulse far away
from the host, the ideal unitary can remain correct while the physical idle
noise model changes substantially.

Plain source adjacency and public language-level barriers are not enough as a
long-term contract:

- Source order can be changed by Guppy/HUGR/QIR/runtime lowering.
- Public barriers may be optimized away before QIS operation collection.
- A barrier says "do not reorder across this point"; it does not say "this
  local pulse is hosted by that two-qubit gate".

PECOS needs a way to represent this source intent directly and fail loudly when
the intent is dropped or cannot be honored.

## Definition

A hosted operation relationship binds one or more local source operations to one
host source operation:

- `host_id`: stable source identifier for the host operation.
- `host_kind`: generic host category, for example `two_qubit_gate` or
  `measurement`.
- `local_role`: local operation role, for example `basis_prefix`,
  `basis_suffix`, `frame_update`, or `readout_prefix`.
- `local_qubits`: source qubits touched by the local operation.
- `host_qubits`: source qubits touched by the host operation.
- `policy`: requested lowering/scheduling policy.

The relationship is not a physical-device-specific instruction. It is source
intent that runtimes may use to lower better schedules, and that PECOS can use
to validate traces and build diagnostics.

## Initial Policies

Start with strict, observable policies instead of implicit best effort:

- `metadata_only`: preserve provenance, but do not require adjacency.
- `same_runtime_batch`: the local operation and host must be submitted to the
  runtime in the same hosted group or batch.
- `max_idle_time`: the lowered trace must show no more than a configured time
  between the local operation and host on the local qubit.
- `lowering_required`: the host must produce a compatible lowered operation.

For any policy stronger than `metadata_only`, failure should be explicit and
actionable. Silent fallback to unhosted behavior is not acceptable.

## Trace Requirements

Traces should preserve both the local and host sides:

- local lowered operations carry `source_kind`, `source_label`, `host_id`, and
  `local_role`.
- host lowered operations carry `source_kind`, `source_label`, and `host_id`.
- replay code can pair local operations to hosts by exact `host_id`, not by
  nearest-neighbor inference.
- audit tools can report whether provenance is exact, inferred, mismatched, or
  missing.

Exact provenance is necessary but not sufficient. It proves PECOS can identify
the intended host; it does not prove the lowered schedule kept the operations
adjacent or within a noise-model threshold.

## Candidate Implementation Layers

### 1. Metadata-Only Vertical Slice

The smallest useful slice is the current trace-metadata approach:

1. Emit qubit-scoped metadata before each local operation and host operation.
2. Preserve metadata through QIS operation collection and runtime replay.
3. Attach metadata to lowered operations.
4. Require exact host matching in downstream audits.

This slice is useful for diagnostics and DEM cache identity, but it does not
constrain scheduling.

### 2. Barrier-Preserving Lowering

Preserving public barriers into `Operation::Barrier` can create runtime replay
batch boundaries, and PECOS replay should drain at those boundaries. This is a
valid generic improvement, but it still does not express "this local pulse is
hosted by that operation". It should not be the only hosted-operation plan.

Current state:

- SLR QIR codegen can emit QIR barrier calls such as
  `__quantum__qis__barrierN__body`.
- `pecos-qis-ffi-types` and `pecos-qis-ffi` already have an
  `Operation::Barrier` control-flow marker.
- PECOS traced Selene replay drains runtime operations at `Operation::Barrier`
  when that marker is present.
- A minimal Guppy public `barrier(...)` probe is currently optimized away before
  PECOS QIS operation collection. The captured raw operation trace contains
  allocations, gates, measurements, and releases, but no `Barrier` operation.
- The generated surface SZZ/SZZdg path uses a PECOS-owned
  `pecos_qis_runtime_barrier_qubit_hugr` helper for `szz_runtime_barriers`.
  The helper returns its qubit argument to create a Guppy/HUGR data dependency
  and queues a real `Operation::Barrier`, so Selene runtime lowering drains the
  current batch before the following host operation.

So barrier preservation requires a Guppy/HUGR/QIR/QIS bridge that lowers public
barriers or qsystem `RuntimeBarrier` operations into `Operation::Barrier`
instead of dropping them as pass-through no-ops. This is useful, but still
secondary to a hosted-operation relationship because a barrier does not identify
which host operation a local pulse belongs to.

### 3. Explicit Hosted Operation

The stronger long-term abstraction is a QIS-level hosted operation or hosted
group:

```text
HostedGroup {
    host_id,
    locals: [QuantumOp],
    host: QuantumOp,
    policy,
}
```

An equivalent representation could be a pair of begin/end host markers around a
set of normal `QuantumOp`s, if that is easier to thread through existing
collectors. The key requirement is that the runtime receives a relationship,
not just a sequence of independent gates.

## Surface-Code SZZ Example

For SZZ/SZZdg surface-code checks:

1. The source renderer forward-flows data-frame Cliffords.
2. Before an SZZ/SZZdg interaction, it emits any required non-virtual local
   data prefix.
3. The prefix is tagged as `local_role=basis_prefix` with the SZZ/SZZdg
   `host_id`.
4. The SZZ/SZZdg interaction is tagged as the host with the same `host_id`.
5. Runtime trace replay verifies exact provenance and, when requested, a
   bounded prefix-host idle threshold.

This should remain generic. PECOS should not encode downstream device names or
experiment packages into the abstraction.

## Fail-Loud Conditions

PECOS should fail loudly when:

- hosted metadata is attached to a source qubit but never consumed by a
  compatible source operation.
- two hosted metadata maps disagree on a key for the same source operation.
- a required host is optimized away or cannot be matched to a lowered operation.
- exact host provenance is requested but only inferred matching is available.
- a scheduling policy such as `max_idle_time` is exceeded in the lowered trace.

Error messages should include source labels, qubits, host labels, observed idle
duration or tick separation, and the relevant policy.

## Open Questions

- Should hosted groups be a first-class QIS `Operation`, or represented as
  markers plus normal `QuantumOp`s?
- Which Guppy/HUGR constructs can carry hosted relationships without being
  optimized away?
- Do runtime plugins need an explicit hosted-operation callback, or can PECOS
  lower hosted groups into existing runtime APIs with flush boundaries and
  metadata?
- How should hosted operations interact with DEM generation for decoders that
  consume raw hypergraph DEMs versus decomposed DEMs?
- Can the same abstraction cover measurement-hosted readout prefixes and
  two-qubit-gate-hosted basis prefixes?

## Recommended Next Slice

1. Add a minimal barrier-survival diagnostic for public Guppy barriers in QIS
   traces. Current strict-xfail target:
   `test_guppy_barrier_survives_into_qis_operation_trace`. The generated SZZ
   path has a positive regression,
   `test_szz_runtime_barrier_survives_into_qis_operation_trace`, through the
   PECOS runtime-barrier helper.
2. If preserving public barriers is small and generic, implement it as a
   separate quality-of-lowering improvement.
3. Prototype an explicit hosted SZZ prefix relationship in the QIS trace path.
4. Use idle audits to compare hosted and unhosted lowering on representative
   surface-code memory circuits.
5. Keep downstream experiments guarded by exact provenance and prefix-host idle
   thresholds until hosted scheduling is proven by traces.
