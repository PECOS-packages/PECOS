# 002 - Runtime-loop `result_tags` via dataflow-bound measurement provenance

**Status:** Draft — spike pending. Authoritative on the design *shape*; not yet
validated against Selene's actual lowering behavior.

**Author:** (dem-polish working notes; design refined by external review)

**Depends on / extends:** [001 - Tag-referenced detectors for
`DetectorErrorModel.from_guppy`](001-from-guppy-tag-referenced-detectors.md).

## Summary

Proposal 001 delivered sound, source-named `result_tags` detectors in
`DetectorErrorModel.from_guppy` for **straight-line, canonical**
`result(tag, measure(q))` programs. The deferred case — runtime
`for _ in range(comptime(n))` loops where the HUGR is not unrolled (one
static measure op per loop body, but N runtime occurrences) — was marked
"upstream-blocked" and `from_guppy` fails loud rather than silently misbind.

This proposal sketches a PECOS-side path to close that gap. The mechanism is
to extend the QIS trace with a **dataflow-bound measurement-provenance
token**: each static measure op in the HUGR is given a stable static op id
(its `extract_result_tag_measurements` ordinal), and a side-effecting FFI
call attached by **dataflow** to the measurement's result records the
`result_id -> static_op_id` pairing into the per-`ExecutionContext` runtime
state, which is then surfaced in the operation trace. Resolution of
`result_tags` becomes a pure data join: `tag -> static_op_id` (from HUGR,
already implemented) ⋈ `static_op_id -> [MeasIds]` (new, from the trace),
no CFG interpretation required.

The single load-bearing assumption — that Selene's lowering preserves a
dataflow edge between an injected `record_static_measure` call and the
measurement op that produced its input — is what the spike must
falsify or confirm.

## Background — why this is deferred today

Per [proposal 001's authoritative closure section][001-closure]:

- HUGR-side: `pecos_hugr_qis::extract_result_tag_measurements` recovers
  `tag -> static-measure-op` from the compiled HUGR, sound-by-construction
  for the canonical scalar `result(tag, measure(q))` pattern.
- For straight-line programs, the HUGR-traversal ordinal of the static
  measure op equals its trace `MeasId` order (committed-test verified in
  `test_from_guppy_result_tags.py::test_result_tags_match_positional_records`).
- For runtime loops, the HUGR has one static measure op per loop body but
  the trace has N runtime occurrences with distinct `MeasId`s. The static
  binding tells you `tag -> {static_op_id}`, but **nothing in the current
  trace tells you which trace measurements came from which static op**.

Proposal 001 evaluated three forks for closing this — (1) Selene emits
`RecordOutput`+`result_id`, (2) correlate Selene's named-result stream by
order, (3) non-Selene QIS-FFI backend — and concluded "NOT feasible
PECOS-side." That conclusion scoped the spike to *making Selene cooperate*.
This proposal explores a different scoping: **PECOS modifies the HUGR
itself before Selene compiles it**, injecting structural provenance markers
that propagate through Selene's lowering chain as ordinary side-effecting
FFI calls into PECOS-owned shims. Selene does not need to know anything
special about them.

[001-closure]: 001-from-guppy-tag-referenced-detectors.md

## Goal

For a Guppy program with a runtime `for _ in range(comptime(n))` loop body
emitting `result("syn", measure(q))`, allow
`detectors_json='[{"id":0,"result_tags":[{"tag":"syn","iter":k}]}]'` (or an
equivalent shape — final syntax TBD) to resolve to the `k`-th occurrence of
the `syn` static measure op in trace order, where "trace order" is
empirically the iteration order. Provide a sound, reorder-immune
tag-referenced detector for runtime-loop programs without a CFG
interpreter and without upstream Selene changes.

## Design

The design has four parts; each is in a PECOS-owned crate.

### 1. HUGR pass: inject `record_static_measure` after each measure op

In `pecos-hugr-qis`, add a new module (e.g.
`crates/pecos-hugr-qis/src/instrument_provenance.rs`) exposing a pass
`instrument_measurement_provenance(hugr: &mut Hugr)` that:

- Walks `hugr.nodes()` filtering by `is_measurement` (already defined in
  `crates/pecos-hugr-qis/src/result_tags.rs:38`).
- Assigns each measurement op the same stable id `extract_result_tag_measurements`
  uses — its traversal ordinal (see `result_tags.rs:78` for the existing
  numbering).
- Inserts a `tket.qsystem`-or-equivalent `__pecos__rt__record_static_measure`
  call **after** each measure op, taking the measurement's result value (or
  future) as a dataflow input and the static op id as a constant attribute.

The critical structural property: `record_static_measure` consumes the
measurement's result. The dataflow edge is what guarantees lowering preserves
the pairing — Selene's compilation cannot reorder the call across the measure
or drop it without breaking dataflow semantics, because dataflow IRs preserve
dataflow edges by construction.

A `marker-before-measure` variant (`__pecos__rt__set_current_static_op_id(N)`
emitted *before* each measure op, paired with a thread-local read inside the
measure FFI) was considered and rejected: "before" is not a stable semantic
relation unless the IR has an explicit ordering dependency, and standalone
side-effecting calls can be reordered/sunk/hoisted by lowering passes.
TLS-based state is also fragile across parallel-shot batching. Dataflow
binding sidesteps both issues.

### 2. QIS FFI: per-`ExecutionContext` provenance map

In `pecos-qis-ffi`, add a new entry point alongside the existing measurement
FFI (`crates/pecos-qis-ffi/src/ffi.rs:208` is where `mz` queues
`QuantumOp::Measure(qubit, result_id)`):

```rust
extern "C" fn __pecos__rt__record_static_measure(result_id: u64, static_op_id: u64);
```

Implementation: lookup the current `ExecutionContext` (the existing per-shot
isolation primitive — bare TLS is rejected; we need per-context state to
survive parallel-shot batching), and write `result_id -> static_op_id` into
a `Vec<(u64, u64)>` or `HashMap<u64, u64>` on the context.

### 3. Trace schema: surface the map per-shot

Extend `pecos-qis-ffi-types::operations` (the trace event schema, current
`QuantumOp::Measure` at `crates/pecos-qis-ffi-types/src/operations.rs:68`):

- Either add a new top-level trace event `MeasurementProvenance { result_id,
  static_op_id }`, emitted at `record_static_measure` time and consumed
  alongside `Quantum::Measure` events; **or**
- Extend the `Quantum::Measure` variant to carry `Option<u64>` static op id
  populated at flush time by looking up `result_id` in the context's
  provenance map.

The new-event form is less ABI-invasive (existing `Measure` consumers
ignore the new event). Final choice deferred to the spike.

### 4. DEM resolution: extend `resolve_result_tags`

In `pecos-qec` (`crates/pecos-qec/src/fault_tolerance/dem_builder/builder.rs`,
where `resolve_result_tags` lives today):

- `extract_result_tag_measurements(hugr)` already returns `tag -> [static_op_id]`.
- The new trace data gives `static_op_id -> [MeasId₀, MeasId₁, …]` in
  trace (== iteration) order.
- Compose: `tag -> [all MeasIds attributable to that static op]`, with
  optional per-occurrence selection via a richer `result_tags` shape:

  ```json
  [{"id": 0, "result_tags": [{"tag": "syn", "iter": 3}]}]
  [{"id": 0, "result_tags": ["syn:all"]}]     // alternative sugar TBD
  ```

  Final syntax TBD; the resolution is a pure data lookup and adding a richer
  syntax later is non-breaking on top of the existing `result_tags`
  semantics.

The pyo3 binding `resolve_result_tags_for_guppy` and the `from_guppy`
wrapper in `python/quantum-pecos/src/pecos/qec/dem.py` would need only the
extra `static_op_id -> [MeasId]` map to be passed in (already obtainable
from the trace consumption in `decode.py`).

## Why dataflow, not "marker before"

Critique from review (paraphrased):

> MLIR's locations/debug attrs are propagated by compiler discipline; your
> FFI marker is an executable side effect. "Before" is not a stable
> semantic relation unless the IR has an explicit ordering dependency.
> Treat it as part of semantics for tracing, with tests that prove it
> survives lowering. … TLS is acceptable only if you can prove marker and
> measure are ordered on the same worker thread and reset correctly. A
> dataflow-attached call is much less fragile.

The honest framing is: this is a **lowered provenance token**, not debug
metadata. We are extending the program's runtime semantics (the trace now
records provenance), and the soundness of that extension rests on dataflow
preservation, which dataflow IRs do by construction.

## Critical assumption (the one thing the spike must answer)

> Can PECOS inject a side-effecting call that consumes a measurement result
> into the HUGR such that Selene's full lowering chain (HUGR -> LLVM IR ->
> compiled binary -> runtime execution) preserves the dataflow edge,
> ensuring the call fires once per dynamic measurement with the correct
> `result_id`?

If yes: the rest is straightforward engineering. If no: PECOS-side closure
is infeasible and proposal 001's "upstream-blocked" conclusion stands — the
remaining path is upstream `tket-qsystem` measurement provenance.

The spike is shaped to answer this question minimally and decisively.

## Spike plan (minimum scope to answer the critical question)

1. **HUGR pass prototype.** Hand-write or programmatically construct a HUGR
   for a Guppy program with two distinct measurements inside a runtime loop
   body (sufficient to expose any "single static op assumption" we might
   accidentally rely on). Apply
   `instrument_measurement_provenance` (which need only be a sketchy
   first-pass; this is a spike).
2. **Run through Selene, not just analysis.** Pass the *mutated*
   `pecos.Hugr(...)` to `pecos.sim(...)` via the same trace path
   `from_guppy` uses (see `python/quantum-pecos/src/pecos/qec/surface/decode.py:719`).
   The mutated HUGR must traverse Selene's full lowering, not just
   `extract_result_tag_measurements`.
3. **Assert trace pairing.** Capture the trace; assert it contains exactly
   the expected `result_id -> static_op_id` pairs for:
   - a straight-line two-measure program (control: must agree with the
     existing committed `extract_result_tag_measurements` ordering);
   - a single-static-measure runtime-loop body (the headline case: N pairs);
   - a two-static-measures-per-iteration loop body (catches single-op
     assumptions and exposes ordering questions);
   - a branch (`if cond: measure(qa) else: measure(qb)`) — closes
     dynamic-shape control with provenance, even if measurement-dependent
     control flow remains unsupported elsewhere (see "Out of scope").
4. **LLVM IR inspection.** At the optimization level Selene actually uses,
   inspect the lowered LLVM to confirm `record_static_measure` is inside
   the loop, data-dependent on the measurement's result, and not
   dead-code-eliminated / hoisted / fused. This is the falsification step
   for the critical assumption.
5. **Parallel-shot isolation.** Run the multi-shot batching path; assert
   per-`ExecutionContext` provenance maps stay isolated (no cross-shot
   leakage).

Outcome: either a green spike that proves the design works under Selene's
actual lowering, or a falsifying observation that pins down the precise
behavior that breaks it (and so guides whether upstream is the only path).

## Soundness scope

This design closes:

- **Per-occurrence runtime-loop tag binding** (the proposal-001 deferred
  item) for fixed/comptime-bounded loops, where execution order is
  deterministic.
- **Branch-as-observed provenance**: dynamic-shape branches with no
  measurement-dependent control flow get correct per-execution provenance
  (which static measure op fired).

This design does **not** close:

- **Measurement-dependent dynamic control flow.** `from_guppy` traces one
  ideal execution; a Guppy program whose quantum operations depend on a
  measurement *outcome* still yields a DEM from a single sampled branch,
  silently wrong and seed-dependent. Provenance tells you *what fired in
  this trace*, not *what would fire across all possible measurement
  outcomes*. Sound treatment of measurement-dependent control still needs
  static rejection (HUGR conditional-on-measurement analysis) or
  branch-aware DEM construction. Out of scope here.
- **Soundness of the surrounding fault model under non-deterministic
  control.** Same reason.

## Out of scope / explicitly rejected alternatives

- **TLS-marker `set_current_static_op_id` before the measure op.** Rejected
  on the design-review grounds above: "before" is not a stable semantic
  relation; lowering can reorder/sink/hoist; TLS is fragile across parallel
  shots.
- **Selene named-result-stream correlation by order.** Same class of
  order-dependent mechanism that proposal 001 excised (it was unsound for
  the straight-line case via the runtime read/store linkage, by the same
  argument). It might work for a narrow loop pattern but is brittle around
  computed results, repeated reads, arrays, CSE, and Selene's scheduling.
- **CFG interpreter inside `pecos-hugr-qis`.** A different solution to a
  different half of the problem: CFG interpretation gives all-paths
  reasoning needed for sound dynamic-control DEM semantics. Runtime
  provenance gives what-fired reasoning for fixed loops and branches as
  observed. For surface-code-style fixed loops, runtime provenance is
  simpler and likely more robust; for dynamic control, neither suffices in
  isolation. Punted to a future proposal.

## Open questions

1. **Final `result_tags` syntax for per-occurrence selection.** The
   resolution is a pure data lookup; the syntax (e.g. `{"tag":"syn","iter":k}`
   vs `"syn:k"` vs `"syn:all"`) is a JSON-schema decision deferred to the
   spike. Prefer something extensible and unambiguous to misuse.
2. **Trace event shape.** New top-level `MeasurementProvenance` event vs.
   extended `Measure` variant. Choose at spike time based on Selene's
   actual behavior (the new-event form is less ABI-invasive).
3. **HUGR pass position.** Should
   `instrument_measurement_provenance` run as part of `from_guppy`'s
   `guppy_to_hugr` step, or as a separate explicit pass callers opt into?
   The former is invisible to callers (preferred for `result_tags` users);
   the latter keeps the trace clean for non-`result_tags` consumers.
4. **Whether to use the same provenance for `meas_ids`.** Today
   `meas_ids` resolves against stamped `MeasId`s positionally. With
   provenance available, `meas_ids` could be redefined as stamped MeasIds
   tagged by static op — but the existing semantics are sound and the
   redundancy discipline (`records`/`meas_ids` alternatives) works. Don't
   change without reason.

## Effort estimate (spike)

Roughly:

- HUGR pass prototype: 1–2 days (existing `pecos-hugr-qis` machinery covers
  most of it; need to investigate `tket`'s op-construction API for the new
  FFI call type).
- FFI + trace schema additions: 1 day.
- End-to-end through Selene + LLVM IR inspection + parallel-shot stress:
  2–3 days.

Total spike: ~1 work week, with clear go/no-go signal at the end.

If the spike succeeds: productionizing it (resolver extension, syntax
finalization, tests, docs) is another ~1 work week.

If the spike fails: the failure mode tells us exactly what upstream
`tket-qsystem` measurement-provenance support PECOS would need, which is
useful even if PECOS-side closure is abandoned.

## What this proposal does NOT change

- `dem-polish` is unchanged. Today's `from_guppy` continues to support
  sound straight-line `result_tags` and fails loud on runtime-loop programs
  using `result_tags`. Positional `records`/`meas_ids` continue to work
  for all programs (including the surface code). This proposal is
  forward-looking work, not a fix to merged code.

## Code paths the spike touches (reference)

- `crates/pecos-hugr-qis/src/result_tags.rs` — existing
  `extract_result_tag_measurements`, `measurement_op_count`,
  `is_measurement`. The new pass module lives alongside.
- `crates/pecos-hugr-qis/src/lib.rs` — re-exports.
- `crates/pecos-qis-ffi/src/ffi.rs:208` — current measurement FFI; new
  `__pecos__rt__record_static_measure` FFI added here.
- `crates/pecos-qis-ffi-types/src/operations.rs:68` — `QuantumOp::Measure`
  trace event; new `MeasurementProvenance` event or extension added here.
- `crates/pecos-qis/src/ccengine.rs` — `ExecutionContext`; provenance map
  attached as new field.
- `crates/pecos-qec/src/fault_tolerance/dem_builder/builder.rs` — existing
  `resolve_result_tags`; extended to consume `static_op_id -> [MeasId]`.
- `python/pecos-rslib/src/dag_circuit_bindings.rs` — existing
  `resolve_result_tags_for_guppy` pyo3 binding; extended argument.
- `python/quantum-pecos/src/pecos/qec/dem.py` — existing `from_guppy`
  wrapper; passes the new provenance map through.
- `python/quantum-pecos/src/pecos/qec/surface/decode.py:719` — existing
  trace path; spike must use this same path (`pecos.sim(...)` on the
  mutated HUGR), not the analysis-only `extract_result_tag_measurements`
  path.
- `python/quantum-pecos/tests/qec/test_from_guppy_result_tags.py` — new
  tests for the loop-occurrence case once the spike succeeds.
