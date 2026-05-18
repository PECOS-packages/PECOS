# 001 - Tag-referenced detectors for `DetectorErrorModel.from_guppy`

**Status:** Draft

**Author:** (dem-polish working notes)

## Summary

`DetectorErrorModel.from_guppy(...)` builds a circuit-level DEM by tracing a
Guppy program through Selene/QIS and replaying the captured gate stream into a
`TickCircuit`. Detectors and observables are supplied by the caller as JSON and
reference measurements **positionally** (`records` = negative offsets, or
`meas_ids` = the sequential MeasIds assigned in trace order).

Positional references are correct only if the caller's assumed measurement
order matches the order the *post-compilation* trace emits. Guppy/Selene
lowering may reorder measurements, which would silently misalign detectors and
produce a wrong-but-plausible DEM. This proposal captures Guppy `result(tag,
value)` **tag identities** through to the `TickCircuit` so detectors can
reference measurements by stable tag, immune to reordering.

## Background / current state

- `measure()` intrinsically allocates the result slot: each measured qubit is
  backed by exactly one `Operation::AllocateResult { id }`, strictly
  interleaved 1:1 with its `Operation::Quantum(Measure)` in the lowered trace.
  The replay (`_replay_lowered_qis_trace_into_tick_circuit`) pairs the k-th
  `AllocateResult` with the k-th measurement (global trace order) and stamps
  that id as the MeasId. This is deterministic and self-consistent *within a
  trace*, but the mapping from a *logical* measurement to its trace position is
  not guaranteed stable across compilation.
- `result(tag, value)` carries a stable name. In the **direct QIS FFI path**
  this becomes `Operation::RecordOutput { result_id, register_name }`
  (`crates/pecos-qis-ffi/src/ffi.rs:604,626`) and is serialized into the
  operation trace.
- **However, under `selene_engine()` (the only engine that runs Guppy/HUGR in
  PECOS today), `RecordOutput` is never emitted into the operation trace**
  (empirically: 0 `RecordOutput` ops for a surface program with many
  `result(...)` calls). Selene routes `result(...)` to its per-shot
  *named-result stream* (`store_named_bool` -> `get_named_results()`,
  `crates/pecos-qis/src/ccengine.rs:1191`), which exposes `name -> values`,
  **not** `name -> result_id`. So tag<->measurement linkage is unavailable
  from anything `capture_operation_trace()` currently returns under Selene.
- The surface `traced_qis` path does not use tags either; its reorder-safety
  comes from validating traced measurement order against an abstract reference
  circuit (`decode.py`, `_build_surface_tick_circuit_for_native_model`). A
  general `from_guppy` has no such reference.

## Goal

Allow `detectors_json`/`observables_json` to reference measurements by a stable
`result(...)` tag (e.g. `{"id": 0, "result_tags": ["syn:r0:s3"]}`), resolved to
MeasIds during replay, so detectors survive Guppy/Selene measurement
reordering.

## Open question / fork (must be resolved before implementation)

The feasibility hinges on getting tag<->measurement linkage out of the Selene
path:

1. **Emit `RecordOutput` (with `result_id`) into the Selene operation trace.**
   Requires Selene's `result()` lowering to produce a tag-bearing op carrying
   the `result_id` (which links to `AllocateResult`/`Measure`). Selene is an
   external Quantinuum component; feasibility/locus of change is unknown and
   must be investigated. *Cleanest if feasible.*
2. **Correlate the named-result stream with the op trace.** Use
   `get_named_results()` (`tag -> values`) plus the op trace
   (`result_id -> measurement`). The named-result entries do not currently
   carry `result_id`; matching by order/value reintroduces the very
   order-dependence we are trying to remove. *Likely unacceptable.*
3. **Non-Selene QIS-FFI trace backend.** The direct QIS FFI path *does* emit
   `RecordOutput`+tag. Investigate whether a Guppy/HUGR program can be traced
   without the Selene runtime. If so, the feature becomes mostly Python-side.
   *Risk: this path may not run HUGR.*

## Sketch of the end-to-end change (assuming a tag-bearing op is available)

1. **Trace capture**: ensure an op carrying `(result_id, tag)` reaches
   `capture_operation_trace()` chunks. (Rust: `pecos-qis` / serialization in
   `crates/pecos-qis/src/ccengine.rs:851`; Python binding `sim.rs:674`.)
2. **Replay** (`decode.py`): while building `meas_ids_in_order`, also build
   `tag -> MeasId` (via `result_id`). Attach as TickCircuit metadata
   (e.g. `set_meta("meas_tags", json)`).
3. **DemBuilder detector parsing** (`crates/pecos-qec/.../dem_builder/builder.rs`,
   `parse_detectors_json`/`ParsedDetector`/`extract_records`): accept an
   optional `result_tags` field and resolve it against the `meas_tags`
   metadata to record indices/MeasIds.
4. **`from_guppy` API**: document tag-referenced detectors; keep positional
   references working for back-compat. Update `_validate_measurement_contract`
   to validate tags against the captured tag set.
5. **Tests**: a Guppy program where compilation reorders measurements;
   positional detectors give a wrong DEM, tag-referenced detectors give the
   correct one.

## Out of scope / already done in dem-polish

- `from_guppy` itself, the strict global-order replay rework (removed the
  buggy program-vs-slot qubit-match heuristic + silent fallback), and the
  corrected `result()` semantics in docs are landed independently of this
  proposal.

## Decision needed

Pick fork (1), (2), or (3) after a feasibility spike on Selene's `result()`
lowering and the named-result stream. No multi-crate implementation should
start before that spike concludes.
