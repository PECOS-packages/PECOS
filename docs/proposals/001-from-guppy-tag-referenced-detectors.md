# 001 - Tag-referenced detectors for `DetectorErrorModel.from_guppy`

**Status:** Partially delivered — see "Final outcome (dem-polish)" at the
bottom. The sections between here and there record the investigation history
(including a runtime approach that was implemented then **proven unsound and
removed**); the final section is authoritative.

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

## Empirical findings (dem-polish spike)

Confirmed by direct probing of `capture_operation_trace()` under
`selene_engine()`:

- A Guppy program with `result("UNIQTAG_A", ...)` / `result("UNIQTAG_B", ...)`
  produces an operation trace in which the tag string **does not appear
  anywhere** (full chunk JSON searched). `RecordOutput` count is **0**.
- Op kinds present: `AllocateQubit`, `AllocateResult`, `Quantum`,
  `ReleaseQubit` only. `AllocateResult` **is** present (so the QIS FFI
  `__quantum__rt__result_allocate` is invoked), but
  `__quantum__rt__result_record_output` (which would queue
  `RecordOutput { result_id, register_name }`) is **not** invoked.
- Conclusion: under Selene, Guppy `result()` does **not** lower to the QIS FFI
  result-record symbol. The tag reaches only Selene's per-shot named-result
  stream (`tag -> value`), with **no `result_id` linkage**, so it cannot be
  associated with a measurement from anything `capture_operation_trace()`
  currently exposes.

## Consequence

Source-stable, reorder-proof MeasIds **cannot** be achieved from the
operation-trace path without a Rust/runtime change in `pecos-qis`/
`pecos-qis-ffi`. The Python `from_guppy` work (committed) is faithful to the
*post-compilation traced order* only; that is explicitly insufficient for the
reorder-robustness requirement.

## Required spike (Rust)

1. Determine the exact QIR/runtime symbol Guppy `result(tag, value)` lowers to
   under Selene, and whether it exposes the result pointer/id at a point PECOS
   can intercept (FFI shim in `crates/pecos-qis-ffi/src/ffi.rs`, or the Selene
   runtime integration in `crates/pecos-qis/src/selene_runtime.rs`).
2. If interceptable: add a trace op (or extend an existing one) carrying
   `(result_id, tag)` into the serialized operation trace
   (`crates/pecos-qis/src/ccengine.rs` `OperationTraceChunk`).
3. Consume it in `_replay_*_qis_trace_into_tick_circuit` to build
   `tag -> MeasId`; attach as TickCircuit metadata.
4. Extend DemBuilder detector parsing to resolve a `result_tags` field
   (`crates/pecos-qec/.../dem_builder/builder.rs`).
5. `from_guppy`: accept tag-referenced detectors; keep positional for
   back-compat.

Feasibility hinges on step 1, which involves Selene/Guppy lowering conventions
that may live outside this repo. No multi-crate implementation should start
before step 1 concludes.

## Spike conclusion (step 1 resolved -- NOT feasible PECOS-side)

Direct reading of the FFI surface (`crates/pecos-qis-ffi/src/ffi.rs`):

- `measure()` -> `___lazy_measure(qubit)` (l.435) allocates and returns the
  `result_id`; `___read_future_bool(result_id)` (l.467) consumes it and
  returns a plain `bool`.
- Guppy `result(tag, bool)` lowers to the Selene-style `print_bool(label_ptr,
  label_len, value: bool)` (l.668) / `print_bool_selene` (l.824) /
  `print_bool_arr_selene` (l.874). These receive **only the tag string and
  the concrete bool value** -- the `result_id` is not a parameter and is
  structurally absent at the record site.
- `__quantum__rt__record` (l.336) only logs; `__quantum__rt__result_record_output`
  (l.604, the QIR convention that *does* carry `result`+tag) is **not invoked**
  by Guppy/Selene-lowered programs.

Therefore the tag and the measurement identity (`result_id`) are never
co-present at any single interceptable call. Adjacency-pairing
`___read_future_bool(result_id)` with a following `print_bool(tag,...)` is
fragile (intervening classical logic / conditionals) and **fundamentally
impossible for array-valued `result()`**, which records many measurements
under one tag with no per-element identity (surface round syndromes and final
readout use exactly this form).

**Result-record lowering must change upstream (Guppy/tket2/Selene) to carry
the QIS result pointer/id** (e.g. adopt `__quantum__rt__result_record_output(
result, tag)`), or expose a result-id-bearing measurement-tagging API. This is
out of scope for `pecos-qis`. PECOS-side options that remain:

- **Reference-order safeguard** (generalize the surface path's traced-vs-
  reference measurement-order equality check to `from_guppy`; caller supplies
  the expected order). Robust against silent reordering corruption; no tags.
- **Documented positional contract** (status quo of the committed work).

Recommend raising the lowering gap with the Guppy/Selene owners; track here
until upstream provides a result-id-bearing record.

## CORRECTION: feasible PECOS-side via ExecutionContext read->name linkage

The "not feasible" conclusion above was wrong. The connection does not need to
exist at a single FFI call; it can be reconstructed and maintained in QIS code
because `ExecutionContext` (`crates/pecos-qis-ffi/src/lib.rs:54`) already holds
both halves:

- `measurement_results: Mutex<Vec<Option<bool>>>` -- values indexed by
  `result_id` (QIS measurement identity).
- `store_named_bool`/`store_named_array` -> `get_named_results()` -- the
  Selene-returned tagged results.

`___read_future_bool(result_id)` (ffi.rs:467) is invoked to obtain each
measurement bool immediately before the `print_bool*` ->
`store_named_*` that records it under its tag. So the robust linkage is:

1. `ExecutionContext`: add `pending_read_result_ids: Mutex<Vec<usize>>` and
   `named_result_ids: Mutex<BTreeMap<String, Vec<usize>>>`.
2. `___read_future_bool(result_id)`: push `result_id` to the pending buffer
   (when an execution context is present).
3. `store_named_bool`/`store_named_array`: drain the pending buffer into
   `named_result_ids[name]`.
4. Expose `get_named_result_ids() -> {tag: [result_id, ...]}` and surface it
   through the trace-capture plumbing (`python/pecos-rslib/src/sim.rs`,
   `crates/pecos-qis/src/ccengine.rs`).
5. Replay builds `tag -> MeasId` (since `result_id == AllocateResult id ==
   MeasId` under the strict replay) and attaches it as TickCircuit metadata.
6. DemBuilder detector parsing resolves an optional `result_tags` field
   against that metadata.
7. `from_guppy` accepts tag-referenced detectors; positional kept for
   back-compat.

This is source-stable and immune to measurement reordering.

## Implemented (dem-polish)

Delivered exactly as above:

- `ExecutionContext` (`crates/pecos-qis-ffi/src/lib.rs`):
  `pending_read_result_ids` + `named_result_ids`; `___read_future_bool`
  records the read; `store_named_bool`/`store_named_array` attribute pending
  reads to the tag; `pecos_get_named_result_ids_json` FFI export.
- `DynamicSyncHandle::get_named_result_ids` (default empty) +
  `HeliosSyncHandle` impl; surfaced via an authoritative end-of-shot
  `OperationTraceChunk { stage: "named_result_ids_final", named_result_ids }`
  emitted from `QisEngine::get_results` (per-op-chunk snapshots dropped --
  they missed tail `result(...)` stores).
- Replay (`decode.py::trace_guppy_into_tick_circuit`) attaches
  `tc.set_meta("meas_tags", {tag: [MeasId]})`.
- `build_dem_from_circuit` resolves an optional `result_tags` field on
  detectors/observables into record offsets via `meas_tags` (serde_json;
  hand-rolled detector parser untouched).
- `DetectorErrorModel.from_guppy` accepts `result_tags`, auto-sets
  `num_measurements` when tags are used, and fails loud on unknown tags.

Validated: `meas_tags` == MZ MeasIds; tag-referenced DEM byte-identical to the
positional equivalent; surface positional path byte-identical (Z+X, no
regression); surface LER workflow with distance suppression; unknown-tag
raises; ruff + cargo + 51 pytest pass. The fingerprint idea was dropped (not a
substitute) per the decision to implement the real linkage.

## Scope: tracked Paulis are NOT covered

`result_tags` anchors detectors/observables, which are *measurement*-anchored.
A tracked Pauli (`"kind": "tracked_pauli"` in `observables_json`) references
**qubits** via its `pauli` string, not measurements -- it is a propagated
Pauli frame, not a measurement outcome -- so `result()` tags do not apply.
Its qubit indices are interpreted in the traced (post-compilation) qubit
numbering and are therefore **not** source-stable the way tag-referenced
detectors/observables now are. Guppy exposes no `result()`-equivalent identity
for a qubit, so there is no analogous anchor.

Impact: geometry-derived paths (e.g. the surface builder, which validates
traced-vs-abstract measurement order and derives logical-operator support from
geometry) are unaffected. Hand-authored tracked Paulis for a *general*
`from_guppy` program must use traced qubit numbering and are reorder-fragile.
Decision: documented as a known limitation (in `from_guppy`'s docstring and
here); a qubit-identity anchor is possible future work, not in scope now.

## Final outcome (dem-polish) -- AUTHORITATIVE

This section supersedes the "Implemented" and "CORRECTION" sections above.

What was tried and what landed:

1. **Runtime read->store linkage (ExecutionContext) -- REMOVED.** Implemented,
   then disproved by a foundation test: a program doing all `measure()`s then
   all `result()`s yields `{tag_c: [0,1,2]}` instead of the correct per-tag
   binding, because measurement-future reads are batched before the stores.
   The mechanism (pending_read_result_ids / named_result_ids /
   note_read_result_id / pecos_get_named_result_ids_json / OperationTraceChunk
   field + end-of-shot emission / DemBuilder `resolve_result_tags_in_json` /
   decode.py `meas_tags` / dem.py `result_tags`) was fully excised as unsound.

2. **Sound HUGR extraction -- KEPT (committed).**
   `pecos_hugr_qis::extract_result_tag_measurements` recovers
   `tag -> measurement` from the compiled HUGR by structural wire-tracing
   (proper `ext_op.args()` tag read; value-port-0 reverse walk). Proven sound
   and reorder-immune for straight-line programs by the `scrambled` fixture
   test (the exact case the runtime heuristic failed). It is a self-contained
   building block; it is **not** wired into `from_guppy`.

3. **Loops are the unsolved gap.** A `for _ in range(comptime(n))` loop (the
   surface-code round structure) is **not unrolled in the HUGR** -- it is a
   CFG with one static measure/result op. Static extraction therefore yields
   `tag -> static-measure-op`, not per-round MeasIds. Bridging that to runtime
   per-occurrence MeasIds requires one of:
   - a HUGR CFG abstract interpreter (~= the excluded `HugrEngine`), or
   - `tket-qsystem` lowering carrying measurement provenance (upstream), or
   - reconstructing the deterministic unrolling from the comptime-bounded CFG
     (still requires CFG interpretation).

Net delivered in `from_guppy`: **sound positional `records`/`meas_ids`
detectors only.** These are byte-identical to the reference and LER-correct
for the surface code (verified), but are *order-sensitive* to Guppy/Selene
recompilation. Reorder-robust tag-referenced detectors are **deferred**; the
sound HUGR building block (#2) is committed for the eventual straight-line
wiring, and the loop case needs CFG-interpreter-class machinery or upstream
`tket-qsystem` provenance.

## Update (gap-4): sound result_tags wired into from_guppy (Rust-centric)

The committed HUGR extractor is now wired into `from_guppy` for the
**straight-line** case, with all logic in Rust and a thin Python pass-through
(per architectural review):

- `pecos_qec::fault_tolerance::dem_builder::resolve_result_tags` (Rust): runtime-
  loop guard (static vs traced measurement count), `result_tags`->record
  resolution, unknown-tag validation -- all fail-loud (`Result`/`ValueError`).
- `pecos_rslib.resolve_result_tags_for_guppy` (thin pyo3): HUGR-bytes +
  detectors/observables JSON + traced count -> resolved JSON, or raises.
  Internally calls `pecos_hugr_qis::extract_result_tag_measurements` +
  `measurement_op_count`.
- `from_guppy` (thin Python): if `result_tags` present, ferry
  `guppy_to_hugr(guppy)` + the traced measurement count to the Rust call. No
  tag logic in Python.

Verified: straight-line `result_tags` DEM is byte-identical to the positional
equivalent (proves the Rust chain and that the HUGR measurement ordinal equals
the traced MeasId order for the supported case); unknown tags fail loud;
runtime-loop programs (incl. surface) **fail loud** rather than silently
misbind; surface positional path byte-identical + LER unaffected.

Remaining deferred: per-occurrence tag binding for runtime-loop programs
(needs CFG-interpreter-class machinery or upstream `tket-qsystem`
provenance). `from_guppy` now hard-errors that case instead of being silent.

## External review response (AUTHORITATIVE — supersedes "Update (gap-4)")

An external review found real defects. Resolution:

- **#1 (critical, fixed):** the HUGR extractor over-collected — `result("x",
  m0==m1)` (lowers through `tket.bool:eq`) gave `records:[-2,-1]` and
  `result("x", True)` gave `records:[]`, both silently wrong.
  `pecos_hugr_qis::extract_result_tag_measurements` is now **sound by
  construction**: it accepts ONLY `result_bool <- tket.bool:read <-
  Measure/MeasureFree` (canonical scalar raw measurement). Computed values,
  constants, and array-valued `result()` (`collections.borrow_arr` machinery)
  are deliberately excluded, with regression tests.
- **#2 (fixed):** `from_guppy` now validates detector/observable schema
  (integer id + records/meas_ids) and **fails loud**, instead of letting the
  DEM builder swallow the parse error and return an empty DEM.
- **#3 (doc fixed):** corrected the docstring — hand-authored JSON tracked
  Paulis are **not** supported by the `observables_json` path (the JSON
  observable parser ignores `kind`/`label`/`pauli`; tracked Paulis come only
  from circuit annotations).
- **#4 (moot):** the gap-4 user-facing `result_tags` wiring is **reverted**
  (dem.py thin block, `pecos_rslib.resolve_result_tags_for_guppy`,
  `pecos_qec::resolve_result_tags`). With no `guppy_to_hugr` call in
  `from_guppy`, the wrapper-input regression no longer exists.
- **#5 (fixed, broader than gap-4):** the lowered-replay no longer assumes a
  strict AllocateResult/Measure 1:1 interleave. `Quantum.Measure` carries
  `[qubit, result_id]`; the replay now reads that `result_id` directly (== the
  MeasId), so batched allocate-allocate-measure-measure is handled and the
  overstated invariant is gone.
- **#6 (fixed):** the non-lowered replay now stamps the real `result_id` via
  `mz_with_ids` instead of discarding it and relying on
  `assign_missing_meas_ids()` to invent sequential ids.
- **#7 / overstatements (corrected):** "proven sound for straight-line" and
  "tag DEM == positional-equivalent" claims are withdrawn. The only retained,
  tested guarantee is the narrow `extract_result_tag_measurements` contract
  above; it is a building block, **not wired into `from_guppy`**. Whether HUGR
  traversal order equals trace MeasId order in general remains unproven and is
  no longer relied upon by any shipped path.

Net shipped: sound positional `from_guppy` (records/meas_ids, schema-validated
fail-loud, `D0`/`L0`), the corrected strict/non-lowered replay (#5/#6), and a
sound-but-narrow standalone HUGR extractor with tests. Tag-referenced
detectors are NOT exposed to users; the loop case remains deferred (CFG-
interpreter-class machinery or upstream `tket-qsystem` provenance).
