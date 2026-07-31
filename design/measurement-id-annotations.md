# Annotations should reference measurements by identity

Status: proposed, revised after two adversarial design reviews (both
NEEDS-ADJUSTMENT; core claim upheld, plan corrected). Step 3 of #387. Depends on
#391 (merged as `cdfddba1b`), which made every `DagCircuit` measurement carry a
unique `MeasId`.

## The defect

`AnnotationKind::Detector` and `AnnotationKind::Observable` both store
`measurement_nodes: Vec<usize>`. That field means two different things: record
indices in a `TickCircuit` (`tick_circuit.rs:2263`, storing
`TickMeasRef::record_idx`) and DAG node ids in a `DagCircuit`
(`dag_circuit.rs:1833`, storing `MeasRef::node`). One type, two meanings,
translated on every conversion. Five separate fixes have each picked a key space
and relocated the failure rather than removing it.

A node id also cannot name a measurement, because a batched measurement gate is
one node covering several. Today that ambiguity is resolved three different
wrong ways:

- `dem_builder/builder.rs:3365` and `dem_builder/sampler.rs:1109` --
  `node_to_meas_idx.entry(node).or_insert(...)`, first qubit wins, rest dropped.
- `influence_builder.rs:200` -- **over-inclusion**: `PauliString::zs(&qubits)`
  over *every* qubit on the node. (An earlier draft called this first-qubit-wins.
  It is the opposite failure.)
- `dag_circuit.rs:1928` `pauli_from_measurement_nodes` spreads Z over all qubits
  of a batched node, so **the stored annotation Pauli is itself corrupted**, not
  merely the readout list.

There is a live bug beyond ambiguity. `TickCircuit::meas_ref` fabricates
`record_idx: meas_id.index()` (`tick_circuit.rs:1516`) and its own doc admits
that for external ids "the record index and the id genuinely differ... that case
cannot be resolved here" (`:1501`). Meanwhile `From<&TickCircuit>` builds
`meas_record_to_node` by positional counting in **storage** order (`:3503`), so
an annotation built from `mz_with_ids` misses the map and is silently dropped by
the `filter_map` at `:3548`/`:3558`.

## The change

```rust
Detector   { measurement_ids: Vec<MeasId>, coords: Vec<f64> },
Observable { measurement_ids: Vec<MeasId> },
```

`MeasId` travels on the gate itself, and both conversions clone gates, so the id
is invariant under conversion in both directions. The annotation becomes a copy
and the translation step -- where a defect has appeared every time it has been
touched -- stops existing.

## Why not the cheaper newtype

Keeping `Vec<usize>` and decreeing it always means record indices, with a
`RecordIdx` newtype, fails on a concrete witness: "record index" is *itself* two
different numbers inside `TickCircuit`. Allocation order (the `mz` counter) and
storage order (tick/batch iteration) diverge under `tick_at` out-of-order filling
and compatible-batch merging -- documented at `tick_circuit.rs:1495` and live in
the conversion, which counts storage order while annotations store
allocation-order indices. A `RecordIdx` newtype would type-bless both disagreeing
numbers as one space. Identity on the gate is the minimal construct invariant
under both reorderings.

## Ordering: there is no single canonical order, and that is fine

An earlier draft claimed ordinals retreat to export boundaries. That was wrong.
Dense ordinals already exist *inside* the analysis:

- `DagFaultInfluenceMap.measurements` is dense and ordered by `MeasId` rank
  (`propagator/dag.rs:619`, extraction at `:1921`).
- eeg's `expanded.measurement_qubit` is dense in **expansion** order
  (`exp/pecos-eeg/src/expand.rs:79`).
- The Guppy binding documents `runtime_meas_ids` as ids in **runtime execution
  order** (`dag_circuit_bindings.rs:1841`), which is not numeric id order.

External ids `[10, 3]` therefore give id-rank order `[3, 10]` in the influence
map and storage order `[10, 3]` in eeg. Both are correct for their own purpose.

**Rule:** every boundary that needs an ordinal computes its own, privately, from
an explicit `MeasId -> ordinal` map. Ordinals are never interchanged between
boundaries and never inferred from an id's numeric value. Nothing is ever
`Vec`-indexed by a raw `MeasId`.

`MeasId`'s own doc currently says it is "directly usable as an array index"
(`meas_id.rs:24`) and its example does `outcomes[m0.0]`. That contract is false
for external ids and must be corrected in this series, or it will seed the next
round of the same bug.

## What has to change with it

**1. The annotation API must stop discarding the qubit.**
`DagCircuit::detector(&[impl Into<usize>])` plus `MeasRef: Deref<Target = usize>`
loses the qubit at the call boundary. It becomes `&[MeasRef]`, `MeasRef` gains
`meas_id`, and the `Deref` goes -- it is what let a node id stand in for a
measurement. Dropping `Deref` compiles today because `From<MeasRef> for usize`
(`dag_circuit.rs:375`) already satisfies the existing bound.

`TickMeasRef.record_idx` is **removed**, not kept. It is a documented lie for
external ids, and two callers already do `record_idx - num_measurements`
arithmetic on it (`crates/pecos/tests/neo_surface_ler_test.rs:211`,
`crates/benchmarks/benches/modules/fault_catalog.rs:453`) which is already wrong
for those ids. They migrate to the meas_ids JSON path.

**2. `From<&TickCircuit> for DagCircuit` is removed, not supplemented.**
Keeping it alongside `TryFrom` is not a preference -- it does not compile. std's
blanket `impl<T, U: Into<T>> TryFrom<U> for T` makes the pair a coherence error
(E0119, verified). `From<&DagCircuit> for TickCircuit` -- the other direction --
becomes a genuine copy and stays infallible.

This also deletes the `to_dag_circuit` pre-scan, a second implementation of the
uniqueness invariant that has already drifted once and is structurally
incomplete: it sees only *supplied* ids, so `add_gate("MZ",[0])` followed by
`tick().mz([1])` still panics.

**3. Resolution distinguishes three failures, all loud.**
An id may be unknown, may name a *removed* measurement (`remove_gate` reserves
the id forever, `dag_circuit.rs:646`, so this is unambiguous), or may name a
measurement that consumes **no record** -- a `MeasureLeaked` carrying a supplied
id is accepted and reserved today. Each needs its own error.

Validation must also reject a record-consuming gate where
`meas_ids.len() != qubits.len()` and non-empty: the batch merge machinery permits
it, and position-based lookup (`tick_circuit.rs:1511`) then pairs the wrong id
with the wrong qubit silently.

**4. Silent drops become errors** at `tick_circuit.rs:3548`/`:3558`,
`dem_builder/builder.rs:3376`, `dem_builder/sampler.rs:1151`, and
`exp/pecos-eeg/src/builder.rs:236` (`if record_idx < num_meas`).

**5. The Python boundary needs its own migration.** The compile-error safety
argument is void there -- everything is `usize`.
`dag_circuit_bindings.rs:996` `extract_measurement_nodes` accepts plain ints as
raw node indices; after the pivot an int would be silently reinterpreted from
node space to id space, so the plain-int path is **rejected** and the py `mz()`
return shape changes (breaking 2-tuple destructuring).
`PyDagCircuit::annotations()` (`:1522`) omits the field today and must not gain
it in the old space mid-migration.

## Migration order

1. `MeasRef` gains `meas_id`, drops `Deref`; `TickMeasRef` drops `record_idx`.
2. Remove `From<&TickCircuit> for DagCircuit`, add `TryFrom`; bindings route
   through it; delete the pre-scan.
3. **Pre-land the id-resolution helpers, tested, alongside the existing
   node-based paths**: `DagCircuit::find_measurement(MeasId) -> Option<MeasRef>`
   (an O(gates) scan, deliberately *not* a maintained index, which `gate_mut`
   could desync -- `dag_circuit.rs:677`); id -> influence-map index via
   `influence_map.meas_ids`; eeg's id -> expansion-rank map.
4. **The pivot, one commit:** the field type, the `detector`/`observable`
   signatures, every consumer switched to the step-3 helpers, fail-loud
   validation, the Python boundary, and eeg -- all together. These cannot be
   separated: the consumers pattern-match the field and cannot compile without
   new resolution, so splitting them means either a huge judgment-laden commit or
   mechanical `MeasId(x)` fixes that compile and are wrong.
5. Presentation: each export boundary computes its own ordinal.

The compile barrier is weaker than an earlier draft claimed. `MeasId` is
`pub struct MeasId(pub usize)` with `From<usize>` (`meas_id.rs:41,58`), so the
mechanical fix at each error site is `MeasId(x)` or `.0`, which compiles and
silently re-conflates. Remove `From<usize> for MeasId`, keep an explicit
constructor, and rely on the de-aliased tests below rather than on the compiler.

Scope cut, stated explicitly: the DEM builder's id plumbing stays untyped --
`influence_map.meas_ids: Vec<usize>`, `resolve_result_tags(&[usize], &[usize])`
(`builder.rs:3413`), JSON `meas_ids: Vec<usize>`. Conflation can persist there
and the tests must cover it.

## Test strategy

The reason conflation bugs have been invisible: in virtually every existing test
`meas_id == record_idx == rank`. Three spaces, one number.

Every tier needs at least one case with **non-positional, non-contiguous** ids
(the `mz_with_ids` pattern, e.g. 9, 1, 5):

- Tick <-> Dag round-trip preserving annotation ids.
- DEM build equivalence: the same physical circuit with positional ids and with
  scrambled ids must produce byte-identical DEMs.
- Sampler, `InfluenceBuilder`, and eeg agreement on the same circuit.
- A batched `Gate::mz(&[0,1])` added via `add_gate`, with a detector referencing
  exactly one of its measurements -- the case that was previously inexpressible.
- Duplicate ids within one annotation still **cancel**, not error and not
  dedupe: `influence_builder.rs:1232` asserts this deliberately, matching the
  stim XOR convention. `Vec`, not a set.
- Removed, unknown, and record-less (`MeasureLeaked`) references each producing
  their own error.
- Python: `mz_with_ids` annotations, and `ValueError` rather than
  `PanicException` on every rejection path.
- A regression proving a sparse id causes no id-sized allocation.

**Acceptance gate:** the end-to-end DEM verification harness from #391, re-run at
the merge commit. It is the only demonstrated detector of resolution-semantics
drift in this subsystem.

Nothing to migrate on disk: no serde derives on `PauliAnnotation`,
`AnnotationKind`, or `MeasId`, and the detectors/observables JSON already speaks
both `records` and `meas_ids` with a consistency check.

## Out of scope

#394 (Pauli clearing at non-destructive measurements), #395 (leakage-aware
history), #396 (channel conventions). None blocks this, and folding any in would
make the DEM comparisons that validate this change impossible to interpret.

## Related

`measurement-id-system.md` in the pecos-docs vault (`~/Repos/pecos-docs/design/`)
is the origin of the identity-over-ordering direction. It is not in this repo;
an earlier draft cited it as though it were.
