# Annotations should reference measurements by identity

Status: implemented. Revised after three adversarial design reviews (core
claim upheld, plan corrected twice). Step 3 of #387. Depends on #391 (merged as
`cdfddba1b`), which made every `DagCircuit` measurement carry a unique `MeasId`.
Steps 1-3 below are merged (#397, #401, #403); step 4 -- the pivot -- is built,
with the deviations from the plan, and the closures from the pivot's
adversarial review round, recorded in "As built" at the end.

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

(An earlier draft claimed batch merging permits a gate where
`meas_ids.len() != qubits.len()`. That is false: `Gate::validate` rejects it and
`gates.rs:153` refuses to merge such gates. The claim came from a reviewer
finding explicitly labelled as reasoned rather than verified, and was written in
without checking.)

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

1. `MeasRef` gains `meas_id`, drops `Deref`. **Done, PR #397.**
   `TickMeasRef` cannot drop `record_idx` here, as an earlier draft said it
   could: `TickCircuit::detector` stores that field into
   `measurement_nodes: Vec<usize>`, so removing it before the field changes type
   would force storing `meas_id.index()` as a bare `usize` -- the exact
   conflation this change removes. It moves into the pivot (step 4).
2. Remove `From<&TickCircuit> for DagCircuit`, add `TryFrom`; bindings route
   through it; delete the pre-scan. **Done, PR #401.** The owned
   `From<TickCircuit>` went with it, and all three production callers already
   returned `Result`, so no public signature changed.
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
`resolve_result_tags(&[usize], &[usize])`
(`builder.rs:3413`), JSON `meas_ids: Vec<usize>`. Conflation can persist there
and the tests must cover it.

**Result tags, when that seam is typed (follow-on, not this series):** tags
remain supported as names layered over identity, never as a second key space.
The durable shape is `tag -> Vec<MeasId>` in emission order -- equivalently
`(tag, occurrence) -> MeasId` -- because a tag emitted inside a loop or a
repeat-until-success gadget names one measurement *per emission*, so the tag
alone is not unique; the id always is. Flattening this is the `_selene_harness`
defect in #72, where internal RUS measurements shifted record positions under
the `measurement_N` tags. The tag table is boundary-owned, built where tags
enter (Guppy/Selene `result(...)`, detector/observable JSON), and is never
consulted inside the analysis -- the same rule as ordinals. Numeric Guppy
result ids already follow this design degenerately: `mz_with_ids` makes the
external id *be* the `MeasId`, a one-element association per emission.

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

## Resolved by the helper round (PR #403 review)

- **The error taxonomy** is now code: `MeasResolveError::{Unknown, Removed,
  RecordLess, Inconsistent}`. `Inconsistent` was added by the review -- a
  `gate_mut` edit can leave an id held-but-unreserved or held twice, and a
  resolver that silently returned the first holder would launder the desync.
  Honesty limits are documented on the resolver: a bare id cannot detect a
  reference from a *different* circuit (ids are circuit-local numbers; a
  colliding foreign id resolves to the local measurement), and `Removed`
  cannot be told apart from an id overwritten via `gate_mut`. Rejecting
  foreign references needs validation where a reference *enters* a circuit.
- **The forgery seam: close it.** The review's verdict, judged against the
  helpers rather than on paper: they make forgery **more** dangerous, because a
  forged `MeasId(x)` that collides with a real id now resolves to a real
  `MeasRef`. Resolution rejects absent numbers but silently accepts the common
  collision case, so the seam cannot be left open. The Rust half is done on
  this branch: the tuple field is private, both `From` impls are gone, and the
  single integer route is `MeasId::from_raw`, whose doc names the boundaries it
  belongs at -- every use is now greppable and auditable. Opaque Python
  references (replacing integer tuples) ride with the pivot's Python-boundary
  migration, where the `mz()` return shape changes anyway.

## Pivot API consequences (from the same review)

- Annotation construction and the DEM builder must reject all four resolver
  variants loudly, wrapped with annotation label/index context.
- `DemSampler::with_circuit_annotations` returns `Self`; it must become
  fallible or hold a pending error for `build`.
- `InfluenceBuilder::with_circuit_annotations` and `build` are infallible and
  need an error channel. Its separate circuit argument should also go, so
  annotations cannot come from a circuit other than `self.dag`.
- eeg needs its own `UnsupportedMeasurementKind`-style error: its expansion is
  `MZ`-only, so an absent id is ambiguous between unknown and unsupported until
  `build`/`build_dem_string`/`summary` can report.
- The influence map's mixed-stamping sentinel (`MeasId(usize::MAX)` for id-less
  entries) is guarded at `meas_index_of`, but the representation itself should
  become explicit (`Option<MeasId>`) when the pivot touches those types.

## Unresolved, deliberately

- **`InfluenceBuilder::run_symbolic_simulation` measures only `qubits[0]` of a
  batched gate** and maps a node to one measurement index. That is a live defect
  independent of this design and is filed as #398; the identity pivot does not
  fix it, contrary to what an earlier draft implied.

Also carried forward: removing `From<&TickCircuit> for DagCircuit` must remove
the owned `From<TickCircuit>` at `tick_circuit.rs:3582` too, and
`From<&DagCircuit> for TickCircuit` is not proven infallible -- `gate_mut` can
desync invariants the conversion then unwraps on.

## Out of scope

#394 (Pauli clearing at non-destructive measurements), #395 (leakage-aware
history), #396 (channel conventions). None blocks this, and folding any in would
make the DEM comparisons that validate this change impossible to interpret.

## Related

[`measurement-id-system.md`](measurement-id-system.md) is the origin of the
identity-over-ordering direction. Originally written in an external notes vault
and cited from there by an earlier draft of this document; now ported into this
directory so the citation chain is complete.

## As built

The pivot landed as designed -- `measurement_ids: Vec<MeasId>` on both
variants, `&[MeasRef]`/`&[TickMeasRef]` fallible annotation constructors, both
conversions copying annotations verbatim, consumers on the step-3 resolvers,
the record-index maps (`meas_record_to_node`, `dag_node_to_record_indices`,
`node_to_meas_idx`) deleted -- with three deviations:

**1. Construction validation is an O(1) lookup, not a `find_measurement`
scan.** A reference carries its node (or tick/batch), so validation checks the
gate it names directly: exists, consumes a record, holds the (qubit, id) pair.
The O(gates) scan per reference would have made surface-code builders
quadratic. The honesty limit is unchanged: a foreign reference that agrees
structurally cannot be detected.

**2. Python keeps tuple handles instead of opaque reference objects.** The
seam the opaque-object plan closed was integer *id* forgery -- and the tuples
never contain an id. A dag handle is `(node, qubit)`, a tick handle is
`(tick, gate_idx, qubit)`; both are resolved against the circuit at
annotation construction, and the recovered id comes from the gate itself.
Plain ints are rejected with a `TypeError`. Forging a tuple is node-space
forgery, caught by resolution unless the forger names a real measurement --
the same limit the Rust `MeasRef` has.

**3. One order per influence map.** The de-aliased tests exposed that
`InfluenceBuilder` maps carried two internal orderings: `detectors` (and the
propagation data, and the sampler's raw channels) in symbolic-replay order,
`measurements`/`meas_ids` re-sorted by id rank. `meas_index_of` therefore did
not index the raw stream -- invisible while ids were positional, wrong the
moment they were not. `InfluenceBuilder` now emits `measurements`/`meas_ids`
in replay order, aligned with everything else in the map;
`DagFaultAnalyzer` maps were already internally consistent (id-rank
throughout). The rule stands: orderings differ *between* maps and are never
interchanged; within one map there is exactly one.

The de-aliased matrix lives in `crates/pecos-qec/tests/meas_id_pivot_tests.rs`
plus the eeg byte-equivalence test and the Python `mz_with_ids` tier; every
guard is mutation-tested.

Closed by the adversarial round (two independent reviews):

- **Duplicate ids are refused at every id-resolving consumer.** `TickCircuit`
  permits duplicate supplied ids (and `try_add_gate` does not advance the mint
  counter, so supplied/minted collisions are constructible); `gate_mut` can
  duplicate ids on a `DagCircuit`. eeg's expansion and the sampler's
  annotation ingestion now refuse such circuits loudly, mirroring the guard
  the DEM JSON path already had. The influence builder was already safe via
  `find_measurement`'s `Inconsistent`.
- **`measurement_order` cannot be combined with stamped ids.** The
  qubit-occurrence heuristic behind a supplied order silently mis-binds on
  non-positional ids; ids already define the mapping. Both the DEM builder and
  the sampler reject the combination; the escape hatch remains for id-less
  legacy circuits only.
- **The record-arithmetic callers got the real migration** the plan asked for:
  the three files now emit `meas_ids` JSON and read out in id space; the
  `record_idx - num_measurements` arithmetic is gone, not renamed.
- Validation cost is stated honestly: linear in the named gate's batch width,
  never in the circuit.
