# Annotations should reference measurements by identity

Status: proposed. Step 3 of #387. Depends on #391 (merged), which made every
`DagCircuit` measurement carry a unique `MeasId`.

## The defect

`AnnotationKind::Detector` and `AnnotationKind::Observable` both store
`measurement_nodes: Vec<usize>`. That field means two different things:

- in a `TickCircuit` it holds **measurement record indices** (`tick_circuit.rs`,
  `detector()` stores `TickMeasRef::record_idx`);
- in a `DagCircuit` it holds **DAG node ids** (`dag_circuit.rs`, `detector()`
  stores `MeasRef::node`).

One type, two meanings, translated on every conversion, with nothing in the type
system distinguishing them. Five separate fixes have each picked a key space and
relocated the failure rather than removing it. The conversion between the two
spaces is where the defects keep appearing.

A node id also cannot name a measurement. A batched measurement gate is one node
covering several measurements, so `Detector { measurement_nodes: [n] }` is
ambiguous whenever `n` measures more than one qubit. Three consumers resolve that
ambiguity by taking the first qubit and silently discarding the rest:

- `dem_builder/builder.rs:3367` -- `node_to_meas_idx.entry(node).or_insert(...)`
- `dem_builder/sampler.rs:1141`
- `influence_builder.rs:200`

## The change

Replace the field with an identity:

```rust
Detector   { measurement_ids: Vec<MeasId>, coords: Vec<f64> },
Observable { measurement_ids: Vec<MeasId> },
```

`MeasId` means the same thing in both representations, so conversion **copies**
the annotation instead of translating it. The translation step -- the thing that
has produced a defect every time it has been touched -- stops existing.

Ordinals do not disappear; they move to where they belong. A DEM record offset
is an export format, computed at the boundary that emits it and never read back
to identify a measurement. That is the direction `design/measurement-id-system.md`
already settled on: values carry their identity through the transformation, and
the negative-offset convention becomes import/export only.

## Why the migration is safe to attempt

Changing the field's type is a compile error at every consumer. There is no
partial state that builds, so nothing can silently keep using the old meaning.
That property is what the five incremental fixes lacked.

## What has to change with it

**1. The annotation API must stop discarding the qubit.**

`DagCircuit::detector(&[impl Into<usize>])` accepts anything convertible to
`usize`, and `MeasRef: Deref<Target = usize>` derefs to the *node*. The qubit is
lost at the call boundary, so the batched-node fix is unreachable without
changing this signature. It becomes:

```rust
pub fn detector(&mut self, measurements: &[MeasRef]) -> usize
```

`MeasRef` already carries `{ node, qubit }` and needs `meas_id` added. The
`Deref<Target = usize>` impl should go: it is what let a node id be used where a
measurement was meant.

The `TickCircuit` side is easier than it looks. `TickCircuit::detector` already
takes `&[TickMeasRef]`, and `TickMeasRef` already carries `meas_id` -- it stores
`record_idx` and throws the id away. It only has to keep what it already has.

**2. `From<&TickCircuit> for DagCircuit` must become fallible.**

Both reviewers asked for this in three consecutive rounds of #391 while it stayed
out of scope. It is now blocking:

- the conversion cannot report failure, so `DagCircuit`'s uniqueness check
  panics, and Python sees a `PanicException` (a `BaseException`, which
  `except Exception` does not catch);
- the binding therefore pre-scans for duplicates in front of the conversion,
  which is a **second implementation of the same invariant**. It has already
  drifted once, disagreeing about `usize::MAX`;
- the pre-scan is also structurally incomplete. It sees only *supplied* ids, so
  a circuit mixing an id-less measurement with a stamped one still panics:

```python
tc.add_gate("MZ", [0])   # generic add_gate reserves no record
tc.tick().mz([1])        # mints record 0
tc.to_dag_circuit()      # PanicException: reuses MeasId(0)
```

`TryFrom<&TickCircuit> for DagCircuit` lets the binding return `ValueError` and
deletes the duplicate. Open question below on whether `From` should remain.

**3. Silent drops at annotation resolution must become loud.**

`filter_map` currently discards unresolvable references at `tick_circuit.rs:3483`
(both arms), `dem_builder/builder.rs:3315`, and `dem_builder/sampler.rs:1149`
and `:1173`. An annotation that loses all of its references becomes an output
with no propagation terms -- not merely empty, but also suppressing the correct
record-based fallback in `build_dem_from_circuit`.

Under identity references an unresolvable id is unambiguously a bug, so these
become errors at the boundaries that can report: `build_dem_from_circuit`, the
sampler, and the new `TryFrom`.

`exp/pecos-eeg/src/builder.rs:236` has the same shape (`if record_idx < num_meas`
silently skips) and should follow.

## Migration, in order

1. `MeasRef` gains `meas_id`; drop its `Deref`. `TickMeasRef` already has one.
2. `TryFrom<&TickCircuit> for DagCircuit`; bindings route through it; delete the
   `to_dag_circuit` pre-scan.
3. The pivot, one commit: `measurement_nodes: Vec<usize>` becomes
   `measurement_ids: Vec<MeasId>`; fix every compile error; conversions copy.
4. Consumers resolve by id. The three first-qubit-wins sites become correct for
   batched nodes as a consequence, not as separate work.
5. Silent drops become errors.
6. Presentation: record offsets computed at export only.

## Open questions for review

**Q1.** Should `From<&TickCircuit> for DagCircuit` be kept alongside `TryFrom`,
panicking as today, or removed outright? Keeping both preserves callers but
leaves a panicking path; removing it is a breaking change to an std trait impl
that several call sites use.

**Q2.** Is `Vec<MeasId>` right, or should it be an ordered set? Duplicate ids in
one annotation are meaningless (XOR of a value with itself), and nothing
currently rejects them.

**Q3.** What happens to an annotation whose measurement is removed by
`remove_gate`? Today the reference dangles and is silently dropped. Options:
reject the removal, invalidate the annotation, or make resolution fail loudly.

**Q4.** `exp/pecos-eeg` treats `measurement_nodes` as record indices and indexes
`expanded.measurement_qubit[record_idx]` -- a dense array sized by the
high-water record count. Legal sparse ids (external Guppy ids up to
`usize::MAX - 1`) would make that an enormous allocation. Does eeg migrate to
ids, or keep an explicit export-boundary ordinal?

**Q5.** Is there a consumer that genuinely needs the *node*, not the
measurement? If so, `MeasRef` carrying both is the answer, but the annotation
should still store only the id.

## Explicitly out of scope

#394 (clearing a propagating Pauli at a non-destructive measurement is wrong,
pre-existing, four sites hold three opinions), #395 (leakage-aware history where
a record depends on a leaked result), #396 (DemBuilder vs fault-sampler channel
conventions). None blocks this and folding any of them in would make the DEM
comparisons that validate this change impossible to interpret.
