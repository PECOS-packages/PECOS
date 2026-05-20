# 006 - Linear-combination `result()` support (XOR/EQ/NOT chains)

**Status:** Draft — narrow refinement of 001's extractor. Worth capturing
because 001's closure-section claim *"equality is not parity"* is actually
incorrect for the specific case of `bool:eq`/`bool:xor`/`bool:not` chains
over raw measurement bits, and a small, sound extension recovers a
legitimate user-facing pattern.

**Author:** (dem-polish working notes)

**Depends on / extends:** [001 - Tag-referenced detectors for
`DetectorErrorModel.from_guppy`](001-from-guppy-tag-referenced-detectors.md)
(which excludes "computed values" by construction).

## Summary

`extract_result_tag_measurements` rejects `result("x", m0 == m1)` and
similar computed values by construction, citing — in 001's closure
section — that "equality is not parity." That blanket exclusion is **sound
but overly conservative**: for the specific case of a chain of `tket.bool:eq`
(boolean equality), `tket.bool:xor`, and `tket.bool:not` over raw
measurement bits (each via `tket.bool:read` of a `Measure`/`MeasureFree`
op), the resulting boolean value's *flip behavior under DEM error
mechanisms* is exactly the XOR-parity of the underlying measurements'
flip behaviors — which is the same as a detector with `records` on those
measurements. The semantics is sound; only the user-visible *value* of
the computed result differs from the parity (one is the negation of the
other), but DEM mechanisms describe *flip conditions*, not values.

This proposal extends `extract_result_tag_measurements` to soundly resolve
the linear (XOR-closed) subset of `tket.bool` computations, returning the
parity-equivalent record-offset list. `tket.bool:and` and `tket.bool:or`
remain excluded (they're genuinely non-linear in the error mechanisms and
not representable as a DEM detector).

This is the smallest proposal in the dem-polish follow-up set. It refines
existing committed behavior; no new trace data, FFI, or wiring.

## Background — why "equality is not parity" is too strong

Consider `b = m0 == m1`:

- Ideal values: `m0_i`, `m1_i`; ideal `b` = `(m0_i == m1_i) = NOT(m0_i
  XOR m1_i)`.
- With error mechanisms `e0`, `e1` cumulatively flipping `m0`, `m1`:
  observed `m0 = m0_i XOR e0`, observed `m1 = m1_i XOR e1`. Observed
  `b = NOT((m0_i XOR e0) XOR (m1_i XOR e1)) = NOT(m0_i XOR m1_i) XOR
  (e0 XOR e1)`.
- Observed `b` XOR ideal `b` = `e0 XOR e1`.

So `b` flips relative to its ideal value iff the parity of `e0` and `e1`
is odd — **the same flip condition as a DEM detector with records on
`m0` and `m1`**. A detector tagged `result("x", m0 == m1)` resolved to
`records: [m0_ord, m1_ord]` is sound.

The same holds for any chain built from `eq`, `xor`, `not` over raw
measurement bits: each such op preserves XOR-linearity, and the resulting
boolean's flip condition is the XOR of the underlying error mechanisms
(with cancellation for repeated occurrences — `m0 == m0` is constant True
regardless of errors, so its flip condition is zero, mapping to an empty
record set, which is correctly *not* a detector — sound rejection).

`and`/`or` are different: their value depends non-linearly on the
operands, so their flip condition depends on the *intended values* of the
operands, not just on the error mechanisms. They cannot be represented
as a DEM detector with a fixed record list.

## Goal

Extend `extract_result_tag_measurements` so the following Guppy patterns
soundly resolve through `result_tags`:

```python
result("x", m0 == m1)              # parity of m0, m1
result("y", not m0)                 # flip-equivalent to m0 alone
result("z", m0 ^ m1 ^ m2)           # parity of m0, m1, m2 (if `^` lowers to tket.bool:xor)
result("w", (m0 == m1) == m2)       # parity of m0, m1, m2 (associativity)
result("v", m0 == m0)               # empty set; rejected as not-a-detector
```

All of these should produce `tag → [ord_i_0, ord_i_1, …]` (the symmetric-
difference set of the measurement ordinals, with even-count duplicates
canceling). The downstream resolver already accepts this shape unchanged.

`and`/`or` continue to be rejected fail-loud (already are; no change).
Mixing `and`/`or` anywhere in the chain rejects the whole chain — the
walk-back terminates as soon as it sees a non-linear op.

## Design

Single change site: `extract_result_tag_measurements` in
`crates/pecos-hugr-qis/src/result_tags.rs`.

Current behavior: the walk-back from `result_bool` accepts exactly
`tket.bool:read ← Measure/MeasureFree`. New behavior: the walk-back is
an XOR-symmetric-difference set accumulator that traverses:

- `tket.bool:read ← Measure/MeasureFree` → emit the measurement's
  ordinal into the accumulator.
- `tket.bool:not(x)` → recurse into `x` (NOT preserves XOR behavior).
- `tket.bool:eq(a, b)` → recurse into both; symmetric-difference accumulate.
- `tket.bool:xor(a, b)` (if Guppy lowers `^` this way; TBD by inspection) →
  recurse into both; symmetric-difference accumulate.
- Anything else (`tket.bool:and`, `tket.bool:or`, `Const`,
  `collections.borrow_arr`, computed values not in the above set) →
  bail out, exclude this tag from the result map (same as today).

Symmetric-difference semantics: an ordinal appearing twice in the chain
cancels (e.g. `m0 == m0` → empty set, which is then *not* added to the
output because empty record sets are not detectors — sound).

Implementation: a recursive `walk_linear(node, visited) -> Option<BTreeSet<usize>>`
where the set is the symmetric-difference of all measurement ordinals
contributing to `node`'s value, and `None` means "non-linear / bailout."

## Critical assumption (the one thing the spike must answer)

> **Guppy's `==`, `!=`, `^`, and `not` over boolean values lower to
> `tket.bool:eq`, `tket.bool:xor`, and `tket.bool:not` respectively
> (or some specific set we can enumerate), and these are the *only*
> linear-boolean ops Guppy uses for measurement-derived booleans.**

If yes → straightforward implementation. If Guppy lowers `^` through
`tket.bool:and` + `tket.bool:not` (or some other non-direct route), the
walk-back has to recognize that compound pattern. The spike inspects
representative Guppy programs to enumerate the actual lowering.

## Spike plan

1. **Enumerate Guppy's boolean-op lowering.** Compile a small Guppy program
   `result("x", m0 == m1)`, `result("y", not m0)`,
   `result("z", m0 ^ m1)` (if Guppy supports `^` on booleans), and any
   other XOR-equivalent forms. Inspect the resulting HUGR to identify the
   `tket.bool:*` ops involved.
2. **Implement `walk_linear`** in `pecos-hugr-qis/src/result_tags.rs`.
   The set of recognized linear ops is what step 1 enumerated.
3. **Foundation tests** in `pecos-hugr-qis` mirroring the existing
   `scrambled` / `computed` / `arr` style: hand-built or compiled-from-
   Guppy HUGRs covering the cases listed in "Goal" plus the rejection
   cases (any `and`/`or` in the chain).
4. **Replace** `computed_and_constant_tags_are_excluded` (in
   `crates/pecos-hugr-qis/src/result_tags.rs` tests) to assert the
   refined semantics: linear shapes resolve; `and`/`or` / constants /
   empty-symmetric-difference cases stay excluded.
5. **End-to-end via `from_guppy`**: add an asymmetric correspondence test
   à la 001's `test_result_tags_match_positional_records` — a Guppy
   program where `result("eq", m0 == m1)` resolves to a detector
   byte-identical to the `records: [m0_ord, m1_ord]` form, with
   pre-measurement gate asymmetry so the test is load-bearing.

## Soundness scope

Covers:
- Boolean equality/xor/negation chains over raw measurement bits.
- Repeated measurement references with XOR cancellation
  (`m0 == m0` → empty set → not a detector, soundly rejected).

Does **not** cover:
- `and`/`or` (non-linear in error mechanisms; rejected fail-loud).
- Computed values mixing classical constants with measurements (e.g.
  `result("x", m0 == True)`) — this simplifies to `result("x", m0)` in
  value space and `record_offset(m0)` in flip space, so it *is* soundly
  resolvable. Borderline: include in the spike if cheap, else defer.
- Array-valued computations (e.g. `result("xs", [m0 == m1, m2 == m3])`) —
  inherits 005's scope.
- Linear ops *inside* runtime loops — inherits 002's scope (need
  per-occurrence binding for the underlying measurements first).

## Out of scope / explicitly rejected

- **`tket.bool:and`/`or` support via case analysis on intended values.**
  Theoretically you could split the DEM into multiple sub-mechanisms
  conditioned on intended values, but that's 004's branch-aware DEM
  territory, not a refinement of the extractor.
- **Generalized rational-combination support.** DEM is XOR-linear by
  definition; non-XOR-linear computations aren't representable. Sticking
  to the XOR-closed subset is the entire soundness story.

## Open questions

1. **What ops does Guppy actually use for `==`, `^`, `!=`, `not` on
   booleans?** Step 1 of the spike. Until known, the proposal is
   "support whichever XOR-linear ops Guppy emits."
2. **Should `m0 == True` simplify to `m0`?** Borderline — yes if it's
   easy. (`tket.bool:eq` with one operand a `Const` simplifies to either
   the other operand or its negation, both of which are linear.)
3. **Naming.** "Linear-combination" or "XOR-closed" or "parity-
   equivalent" — pick one and stick with it. I've used "linear-
   combination" throughout; "XOR-closed" is more precise.

## Code paths the spike touches

- `crates/pecos-hugr-qis/src/result_tags.rs` — `extract_result_tag_measurements`
  extended with the `walk_linear` recursion; existing
  `computed_and_constant_tags_are_excluded` test updated to reflect the
  refined semantics.
- `crates/pecos-hugr-qis/tests/fixtures/` — potentially a new HUGR fixture
  for the positive linear-combination case.
- `python/quantum-pecos/tests/qec/test_from_guppy_result_tags.py` — add an
  end-to-end correspondence test for `result("eq", m0 == m1)`.

No change to `resolve_result_tags`, the pyo3 binding, or `dem.py` — the
output shape (`tag → [multiple ordinals]`) is unchanged.

## Effort estimate

- Spike step 1 (enumerate Guppy's boolean lowering): 0.5 day.
- `walk_linear` implementation + foundation tests: 1–1.5 days.
- End-to-end correspondence test: 0.5 day.
- Update closure-section claim in proposal 001 ("equality is not parity"
  → "XOR-closed boolean computations resolve to the parity of their
  contributing measurements; AND/OR remain excluded"): 0.25 day.

Total: ~2.5 days. The smallest of the dem-polish follow-ups, and
self-contained.

## Cost/benefit honesty

This is a **narrow** refinement. Users with `result("x", m0 == m1)` can
already express the equivalent detector as `records: [-2, -1]` directly
without `result_tags`. The benefit is purely:

- Source-anchored, reorder-immune naming for computed booleans (e.g.
  syndrome equality checks).
- Removing an over-conservative exclusion that a careful reader of 001's
  closure might (correctly) recognize as too strong.

The cost is moderate-to-low: ~2.5 days of focused work, small surface
change, no new APIs, well-bounded test surface. It's the kind of
refinement that's worth doing if there's user demand for the pattern
(common in QEC syndrome processing) but skippable if not.

## What this proposal does NOT change

- `dem-polish` is unchanged. Today's extractor continues to fail-loud-
  exclude computed values; the `computed_and_constant_tags_are_excluded`
  test stays green until the extractor is extended.
- 002, 003, 004, 005 are all independent of 006 (and vice versa).
