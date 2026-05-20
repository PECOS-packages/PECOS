# 005 - Array-valued `result()` support in `result_tags`

**Status:** Draft — spike pending. Smallest of the 002/003/004/005 follow-up
set. Composes with 002 (depends on 002 if runtime-loop arrays are needed;
straight-line arrays can be done independently).

**Author:** (dem-polish working notes)

**Depends on / extends:** [001 - Tag-referenced detectors for
`DetectorErrorModel.from_guppy`](001-from-guppy-tag-referenced-detectors.md)
(which excludes array-valued `result()` by construction);
composes with [002 - Runtime-loop `result_tags` via dataflow-bound
measurement provenance](002-runtime-loop-result-tags-via-dataflow-provenance.md)
for the runtime-loop case.

## Summary

`extract_result_tag_measurements` (in
`crates/pecos-hugr-qis/src/result_tags.rs`) is **sound by construction**:
it accepts *only* the canonical pattern
`tket.result:result_bool ← tket.bool:read ← Measure/MeasureFree`. Among the
three deliberately-excluded shapes (computed values, constants, array-
valued), **array-valued `result()` is the only one that's a usability gap
rather than a category error**: it's a legitimate, common pattern (e.g.
`result("round_0_syndrome", measure_array(ancillas))`) that the extractor
rejects today only because `tket.result:result_array_bool` lowers through
`collections.borrow_arr` machinery that doesn't cleanly expose per-element
measurement provenance to the static extractor.

This proposal extends `extract_result_tag_measurements` to recognize
`result_array_bool` and walk back through `borrow_arr` ops to the
individual `Measure` op(s) that produced each array element. The result is
that a tag bound to an array of measurements resolves to a list of record
offsets (one per element, in array order), exactly the same shape as a
multi-tag detector or a multi-record detector.

If the per-element static walk turns out to be infeasible from
`borrow_arr` alone, the proposal falls back to 002's runtime measurement-
provenance mechanism: each array element's `Measure` op carries a
`record_static_measure` call, and the trace tells us per-element which
static op fired. Either path produces the same end-user surface.

## Background

- Foundation test `array_valued_tag_is_excluded` in
  `crates/pecos-hugr-qis/src/result_tags.rs` pins the current exclusion
  using the `arr.hugr` fixture (`result("pair", measure_array(qs))`).
- The 001 closure justifies this as "array-valued `result(...)` (`result_array_bool`
  lowers through `collections.borrow_arr` machinery that does not cleanly
  expose per-element measurement provenance). Resolving those structurally
  would silently misbind … so they are not returned."
- The user-facing impact: a natural Guppy idiom for a round of QEC syndrome
  measurement —

  ```python
  @guppy
  def round() -> None:
      syndrome = measure_array(ancillas)
      result("round_0_syn", syndrome)
  ```

  — currently can't be referenced via `result_tags`; the user must either
  break the array apart into individual `result("…_0", measure(a0))`
  calls, or use positional `records`. Both are workable but verbose.

## Goal

Allow `result_tags` to reference array-valued `result()` outputs, with each
element mapped to its corresponding measurement. The expected surface is
unchanged from 001's scalar case — `result_tags: ["round_0_syn"]` simply
expands to the list of records the array elements correspond to.

Equivalent to: an array-valued tag is sugar for the per-element list of
individual scalar tags, *as if* the program had written

```python
result("round_0_syn[0]", measure(a0))
result("round_0_syn[1]", measure(a1))
…
```

Per-element selection syntax (e.g. `result_tags: [{"tag": "round_0_syn",
"index": 3}]`) is a natural extension but TBD; the headline case is the
whole array.

## Design

Two viable paths.

### Path A: pure HUGR-side resolution (cheapest if it works)

Extend `extract_result_tag_measurements` to recognize the
`tket.result:result_array_bool` pattern:

- The op carries the tag in its `args()` (same as `result_bool`).
- Its value input is wired (via `tket.bool:read` of each element? or via
  `collections.borrow_arr` ops? — to be determined by inspection of
  `arr.hugr`) from a structure that ultimately traces back to N `Measure`
  ops, one per array element.
- The walk-back must traverse `borrow_arr`'s element-access pattern
  unambiguously to identify which `Measure` op feeds which array index.

If `borrow_arr`'s element-access pattern preserves a clean source-element
correspondence (which is the question this spike must answer for the
static path), the extractor returns `tag → [ord_0, ord_1, …, ord_{N-1}]`
in array order — same data shape as the multi-occurrence `tag → [ord_0,
…]` proposal 002 produces for loops, and 001's existing resolver consumes
without modification.

### Path B: runtime provenance via 002

If the static walk through `borrow_arr` is too tangled (or if the
correspondence isn't sound under Selene's lowering), 002's
`record_static_measure` mechanism gives us the data we need without any
new static analysis: each underlying `Measure` op carries a
`record_static_measure(result_id, static_op_id)` dataflow-bound call; the
trace surfaces `static_op_id → [MeasIds]`; the extractor only needs to
identify the *static op id of each Measure feeding the array result*,
which doesn't require resolving `borrow_arr`'s indexing semantics — only
identifying the set of measure-op ids associated with the array tag.

Path B is structurally cleaner and composes with 002. The cost is a
dependency on 002 landing first (or being co-implemented).

## Critical assumption (the one thing the spike must answer)

> **Can the structural walk-back from a `tket.result:result_array_bool` op,
> through `collections.borrow_arr` machinery, identify per-element which
> `Measure` op produced each element of the array, in array index order?**

Falsifiable by reading `arr.hugr` (committed fixture in
`crates/pecos-hugr-qis/tests/fixtures/`) at the HUGR level and tracing the
dataflow. If yes → Path A. If no → Path B (and 002 must land first).

## Spike plan

1. **Read the existing `arr.hugr` fixture** and document the actual
   `borrow_arr` lowering pattern: which ops sit between the `Measure` ops
   and the `result_array_bool` consumer, and what relations between them
   constitute the "this measure feeds element k" link.
2. **Path A prototype**: extend `extract_result_tag_measurements` to walk
   the `result_array_bool` → `borrow_arr` → `Measure` chain. Foundation-
   test on `arr.hugr` (asserting the array maps to the expected
   per-element ordinals).
3. **If Path A unambiguous**: wire into `resolve_result_tags` (no change
   needed — already accepts `tag → [multiple ordinals]`); add a
   `from_guppy` test verifying `result_tags: ["arr_tag"]` resolves to the
   same DEM as the per-element scalar form.
4. **If Path A ambiguous or unsafe**: punt to Path B, document the
   blocking observation, defer until 002 lands.

## Soundness scope

Covers:
- Straight-line `result(tag, measure_array(qs))` for fixed-length comptime
  arrays. (Length must be comptime-known for the trace to have a fixed
  number of measurements; runtime-length arrays would need 002's machinery
  even more.)

Does **not** cover:
- Computed array values like `result("foo", [m0 ^ m1, m2 ^ m3])` —
  inherits 006's scope question (linear-combination resolution) over
  arrays.
- Array results inside runtime loops — composes with 002. Until 002
  lands, runtime-loop array results fail loud at the loop guard.
- Per-element value access in the tag itself (`result_tags: [{"tag":
  "syn", "index": 3}]`) — natural extension if useful, but the headline
  whole-array form is sufficient for the most common pattern (a detector
  spanning a whole syndrome).

## Out of scope / explicitly rejected

- **Treating array-valued `result()` as a single opaque parity** (i.e.
  resolving to a single record). The semantics in Guppy is N independent
  values, not their parity; collapsing them would silently misbind.
- **Supporting heterogeneous arrays** (e.g. `[m0, True, m1 ^ m2]`). The
  array element walk must terminate at raw `Measure` ops for each
  element; mixing in constants or computed values falls under 001's
  deliberate exclusions (or, partially, 006's linear-combination
  refinement).

## Open questions

1. **`borrow_arr` semantics.** Is the static walk Path A asks for actually
   feasible? Spike step 1 answers this.
2. **Per-element syntax.** Is `result_tags: [{"tag": "syn", "index": 3}]`
   wanted, or just the whole-array form? Defer until a user requests it.
3. **Composition with 002 if both land.** If 002 has injected
   `record_static_measure` for every measure op, Path B becomes the
   simpler implementation regardless of whether Path A would have worked.
   In that case, drop Path A in favor of consistency with the rest of
   the 002 mechanism.

## Code paths the spike touches

- `crates/pecos-hugr-qis/src/result_tags.rs` — extend
  `extract_result_tag_measurements` to recognize `result_array_bool` /
  `borrow_arr`.
- `crates/pecos-hugr-qis/tests/fixtures/arr.hugr` — existing fixture used
  as the structural reference; possibly add a new fixture for the
  positive case.
- `crates/pecos-qec/src/fault_tolerance/dem_builder/builder.rs` — no
  change needed (the resolver already handles `tag → [multiple
  ordinals]`).
- `python/quantum-pecos/tests/qec/test_from_guppy_result_tags.py` — add a
  test for array-valued `result()` resolution, mirroring the asymmetric
  correspondence pattern (`result_tags: ["arr_tag"]` equals the
  per-element record list, verified via a load-bearing asymmetric
  fixture).

## Effort estimate

- Path A: spike step 1 (read `arr.hugr`): 0.5 day. If feasible: extractor
  extension + tests: ~2 days. Total: ~2.5 days.
- Path B (if Path A infeasible): waits on 002. Implementation on top of
  002 once it lands: ~1 day (the per-measurement provenance is already
  available; only need to extract the static-op set associated with the
  array tag from the HUGR).

The smallest of the 002/003/004/005 set; can ship independently if Path A
works.

## What this proposal does NOT change

- `dem-polish` is unchanged. Today's `from_guppy` continues to reject
  array-valued `result()` in `result_tags` via the extractor's deliberate
  narrowing; the rejection is committed-test pinned and that test stays
  green until the extractor is extended.
