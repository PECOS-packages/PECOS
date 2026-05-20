# 003 - Hand-authored tracked Paulis in `observables_json`

**Status:** Draft — spike pending. Captures the design question and the
soundness assumption that distinguishes this from the existing
positional/annotation-only path.

**Author:** (dem-polish working notes)

**Depends on / extends:** [001 - Tag-referenced detectors for
`DetectorErrorModel.from_guppy`](001-from-guppy-tag-referenced-detectors.md)
(which explicitly out-of-scoped this); applies the same
structural-HUGR-binding pattern.

## Summary

Today `observables_json` actively rejects `{"kind": "tracked_pauli", ...}`
entries. Tracked Paulis are produced only from **circuit annotations** (e.g.
`dag.tracked_pauli(PauliString.from_str("X0 Z2"), label="x_check")`), never
from caller JSON. The rejection is committed-test pinned
(`test_from_guppy_rejects_json_tracked_pauli_observables`) and the
`from_guppy` docstring documents it as a hard limitation. The reason isn't
schema laziness — it's that **qubit identity through Guppy/Selene
compilation is not stable enough today** to safely accept positional qubit
references from caller text. A positional `"X0 Z2"` written against the
user's mental model of the program is not guaranteed to mean the same qubit
the trace actually allocates first.

This proposal lays out a path to soundly accept hand-authored tracked Paulis
in `observables_json` by giving qubits the same treatment proposal 001 gave
measurements: a stable, structural HUGR-derived qubit identifier that
travels through compilation. The MLIR-pattern shape from 002 also applies if
qubit identity needs to track runtime-loop instances of `qubit()`
allocations.

## Background

- `from_guppy` docstring (`python/quantum-pecos/src/pecos/qec/dem.py`)
  observable section: *"hand-authored JSON tracked Paulis are NOT supported
  by this path. … A `{"kind": "tracked_pauli", ...}` entry here is rejected
  by the builder."*
- Rust parser (`reject_tracked_pauli` in
  `crates/pecos-qec/src/fault_tolerance/dem_builder/builder.rs`) errors fail-
  loud on `kind=="tracked_pauli"` for both detectors and observables.
- The existing annotation path
  (`dag.tracked_pauli(PauliString.from_str(...))`) builds tracked Paulis
  from in-Rust qubit indices in the circuit-construction order. For the
  surface builder this is well-defined (the builder controls qubit
  numbering); for `from_guppy`, the program is compiled and traced through
  Selene and the qubit numbering is post-trace allocation order. Caller text
  written against source-level intent may not agree.
- Prior dem-polish commit `46243b0d` ("Document tracked-Pauli qubit-
  numbering limitation in `from_guppy`") records exactly this concern.

## Goal

Allow `observables_json` to include hand-authored tracked Paulis, e.g.

```json
[{"id": 0, "kind": "tracked_pauli", "label": "X_logical",
  "pauli": [{"qubit_ord": 0, "pauli": "X"},
            {"qubit_ord": 2, "pauli": "Z"}]}]
```

resolved soundly against the traced circuit — i.e. `qubit_ord: 0` means
*the qubit allocated by the 0th `AllocateQubit` op in HUGR traversal order*,
which is the same reorder-immune binding `extract_result_tag_measurements`
gives for measurements.

Alternatively (or additionally) allow a Pauli-string form `"+X0 Z2"` where
the bare integer is reinterpreted as the same HUGR-derived qubit ordinal,
once the correspondence to traced slot is committed-test verified.

## Design

Four parts, mirroring proposal 001's measurement-tag work.

### 1. HUGR pass: structural qubit ordinals

New module in `pecos-hugr-qis` (e.g. `qubit_ordinals.rs`):

```rust
pub fn extract_qubit_allocation_ordinals<H: HugrView<Node = Node>>(
    hugr: &H,
) -> BTreeMap<usize, Node>;  // ordinal -> AllocateQubit node

pub fn qubit_allocation_count<H: HugrView<Node = Node>>(hugr: &H) -> usize;
```

Numbering: HUGR traversal ordinal of `AllocateQubit` ops (parallel to how
`extract_result_tag_measurements` numbers `Measure` ops). Sound by
construction — purely structural.

### 2. Rust parser: accept tracked_pauli observables

Extend `parse_single_observable` in
`crates/pecos-qec/src/fault_tolerance/dem_builder/builder.rs`:

- Detect `kind=="tracked_pauli"`.
- Parse `pauli` as either:
  - A list of `{"qubit_ord": N, "pauli": "X" | "Y" | "Z"}` objects, **or**
  - A string `"+X0 Z2"` whose integers are interpreted as qubit ordinals
    (post-resolution).
- Parse `label` as an optional string.

Schema validation: integer ordinals only, single Pauli letter per qubit
(X/Y/Z), no duplicate qubit ordinals within a single observable.

### 3. Resolver: ordinal → traced qubit slot

Same shape as `resolve_result_tags` for measurements:

```rust
pub fn resolve_tracked_pauli_qubit_ordinals(
    observables_json: &str,
    qubit_ord_to_slot: &BTreeMap<usize, usize>,
    static_qubit_count: usize,
    traced_qubit_count: usize,
) -> Result<String, DemBuilderError>;
```

Resolves each `qubit_ord` to a traced qubit slot; rewrites
`{"qubit_ord": N, "pauli": "X"}` entries into the slot-indexed form the
existing tracked-Pauli machinery already consumes. Same fail-loud
discipline: unknown ordinal → error; static-vs-traced count mismatch → loop
guard error (the qubit analog of 002's measurement loop guard, for runtime-
loop qubit allocation).

### 4. `from_guppy` wiring (thin, in `dem.py`)

Already compiles the HUGR when `result_tags` is requested; do the same when
any observable has `kind=="tracked_pauli"`. Pass HUGR bytes + traced qubit
count to a pyo3 `resolve_tracked_pauli_for_guppy`. The downstream Rust DEM
builder consumes already-resolved tracked Paulis.

## Critical assumption (the one thing the spike must answer)

> **For a straight-line Guppy program, the HUGR-traversal ordinal of the
> `i`-th `AllocateQubit` op equals the trace allocation order of the qubit
> slot that `qubit()` produced.**

This is the qubit-side analog of the measurement-side property 001 left
"unproven and no longer relied upon," and that 001's wiring then proved by
committed cross-check for the supported scope. The same test pattern
applies here: write a Guppy program with N source-scrambled `qubit()`
allocations + a tracked Pauli on a specific subset; the DEM via
`qubit_ord`-resolved tracked Pauli must equal the DEM built via the
positional annotation form for the same qubits. If they match across an
asymmetric program, the correspondence is committed-test-verified for the
supported scope.

If the correspondence fails: this proposal needs 002's measurement-
provenance mechanism extended to qubits (a `record_static_qubit_alloc`
FFI), and effort grows considerably.

## Spike plan

1. Build the `extract_qubit_allocation_ordinals` HUGR pass and a
   `qubit_allocation_count` companion. Foundation-test on a scrambled
   straight-line Guppy program (mirror of 001's
   `scrambled_three_measurements`).
2. Prototype the parser + resolver minimally; bypass `from_guppy` wiring.
3. **Correspondence cross-check**: write a Guppy program with three qubits
   in scrambled allocation order, each with a distinct gate history (so the
   DEMs for tracked Paulis on each are distinguishable, à la 002's
   asymmetric scrambled test). Build a tracked Pauli via the new ordinal
   form and via the existing positional annotation form; assert byte-
   identical DEMs.
4. Wire into `from_guppy`, replace the rejection-of-`tracked_pauli` test
   with a positive test, add unknown-ordinal / loop-guard / non-@guppy /
   wrapper-input fail-loud tests (mirroring `test_from_guppy_result_tags.py`).

## Soundness scope

Covers:
- Straight-line Guppy programs with statically-allocated qubits.
- Caller observables of the form *"this logical observable is the parity of
  Pauli operators on these qubits"* — the canonical Stim-style tracked
  Pauli.

Does **not** cover:
- Runtime-loop qubit allocation (`for _ in range(comptime(n)): q = qubit();
  …`). Same gap structure as 002's measurement case: one static
  `AllocateQubit` op, N traced slots, no static-op → traced-slot
  correspondence without 002-style provenance. The loop guard rejects this
  case fail-loud; closing it composes with 002 (extend provenance to qubits)
  if/when 002 lands.
- Source-named qubits (`{"qubit_name": "qa"}`). Would require Guppy to
  expose source-level qubit names through the HUGR. Orthogonal extension.
- Dynamic qubit allocation under measurement-dependent control flow.
  Inherits 004's scope.

## Out of scope / alternatives considered

- **Just accept the existing `PauliString.from_str("X0 Z2")` form
  positionally without HUGR resolution.** Rejected: this is exactly the
  fragile path 001 set out to fix for measurements; doing it for qubits
  would reintroduce the same silent-misbind risk on
  Guppy/Selene-recompilation.
- **Surface source qubit names from Guppy.** Would be cleaner from a UX
  perspective (`"qubit_name": "qa"`) but depends on Guppy preserving
  variable names through HUGR generation, which is upstream and not
  guaranteed. The HUGR ordinal form is available today.
- **Make tracked Paulis reference measurements (records/meas_ids/result_tags)
  instead of qubits.** That's a different conceptual model — tracked
  Paulis are physical observables on qubits, not on measurement records.
  Conflating them would be a category error.

## Open questions

1. **`pauli` JSON shape.** Object list vs string-with-integers vs both?
   String form is concise but ambiguous (`"X10"` could be `X on qubit 10`
   or `X10 (rank-10 Pauli)`); object form is verbose but unambiguous.
   Preference: support both, with the object form as the canonical / safer
   one.
2. **Should the `result_tags` HUGR-compile in `from_guppy` be shared with
   the new tracked-Pauli HUGR-compile?** Both need `guppy_to_hugr(guppy)`;
   compile once if either is present. Trivial optimization.
3. **Naming.** `qubit_ord` vs `qubit_ordinal` vs `qubit_id`. The last
   collides with PECOS's internal `QubitId` type (not the same thing).
   Prefer `qubit_ord` for clarity.

## Code paths the spike touches

- `crates/pecos-hugr-qis/src/qubit_ordinals.rs` — new (parallel to
  `result_tags.rs`).
- `crates/pecos-hugr-qis/src/lib.rs` — re-export
  `extract_qubit_allocation_ordinals`, `qubit_allocation_count`.
- `crates/pecos-qec/src/fault_tolerance/dem_builder/builder.rs` —
  `parse_single_observable` extension; new `resolve_tracked_pauli_qubit_ordinals`.
- `python/pecos-rslib/src/dag_circuit_bindings.rs` — new pyo3 wrapper
  `resolve_tracked_pauli_for_guppy`.
- `python/quantum-pecos/src/pecos/qec/dem.py` — `from_guppy` thin pass-
  through; share `guppy_to_hugr` compile with `result_tags` when both are
  present.
- `python/quantum-pecos/tests/qec/test_from_guppy_dem.py` — replace
  `test_from_guppy_rejects_json_tracked_pauli_observables` with a positive
  test of the new form; add unknown-ordinal / loop-guard / asymmetric
  correspondence tests.
- `python/quantum-pecos/tests/qec/test_from_guppy_result_tags.py` may grow
  a mixed test (result_tags detector + tracked-Pauli observable on the
  same scrambled program).

## Effort estimate

- HUGR pass + foundation tests: 1 day.
- Parser/resolver/binding: 1–2 days.
- Wiring + correspondence test + asymmetric/edge-case tests: 2 days.
- Documentation updates + proposal 001 closure addendum: 0.5 day.

Total spike + production: ~1 work week. Significantly smaller than 002
because it does **not** require trace-schema or FFI changes — purely HUGR-
side analysis and JSON-side handling, plus a single critical-assumption
correspondence test analogous to one we've already done for measurements.

## What this proposal does NOT change

- `dem-polish` is unchanged. Today's `from_guppy` continues to fail loud on
  hand-authored tracked Paulis in `observables_json`; circuit-annotation
  tracked Paulis continue to work unchanged.
- Proposal 002's runtime-loop closure is independent. If 002 lands first
  and adds measurement provenance, this proposal naturally extends to add
  qubit-allocation provenance using the same mechanism.
