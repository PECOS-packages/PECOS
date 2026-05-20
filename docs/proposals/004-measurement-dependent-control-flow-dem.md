# 004 - Sound DEM construction for measurement-dependent control flow

**Status:** Draft — spike pending. Two options laid out (static rejection +
branch-aware DEM); strong recommendation to ship option A first.

**Author:** (dem-polish working notes)

**Depends on:** [001 - Tag-referenced detectors for
`DetectorErrorModel.from_guppy`](001-from-guppy-tag-referenced-detectors.md)
(which documents the limitation), and overlaps with
[002 - Runtime-loop `result_tags` via dataflow-bound measurement
provenance](002-runtime-loop-result-tags-via-dataflow-provenance.md)
(which explicitly does *not* close this).

## Summary

`DetectorErrorModel.from_guppy` traces **one ideal execution** of a Guppy
program and builds a DEM from that trace. A program with
**measurement-dependent quantum control flow** — e.g. `if measure(q):
x(other)` — yields a DEM built from a single sampled branch: silently
wrong, seed-dependent, undefined. The `from_guppy` docstring explicitly
documents this as unsupported, and `from_guppy` does **not currently reject
such programs** — it builds a DEM that callers must not rely on.

A prior dem-polish round attempted a runtime-trace heuristic
(`reject_dynamic_control`) to detect this case; it false-positived on the
standard surface code (which has statically-scheduled post-measurement
gates per round, indistinguishable from genuine measurement-dependent
feedback in the runtime trace) and was reverted. The reverted-guard
analysis is committed in proposal 001's "Second external review +
outcome" section. The proposal recommendation that came out of that
analysis is **sound detection requires static HUGR analysis, not a runtime-
trace heuristic**.

This proposal lays out two viable sound paths and recommends starting with
**option A: static rejection** to close the silent-misbind hole, with
**option B: branch-aware DEM construction** as a separate larger
follow-up if/when there's user demand.

## Background — why this is currently a silent-wrong hole

- `from_guppy` runs `pecos.sim(program).classical(pecos.selene_engine())
  .quantum(pecos.stabilizer()).qubits(N).seed(s).capture_operation_trace()`.
- The trace is the gates that fired on this one sampled execution. For a
  measurement-dependent branch, exactly one branch's gates are in the
  trace; the other branches are invisible.
- The DEM is then built over those traced gates as if they were a static
  circuit. The "fault propagation" calculation is structurally fine on the
  traced circuit, but **the DEM does not model the gates that would have
  fired in other branches** — so the result is correct *only* for the one
  branch that happened to fire, and wrong for any decoding scenario where a
  different branch was active.
- For typical Guppy QEC workflows this hole is hypothetical (programs are
  straight-line or have only statically-scheduled feedback), but the
  language allows it, the docstring says it's "unsupported / undefined,"
  and no machinery enforces that.

## The sound vs unsound boundary (why the reverted guard failed)

Runtime trace observation:
- Surface code: ancilla measure → statically-scheduled per-round gate on
  ancilla qubit (e.g. ancilla reset / classical update) → next round
  measure. The trace shows "Measure(q_a, r0); … other gates; Measure(q_a,
  r1)" with classical ops between. Indistinguishable from "Measure(q_a,
  r0); if r0: X(q_other); Measure(q_a, r1)" at the trace level.
- The reverted `reject_dynamic_control` heuristic looked for non-MZ
  quantum ops in `pending_continue` chunks after a measurement and rejected
  them. It false-positive-rejected the surface code; the only way to
  avoid that false positive was to admit the false-negatives.

Static HUGR observation:
- The HUGR dataflow makes "Quantum op X depends on Measure op M's result"
  a **structural property**: there is a dataflow path from M's classical
  output (through `tket.bool:read`, possibly through `tket.bool:eq` /
  classical computations, into a `Conditional` op whose body contains X).
- The surface code's post-measurement gates have **no dataflow edge** from
  any Measure op's output to their control inputs — they're statically
  scheduled, not conditional. The structural check classifies them
  correctly as not-measurement-dependent.
- A `Conditional` whose discriminant traces back to a Measure op
  unambiguously *is* measurement-dependent control flow.

This is the same sound-vs-unsound boundary 001's `result_tags` work
established for measurement identity: **structural (HUGR) is sound,
behavioral (runtime trace) is not**.

## Goal

Close the silent-misbind hole. Either:

- **(A)** Soundly **reject** Guppy programs whose DEM cannot be built
  faithfully (measurement-dependent quantum control flow), with a clear,
  actionable error message.
- **(B)** Soundly **build a DEM** that captures all reachable branches and
  is correct for any execution path the program could take.

Option A is mandatory for soundness. Option B is a feature on top.

## Option A: static rejection (recommended first)

A HUGR pass:

```rust
pub fn detect_measurement_dependent_quantum_ops<H: HugrView<Node = Node>>(
    hugr: &H,
) -> Vec<MeasurementDependentOp>;

pub struct MeasurementDependentOp {
    pub quantum_op: Node,
    pub measure_source: Node,
    pub via_path: Vec<Node>,
}
```

It walks every Quantum op (or every op that PECOS classifies as a quantum
operation), checks whether any of its control/discriminant inputs has a
dataflow ancestor that is a Measure op. The reverse-walk is bounded: stop
at a Measure (positive — measurement-dependent), at a function input or
`Const` (negative — independent), at a comptime classical value (negative —
not a runtime measurement).

Wiring: `from_guppy` runs this check after `guppy_to_hugr`. If any
measurement-dependent Quantum op is detected, fail loud:

```
ValueError: from_guppy cannot soundly build a DEM for a program with
measurement-dependent quantum control flow. Detected: Quantum op `x(q1)`
at … is conditional on Measure result from … (path: …). DEM construction
traces one ideal execution and does not model alternate branches; build
the static-circuit-equivalent program explicitly, or use
`pecos.sim`-based sampling for measurement-dependent dynamics.
```

This **closes the silent-misbind hole**. It does not enable the feature.

### Cases the spike must validate

| Program | Expected |
|---|---|
| straight-line `qubit() ; measure(q)` | not flagged |
| `if measure(q1) == measure(q2): x(q3)` (computed conditional) | flagged via `tket.bool:eq` |
| `if measure(q): x(other)` | flagged |
| surface code (`make_surface_code(...)`) | **not flagged** (no Measure→Quantum dataflow edge) |
| `if comptime_const: x(q)` | not flagged (comptime, not measurement-derived) |
| `for _ in range(comptime(n)): x(q)` | not flagged (comptime loop, not measurement-dependent) |
| `if measure(q): result("x", True)` (classical-only conditional) | **not flagged** (no Quantum op in the conditional body) |

The last row is important: measurement-dependent **classical** updates
(producing more `result()` tags) are common and don't affect DEM
construction. The check is specifically "any Quantum op that's
measurement-conditional."

## Option B: branch-aware DEM (separate follow-up)

For a `Conditional` op whose discriminant depends on a measurement, both
branches can fire on different shots. A sound DEM must include the fault
mechanisms from both branches' Quantum ops, conditioned on the relevant
measurement outcomes.

Strategies:

- **Static enumeration.** Walk the HUGR, identify all measurement-
  dependent `Conditional` ops, enumerate the cross product of branches
  (2^k for k boolean measurements). For each combination, generate a
  hypothetical static circuit by inlining the chosen branches, build a per-
  combination DEM, combine. Tractable for small `k`; explodes for
  surface-code-scale measurement counts. Likely scoped to "k ≤ small N"
  with a guard.
- **CFG abstract interpreter.** A proper symbolic execution of the HUGR
  treating measurement outcomes as symbolic boolean inputs. This is
  essentially the excluded `HugrEngine`. Substantial.
- **Path summarization.** For specific structural patterns (e.g.
  syndrome-conditional Pauli correction), the branch effect on the fault
  model is summarizable analytically. Pattern-specific, not general.

Option B's design space is large and depends on actual user demand. **The
recommendation is to defer it.** Option A alone is a complete soundness
fix; option B is a feature whose cost/benefit needs concrete use cases to
evaluate.

## Critical assumption (the one thing option A's spike must answer)

> **The structural HUGR analysis `detect_measurement_dependent_quantum_ops`
> can distinguish measurement-conditional Quantum ops from
> statically-scheduled post-measurement Quantum ops with no false positives
> on real QEC workflows (surface code in particular) and no false negatives
> on genuine dynamic-control programs.**

Falsifiable: run on the cases in the table above; surface code is the
critical non-false-positive case (the same one that killed the reverted
runtime-trace guard); `if measure(q): x(other)` is the critical true-
positive case.

If the HUGR dataflow analysis can't cleanly classify e.g. `comptime`
conditionals or `tket.bool` chains, the spike answer is the specific
HUGR-op pattern that needs special-casing — and the scope of the special-
case set determines feasibility.

## Spike plan (option A only)

1. **Catalog the HUGR-op shapes involved**: `Conditional` op, `tket.bool:*`
   ops, `Const`, function inputs, `comptime`-derived constants. Verify how
   Guppy lowers each branch pattern.
2. **Implement `detect_measurement_dependent_quantum_ops`** in
   `pecos-hugr-qis` as a reverse-dataflow walk from each Quantum op's
   control/discriminant inputs. Terminate at Measure (positive), at
   Const/Input/comptime-constant (negative).
3. **Foundation tests** in `pecos-hugr-qis` on hand-built or compiled-from-
   Guppy fixture HUGRs covering the case table.
4. **Wire into `from_guppy`** — call the analysis on the HUGR after
   `guppy_to_hugr` (and share that compile with `result_tags` /
   tracked-Pauli paths if either is also active). Fail loud on detection.
5. **Update tests**: replace
   `test_from_guppy_dynamic_control_is_unsupported_and_unguarded` (which
   currently asserts no guard rejects, pinning the absence of a
   detector) with `test_from_guppy_statically_rejects_measurement_dependent_quantum_control`
   (asserts the new sound detector rejects dynamic programs *and* accepts
   the surface code — the critical regression test).
6. **Sanity sweep**: run the full qec pytest to confirm no existing
   workflow trips the new check.

## Soundness scope

Option A covers:
- Closing the silent-wrong-DEM hole for measurement-dependent quantum
  control flow. After A, `from_guppy` either produces a correct DEM (for
  programs without such control) or refuses fail-loud (for programs with
  it).

Option A does **not** cover:
- Building correct DEMs for measurement-dependent programs. That's
  option B.
- Measurement-dependent **classical** control (e.g.
  measurement-dependent `result()` outputs that don't change the quantum
  state). Such programs are not flagged because they don't affect DEM
  construction. The trace simply records different `result()` values per
  shot, which is fine.

## Relationship to 001 and 002

- **001** introduced the soundness discipline (structural HUGR vs runtime
  heuristic) and explicitly noted measurement-dependent control as a
  separate deferred case requiring static analysis. This proposal closes
  that.
- **002** addresses *measurement identity through runtime loops* — the
  question "which trace measurement came from which static measure op?".
  That's orthogonal to *whether a Quantum op is measurement-conditional*.
  002's `record_static_measure` mechanism doesn't help here, and 004's
  static analysis doesn't help 002 — different problems, different
  mechanisms. They can be implemented independently and in either order.
- **003** (tracked Paulis in JSON) is also independent. Programs with
  measurement-dependent quantum control flow that *also* use hand-authored
  tracked Paulis would be rejected by 004's check before 003's resolution
  fires.

## Out of scope / alternatives considered

- **Runtime guards.** Already-attempted-and-reverted. Cannot
  soundly distinguish surface code from genuine dynamic control.
- **Documentation-only.** "Don't write programs with measurement-dependent
  control flow if you use `from_guppy`" is the current state. It's
  inadequate because the resulting silent-wrong DEM is a correctness defect
  callers may not notice; the language allows the construct and `from_guppy`
  produces *some* DEM that callers may use.
- **Optional accept-anyway flag** (e.g. `unsafe_allow_dynamic_control`).
  Tempting but contrary to the project's "fail loud, no silent-wrong"
  values. If a user genuinely needs DEM for a specific dynamic program,
  they should expand it into the static-circuit-equivalent themselves (the
  error message suggests this), or option B should be built.

## Open questions

1. **What exactly counts as a "Quantum op" for the check?** Probably
   anything classified by `pecos_hugr_qis`'s existing
   measurement/quantum-op recognition (`is_measurement` and any analog for
   Pauli/Clifford ops). Should `Measure` ops themselves under a measurement-
   dependent conditional also be flagged? (Today: yes — a conditional
   measurement is measurement-dependent.)
2. **Should the check run *unconditionally* in `from_guppy`, or only when
   no measurement-dependent feature is explicitly opted into?** Probably
   unconditional — the cost is a HUGR walk, and the goal is to close the
   silent-wrong hole for every caller.
3. **`from_circuit` and `DemSampler`.** Should the analogous check run on
   circuits not built via `from_guppy`? Probably not — `from_circuit`
   consumes an already-constructed circuit that doesn't have HUGR-level
   conditional ops at all (it's a flat TickCircuit). Measurement-
   dependent control is a Guppy-source concern.
4. **Sharing the HUGR compile with `result_tags` / proposal 003.** If any
   of {result_tags, tracked_pauli, the new check} is active, `guppy_to_hugr`
   runs once. The new check fires *first* (cheap, fail-loud short-circuit
   before resolving tags).

## Code paths the spike touches

- `crates/pecos-hugr-qis/src/conditional_on_measurement.rs` — new module
  implementing the analysis.
- `crates/pecos-hugr-qis/src/lib.rs` — re-export.
- `python/pecos-rslib/src/dag_circuit_bindings.rs` — new pyo3 wrapper
  `detect_measurement_dependent_quantum_ops_for_guppy`.
- `python/quantum-pecos/src/pecos/qec/dem.py` — `from_guppy` runs the
  check after `guppy_to_hugr` (whenever the HUGR is compiled), or
  unconditionally; raises `ValueError` with a clear message on detection.
- `python/quantum-pecos/tests/qec/test_from_guppy_dem.py` — replace
  `test_from_guppy_dynamic_control_is_unsupported_and_unguarded` with the
  positive rejection test; ensure surface-code and `result_tags`
  byte-identical tests continue to pass (critical regression check).

## Effort estimate (option A)

- HUGR analysis + foundation tests: 2–3 days (the reverse-dataflow walk is
  not large, but it must correctly handle the `tket.bool:*`/`Conditional`/
  `Const`/comptime cases).
- Wiring + test updates: 1 day.
- Surface-code regression confirmation + edge cases: 1 day.

Total: ~1 work week, with a clear go/no-go signal (does the analysis
classify the case-table cases correctly?).

Option B is a much larger separate proposal if/when it becomes needed.

## What this proposal does NOT change

- `dem-polish` is unchanged. Today's `from_guppy` continues to build a DEM
  (silently wrong) for measurement-dependent control programs and is
  documented as not-supported for such programs. This proposal adds the
  static guard.
- 002 and 003 are independent and can land in any order relative to 004.
