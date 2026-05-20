# PECOS Proposals

This directory contains architectural proposals and design explorations for PECOS. These documents capture ideas that may influence future development directions.

## Status Labels

- **Draft** - Initial exploration, gathering feedback
- **Under Discussion** - Being actively considered
- **Accepted** - Approved for implementation
- **Implemented** - Completed and merged
- **Deferred** - Good idea, but not now
- **Rejected** - Decided against

## Proposals

| Folder/File | Status | Summary |
|-------------|--------|---------|
| [001-from-guppy-tag-referenced-detectors.md](001-from-guppy-tag-referenced-detectors.md) | Draft | Capture Guppy `result()` tags so `DetectorErrorModel.from_guppy` detectors are reorder-proof (delivered for the straight-line scope; remaining gaps split out to 002–006) |
| [002-runtime-loop-result-tags-via-dataflow-provenance.md](002-runtime-loop-result-tags-via-dataflow-provenance.md) | Draft | Close 001's runtime-loop deferral PECOS-side via a dataflow-bound `record_static_measure` FFI injected into the HUGR before Selene compiles it; spike pending |
| [003-hand-authored-tracked-paulis-in-observables-json.md](003-hand-authored-tracked-paulis-in-observables-json.md) | Draft | Soundly accept hand-authored tracked-Pauli observables in `observables_json` by giving qubits structural HUGR ordinals (the same MLIR-pattern proposal 001 applied to measurements); spike pending |
| [004-measurement-dependent-control-flow-dem.md](004-measurement-dependent-control-flow-dem.md) | Draft | Close the silent-wrong-DEM hole for Guppy programs with measurement-dependent quantum control flow via a static HUGR dataflow analysis (option A: rejection); branch-aware DEM construction (option B) sub-scoped, deferred |
| [005-array-valued-result-support.md](005-array-valued-result-support.md) | Draft | Extend `extract_result_tag_measurements` to recognize `tket.result:result_array_bool` so `result(tag, measure_array(qs))` resolves as a list of records; spike pending. Smallest of 002–006; composes with 002 for runtime-loop arrays |
| [006-linear-combination-result-support.md](006-linear-combination-result-support.md) | Draft | Extend the extractor to soundly resolve XOR-closed `bool:eq`/`xor`/`not` chains over raw measurements (`result("x", m0 == m1)` → records:[m0_ord, m1_ord]); narrow refinement of 001's "computed values excluded" rule |

## Relationships and what is *not* separately proposed

The dem-polish follow-ups split 001's residual scope as follows:

| 001 deferred / out-of-scope item | Follow-up |
|---|---|
| Runtime-loop `result_tags` | 002 |
| Hand-authored tracked Paulis in JSON | 003 |
| Measurement-dependent quantum control flow | 004 (option A: static rejection) |
| Array-valued `result()` in `result_tags` | 005 |
| Linear-combination (XOR/EQ/NOT) `result()` in `result_tags` | 006 |

Items intentionally **not** given a separate proposal:

- **Branch-aware DEM construction** for measurement-dependent control flow
  is sub-scoped as "option B" of 004, deferred until a concrete use case
  motivates the substantial design space (CFG abstract interpretation,
  branch enumeration cost, semantic combination of per-branch DEMs).
- **Selene-side cooperation** for direct measurement provenance is the
  alternative 002 set aside. It would be an upstream proposal, not a
  PECOS one.
- **HUGR CFG abstract interpreter** (`HugrEngine`-equivalent) is the
  alternative both 002 and 004 set aside. Explicitly excluded as a wrong
  direction for the dem-polish scope; substantial work that duplicates
  what upstream `tket-qsystem` is expected to provide eventually.
- **Source-named qubit / measurement references** (`{"qubit_name": "qa"}`)
  depend on upstream Guppy preserving source-level variable names through
  HUGR generation. Mentioned as out-of-scope in 003 and elsewhere.
- **Genuinely non-linear computed `result()`** (`and`/`or`) is a category
  error — not representable as a DEM detector. 001's exclusion is correct
  for this case; 006 only relaxes it for the linear sub-case.

## Contributing

When adding a new proposal:

1. For a single document: Create `NNN-short-title.md`
2. For a multi-document exploration: Create a folder with `README.md` and related docs
3. Add an entry to this README
4. Open for discussion
