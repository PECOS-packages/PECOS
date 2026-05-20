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
| [001-from-guppy-tag-referenced-detectors.md](001-from-guppy-tag-referenced-detectors.md) | Draft | Capture Guppy `result()` tags so `DetectorErrorModel.from_guppy` detectors are reorder-proof (delivered for the straight-line scope; runtime-loop case deferred to 002) |
| [002-runtime-loop-result-tags-via-dataflow-provenance.md](002-runtime-loop-result-tags-via-dataflow-provenance.md) | Draft | Close 001's runtime-loop deferral PECOS-side via a dataflow-bound `record_static_measure` FFI injected into the HUGR before Selene compiles it; spike pending |
| [003-hand-authored-tracked-paulis-in-observables-json.md](003-hand-authored-tracked-paulis-in-observables-json.md) | Draft | Soundly accept hand-authored tracked-Pauli observables in `observables_json` by giving qubits structural HUGR ordinals (the same MLIR-pattern proposal 001 applied to measurements); spike pending |
| [004-measurement-dependent-control-flow-dem.md](004-measurement-dependent-control-flow-dem.md) | Draft | Close the silent-wrong-DEM hole for Guppy programs with measurement-dependent quantum control flow via a static HUGR dataflow analysis (option A: rejection); branch-aware DEM construction (option B) deferred |

## Contributing

When adding a new proposal:

1. For a single document: Create `NNN-short-title.md`
2. For a multi-document exploration: Create a folder with `README.md` and related docs
3. Add an entry to this README
4. Open for discussion
