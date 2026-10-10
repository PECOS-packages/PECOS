# Changelog

PECOS uses GitHub to manage both Python and Rust releases.

Please see our [GitHub releases page](https://github.com/PECOS-packages/PECOS/releases) for the changelog.

## Unreleased

### Deprecations

- Deprecated `DemBuilder::with_measurement_order`,
  `DemSamplerBuilder::with_measurement_order`, `SamplingEngineBuilder::with_measurement_order`,
  `MemBuilder::with_measurement_order`, `DemStabSimBuilder::measurement_order`, and
  `MemStabSimBuilder::measurement_order`, plus the Python `DemBuilder.with_measurement_order`
  and `DemSamplerBuilder.with_measurement_order` methods. Circuit-built influence maps
  already use emission order; express a different record frame with `meas_ids` in metadata.
  Existing mapping behavior and validation remain unchanged.

### Decoder changes

- `windowed` specs with a buffer previously returned the result of a monolithic
  decode by `sandwich_phase2` (correlated PyMatching by default). Callers who want
  that result should request the monolithic decoder directly. `windowed` is now
  a streaming decoder and requires `inner`, `buffer`, and `step`. It commits whole local
  correction components and carries their full detector incidence. The old
  modes and their tuning options have been removed.
- `PyMatchingDecoder`'s edge decode now applies correlations when the decoder
  was built with them.
- `DetailedDecoder::decode_to_edges` for `PyMatchingDecoder` now returns the
  correction edges with their weights and observables. Previously it returned
  matched detection-event pairs with zero weights and no observables.


### Python breaking changes

- `pecos.quantum.hugr_to_dag_circuit`, `hugr_op_to_gate_type`,
  `gate_type_to_hugr_op`, `is_quantum_operation`, and
  `pecos_rslib.HugrConversionError` have been removed, including their
  `pecos_rslib.quantum` and `pecos.quantum` re-exports.
- `pecos.experimental.execute_hugr_symbolic` and `execute_hugr_symbolic_noisy`
  have been removed. Use `trace_program_to_tick_circuit(...).to_dag_circuit()`
  with `execute_dag_circuit_symbolic` or `execute_dag_circuit_symbolic_noisy`.
- `pecos_rslib.resolve_result_tags_for_guppy` now takes the Python-computed
  tag-occurrence map and static measurement count instead of HUGR bytes.
  `extract_result_tag_measurements_for_guppy` and
  `guppy_hugr_has_nontrivial_control_flow` have been removed; static result-tag
  analysis and control-flow checks now run in Python.
- The symbolic circuit executor now accepts rotations at Clifford angles, and
  `QAlloc` prepares the qubit in |0>. Preparation noise applies an X error to
  Z-basis preparations and a Z error to X-basis preparations, after preparation.
  Resets clear incoming faults, and measurement-and-reset operations apply
  measurement noise before reset and preparation noise afterwards. Gate noise
  remains attached to each physical gate before Clifford lowering. Noisy symbolic
  statistics for Guppy programs differ from the removed HUGR route because noise
  now attaches to each native QIS gate: for example, one Guppy `h` becomes two
  native gates. Measurement records follow the circuit's measurement order
  (runtime order for a traced program), including independent measurements.
- `pecos_rslib.Hugr` and `pecos_rslib.programs.Hugr` have been removed. Use
  `pecos.Hugr`/`pecos.Guppy` or
  `pecos_rslib.Qis(pecos_rslib.hugr_lowering.compile_hugr_to_qis(...))`.
- `pecos_rslib_llvm.compile_hugr_to_qis` has been removed. Use
  `pecos_rslib.hugr_lowering.compile_hugr_to_qis`.
- Guppy/HUGR programs no longer run on the neo stack: `.stack("neo")` raises
  `ValueError`. HUGR reaches the engines stack as QIS. HUGR-to-QIS lowering now
  happens when `sim()` is called, so compilation errors raise from `sim()`.
- `keep_intermediate_files` now keeps `program.ll`, the QIS source, instead of
  `program.hugr`.
- QIS simulation builders can now be run more than once.
- `sim(Qis(...))` no longer resolves the default Selene runtime or loads the
  program when `sim()` is called. Both happen at the first `run()`, `build()`,
  or `capture_operation_trace()` unless `.classical()` supplies an engine, so a
  missing runtime or an invalid QIS program now raises from that call.
- `.foreign_object()` on QIS programs, including lowered Guppy/HUGR programs,
  is refused pending issue #854.
- `LogicalCircuitBuilder.build_algorithm_descriptor(buffer=0)` now rejects a
  non-terminal segment when its source-tracked detector model requires forward
  look-ahead. Omit `buffer` to derive the safe minimum automatically, or pass at
  least the reported number of rounds. Segment dictionaries now distinguish the
  backward-compatible commit count (`num_detectors`, also available as
  `num_commit_detectors`) from the detector count in the halo-bearing segment DEM
  (`num_window_detectors`).
- `WindowedLogicalSubgraphDecoder(dem, stab_coords, step, buffer)` now requires
  `step` and `buffer`; they previously defaulted to 8 and 4.

### Rust breaking changes

- `DagCircuit::as_dag_mut` has been removed; use the gate APIs to preserve
  circuit bookkeeping.
- The `pecos-hugr` crate and `pecos_quantum::hugr_convert` module have been
  removed, including `SimpleHugr`, `SimpleGate`, `HugrConvertError`,
  `NotSimpleError`, `hugr_to_dag_circuit`, `dag_circuit_to_hugr`,
  `hugr_op_to_gate_type`, `gate_type_to_hugr_op`, and `is_quantum_operation`.
  The `pecos-quantum` and `pecos` feature `hugr`, the `pecos_quantum::Hugr`
  re-export, and the `pecos::quantum` HUGR re-exports have also been removed.
  PECOS Rust crates no longer read HUGR; Python lowers Guppy/HUGR programs to QIS.
- The symbolic circuit executor accepts Clifford-angle rotations, prepares |0>
  for `QAlloc`, and applies preparation noise after reset in the preparation
  basis, with incoming faults cleared by resets.
- `pecos_programs::Hugr` and `Program::Hugr` have been removed. Rust accepts QIS;
  HUGR is lowered to QIS at the Python boundary.
- The unused `pecos_qis::ProgramType` enum has been removed.
- The `pecos-hugr-qis` crate, the `pecos-qis` feature `hugr`, and the `pecos`
  feature `hugr-qis` have been removed.
- `QisEngineBuilder::platform` and `QSystemPlatform` have been removed. Select
  Sol at the Python boundary with `compile_hugr_to_qis(..., platform="sol")`.
- In `pecos-experimental`, `hugr_executor`, `execute_hugr`, and
  `HugrExecutionError` have been renamed to `symbolic_executor`,
  `execute_circuit_symbolic`, and `SymbolicExecutionError`. The executor accepts
  circuits and no longer reads HUGR. CY now has the correct phase, and MX leaves
  the measured qubit in its X-basis eigenstate, including during fault propagation.
- Symbolic execution and noisy history building now take `&DagCircuit` and use
  `DagCircuit::insertion_stable_topological_order` to preserve measurement record
  positions. Gate insertion sequences survive removal and node-index reuse,
  and are preserved when cloning a circuit. This opt-in traversal costs
  O(E + V log V); `topological_order` and
  `iter_gates_topo` retain their fast O(V + E) traversal for other consumers.
- The pecos-phir HUGR functions `compile_hugr_via_phir`,
  `compile_hugr_bytes_via_phir`, `hugr_to_phir_mlir`, and
  `PhirEngineBuilder::from_hugr_bytes` have been removed, along with the
  `pecos_phir::Pipeline`, `pecos_phir::InputFormat`, `pecos_phir::prelude::execute_hugr`,
  and `pecos_phir::prelude::execute_guppy` stubs.
- The `pecos::quantum::read_hugr_envelope` re-export has been removed.
- `CliffordGateable::apply_global_phase` replaces the former
  `ArbitraryRotationGateable::apply_global_phase` hook. This is source-breaking
  for out-of-tree implementors that override or call the hook through the old
  trait. Projective backends with `*_up_to_phase` reads retain the no-op
  default; backends exposing exact amplitudes must implement it.

### Rust bug fixes

- Dense rotation-family matrices now use the signed `(-pi, pi]` angle
  representative and agree exactly with the simulators. This changes
  `ToMatrix` output by a global `-1` for stored negative rotation angles,
  negative `theta` in `RXY1Q` and `U3`, composites containing those rotations,
  and the named `SXXdg`, `SYYdg`, and `SZZdg` gates.
- The `CliffordGateable` default decompositions now deliver their residual
  global phases through `apply_global_phase`. Exact-amplitude backends that
  inherit these defaults therefore change state by the required global phase;
  projective backends continue to use the no-op hook.
- `StateVecSoA::g` and `gdg` were the only two-qubit kernels missing the
  `flush_two_qubit` prologue, so they read stale amplitudes whenever a
  single-qubit gate was still queued. Because gate fusion is enabled by
  default, this produced wrong states on the common path.
- `Angle::to_radians_signed`, `to_turns_signed`, and
  `to_half_turns_signed` now choose the principal-value sign from the stored
  fraction instead of a rounded floating-point value. Exactly `HALF_TURN`
  remains positive and maps to `+pi`, `+0.5`, and `+1.0`, respectively; stored
  fractions strictly above it map to the negative representative.

### Python batch decoding

The legacy batch-decode entry points have been removed in favor of the unified
`SampleBatch.decode(...)` and `DemSampler.decode(...)` APIs:

- `SampleBatch.decode_count(dem, decoder)` becomes `SampleBatch.decode(dem, decoder).num_errors`.
- `SampleBatch.decode_each(dem, decoder)` becomes
  `SampleBatch.decode(dem, decoder, predictions=True).predictions`.
- `SampleBatch.decode_count_parallel(dem, decoder, num_workers=N)` becomes
  `SampleBatch.decode(dem, decoder, workers=N).num_errors`.
- `SampleBatch.decode_count_batch(dem)` becomes
  `SampleBatch.decode(dem, pymatching(correlated=False)).num_errors`.
- `SampleBatch.decode_stats(dem, decoder)` becomes
  `SampleBatch.decode(dem, decoder, timing=True)`; read counts from the result and timing from `.stats`.
- `SampleBatch.decode_stats_parallel(dem, decoder, num_workers=N)` becomes
  `SampleBatch.decode(dem, decoder, workers=N, timing=True)`.
- `DemSampler.sample_decode_count(dem, shots, decoder, seed=seed)` becomes
  `DemSampler.decode(dem, shots, decoder, seed=seed).num_errors`.
- `DemSampler.sample_decode_count_parallel(dem, shots, decoder, seed=seed, num_workers=N)` becomes
  `DemSampler.decode(dem, shots, decoder, seed=seed, workers=N).num_errors`.

`DemSampler.decode` uses a new canonical sampling ABI. A fixed seed therefore
produces a deliberately different shot stream from the removed sampling
methods, so seeded results are not comparable across this change. In return,
the new stream is reproducible independently of worker count.

The legacy `"pymatching"` type string continues to mean correlated matching.
The typed `pymatching(correlated=...)` factory requires callers to choose the
correlation mode explicitly.
