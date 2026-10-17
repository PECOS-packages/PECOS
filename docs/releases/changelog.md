# Changelog

PECOS uses GitHub to manage both Python and Rust releases.

Please see our [GitHub releases page](https://github.com/PECOS-packages/PECOS/releases) for the changelog.

## Unreleased

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


### Python additions

- Rendered surface-code SZ teleportation uses `correction="physical"` by default
  to execute the conditional logical Z (the Guppy/Helios idiom), or
  `correction="record"` to only record `sz_correction` for frame tracking.
  These string modes replace the boolean switch; other values raise `ValueError`.
  Both modes output the frame bit.
  The builder's observable mask remains the DEM in all cases; exported
  `crossed_teleportations` will support a future executed-run transcription.
  Mask extraction on rows from a program that executed the correction counts it
  twice.

- `t_teleportation_decision_model(patch, rounds_before)` models the first T/TDG
  teleportation correction decision. `detection_events` transcribes a program's
  records into its detectors, and `build_dem` builds its DEM from a native trace
  of a Clifford stand-in, so the DEM's noise matches the program's per native
  gate, including single-qubit noise.
- `make_surface_t_teleportation(..., forced_decision=...)` forces the first T/TDG
  teleportation correction. `TeleportationDecisionModel.run_decoded` runs both
  forced variants through a caller-supplied runner bound to one seed and selects
  each shot by the decoded decision, giving per-shot decoded feed-forward results
  (`DecodedDecisionRun`) at twice the shot cost. `decode_decisions` decodes the
  first decision from program results.

### Python breaking changes

- Removed `pecos.execute_llvm`; use `pecos_rslib.hugr_lowering.compile_hugr_to_qis`
  for lowering and `pecos.sim` for execution.
- Removed `pecos.compilation_pipeline.compile_guppy_to_llvm`; compose
  `compile_guppy_to_hugr` with `compile_hugr_to_qis` instead.
- Removed `pecos.engines.compile_hugr_to_qis`; use
  `pecos_rslib.hugr_lowering.compile_hugr_to_qis` instead.
- Removed `pecos.engines.get_compilation_backends`; use `compile_hugr_to_qis`
  directly to compile with the required compiler.
- Removed `pecos.get_guppy_backends`; use `pecos.sim` directly to run programs.
- Removed the HUGR converters from `pecos.circuit_converters`: `hugr_to_dag`,
  `guppy_to_dag`, `dag_to_gate_sequence`, `UnsupportedHugrStructureError`,
  `hugr_to_ast` and `guppy_to_ast`. For a DAG, use the QIS trace route
  `trace_program_to_tick_circuit(...).to_dag_circuit()`; to run a program, use
  `pecos.sim`. SLR AST conversion is unchanged.
- Guppy compilation (`compile_guppy_to_hugr`, `pecos.Guppy`, `pecos.sim`) accepts
  only `@guppy` function definitions (`guppylang.defs.GuppyFunctionDefinition`) and
  raises `ValueError` for anything else. Objects returned by `with_opt_level(...)`
  or `with_minimal_opt()` are no longer accepted.
- Removed `pecos.protocols.GuppyCallable`; annotate Guppy definitions with
  `guppylang.defs.GuppyFunctionDefinition` and use `compile_guppy_to_hugr` or `pecos.sim`.

- Seeded samples change for every engine on the shared Monte Carlo loop (including
  QIS/Guppy, QASM and PHIR). A seed now determines each shot from its global index,
  independently of the worker count and random draws in earlier shots.

- Removed `DemBuilder.with_measurement_order` and `DemSamplerBuilder.with_measurement_order`
  from `pecos_rslib.qec`, including their type stubs. Write detector/observable records
  in influence-map (emission) order, or reference stamped `meas_ids` in metadata.
  For a foreign record frame, permute sampled outcomes after sampling.

- `TickCircuit.add_observable` now accepts `reference=0` or `reference=1` to store
  the optional noiseless observable value, with the JSON parser's integer-only validation.
  Omitting it still records an unknown reference, which observable-flip extraction
  refuses. CSS memory producers `bb_memory_circuit` and `coloration_memory_circuit`
  now emit their structurally derived references. Arbitrary Guppy specs and
  coin-toss output inference leave unknown references absent; typed specs preserve
  an explicit `Observable(..., metadata={"reference": value})`.

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
- `extract_detection_events_and_observables` no longer treats an observable with no
  noiseless reference as having a reference of zero, and raises when its lazy
  observable-flip sequence is accessed. The second result is now a reusable lazy
  `Sequence`; use `list(...)` to materialize it. Detector-only callers can discard it.
  Missing references include observables carried only by circuit annotations, such as one added with
  `TickCircuit.observable`, since an annotation has no reference slot. Previously such
  an observable extracted flips from its raw parity, which is wrong whenever its
  noiseless value is one.
- `extract_detection_events_and_observables` now returns declared observable ids
  rather than positions in the metadata list. Callers using these ids as array
  indices need a width of maximum observable id plus one, not the entry count.
- `extract_detection_events_and_observables` now raises
  `MissingObservableReferenceError` for observable metadata missing `id` or
  `reference`, with non-integer or duplicate ids, or with a reference other than
  integer 0 or 1 (booleans are rejected). Metadata carrying no reference is no
  longer accepted when consuming observable flips; the surface producers
  `LogicalCircuitBuilder.to_tick_circuit` and `TickCircuitRenderer` (including
  `build_memory_circuit` and `generate_tick_circuit_from_patch`) currently emit this field.
- `pecos.qec.empirical_correlation_table` now reports reference-relative observable
  flip rates instead of raw parity rates, uses declared observable ids instead of
  metadata positions in its keys, and raises `MissingObservableReferenceError`
  for reference-free metadata that it previously tabulated. Unlike the empirical
  function, `exact_correlation_table` still reports raw parity rates.
- `MissingObservableReferenceError` is now exported from `pecos.qec` as well as
  `pecos.qec.surface`. `ObservableReferenceDisagreementError` has been removed
  along with the probabilistic two-seed reference check; surface producers
  use one replay to compute a reference. LogicalCircuitBuilder guards determinism;
  TickCircuitRenderer trusts the caller to supply a deterministic readout.
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

- Python bindings that turn a float into an angle raise `ValueError` for NaN or
  infinite input, including `QuantumCircuit.append`, which stays atomic.
- Detector and observable annotations no longer carry a derived Pauli string:
  `annotations()` dicts drop `"pauli"` for detectors and observables, and the
  observable `pauli` field is removed from `circuit_definitions()`. Use
  `measured_pauli(meas_id)` / `measured_paulis(meas_ids)` on `DagCircuit` and
  `TickCircuit` for each measurement's basis and qubit. The plural method returns
  `(basis, qubit)` pairs aligned with its input IDs.
- `PauliProp` no longer has a fixed width; its constructor takes only
  `track_sign`. Pickles of `PauliProp` objects from earlier versions do not load.
- A qubit listed twice in a Pauli string is rejected by every constructor and
  parser, and `to_dense_str(n)` raises when the support does not fit in `n`.
- `QuantumCircuit` raises `QubitConflictError` when a different gate is added on
  a qubit already used in the same tick, instead of dropping the gate.
- `StateVecEngine` keeps the qubit count it was constructed with; addressing a
  higher qubit raises instead of resizing the simulator.

- Simulation `build()` and `run()` map input errors to `ValueError`, with the
  message as `"<context>: <detail>"` (no `Input error:` prefix); genuine
  execution failures remain `RuntimeError`.

### Python bug fixes

- Fixed Guppy result tags binding to the wrong measurement when HUGR node order
  differed from runtime order, such as with inlined helpers. Tags now bind from
  the lowered program that is executed. Errors for uncertifiable `result_tags` /
  `result_ref` programs have changed text: they previously mentioned runtime loops
  and now explain how to use positional records or measurement IDs instead.

- `StateVecEngine` no longer resets every qubit mid-shot when a later batch
  addresses a higher qubit (#1252).
- Weighted two-qubit noise labels (`p2_weights`) apply `"AB"` as `A` on the
  gate's first operand and `B` on its second; the sampling engine previously
  read them by absolute qubit index and dropped weighted two-qubit noise for
  gates not on qubits 0 and 1.
- `PauliStabilizerGroup` rejects generator sets that generate -I, and group
  membership, order and element enumeration account for dependent generators.
- A single-qubit PHIR-JSON gate listing several qubits applies to every listed
  qubit in the Rust PHIR-JSON engine, not only the first.

### Rust breaking changes

- Seeded samples change for every engine on the shared Monte Carlo loop. Each shot
  is reseeded from the run seed and its global index before resetting the engine.
- `WorkerSeedRecord.seed` is replaced by `first_shot`, the worker's first global
  shot index. `SeedReport.base_seed` now drives replay; `root_seed` is provenance
  only. Old serialized reports must be regenerated. Shot-seed derivation is stable,
  but component derivation uses `DefaultHasher`, so seeded samples are not promised
  stable across Rust releases.

- Removed `DemBuilder::with_measurement_order`,
  `DemSamplerBuilder::with_measurement_order`, `MemBuilder::with_measurement_order`, `DemStabSimBuilder::measurement_order`,
  `MemStabSimBuilder::measurement_order`, `MeasurementNoiseModel::with_measurement_order`,
  `MeasurementNoiseModel::set_measurement_order`, and the public
  `MeasurementNoiseModel::im_to_tc_order` field. Write detector/observable records in
  influence-map order (emission order for circuit-built maps), or reference stamped
  `meas_ids` in metadata. For a foreign record frame, permute sampled outcomes after sampling.

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
  positions. Gate insertion keys survive removal and node-index reuse,
  and are preserved when cloning a circuit. Pass replacements inherit the
  original key with a replacement index appended; nested replacements extend
  that path again. This preserves their original position relative to other
  gates and keeps metadata history consistent with the same ordering.
  This opt-in traversal costs O(E + D V log V), where D is one plus replacement
  nesting depth (O(E + V log V) for bounded depth); `topological_order` and
  `iter_gates_topo` retain their fast O(V + E) traversal for other consumers.
- `DagCircuit::connect` and `connect_all` return `DagWireError` (re-exported from
  `pecos_quantum` and `pecos::quantum`, along with `DagGateError`) instead of
  `DagWouldCycleError`. Besides cycles, they now reject a missing gate and an
  exact duplicate `(from, to, qubit)` wire; `connect_all` checks every shared
  qubit before adding any wire. Python's `DagCircuit.connect` and `connect_all`
  raise `ValueError` for these.
- `DagCircuit::remove_gate` bridges each qubit wire through the removed gate,
  so its predecessor and successor stay ordered. Code that removed a gate and
  then reconnected its neighbours with `connect` now gets
  `DagWireError::Duplicate`; drop the reconnection.
- Auto-wired insertion (`add_gate_auto_wire` and the gate builders) wires each
  new gate after every gate in its qubits' append frontier. `add_gate` leaves
  the frontier unchanged; `connect` from a frontier gate moves it to the tails
  of the destination's chain; removing a frontier gate puts its predecessors
  back. `.meta()` and `last_added_node` follow auto-wired insertion history,
  survive removal, and are not moved by unwired insertion. Auto-wiring errors
  are returned as `DagGateError::Wiring` instead of being discarded.
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

- Float-to-angle conversion is fallible only: `Angle::from_radians`,
  `from_turns`, `epsilon_from_radians`, `epsilon_from_turns`,
  `abs_diff_eq_radians`, `abs_diff_eq_turns` and `impl From<f64> for Angle` are
  removed in favour of `try_from_radians` / `try_from_turns` (which return
  `NonFiniteAngleError`) and the compile-time `radians!` / `turns!` macros.
  `from_turn_ratio` rejects ratios above one full turn at every width.
- ByteMessage gate payloads encode angles as exact `u64` fractions (protocol
  version 5); scheduled batches and events use versions 6 and 7. Older versions
  are rejected with an error naming the expected version. A `Custom` gate with
  non-angle parameters is rejected by `Gate::validate`.
- Every simulator has a fixed qubit count: `StateVectorEngine` no longer grows,
  `StateVectorSimulator::create_with_rng` is removed, and `QuantumEngineBuilder`
  has a required `num_qubits()` accessor. `SimBuilder` and the `MonteCarloEngine`
  convenience constructors return an error when a dynamic classical engine
  reports 0 qubits before execution, and `SimBuilder` rejects a simulation qubit
  count that differs from the quantum builder's. `pecos run` takes `--qubits N`,
  required for QIS programs.
- `MonteCarloEngine::new_with_defaults` / `new_with_depolarizing_noise` and the
  `with_depolarizing_noise` setters of `MonteCarloEngineBuilder` and
  `HybridEngineBuilder` return `Result`. Hybrid and Monte Carlo builder setters
  replace only the component they name; `HybridEngineBuilder` composes a fresh
  `QuantumSystem` at build (`QuantumSystem::into_components` is new).
- `PauliString` stores sorted, identity-free, distinct-qubit entries.
  `with_phase_and_paulis` and `from_decomposed` return `Result`;
  `ParsePauliStringError` is an enum with `DuplicateQubit`;
  `pauli_str(Option<usize>)` / `to_dense_str(Option<usize>)` become `pauli_str()`
  / `to_dense_str()` plus fallible `*_width(n)` forms. Pauli group order,
  element and centralizer queries, generator removal and multiplication,
  `StabilizerCode::distance` / `logical_operators` and controlled-rotation
  lowerings (including `TickHandle::crz`) are fallible.
  `PauliString::to_flat_matrix` returns `Result`, rejecting unsupported widths
  and support outside the width.
- `PauliAnnotation.pauli` is removed (`AnnotationKind::TrackedPauli { pauli }`
  carries a tracked Pauli), and `DemOutputMetadata.pauli` is
  `Option<PauliString>`.
- pecos-neo probability-, weight-, angle- and rate-valued constructors, setters
  and builders return `ValidationResult` and reject out-of-range or NaN values
  instead of clamping them; `clamp_probability` is removed. `OverRotation`
  scales the gate's signed principal rotation; angle-dependent probabilities
  use its signed principal magnitude. A positive `p_meas_crosstalk_local` without
  a neighbour function now returns a build `ValidationError` instead of warning
  and ignoring the setting. At-most-one Pauli rate sums allow only accumulated
  floating-point rounding (`n * f64::EPSILON` for `n` terms), replacing the
  previous `1e-9` tolerance; individual rates remain strictly validated.
- pecos-engines depolarizing and biased-depolarizing probability errors name the
  invalid parameter and its value, including uniform probabilities.
- Foreign C API: all `pecos_circuit_*` gate appenders
  return `int32_t` (`0` on success, `-1` on invalid input).
- `try_asin` / `try_acos` return `InverseTrigError`, distinguishing non-finite
  input from finite values outside [-1, 1]. Parsing integer multiples of pi
  wraps exactly, including multiples greater than a full turn.
- QuiZX export uses a 2^61 half-turn grid to keep simplifier arithmetic in range.
  Fractions divisible by four are exact; others round by at most two ticks,
  with ties away from zero on the signed principal representative.

### Rust bug fixes

- Gates appended to a `DagCircuit` after `remove_gate`, or after converting from
  a `TickCircuit`, could be placed before existing gates on the same qubit.
  After a removal an append could also panic, or wire from an unrelated gate
  that reused the freed slot. Removing a gate also dropped the order between its
  neighbours.
- `DagCircuit::wires` reported the same qubit for every parallel wire between
  two gates.
- `CancelInverses`, `MergeAdjacentRotations`, and `PeepholeOptimize` no longer
  rewrite across a manual bypass wire. `SimplifyRotations` rewires every wire of
  a gate it splits into per-qubit gates, and does not split a gate that has
  manual wires labeled outside its qubits.
- The surface-code `DagCircuitRenderer` added SX, SXdg, T, and Tdg without
  wires, so they could be ordered out of program order. Python `DagCircuit` now
  has `sx` and `sxdg` builders.
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

- Angle conversion from floats is exact (correctly rounded) for every width and
  both signs; negative inputs were less precise than positive ones (#1304).
  Internal QASM, PHIR and scheduled-transport pipelines no longer round-trip
  angles through `f64`.
- NaN or infinite angles at QASM, PHIR-JSON, textual IR, QIS, scheduled-noise
  and Selene ingress return errors instead of panicking (#1296).
- pecos-neo idle-noise configurations whose rate could overflow over an idle
  duration are rejected (#1298), and out-of-range probabilities raise instead of
  being clamped (#1299).
- Builders seeded from an existing `HybridEngine` kept only its classical engine,
  replacing the quantum engine with a state vector and dropping the noise model.

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
