# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Contract tests for the planned ``SampleBatch.decode`` entry point."""

from __future__ import annotations

import math
import threading

import pytest

from pecos_rslib.decoders import (
    bp_osd,
    fusion_blossom,
    mwpf,
    pecos_uf,
    pymatching,
    relay_bp,
    tesseract,
    tesseract_trellis,
)
from pecos_rslib import TickCircuit
from pecos_rslib.qec import DagFaultAnalyzer, DemSampler, SampleBatch

DEM = "error(0.1) D0 L0\n"


def _batch(num_shots: int = 8) -> SampleBatch:
    rows = [[shot & 1] for shot in range(num_shots)]
    truth_masks = [int(shot % 3 == 0) for shot in range(num_shots)]
    return SampleBatch(rows, truth_masks)


def _aperiodic_batch(num_shots: int) -> SampleBatch:
    # Order-restoration tests need predictions whose pattern shares no period
    # with the chunk size: with period-2 rows every chunk's prediction slice
    # is identical and permuting chunks is invisible. Period 97 is coprime to
    # every plausible chunk size.
    rows = [[1 if shot % 97 == 0 else 0] for shot in range(num_shots)]
    truth_masks = [int(shot % 3 == 0) for shot in range(num_shots)]
    return SampleBatch(rows, truth_masks)


@pytest.mark.parametrize(
    ("legacy", "factory"),
    [
        ("pymatching", lambda: pymatching(correlated=True)),
        ("pymatching_uncorrelated", lambda: pymatching(correlated=False)),
        ("tesseract", lambda: tesseract(preset="fast")),
        ("tesseract_trellis", lambda: tesseract_trellis()),
        ("bp_osd", bp_osd),
        ("fusion_blossom_serial", lambda: fusion_blossom(solver="serial")),
        ("pecos_uf", pecos_uf),
    ],
)
def test_spec_and_legacy_string_match_frozen_count(legacy, factory) -> None:
    batch = _batch()
    # Behavior pin: all listed decoders predict the one detector's L0 edge,
    # producing five mismatches against _batch's deliberately different truth.
    assert batch.decode(DEM, factory(), workers=1).num_errors == 5
    assert batch.decode(DEM, legacy, workers=1).num_errors == 5


def test_native_and_generic_predictions_are_ordered_wide_python_ints() -> None:
    dem = "error(0.1) D0 L64\n"
    batch = SampleBatch([[1], [0], [1]], [1 << 64, 0, 1 << 64])
    spec = pymatching(correlated=True)

    native = batch.decode(dem, spec, predictions=True)
    generic = batch.decode(dem, spec, workers=1, predictions=True)

    assert native.execution_path == "native_batch"
    assert native.predictions == generic.predictions
    assert native.predictions == [1 << 64, 0, 1 << 64]
    assert native.num_errors == generic.num_errors == 0


def test_native_batch_honors_both_pymatching_correlation_modes() -> None:
    dem = "error(0.01) D0 D1 ^ D2 D3 L0\n" "error(0.1) D2\n" "error(0.1) D3\n"
    batch = SampleBatch([[1, 1, 1, 1], [0, 0, 0, 0]], [0, 0])
    predictions_by_mode = []
    for correlated in [True, False]:
        spec = pymatching(correlated=correlated)
        native = batch.decode(dem, spec, predictions=True)
        per_shot = batch.decode(dem, spec, workers=1, predictions=True)
        assert native.execution_path == "native_batch"
        assert native.predictions == per_shot.predictions
        predictions_by_mode.append(native.predictions)

    assert predictions_by_mode[0] != predictions_by_mode[1]


@pytest.mark.parametrize("correlated", [False, True])
@pytest.mark.parametrize(("workers", "execution_path"), [(1, "sequential"), (3, "parallel"), (None, "native_batch")])
@pytest.mark.parametrize("predictions", [False, True])
def test_pymatching_without_observables(correlated, workers, execution_path, predictions) -> None:
    dem = "error(0.1) D0 D1"
    batch = SampleBatch([[0, 0], [1, 1]], [0, 0], num_observables=0)
    result = batch.decode(dem, pymatching(correlated=correlated), workers=workers, predictions=predictions)
    reference = batch.decode(dem, bp_osd(), workers=workers, predictions=predictions)

    assert result.execution_path == execution_path
    assert result.num_shots == reference.num_shots == 2
    assert result.num_errors == reference.num_errors == 0
    assert result.predictions == reference.predictions == ([0, 0] if predictions else None)

    sampler = DemSampler.from_dem_string(dem)
    sampled = sampler.decode(
        dem, 64, pymatching(correlated=correlated), seed=1, workers=workers, predictions=predictions
    )
    sampled_reference = sampler.decode(dem, 64, bp_osd(), seed=1, workers=workers, predictions=predictions)

    assert sampled.execution_path == execution_path
    assert sampled.num_shots == sampled_reference.num_shots == 64
    assert sampled.num_errors == sampled_reference.num_errors == 0
    assert sampled.predictions == sampled_reference.predictions == ([0] * 64 if predictions else None)


def test_prediction_and_timing_requests_can_be_combined() -> None:
    result = _batch().decode(
        DEM,
        pymatching(correlated=False),
        predictions=True,
        timing=True,
    )
    assert result.execution_path != "native_batch"
    assert result.predictions == [0, 1, 0, 1, 0, 1, 0, 1]
    assert result.stats is not None
    assert result.stats.num_shots == 8
    assert result.stats.num_timing_samples == result.num_shots
    assert result.stats.wall_elapsed > 0
    assert result.stats.summed_decode_elapsed == result.stats.total_seconds
    assert len(result.stats.quantiles) == 21


def test_parallel_predictions_and_timing_preserve_every_shot() -> None:
    batch = _aperiodic_batch(1024)
    spec = tesseract(preset="fast")
    sequential = batch.decode(
        DEM,
        spec,
        workers=1,
        predictions=True,
        timing=True,
    )
    parallel = batch.decode(
        DEM,
        spec,
        workers=4,
        predictions=True,
        timing=True,
    )

    assert parallel.execution_path == "parallel"
    assert parallel.predictions == sequential.predictions
    assert sequential.stats is not None
    assert parallel.stats is not None
    assert sequential.stats.num_timing_samples == batch.num_shots
    assert parallel.stats.num_timing_samples == batch.num_shots


def test_result_fields_and_interval_domain() -> None:
    batch = SampleBatch([[0], [0], [0], [0]], [0, 1, 0, 1])
    result = batch.decode(DEM, "pymatching", predictions=False, timing=False)

    assert result.num_shots == 4
    assert result.num_errors == 2
    assert result.logical_error_rate == 0.5
    assert result.workers_used == 1
    assert result.reproducibility_warnings == []
    assert result.sampling_seed_used is None
    assert result.predictions is None
    assert result.stats is None
    assert "native_batch" in repr(result)
    for alpha in [1e-6, 0.05, 0.5]:
        lo, hi = result.interval(alpha)
        assert 0 <= lo <= hi <= 1
    for alpha in [0.999e-6, 0.500001, math.nan, math.inf, -math.inf]:
        with pytest.raises(ValueError, match="alpha"):
            result.interval(alpha)

    empty = DemSampler.from_dem_string(DEM).sample_batch(0, seed=1).decode(DEM, "pymatching")
    assert empty.num_shots == 0
    assert empty.logical_error_rate == 0.0
    with pytest.raises(ValueError, match="num_shots"):
        empty.interval()

    known = SampleBatch([[0], [0]], [0, 1]).decode(DEM, "pymatching")
    assert known.interval() == pytest.approx((0.06083027592009732, 0.9391697240799026))


def test_explicit_and_auto_worker_contracts() -> None:
    batch = _batch(1024)
    spec = tesseract(preset="fast")
    sequential = batch.decode(DEM, spec, workers=1)
    parallel = batch.decode(DEM, spec, workers=4)
    automatic = batch.decode(DEM, spec)

    assert sequential.num_errors == parallel.num_errors == automatic.num_errors
    assert sequential.execution_path == "sequential"
    assert parallel.execution_path == "parallel"
    assert parallel.workers_used == 4
    assert automatic.execution_path == "parallel"
    assert automatic.workers_used >= 2
    for result in [sequential, parallel, automatic]:
        assert result.logical_error_rate == result.num_errors / result.num_shots
        assert result.reproducibility_warnings == []
        assert result.predictions is None
        assert result.stats is None
        assert result.sampling_seed_used is None
        assert result.execution_path in repr(result)
    with pytest.raises(ValueError, match="workers"):
        batch.decode(DEM, spec, workers=0)

    assert _batch().decode(DEM, spec).execution_path == "sequential"


def test_explicit_workers_are_bounded_by_the_shot_count() -> None:
    # A worker beyond one per shot could only idle, so the pool never grows past the batch.
    batch = _batch(2)
    result = batch.decode(DEM, tesseract(preset="fast"), workers=64)
    assert result.execution_path == "parallel"
    assert result.workers_used == 2
    assert result.num_errors == batch.decode(DEM, tesseract(preset="fast"), workers=1).num_errors
    empty = DemSampler.from_dem_string(DEM).sample_batch(0, seed=1).decode(DEM, tesseract(preset="fast"), workers=64)
    assert (empty.execution_path, empty.workers_used, empty.num_shots) == ("parallel", 1, 0)


def test_wall_clock_limited_mwpf_requires_explicit_parallel_opt_in() -> None:
    spec = mwpf(timeout=0.5)
    batch = _batch()
    automatic = batch.decode(DEM, spec)
    assert automatic.execution_path == "sequential"
    parallel = batch.decode(DEM, spec, workers=4)
    assert parallel.execution_path == "parallel"
    assert parallel.workers_used == 4
    assert parallel.reproducibility_warnings


def test_history_dependent_relay_bp_planning() -> None:
    batch = _batch()
    spec = relay_bp()
    automatic = batch.decode(DEM, spec)
    assert automatic.execution_path == "sequential"
    # Behavior pin for the history-dependent planner path.
    assert automatic.num_errors == 5
    assert batch.decode(DEM, spec, workers=1).num_errors == automatic.num_errors
    with pytest.raises(ValueError, match=r"worker count 4"):
        batch.decode(DEM, spec, workers=4)


def test_dimension_mismatch_is_value_error_on_native_path() -> None:
    batch = SampleBatch([[0, 0]], [0])
    with pytest.raises(ValueError, match=r"2 detectors.*1"):
        batch.decode(DEM, pymatching(correlated=True))


def test_dem_none_requires_corpus_and_loaded_corpus_uses_embedded_dem(tmp_path) -> None:
    generated = DemSampler.from_dem_string(DEM).sample_batch(16, seed=7)
    with pytest.raises(ValueError, match="no embedded DEM"):
        generated.decode(decoder="pymatching")

    path = tmp_path / "decode.pecos"
    generated.save(path, dem=DEM)
    loaded = SampleBatch.load(path)
    # The corpus was sampled from the same single-mechanism DEM, so decoding is exact.
    assert loaded.decode(decoder=pymatching(correlated=True)).num_errors == 0


def test_embedded_dem_guard_and_opt_out(tmp_path) -> None:
    path = tmp_path / "guard.pecos"
    _batch(2).save(path, dem=DEM)
    loaded = SampleBatch.load(path)
    different = "error(0.2) D0 L0\n"
    with pytest.raises(ValueError, match="differs from the DEM embedded"):
        loaded.decode(different, "pymatching")
    # Behavior pin: both supplied shots disagree with this DEM's predictions.
    assert (
        loaded.decode(
            different,
            pymatching(correlated=True),
            allow_dem_mismatch=True,
        ).num_errors
        == 2
    )


def test_hybrid_spec_uses_embedded_full_and_argument_decomposed_dem() -> None:
    full = "error(0.1) D0 L0\n"
    decomposed = "error(0.1) D0\n"
    batch = SampleBatch([[1]], [0])

    assert (
        batch.decode(
            decomposed,
            f"belief_matching_hybrid:{full}",
            workers=1,
        ).num_errors
        == 0
    )


def test_raw_measurement_error_precedes_invalid_decoder() -> None:
    circuit = TickCircuit()
    circuit.tick().pz([0])
    circuit.tick().mz([0])
    influence_map = DagFaultAnalyzer(circuit.to_dag_circuit()).build_influence_map()
    batch = DemSampler.raw_uniform(influence_map, 0.01).sample_batch(1, seed=3)

    with pytest.raises(ValueError, match=r"raw-measurement.*not detector events"):
        batch.decode(DEM, "not_a_decoder", allow_dem_mismatch=True)


def test_gil_is_released_during_decode() -> None:
    batch = _batch(100_000)
    started = threading.Event()
    stop = threading.Event()
    progress = [0]

    def worker() -> None:
        started.set()
        while not stop.is_set():
            progress[0] += 1

    thread = threading.Thread(target=worker)
    thread.start()
    started.wait()
    before = progress[0]
    try:
        batch.decode(DEM, pymatching(correlated=True))
    finally:
        stop.set()
        thread.join()
    assert progress[0] - before > 100


def test_parallel_chunk_scheduling_preserves_order_and_counts() -> None:
    # Workers pull dynamic chunks rather than one fixed slice each, so shot
    # order has to be restored from the chunk index. Oversubscribing workers
    # makes any completion-order leak show up; the aperiodic fixture makes a
    # chunk permutation change the prediction list instead of hiding.
    batch = _aperiodic_batch(4096)
    spec = tesseract(preset="fast")
    sequential = batch.decode(DEM, spec, workers=1, predictions=True, timing=True)

    for workers in (2, 7, 64):
        parallel = batch.decode(DEM, spec, workers=workers, predictions=True, timing=True)
        assert parallel.num_errors == sequential.num_errors
        assert parallel.predictions == sequential.predictions
        assert parallel.stats.num_timing_samples == batch.num_shots


def _sector_dem(*, labelled=True):
    from pecos_rslib.qec import DetectorErrorModel
    from pecos_rslib.quantum import TickCircuit

    circuit = TickCircuit()
    circuit.tick().pz([0, 1, 2, 3, 4])
    circuit.tick().mz([0, 1, 2, 3, 4])
    records = [[-5, -3, -1], [-5, -2, -1], [-4, -3, -1], [-4, -2, -1]]
    for i, (refs, sector) in enumerate(zip(records, ["X", "Z", "X", "Z"], strict=True)):
        circuit.add_detector(refs, detector_id=i, fault_sector=sector if labelled else None)
    return DetectorErrorModel.from_circuit(circuit, p1=0, p2=0, p_meas=0.01, p_prep=0)


def _repeated_detector_dem(count):
    from pecos_rslib.qec import DetectorErrorModel
    from pecos_rslib.quantum import TickCircuit

    circuit = TickCircuit()
    circuit.tick().pz([0])
    circuit.tick().mz([0])
    for i in range(count):
        circuit.add_detector([-1], detector_id=i, fault_sector="X")
    return DetectorErrorModel.from_circuit(circuit, p1=0, p2=0, p_meas=0.01, p_prep=0)


def _report_counts(report):
    return (
        report.input_hyperedges,
        report.projected,
        report.zero_probability_omitted,
        report.dropped_indices,
        report.cross_sector_components,
        report.label_conflicts,
        report.unlabelled_detectors,
        report.distinct_sectors,
    )


def test_contract_object_sectors_change_projected_components():
    dem = _sector_dem()
    batch = SampleBatch([[0, 0, 0, 0]], [0], num_observables=0)
    report = batch.decode(dem, pymatching(correlated=True)).projection
    text_report = batch.decode(str(dem), pymatching(correlated=True)).projection
    # Each of five measurement faults has p=.01: 01, 0123, 02, 13, 23.
    # All have empty observable support. The four-detector fault has two covers.
    # Sectors choose 02+13; no sectors tie-break lexicographically to 01+23.
    # Only passthrough 01 and 23 cross known sectors: 2. Text has no known
    # sectors, so its cross count is 0 even though its chosen pairs differ.
    assert _report_counts(report) == (1, 1, [], [], 2, 0, 0, 2)
    assert _report_counts(text_report) == (1, 1, [], [], 0, 0, 4, 0)
    labelled = dem.build_decoder(pymatching(correlated=True))
    unlabelled = _sector_dem(labelled=False).build_decoder(pymatching(correlated=True))
    assert labelled.projection == report
    assert unlabelled.projection == text_report
    assert "D0 D2 ^ D1 D3" in labelled.dem
    assert "D0 D1 ^ D2 D3" in unlabelled.dem


@pytest.mark.parametrize(("workers", "path"), [(1, "sequential"), (2, "parallel"), (None, "native_batch")])
def test_contract_projection_on_batch_and_fused_paths(workers, path):
    dem = _sector_dem()
    sampler = dem.to_sampler()
    batch = sampler.sample_batch(2050, seed=41)
    result = batch.decode(dem, pymatching(correlated=True), workers=workers, predictions=True)
    fused = sampler.decode(dem, 2050, pymatching(correlated=True), workers=workers, seed=41, predictions=True)
    assert result.execution_path == fused.execution_path == path
    # The five faults 01, 0123, 02, 13, 23 contain one hyperedge, covered by
    # 02+13. Only passthrough 01 and 23 cross sectors; all four ids are labelled.
    assert _report_counts(result.projection) == (1, 1, [], [], 2, 0, 0, 2)
    assert _report_counts(fused.projection) == (1, 1, [], [], 2, 0, 0, 2)
    assert result.predictions == fused.predictions
    assert result.num_errors == fused.num_errors


def test_contract_report_value_semantics_and_read_only_properties():
    dem = _sector_dem()
    report = dem.build_decoder(pymatching(correlated=True)).projection
    assert report == dem.build_decoder(pymatching(correlated=True)).projection
    assert report != object()
    assert "ProjectionReport" in repr(report)
    report.dropped_indices.append(99)
    report.zero_probability_omitted.append(99)
    assert report.dropped_indices == report.zero_probability_omitted == []
    with pytest.raises(AttributeError):
        report.projected = 999
    import pickle

    with pytest.raises(TypeError):
        pickle.dumps(report)


def test_contract_object_identity_and_embedded_text(tmp_path):
    dem = _sector_dem()
    batch = dem.to_sampler().sample_batch(4, seed=2)
    path = tmp_path / "sectors.npz"
    batch.save(path, dem=str(dem))
    loaded = SampleBatch.load(path)
    assert loaded.decode(dem, pymatching(correlated=True)).projection.unlabelled_detectors == 0
    assert loaded.decode(None, pymatching(correlated=True)).projection.unlabelled_detectors == 4
    wrong_path = tmp_path / "wrong.npz"
    batch.save(wrong_path, dem=str(dem) + "\n")
    wrong = SampleBatch.load(wrong_path)
    with pytest.raises(ValueError, match="DEM"):
        wrong.decode(dem, pymatching(correlated=True))
    with pytest.raises(ValueError, match="DEM"):
        wrong.compare_decoders(dem, "pymatching", "pymatching")
    assert wrong.decode(dem, pymatching(correlated=True), allow_dem_mismatch=True).projection.unlabelled_detectors == 0
    assert (
        wrong.compare_decoders(
            dem,
            "pymatching",
            "pymatching",
            allow_dem_mismatch=True,
        ).dut_projection.unlabelled_detectors
        == 0
    )


def test_compatibility_graphlike_model_has_no_projection():
    dem = _repeated_detector_dem(1)
    batch = dem.to_sampler().sample_batch(8, seed=3)
    assert batch.decode(dem, pymatching(correlated=True)).projection is None
    assert batch.decode(str(dem), pymatching(correlated=True)).projection is None
    assert dem.build_decoder(pymatching(correlated=True)).projection is None
    assert dem.to_sampler().decode(dem, 8, pymatching(correlated=True), seed=3).projection is None
    comparison = batch.compare_decoders(dem, "pymatching", "pymatching")
    assert comparison.dut_projection is comparison.reference_projection is None


def test_contract_comparison_retains_both_preparation_reports():
    dem = _sector_dem()
    batch = dem.to_sampler().sample_batch(8, seed=3)
    result = batch.compare_decoders(dem, "pymatching", "pymatching_uncorrelated")
    assert (
        result.dut_projection
        == result.reference_projection
        == dem.build_decoder(pymatching(correlated=True)).projection
    )
    assert result.total_shots == 8
    assert result.counts == [[8, 0, 0], [0, 0, 0], [0, 0, 0]]
    text = batch.compare_decoders(str(dem), "pymatching", "pymatching")
    assert text.dut_projection.unlabelled_detectors == text.reference_projection.unlabelled_detectors == 4
    graph_vs_hyper = batch.compare_decoders(dem, "pymatching", "tesseract")
    assert graph_vs_hyper.dut_projection == result.dut_projection
    assert graph_vs_hyper.reference_projection is None
    hyper_vs_graph = batch.compare_decoders(dem, "tesseract", "pymatching")
    assert hyper_vs_graph.dut_projection is None
    assert hyper_vs_graph.reference_projection == result.reference_projection


@pytest.mark.parametrize("fused", [False, True])
def test_contract_provider_rejects_object_before_build(fused):
    class Provider:
        _pecos_decoder_api_version = 1
        history_dependent = False
        wall_clock_dependent = False
        builds = 0

        def _pecos_build_decoder(self, dem):
            self.builds += 1
            raise AssertionError("must not call provider")

    dem = _sector_dem()
    provider = Provider()
    if fused:
        with pytest.raises(RuntimeError, match="providers require DEM text by protocol"):
            dem.to_sampler().decode(dem, 8, provider)
    else:
        with pytest.raises(RuntimeError, match="providers require DEM text by protocol"):
            dem.to_sampler().sample_batch(8, seed=3).decode(dem, provider)
    assert provider.builds == 0


def test_contract_hybrid_refuses_object_everywhere():
    dem = _sector_dem()
    batch = dem.to_sampler().sample_batch(8, seed=3)
    for call in [
        lambda: batch.decode(dem, "belief_matching_hybrid:error(0.01) D0 D1 D2 D3"),
        lambda: dem.to_sampler().decode(dem, 8, "belief_matching_hybrid:error(0.01) D0 D1 D2 D3"),
        lambda: dem.build_decoder("belief_matching_hybrid:error(0.01) D0 D1 D2 D3"),
        lambda: batch.compare_decoders(dem, "belief_matching_hybrid:error(0.01) D0 D1 D2 D3", "pymatching"),
    ]:
        with pytest.raises(RuntimeError, match="requires.*DEM text.*objects are not supported"):
            call()


@pytest.mark.parametrize("tracked", [False, True])
def test_contract_object_is_standard_text_plus_labels(tracked):
    from pecos_rslib import PauliString
    from pecos_rslib.qec import DetectorErrorModel

    circuit = TickCircuit()
    circuit.tick().pz([0, 1, 2, 3, 4])
    circuit.tick().mz([0, 1, 2, 3, 4])
    records = [[-5, -3, -1], [-5, -2, -1], [-4, -3, -1], [-4, -2, -1]]
    for i, refs in enumerate(records):
        circuit.add_detector(refs, detector_id=i, fault_sector="X" if i % 2 == 0 else "Z")
    circuit.add_observable([-5], observable_id=0)
    if tracked:
        circuit.tracked_pauli(PauliString.from_str("Z"), label="tracked")
    dem = DetectorErrorModel.from_circuit(circuit, p1=0, p2=0, p_meas=0.0123456789, p_prep=0.0234567891)
    assert dem.num_tracked_paulis == int(tracked)
    sampler = dem.to_sampler()
    batch = sampler.sample_batch(2048, seed=719)
    spec = bp_osd()
    obj = batch.decode(dem, spec, predictions=True, workers=1)
    text = batch.decode(str(dem), spec, predictions=True, workers=1)
    assert obj.projection is text.projection is None
    assert obj.predictions == text.predictions
    assert len(set(obj.predictions)) > 1
    assert obj.num_errors == text.num_errors
    built = dem.build_decoder(spec)

    # Compare numeric probabilities and written target order as well: a model
    # can change at the boundary without changing this finite set of predictions.
    def mechanisms(text):
        return [
            (float(probability.removeprefix("error(")), targets)
            for line in text.splitlines()
            if line.startswith("error(")
            for probability, _, targets in [line.partition(") ")]
        ]

    assert mechanisms(built.dem) == mechanisms(str(dem))
    assert [built.decode_syndrome(row).mask for row in batch.detector_events()] == text.predictions
    fused_object = sampler.decode(dem, 2048, spec, seed=719, predictions=True, workers=1)
    fused_text = sampler.decode(str(dem), 2048, spec, seed=719, predictions=True, workers=1)
    assert fused_object.predictions == fused_text.predictions
    assert fused_object.num_errors == fused_text.num_errors
    assert (
        batch.compare_decoders(dem, "bp_osd", "bp_osd").counts
        == batch.compare_decoders(str(dem), "bp_osd", "bp_osd").counts
    )
