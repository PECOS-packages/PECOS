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


def test_wall_clock_limited_mwpf_requires_explicit_parallel_opt_in() -> None:
    spec = mwpf(timeout=0.5)
    batch = _batch()
    try:
        automatic = batch.decode(DEM, spec)
    except ValueError as error:
        if "MWPF decoder is not available" in str(error):
            pytest.skip("MWPF feature is absent from this build")
        raise
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
