# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Contract tests for fused ``DemSampler.decode`` sampling ABI v1."""

from __future__ import annotations

import pytest

pytest.importorskip("pecos_rslib")

from pecos_rslib import TickCircuit  # noqa: E402
from pecos_rslib.decoders import mwpf, pymatching, relay_bp, tesseract  # noqa: E402
from pecos_rslib.qec import DagFaultAnalyzer, DemSampler  # noqa: E402

DEM = "error(0.25) D0 L0\nerror(0.1) D0\n"


def _sampler() -> DemSampler:
    return DemSampler.from_dem_string(DEM)


def test_worker_count_does_not_change_canonical_sample_stream() -> None:
    sampler = _sampler()
    spec = tesseract(preset="fast")
    sequential = sampler.decode(DEM, 1500, spec, seed=7, workers=1, predictions=True)
    parallel = sampler.decode(DEM, 1500, spec, seed=7, workers=2, predictions=True)
    automatic = sampler.decode(DEM, 1500, spec, seed=7, predictions=True)

    assert sequential.execution_path == "sequential"
    assert parallel.execution_path == "parallel"
    assert parallel.workers_used == 2
    assert automatic.execution_path == "parallel"
    assert sequential.num_errors == parallel.num_errors == automatic.num_errors
    assert sequential.predictions == parallel.predictions == automatic.predictions


def test_native_predictions_equal_single_shot_predictions() -> None:
    sampler = _sampler()
    spec = pymatching(correlated=True)
    native = sampler.decode(DEM, 1500, spec, seed=17, predictions=True)
    per_shot = sampler.decode(DEM, 1500, spec, seed=17, workers=1, predictions=True)

    assert native.execution_path == "native_batch"
    assert per_shot.execution_path == "sequential"
    assert native.num_errors == per_shot.num_errors
    assert native.predictions == per_shot.predictions


def test_wide_sampler_truth_and_predictions_do_not_narrow_through_u64() -> None:
    dem = "error(0.25) D0 L64\n"
    sampler = DemSampler.from_dem_string(dem)
    spec = pymatching(correlated=False)
    native = sampler.decode(dem, 64, spec, seed=17, predictions=True)
    per_shot = sampler.decode(dem, 64, spec, seed=17, workers=1, predictions=True)

    assert native.predictions == per_shot.predictions
    assert native.num_errors == per_shot.num_errors == 0
    assert native.predictions is not None
    assert 1 << 64 in native.predictions
    assert set(native.predictions) <= {0, 1 << 64}


def test_fused_stream_golden_values_span_chunk_boundaries() -> None:
    result = _sampler().decode(
        DEM,
        2065,
        pymatching(correlated=False),
        seed=42,
        workers=1,
        predictions=True,
    )

    # Reproducibility ABI v1 drift pins, spanning both canonical 1024-shot chunk
    # boundaries. A change here means the seeded sampling stream moved, which is a
    # breaking change: bump the ABI version and release-note it rather than
    # regenerating these literals.
    assert result.num_errors == 212
    assert result.predictions is not None
    assert {index: result.predictions[index] for index in [0, 1023, 1024, 2047, 2048, 2064]} == {
        0: 1,
        1023: 0,
        1024: 0,
        2047: 1,
        2048: 0,
        2064: 0,
    }


def test_same_seed_prediction_stream_has_prefix_property() -> None:
    sampler = _sampler()
    spec = pymatching(correlated=False)
    predictions = [
        sampler.decode(DEM, shots, spec, seed=42, workers=1, predictions=True).predictions
        for shots in [1025, 1500, 2065]
    ]
    assert predictions[0] is not None
    assert predictions[1] is not None
    assert predictions[2] is not None
    assert predictions[0] == predictions[1][:1025]
    assert predictions[1] == predictions[2][:1500]


def test_resolved_seed_round_trips_and_distinct_seeds_diverge() -> None:
    sampler = _sampler()
    spec = pymatching(correlated=False)
    entropy_seeded = sampler.decode(DEM, 256, spec, predictions=True)
    assert entropy_seeded.sampling_seed_used is not None
    replay = sampler.decode(
        DEM,
        256,
        spec,
        seed=entropy_seeded.sampling_seed_used,
        predictions=True,
    )
    assert replay.sampling_seed_used == entropy_seeded.sampling_seed_used
    assert replay.num_errors == entropy_seeded.num_errors
    assert replay.predictions == entropy_seeded.predictions

    seed_7 = sampler.decode(DEM, 256, spec, seed=7, predictions=True)
    seed_8 = sampler.decode(DEM, 256, spec, seed=8, predictions=True)
    assert seed_7.sampling_seed_used == 7
    assert seed_8.sampling_seed_used == 8
    assert seed_7.predictions != seed_8.predictions


def test_history_and_wall_clock_traits_reuse_batch_planner_contracts() -> None:
    sampler = _sampler()
    relay = sampler.decode(DEM, 32, relay_bp(), seed=3)
    assert relay.execution_path == "sequential"
    assert sampler.decode(DEM, 32, relay_bp(), seed=3, workers=1).num_errors == relay.num_errors
    with pytest.raises(ValueError, match=r"worker count 4"):
        sampler.decode(DEM, 32, relay_bp(), seed=3, workers=4)

    try:
        timed = sampler.decode(DEM, 32, mwpf(timeout=0.5), seed=3, workers=4)
    except ValueError as error:
        if "MWPF decoder is not available" in str(error):
            pytest.skip("MWPF feature is absent from this build")
        raise
    assert timed.execution_path == "parallel"
    assert timed.reproducibility_warnings


def test_empty_runs_still_resolve_seed_and_preflight_decoder() -> None:
    sampler = _sampler()
    result = sampler.decode(DEM, 0, "pymatching", predictions=True)
    assert result.num_shots == 0
    assert result.num_errors == 0
    assert result.logical_error_rate == 0.0
    assert result.workers_used == 1
    assert result.sampling_seed_used is not None
    assert result.predictions == []
    assert result.stats is None
    with pytest.raises(ValueError, match="num_shots"):
        result.interval()

    wrong_dimension = "error(0.1) D0 D1 L0\n"
    with pytest.raises(ValueError, match=r"1 detectors.*2"):
        sampler.decode(wrong_dimension, 0, "pymatching", seed=19)
    with pytest.raises(ValueError, match="not_a_decoder"):
        sampler.decode(DEM, 0, "not_a_decoder", seed=19)


def test_raw_sampler_rejection_precedes_decoder_parsing() -> None:
    circuit = TickCircuit()
    circuit.tick().pz([0])
    circuit.tick().mz([0])
    influence_map = DagFaultAnalyzer(circuit.to_dag_circuit()).build_influence_map()
    sampler = DemSampler.raw_uniform(influence_map, 0.01)

    with pytest.raises(ValueError, match=r"raw-measurement.*not detector events"):
        sampler.decode(DEM, 0, "not_a_decoder", seed=3)


def test_timing_counts_decode_calls_and_combines_with_predictions() -> None:
    result = _sampler().decode(
        DEM,
        33,
        pymatching(correlated=False),
        seed=5,
        predictions=True,
        timing=True,
    )
    assert result.execution_path == "sequential"
    assert result.predictions is not None
    assert len(result.predictions) == result.num_shots
    assert result.stats is not None
    assert result.stats.num_timing_samples == result.num_shots
    assert result.stats.summed_decode_elapsed == result.stats.total_seconds
    assert result.stats.wall_elapsed >= result.stats.summed_decode_elapsed


def test_legacy_sampling_streams_remain_frozen() -> None:
    sampler = _sampler()

    # Stream-stability pins for the pre-ABI legacy paths. These rows and count
    # guard their original continuous and geometric RNG streams, respectively.
    assert sampler.sample_decode_count(DEM, 12, "pymatching", seed=91) == 3
    batch = sampler.sample_batch(12, seed=91)
    assert batch.detector_events() == [
        [True],
        [False],
        [False],
        [False],
        [False],
        [False],
        [False],
        [False],
        [False],
        [False],
        [True],
        [True],
    ]
    assert batch.observable_flips() == [
        [True],
        [False],
        [False],
        [False],
        [False],
        [False],
        [False],
        [False],
        [False],
        [False],
        [True],
        [False],
    ]


def test_worker_count_is_clamped_to_the_number_of_sampling_chunks() -> None:
    # A sampling chunk is the unit of parallel work, so a caller asking for more
    # workers than there are chunks must be told what actually ran -- and must not
    # hand an arbitrary thread count to the OS.
    result = _sampler().decode(DEM, 1500, tesseract(preset="fast"), seed=7, workers=64)

    assert result.execution_path == "parallel"
    assert result.workers_used == 2  # ceil(1500 / 1024)

    single_chunk = _sampler().decode(DEM, 100, tesseract(preset="fast"), seed=7, workers=8)
    assert single_chunk.workers_used == 1

    # Clamping must not perturb the guaranteed stream.
    reference = _sampler().decode(DEM, 1500, tesseract(preset="fast"), seed=7, workers=1, predictions=True)
    clamped = _sampler().decode(DEM, 1500, tesseract(preset="fast"), seed=7, workers=64, predictions=True)
    assert clamped.predictions == reference.predictions
