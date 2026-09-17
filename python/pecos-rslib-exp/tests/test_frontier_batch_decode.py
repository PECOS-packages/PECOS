# Copyright 2026 The PECOS Developers

"""Frontier integration with typed specs and native parallel batch execution."""

import math
import random

import pytest
from pecos_rslib.qec import DemSampler, SampleBatch

exp = pytest.importorskip("pecos_rslib_exp")
frontier = exp.frontier

DEM = "error(0.1) D0 D1 D2 L0\nerror(0.03) D0\nerror(0.03) D1\nerror(0.03) D2\n"


def test_public_spec_and_configuration():
    from pecos.decoders import frontier as public_frontier

    assert public_frontier() == frontier()
    assert frontier().family == "frontier"
    assert not frontier().history_dependent
    assert not frontier().wall_clock_dependent
    assert repr(frontier()) == "frontier()"
    spec = frontier(
        k=128,
        delta=math.inf,
        score_alpha=0.5,
        column_order=[3, 2, 1, 0],
        bp_score_iterations=2,
        merge_indistinguishable=True,
    )
    assert eval(repr(spec), {"frontier": frontier}) == spec  # noqa: S307 - trusted local repr
    assert frontier(metric_mode="frontierLite") == frontier(metric_mode="maxlog_int")


@pytest.mark.parametrize(
    "options",
    [
        {"k": 0},
        {"delta": math.nan},
        {"delta": -1},
        {"score_alpha": math.inf},
        {"score_alpha": -1},
        {"column_order": "unknown"},
        {"column_order": 3},
        {"metric_mode": "unknown"},
        {"int_metric_scale": 0},
        {"metric_mode": "maxlog_int", "delta": math.inf},
        {"metric_mode": "maxlog_int", "score_alpha": 1e-9},
        {"metric_mode": "maxlog_int", "merge_indistinguishable": True},
    ],
)
def test_invalid_options(options):
    with pytest.raises(ValueError, match=r"must|invalid|incompatible|quantizes") as factory_error:
        frontier(**options)
    with pytest.raises(ValueError, match=r"must|invalid|incompatible|quantizes") as direct_error:
        exp.FrontierDecoder.from_dem(DEM, **options)
    assert str(factory_error.value) == str(direct_error.value)
    with pytest.raises(ValueError, match=r"must|invalid|incompatible|quantizes") as committee_error:
        exp.FrontierCommitteeDecoder.from_dem(DEM, **options)
    assert str(factory_error.value) == str(committee_error.value)
    with pytest.raises(ValueError, match=r"must|invalid|incompatible|quantizes") as factor_error:
        exp.FrontierDecoder.from_factors([[(0.9, [], []), (0.1, [0], [0])]], 1, 1, **options)
    assert str(factory_error.value) == str(factor_error.value)


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"k": 2, "delta": 2.0},
        {"bp_score_iterations": 2},
        {"column_order": "time_order"},
        {"column_order": "backward_deadline_reorder"},
        {"column_order": [3, 2, 1, 0]},
        {"merge_indistinguishable": True},
        {"metric_mode": "maxlog_int"},
    ],
)
def test_parallel_predictions_match_sequential(options):
    # Aperiodic rows expose chunk-order errors; exercise several dynamic chunks.
    rng = random.Random(35)
    rows = [[rng.randrange(2) for _ in range(3)] for _ in range(1025)]
    truth = [rng.randrange(2) for _ in rows]
    batch = SampleBatch(rows, truth)
    spec = frontier(**options)
    sequential = batch.decode(DEM, spec, workers=1, predictions=True)
    parallel = batch.decode(DEM, spec, workers=4, predictions=True, timing=True)
    assert sequential.execution_path == "sequential"
    assert parallel.execution_path == "parallel"
    assert parallel.workers_used == 4
    assert parallel.predictions == sequential.predictions
    assert parallel.num_errors == sequential.num_errors
    assert parallel.num_errors == sum(a != b for a, b in zip(parallel.predictions, truth, strict=True))
    assert parallel.stats.num_timing_samples == len(rows)
    assert parallel.reproducibility_warnings == []


def test_auto_execution_wide_observables_and_count_only():
    dem = "error(0.1) D0 D1 D2 L70\n"
    rows = [[i % 2] * 3 for i in range(1024)]
    expected = [(1 << 70) if i % 2 else 0 for i in range(1024)]
    batch = SampleBatch(rows, expected)
    auto = batch.decode(dem, frontier(), predictions=True)
    assert auto.predictions == expected
    assert auto.num_errors == 0
    assert auto.execution_path == "parallel"
    count = batch.decode(dem, frontier(), workers=3)
    assert count.predictions is None
    assert count.num_errors == 0
    empty = DemSampler.from_dem_string(dem).sample_batch(0, seed=1).decode(dem, frontier(), workers=2)
    assert empty.num_shots == 0


@pytest.mark.parametrize("workers", [1, 3])
def test_invalid_order_and_impossible_syndrome_are_errors(workers):
    batch = SampleBatch([[0, 0, 0]], [0])
    with pytest.raises(RuntimeError, match="permutation"):
        batch.decode(DEM, frontier(column_order=[0, 0, 1, 2]), workers=workers)
    with pytest.raises(RuntimeError, match="unexplainable"):
        SampleBatch([[1]], [0]).decode("detector D0\n", frontier(), workers=workers)


def test_fused_sampling_matches_sequential_decoding():
    sampler = DemSampler.from_dem_string(DEM)
    expected = sampler.decode(DEM, 3073, frontier(), seed=17, workers=1, predictions=True)
    fused = sampler.decode(DEM, 3073, frontier(), seed=17, workers=3, predictions=True)
    assert fused.execution_path == "parallel"
    assert fused.workers_used == 3
    assert fused.predictions == expected.predictions
    assert fused.num_errors == expected.num_errors


def test_predictions_match_direct_experimental_binding():
    exp = pytest.importorskip("pecos_rslib_exp")
    rows = [[(i >> j) & 1 for j in range(3)] for i in range(8)]
    for options in (
        {},
        {"k": 2},
        {"metric_mode": "maxlog_int"},
        {"column_order": "time_order"},
        {"bp_score_iterations": 2},
    ):
        direct = exp.FrontierDecoder.from_dem(DEM, **options)
        expected = [direct.decode_syndrome(row).observable_flips.mask for row in rows]
        result = SampleBatch(rows, [0] * len(rows)).decode(DEM, frontier(**options), workers=3, predictions=True)
        assert result.predictions == expected
