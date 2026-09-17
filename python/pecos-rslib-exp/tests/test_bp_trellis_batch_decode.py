# Copyright 2026 The PECOS Developers

"""BpTrellis integration with typed specs and native parallel batch execution."""

import math
import random

import pytest
from pecos_rslib.qec import DemSampler, SampleBatch

exp = pytest.importorskip("pecos_rslib_exp")
bp_trellis = exp.bp_trellis

DEM = "error(0.1) D0 D1 D2 L0\nerror(0.03) D0\nerror(0.03) D1\nerror(0.03) D2\n"


def test_public_spec_and_configuration():
    from pecos.decoders import bp_trellis as public_bp_trellis

    assert public_bp_trellis() == bp_trellis()
    assert bp_trellis().family == "bp_trellis"
    assert not bp_trellis().history_dependent
    assert not bp_trellis().wall_clock_dependent
    assert repr(bp_trellis()) == "bp_trellis()"
    spec = bp_trellis(
        k=128,
        delta=math.inf,
        score_alpha=0.5,
        ordering=[3, 2, 1, 0],
        bp_score_iterations=2,
        merge_indistinguishable=False,
        escalation_ks=[256, 512],
    )
    assert eval(repr(spec), {"bp_trellis": bp_trellis}) == spec  # noqa: S307 - trusted local repr
    assert bp_trellis(escalation_ks=None) == bp_trellis(escalation_ks=[])


@pytest.mark.parametrize(
    "options",
    [
        {"k": 0},
        {"delta": math.nan},
        {"delta": -1},
        {"score_alpha": math.inf},
        {"score_alpha": -1},
        {"ordering": "unknown"},
        {"ordering": None},
        {"escalation_ks": [0]},
    ],
)
def test_invalid_options(options):
    with pytest.raises(ValueError, match=r"must|invalid|incompatible") as factory_error:
        bp_trellis(**options)
    with pytest.raises(ValueError, match=r"must|invalid|incompatible") as direct_error:
        exp.BpTrellisDecoder.from_dem(DEM, **options)
    assert str(factory_error.value) == str(direct_error.value)
    if "escalation_ks" in options:
        assert "escalation_ks[0]" in str(factory_error.value)


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"k": 2, "delta": 2.0},
        {"bp_score_iterations": 2},
        {"ordering": "time_order"},
        {"ordering": "backward_deadline"},
        {"ordering": [3, 2, 1, 0]},
        {"merge_indistinguishable": False},
        {"escalation_ks": [16, 32]},
    ],
)
def test_parallel_predictions_match_sequential(options):
    # Aperiodic rows expose chunk-order errors; exercise several dynamic chunks.
    rng = random.Random(35)
    rows = [[rng.randrange(2) for _ in range(3)] for _ in range(1025)]
    truth = [rng.randrange(2) for _ in rows]
    batch = SampleBatch(rows, truth)
    spec = bp_trellis(**options)
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
    auto = batch.decode(dem, bp_trellis(), predictions=True)
    assert auto.predictions == expected
    assert auto.num_errors == 0
    assert auto.execution_path == "parallel"
    count = batch.decode(dem, bp_trellis(), workers=3)
    assert count.predictions is None
    assert count.num_errors == 0
    empty = DemSampler.from_dem_string(dem).sample_batch(0, seed=1).decode(dem, bp_trellis(), workers=2)
    assert empty.num_shots == 0


@pytest.mark.parametrize("workers", [1, 3])
def test_invalid_order_and_impossible_syndrome_are_errors(workers):
    batch = SampleBatch([[0, 0, 0]], [0])
    with pytest.raises(RuntimeError, match="permutation"):
        batch.decode(DEM, bp_trellis(ordering=[0, 0, 1, 2]), workers=workers)
    with pytest.raises(RuntimeError, match="unexplainable"):
        SampleBatch([[1]], [0]).decode("detector D0\n", bp_trellis(), workers=workers)


def test_fused_sampling_matches_sequential_decoding():
    sampler = DemSampler.from_dem_string(DEM)
    expected = sampler.decode(DEM, 3073, bp_trellis(), seed=17, workers=1, predictions=True)
    fused = sampler.decode(DEM, 3073, bp_trellis(), seed=17, workers=3, predictions=True)
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
        {"escalation_ks": [16, 32]},
        {"ordering": "time_order"},
        {"bp_score_iterations": 2},
    ):
        direct = exp.BpTrellisDecoder.from_dem(DEM, **options)
        expected = [direct.decode_syndrome(row).observable_flips.mask for row in rows]
        result = SampleBatch(rows, [0] * len(rows)).decode(DEM, bp_trellis(**options), workers=3, predictions=True)
        assert result.predictions == expected


@pytest.mark.parametrize("workers", [1, 4])
def test_no_path_escalation_is_used_in_batch_execution(workers):
    dem = "error(0.4) D0\nerror(0.4) D1\nerror(0.1) D0 D1 D2 L0\n"
    options = {"k": 2, "bp_score_iterations": 0, "merge_indistinguishable": False, "ordering": "time_order"}
    batch = SampleBatch([[0, 0, 1]] * 1025, [1] * 1025)
    with pytest.raises(RuntimeError, match="unexplainable"):
        batch.decode(dem, bp_trellis(**options), workers=workers)
    result = batch.decode(
        dem,
        bp_trellis(**options, escalation_ks=[16]),
        workers=workers,
        predictions=True,
    )
    assert result.predictions == [1] * 1025
    assert result.num_errors == 0
