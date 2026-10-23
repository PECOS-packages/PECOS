# Copyright 2026 The PECOS Developers
"""Rust-backed bounded recovery and explicit fallback predictions in Python."""

import math

import pecos_rslib_exp as exp
import pytest

DEM = "error(0.4) D0\nerror(0.4) D1\nerror(0.1) D0 D1 D2 L70"


def decoder(k=1, fallback=None, **kwargs):
    return exp.FrontierDecoder.from_dem(
        DEM,
        k=k,
        delta=0,
        score_alpha=0,
        column_order="time_order",
        fallback=fallback,
        **kwargs,
    )


@pytest.mark.parametrize("workers", [1, 4])
@pytest.mark.parametrize("metric_mode", ["logsumexp_float", "maxlog_int"])
def test_delta_max_caps_retries_and_does_not_change_base(workers, metric_mode):
    engine = decoder(metric_mode=metric_mode)
    report = engine.decode_syndrome([0, 0, 1], on_no_path="report", delta_max=3)
    assert report.deltas_tried == [0, 1, 2, 3]
    assert report.cause == "exhausted"
    assert not report.fallback_used
    assert engine.decode_syndrome([0, 0, 1], on_no_path="report").deltas_tried == [0]
    outcomes = engine.decode_batch([[0, 0, 0], [0, 0, 1]] * 65, workers=workers, on_no_path="report", delta_max=3)
    for i, result in enumerate(outcomes):
        assert result.no_path == bool(i % 2)
        assert result.deltas_tried == ([0, 1, 2, 3] if i % 2 else [0])
    with pytest.raises(RuntimeError, match="unexplainable"):
        engine.decode_from_defects([2], delta_max=3)


@pytest.mark.parametrize("metric_mode", ["logsumexp_float", "maxlog_int"])
def test_retry_success_is_a_normal_result_and_stops_early(metric_mode):
    engine = decoder(k=8, metric_mode=metric_mode)
    result = engine.decode_from_defects([2], delta_max=32)
    assert isinstance(result, exp.FrontierResult)
    assert not result.no_path
    assert not result.fallback_used
    assert len(result.deltas_tried) > 1
    assert result.delta_used == result.deltas_tried[-1] < 32
    assert result.escalation_rungs_used == len(result.deltas_tried) - 1
    assert result.observable_flips.mask == 1 << 70


@pytest.mark.parametrize("workers", [1, 4])
def test_always_answer_feasible_fallback_preserves_verified_witness(workers):
    engine = decoder(fallback="feasible_then_zero")
    rows = [[0, 0, 0], [0, 0, 1]] * 65
    results = engine.decode_batch(rows, workers=workers, on_no_path="fallback", delta_max=3)
    for row, result in zip(rows, results, strict=True):
        if row[2] == 0:
            assert not result.fallback_used
            continue
        assert isinstance(result, exp.FrontierFallback)
        assert result.no_path
        assert result.fallback_used
        assert result.fallback_policy == "feasible_then_zero"
        assert result.status == "fallback:feasible"
        assert result.syndrome_consistent is True
        assert result.mechanism_indices == [0, 1, 2]
        assert result.observable_flips.mask == 1 << 70
        assert result.deltas_tried == [0, 1, 2, 3]
        assert result.failure.cause == "exhausted"
        assert result.transitions == result.failure.transitions
        assert not hasattr(result, "log_evidence")
        assert not hasattr(result, "runner_up_gap")
    # Configuring a fallback does not silently enable it for strict calls.
    with pytest.raises(RuntimeError, match="unexplainable"):
        engine.decode_syndrome([0, 0, 1])


@pytest.mark.parametrize("fallback", ["zero", "feasible_then_zero"])
def test_impossible_syndrome_has_flagged_zero_prediction(fallback):
    engine = exp.FrontierDecoder.from_dem("error(1) D0 L70", fallback=fallback)
    result = engine.decode_from_defects([], on_no_path="fallback", delta_max=100)
    assert result.status == "fallback:zero"
    assert result.fallback_policy == fallback
    assert result.syndrome_consistent is (False if fallback == "feasible_then_zero" else None)
    assert result.mechanism_indices is None
    assert len(result.observable_flips) == 71
    assert result.observable_flips.mask == 0
    assert result.failure.cause == "residual"
    assert result.failure.detector == 0
    assert result.deltas_tried == []
    with pytest.raises(RuntimeError, match="dimensions"):
        engine.decode_syndrome([], on_no_path="fallback")
    with pytest.raises(RuntimeError, match="shot 1"):
        engine.decode_batch([[1], []], on_no_path="fallback", workers=2)


@pytest.mark.parametrize("maximum", [-1, math.nan, math.inf])
def test_invalid_recovery_options_rejected_for_empty_batches(maximum):
    with pytest.raises(ValueError, match="delta_max"):
        decoder().decode_batch([], on_no_path="report", delta_max=maximum)


def test_explicit_fallback_configuration_and_categorical_limits():
    with pytest.raises(ValueError, match="configured fallback"):
        decoder().decode_batch([], on_no_path="fallback")
    with pytest.raises(ValueError, match="fallback must be"):
        decoder(fallback="guess")
    with pytest.raises(ValueError, match="delta_max"):
        exp.FrontierDecoder.from_dem(DEM, delta=16).decode_batch([], delta_max=8)
    factors = [[(0.5, [], []), (0.3, [0, 1], [0]), (0.2, [0, 1], [])]]
    with pytest.raises(RuntimeError, match="independent binary DEM"):
        exp.FrontierDecoder.from_factors(factors, 2, 1, fallback="feasible_then_zero")
    engine = exp.FrontierDecoder.from_factors(factors, 2, 1, fallback="zero")
    result = engine.decode_syndrome([0, 1], on_no_path="fallback", delta_max=100)
    assert result.status == "fallback:zero"
    assert result.failure.cause == "infeasible"
    assert result.deltas_tried == [50]


def recovery_spec(**kwargs):
    return exp.frontier(k=1, delta=0, score_alpha=0, column_order="time_order", **kwargs)


def recovery_rows(result):
    return [
        (r.shot_index, r.undecoded, r.fallback_used, r.syndrome_consistent, r.deltas_tried) for r in result.recovery
    ]


@pytest.mark.parametrize("workers", [1, 4])
@pytest.mark.parametrize("retain", [False, True])
def test_shared_batch_fallbacks_are_scored_against_truth(workers, retain):
    from pecos_rslib.qec import SampleBatch

    rows = [[0, 0, 0], [0, 0, 1], [0, 0, 1]] * 65
    # Identical fallback syndromes have different true logical masks: one is wrong.
    truth = [0, 1 << 70, 0] * 65
    spec = recovery_spec(delta_max=3, on_no_path="fallback", fallback="feasible_then_zero")
    assert eval(repr(spec), {"frontier": exp.frontier}) == spec  # noqa: S307 - trusted local repr
    result = SampleBatch(rows, truth).decode(DEM, spec, workers=workers, predictions=retain, timing=True)
    assert result.num_errors == 65
    assert sum(r.logical_error for r in result.recovery if r.fallback_used) == 65
    assert result.num_scored == len(rows)
    assert result.num_fallback == 130
    assert result.num_undecoded == 0
    assert result.logical_error_rate == pytest.approx(1 / 3)
    assert result.stats.logical_error_rate == result.logical_error_rate
    assert result.predictions == ([0, 1 << 70, 1 << 70] * 65 if retain else None)
    assert recovery_rows(result) == [(i, False, True, True, [0, 1, 2, 3]) for i in range(len(rows)) if i % 3]


@pytest.mark.parametrize("workers", [1, 4])
def test_shared_reporting_does_not_score_missing_predictions(workers):
    from pecos_rslib.qec import SampleBatch

    result = SampleBatch([[0, 0, 0], [0, 0, 1]] * 65, [1, 0] * 65).decode(
        DEM,
        recovery_spec(delta_max=3, on_no_path="report"),
        workers=workers,
        predictions=True,
        timing=True,
    )
    assert result.predictions == [0, None] * 65
    assert result.num_undecoded == 65
    assert result.num_scored == 65
    assert result.num_errors == 65
    assert result.num_fallback == 0
    assert math.isnan(result.logical_error_rate)
    assert math.isnan(result.stats.logical_error_rate)
    with pytest.raises(ValueError, match="undecoded"):
        result.interval()
    assert recovery_rows(result) == [(i, True, False, None, [0, 1, 2, 3]) for i in range(1, 130, 2)]


@pytest.mark.parametrize("on_no_path", ["report", "fallback"])
def test_fused_sampler_recovery_is_worker_independent(on_no_path):
    from pecos_rslib.qec import DemSampler

    sampler = DemSampler.from_dem_string(DEM)
    spec = recovery_spec(delta_max=3, on_no_path=on_no_path, fallback="feasible_then_zero")
    # Fused sampling uses sampling ABI v1, a different stream from sample_batch.
    expected = sampler.decode(DEM, 2051, spec, workers=1, seed=43, predictions=True)
    assert expected.recovery
    for workers in [1, 4]:
        result = sampler.decode(DEM, 2051, spec, workers=workers, seed=43, predictions=True)
        assert result.predictions == expected.predictions
        assert result.num_errors == expected.num_errors
        assert result.num_undecoded == expected.num_undecoded
        assert result.num_fallback == expected.num_fallback
        assert recovery_rows(result) == recovery_rows(expected)


def test_batch_retry_success_and_impossible_fallback_are_distinguished():
    from pecos_rslib.qec import SampleBatch

    result = SampleBatch([[0, 0, 1]], [1 << 70]).decode(
        DEM,
        exp.frontier(k=8, delta=0, score_alpha=0, column_order="time_order", delta_max=32),
        predictions=True,
    )
    assert result.num_errors == 0
    assert result.num_fallback == 0
    assert result.recovery[0].deltas_tried[-1] < 32
    assert len(result.recovery[0].deltas_tried) > 1
    result = SampleBatch([[0]], [1 << 70]).decode(
        "error(1) D0 L70",
        exp.frontier(on_no_path="fallback", fallback="feasible_then_zero"),
        predictions=True,
    )
    assert result.predictions == [0]
    assert result.num_errors == 1
    assert result.num_fallback == 1
    assert result.recovery[0].syndrome_consistent is False


@pytest.mark.parametrize(
    "kwargs",
    [{"delta_max": -1}, {"delta_max": math.inf}, {"on_no_path": "fallback"}, {"on_no_path": "unknown"}],
)
def test_batch_factory_rejects_invalid_recovery_options(kwargs):
    with pytest.raises(ValueError, match=r"delta_max|on_no_path"):
        recovery_spec(**kwargs)
