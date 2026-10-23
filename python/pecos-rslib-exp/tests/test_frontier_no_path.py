# Copyright 2026 The PECOS Developers

"""Direct Frontier reporting preserves outcomes without inventing predictions."""

import math
import random

import pecos_rslib_exp as exp
import pytest


@pytest.mark.parametrize("metric_mode", ["logsumexp_float", "maxlog_int"])
@pytest.mark.parametrize("workers", [1, 4])
def test_mixed_batch_retains_order_and_matches_individual_attempts(metric_mode, workers):
    decoder = exp.FrontierDecoder.from_dem("error(0.1) D0 D1 L70", metric_mode=metric_mode)
    rng = random.Random(35)
    shots = [[rng.randrange(2), rng.randrange(2)] for _ in range(1025)]
    expected = [decoder.decode_syndrome(shot, on_no_path="report") for shot in shots]
    actual = decoder.decode_batch(shots, workers=workers, on_no_path="report")
    assert len(actual) == len(shots)
    for shot, result, single in zip(shots, actual, expected, strict=True):
        assert result.no_path == single.no_path == (shot[0] != shot[1])
        assert result.status == single.status
        assert result.transitions == single.transitions
        assert result.dropped_states == single.dropped_states
        assert result.bp_seconds == single.bp_seconds == 0.0
        if result.no_path:
            assert isinstance(result, exp.FrontierNoPath)
            assert result.status == "no_path"
            assert not hasattr(result, "observable_flips")
            assert not hasattr(result, "log_evidence")
            assert not hasattr(result, "runner_up_gap")
        else:
            assert isinstance(result, exp.FrontierResult)
            assert result.observable_flips.mask == single.observable_flips.mask == shot[0] << 70
            assert result.log_evidence == single.log_evidence
            assert result.logical_masses == single.logical_masses


@pytest.mark.parametrize("workers", [1, 4])
def test_reporting_does_not_swallow_invalid_shots(workers):
    decoder = exp.FrontierDecoder.from_dem("error(0.1) D0 D1")
    # A no-path before an invalid row must not hide that row's error.
    with pytest.raises(RuntimeError) as single:
        decoder.decode_syndrome([1], on_no_path="report")
    with pytest.raises(RuntimeError) as batch:
        decoder.decode_batch([[1, 0], [1], [0, 0]], workers=workers, on_no_path="report")
    assert str(batch.value) == f"shot 1: {single.value}"
    with pytest.raises(RuntimeError, match="Invalid node index"):
        decoder.decode_from_defects([2], on_no_path="report")


@pytest.mark.parametrize("workers", [1, 4])
def test_raise_remains_default_with_first_failure_index(workers):
    decoder = exp.FrontierDecoder.from_dem("error(0.1) D0 D1")
    with pytest.raises(RuntimeError) as single:
        decoder.decode_syndrome([1, 0])
    for options in ({}, {"on_no_path": "raise"}):
        with pytest.raises(RuntimeError) as batch:
            decoder.decode_batch([[0, 0], [1, 0], [0, 1]], workers=workers, **options)
        assert str(batch.value) == f"shot 1: {single.value}"


def test_sparse_reporting_and_forced_residual_have_no_placeholder_prediction():
    decoder = exp.FrontierDecoder.from_dem("error(1) D0 L70")
    report = decoder.decode_from_defects([], on_no_path="report")
    assert isinstance(report, exp.FrontierNoPath)
    assert report.no_path
    assert report.transitions == report.dropped_states == report.bp_seconds == 0
    assert repr(report) == "FrontierNoPath(transitions=0, dropped_states=0)"
    assert not hasattr(report, "observable_flips")
    assert decoder.decode_from_defects([0], on_no_path="report").observable_flips.mask == 1 << 70


def test_pruned_no_path_is_reported_without_retrying_or_changing_scratch():
    dem = "error(0.4) D0\nerror(0.4) D1\nerror(0.1) D0 D1 D2 L0"
    options = {"delta": math.inf, "score_alpha": 0, "column_order": "time_order"}
    decoder = exp.FrontierDecoder.from_dem(dem, k=1, **options)
    result = decoder.decode_syndrome([0, 0, 1], on_no_path="report")
    assert result.no_path
    assert result.transitions > 0
    assert result.dropped_states > 0
    # A retained path exists, but report mode neither changes K nor retries it.
    wider = exp.FrontierDecoder.from_dem(dem, k=8, **options)
    assert wider.decode_syndrome([0, 0, 1]).observable_flips.mask == 1
    assert decoder.decode_syndrome([0, 0, 0]).observable_flips.mask == 0
    with pytest.raises(RuntimeError, match="unexplainable"):
        decoder.decode_syndrome([0, 0, 1])


@pytest.mark.parametrize("metric_mode", ["logsumexp_float", "maxlog_int"])
def test_categorical_factors_report_no_path(metric_mode):
    decoder = exp.FrontierDecoder.from_factors(
        [[(0.5, [], []), (0.3, [0, 1], [0]), (0.2, [0, 1], [])]],
        2,
        1,
        metric_mode=metric_mode,
    )
    assert decoder.decode_syndrome([0, 1], on_no_path="report").no_path
    assert not decoder.decode_syndrome([1, 1], on_no_path="report").no_path


def test_policy_validation_includes_empty_batches():
    decoder = exp.FrontierDecoder.from_dem("detector D0")
    for method, argument in (
        (decoder.decode_syndrome, [0]),
        (decoder.decode_from_defects, []),
        (decoder.decode_batch, []),
    ):
        with pytest.raises(ValueError, match="on_no_path must be"):
            method(argument, on_no_path="identity")
    assert decoder.decode_batch([], workers=4, on_no_path="report") == []
    with pytest.raises(ValueError, match="workers must be"):
        decoder.decode_batch([], workers=0, on_no_path="report")
