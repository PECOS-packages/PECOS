"""Published counts and independently derived Wilson identities for Lavasani's convention."""

import json
from math import sqrt
from pathlib import Path

import pytest
from pecos.qec.memory_benchmark import MemoryCounts, pooled_wilson_memory_estimate

DATA = json.loads((Path(__file__).parent / "fixtures/paper_color_codes/lavasani_memory_statistics.json").read_text())


@pytest.mark.parametrize("row", DATA["rows"], ids=lambda r: f"d{r['distance']}-{r['decoder']}")
def test_hardware_estimates_and_uncertainties_record_the_printed_discrepancy(row):
    estimate = pooled_wilson_memory_estimate([MemoryCounts(row["shots"], row["failures"])], rounds=row["rounds"])
    assert estimate.confidence == pytest.approx(0.6826894921370859)
    if "rounding_discrepancy" in row:
        # Keep the published number and independently count-derived value;
        # changing the estimator or widening the rounding tolerance would hide it.
        assert (row["distance"], row["decoder"]) == (5, "Frontier")
        assert estimate.rate_per_cycle == pytest.approx(row["count_implied_rate"], rel=1e-14)
        assert abs(estimate.rate_per_cycle - row["printed_rate"]) > row["rate_resolution"] / 2
    else:
        assert estimate.rate_per_cycle == pytest.approx(row["printed_rate"], abs=row["rate_resolution"] / 2, rel=0)
    assert estimate.interval[1] - estimate.rate_per_cycle == pytest.approx(
        row["printed_plus"],
        abs=row["plus_resolution"] / 2,
        rel=0,
    )
    assert estimate.rate_per_cycle - estimate.interval[0] == pytest.approx(
        row["printed_minus"],
        abs=row["minus_resolution"] / 2,
        rel=0,
    )


def test_pool_counts_before_normalizing_instead_of_equal_weighting_groups():
    estimate = pooled_wilson_memory_estimate([MemoryCounts(10, 5), MemoryCounts(90, 0)], rounds=20)
    assert estimate.shots == 100
    assert estimate.failures == 5
    assert estimate.pooled_groups == 2
    assert estimate.block_error_rate == 0.05
    assert estimate.rate_per_cycle == 0.05 / 20.5
    merged = pooled_wilson_memory_estimate([MemoryCounts(100, 5)], rounds=20)
    assert estimate.interval == merged.interval


def test_wilson_boundary_and_symmetric_analytic_identities():
    zero = pooled_wilson_memory_estimate([MemoryCounts(10, 0)], rounds=20)
    full = pooled_wilson_memory_estimate([MemoryCounts(10, 10)], rounds=20)
    assert zero.interval == pytest.approx((0.0, 1 / (11 * 20.5)))
    assert full.interval == pytest.approx((10 / (11 * 20.5), 1 / 20.5))
    middle = pooled_wilson_memory_estimate([MemoryCounts(100, 50)], rounds=1, initialization_rounds=0)
    assert middle.interval == pytest.approx((0.5 - 0.5 / sqrt(101), 0.5 + 0.5 / sqrt(101)))
    two_sigma = pooled_wilson_memory_estimate([MemoryCounts(10, 0)], rounds=1, initialization_rounds=0, z_score=2)
    assert two_sigma.interval == pytest.approx((0.0, 2 / 7))
    assert two_sigma.confidence == pytest.approx(0.9544997361036416)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rounds": 0},
        {"rounds": True},
        {"rounds": 20, "initialization_rounds": -1},
        {"rounds": 20, "initialization_rounds": float("nan")},
        {"rounds": 20, "z_score": 0},
        {"rounds": 20, "z_score": float("inf")},
        {"rounds": 20, "z_score": True},
    ],
)
def test_rejects_invalid_exposure_and_confidence(kwargs):
    with pytest.raises(ValueError, match="Pooled estimate requires"):
        pooled_wilson_memory_estimate([MemoryCounts(10, 1)], **kwargs)


def test_requires_sampled_resolved_trials():
    with pytest.raises(ValueError, match="Pooled estimate requires"):
        pooled_wilson_memory_estimate([], rounds=20)
    with pytest.raises(ValueError, match="Unresolved"):
        pooled_wilson_memory_estimate([MemoryCounts(10, 1, 1)], rounds=20)
