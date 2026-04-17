# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Unit tests for the fit math in examples/surface/native_dem_threshold_sweep.py.

These helpers are pure functions with no external dependencies, so they can be
tested directly without running a full sweep. The goal is to catch silent
numerical regressions in the stats path (Wilson intervals, golden-section fit,
linear regression, threshold-scaling gates) that would otherwise only show up
as bad numbers on a real sweep.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType


def _repo_root() -> Path:
    """Return the repo root directory by walking up from this test file."""
    cur = Path(__file__).resolve()
    for candidate in [cur, *cur.parents]:
        if (candidate / "Justfile").is_file() and (candidate / "examples").is_dir():
            return candidate
    msg = f"Could not locate repo root above {cur}"
    raise RuntimeError(msg)


_SWEEP_MODULE_NAME = "_surface_sweep_under_test"


def _load_sweep_module() -> ModuleType:
    """Load the sweep example script as an importable module for white-box testing.

    The example lives outside any Python package, so we load it by path. The
    module is registered in ``sys.modules`` before executing so that
    ``dataclass()`` can resolve ``cls.__module__`` when introspecting string
    annotations (otherwise ``dataclasses.py`` crashes on
    ``sys.modules.get(cls.__module__).__dict__``).
    """
    example_path = _repo_root() / "examples" / "surface" / "native_dem_threshold_sweep.py"
    spec = importlib.util.spec_from_file_location(_SWEEP_MODULE_NAME, example_path)
    if spec is None or spec.loader is None:
        msg = f"Could not load sweep module from {example_path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_SWEEP_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(_SWEEP_MODULE_NAME, None)
        raise
    return module


@pytest.fixture(scope="module")
def sweep() -> ModuleType:
    """Module-scoped handle to the loaded sweep example script."""
    return _load_sweep_module()


def _make_point(
    sweep: ModuleType,
    *,
    distance: int,
    basis: str,
    p: float,
    total_rounds: int,
    logical_error_rate: float,
    num_shots: int = 1000,
) -> object:
    """Build a ``SweepPoint`` consistent with the given logical error rate."""
    num_errors = round(logical_error_rate * num_shots)
    return sweep.SweepPoint(
        backend="test",
        distance=distance,
        basis=basis,
        physical_error_rate=p,
        total_rounds=total_rounds,
        num_shots=num_shots,
        num_logical_errors=num_errors,
        num_raw_errors=None,
        logical_error_rate=logical_error_rate,
        raw_error_rate=None,
    )


def _make_summary(
    sweep: ModuleType,
    *,
    distance: int,
    p: float,
    epsilon_per_round: float,
    basis: str = "X",
    fit_rms: float = 0.0,
) -> object:
    """Build a ``FitSummary`` with the given fitted per-round rate and RMS."""
    projected = sweep.ler_over_rounds(epsilon_per_round, distance)
    return sweep.FitSummary(
        backend="test",
        distance=distance,
        basis=basis,
        physical_error_rate=p,
        num_shots_per_round_point=1000,
        round_values=(distance, distance + 1),
        observed_logical_error_rates=(projected, projected),
        observed_raw_error_rates=(None, None),
        fitted_logical_error_rate_per_round=epsilon_per_round,
        fitted_projected_logical_error_rate_over_d_rounds=projected,
        fit_root_mean_square_error=fit_rms,
    )


# ---------------------------------------------------------------------------
# Wilson interval
# ---------------------------------------------------------------------------


def test_wilson_interval_zero_errors_is_nonzero_upper(sweep: ModuleType) -> None:
    """With no observed errors the lower bound is zero but the upper is not."""
    low, high = sweep._wilson_interval(0, 100)
    assert low == 0.0
    assert 0.0 < high < 1.0


def test_wilson_interval_all_errors_is_nonzero_lower(sweep: ModuleType) -> None:
    """With every trial an error the upper bound is one but the lower is not."""
    low, high = sweep._wilson_interval(100, 100)
    assert high == 1.0
    assert 0.0 < low < 1.0


def test_wilson_interval_brackets_point_estimate(sweep: ModuleType) -> None:
    """The interval must contain the point estimate and lie inside ``[0, 1]``."""
    n = 1000
    k = 37
    low, high = sweep._wilson_interval(k, n)
    point = k / n
    assert low <= point <= high
    assert 0.0 <= low < high <= 1.0


def test_wilson_interval_rejects_zero_trials(sweep: ModuleType) -> None:
    """Zero trials has no defined interval; must raise."""
    with pytest.raises(ValueError, match="num_trials must be positive"):
        sweep._wilson_interval(0, 0)


# ---------------------------------------------------------------------------
# Linear regression
# ---------------------------------------------------------------------------


def test_linear_regression_exact_line(sweep: ModuleType) -> None:
    """Regression of an exact line recovers (slope, intercept)."""
    xs = [0.0, 1.0, 2.0, 3.0]
    ys = [1.0, 3.0, 5.0, 7.0]  # y = 2x + 1
    slope, intercept = sweep._linear_regression(xs, ys)
    assert slope == pytest.approx(2.0)
    assert intercept == pytest.approx(1.0)


def test_linear_regression_rejects_too_few_points(sweep: ModuleType) -> None:
    """Fewer than two points is not a valid regression."""
    with pytest.raises(ValueError, match="at least two"):
        sweep._linear_regression([1.0], [2.0])


def test_linear_regression_rejects_degenerate_x(sweep: ModuleType) -> None:
    """All-equal ``xs`` makes the slope undefined."""
    with pytest.raises(ValueError, match="distinct x"):
        sweep._linear_regression([1.0, 1.0, 1.0], [2.0, 3.0, 4.0])


def test_linear_regression_requires_matching_lengths(sweep: ModuleType) -> None:
    """Mismatched ``xs``/``ys`` lengths is a programmer error."""
    with pytest.raises(ValueError, match="same length"):
        sweep._linear_regression([1.0, 2.0], [3.0])


# ---------------------------------------------------------------------------
# Per-round rate fit
# ---------------------------------------------------------------------------


def test_fit_per_round_rate_recovers_known_rate(sweep: ModuleType) -> None:
    """Fitting points generated from a known per-round rate recovers that rate."""
    true_rate = 3.0e-3
    rounds = [6, 8, 10, 12]
    points = [
        _make_point(
            sweep,
            distance=5,
            basis="X",
            p=0.005,
            total_rounds=r,
            logical_error_rate=sweep.ler_over_rounds(true_rate, r),
        )
        for r in rounds
    ]
    fitted = sweep._fit_per_round_rate(points)
    assert fitted == pytest.approx(true_rate, rel=1e-3)


def test_fit_per_round_rate_single_point_closed_form(sweep: ModuleType) -> None:
    """With one observation the fit uses the closed-form per-round inversion."""
    true_rate = 2.5e-3
    r = 10
    points = [
        _make_point(
            sweep,
            distance=5,
            basis="X",
            p=0.005,
            total_rounds=r,
            logical_error_rate=sweep.ler_over_rounds(true_rate, r),
        ),
    ]
    fitted = sweep._fit_per_round_rate(points)
    assert fitted == pytest.approx(true_rate, rel=1e-6)


def test_fit_per_round_rate_empty_raises(sweep: ModuleType) -> None:
    """Zero observations cannot be fit and must raise a clear error."""
    with pytest.raises(ValueError, match="at least one"):
        sweep._fit_per_round_rate([])


def test_fit_per_round_rate_all_zero_returns_zero(sweep: ModuleType) -> None:
    """All-zero observed logical error rates imply a zero per-round rate."""
    points = [
        _make_point(sweep, distance=5, basis="X", p=0.005, total_rounds=r, logical_error_rate=0.0) for r in (6, 8, 10)
    ]
    assert sweep._fit_per_round_rate(points) == 0.0


# ---------------------------------------------------------------------------
# Noise-dominated fit warning
# ---------------------------------------------------------------------------


def test_fit_rms_warning_emitted_when_rms_exceeds_epsilon(sweep: ModuleType) -> None:
    """Warning fires when the fit residual dwarfs the fitted per-round rate."""
    summary = _make_summary(sweep, distance=3, p=0.006, epsilon_per_round=5.0e-3, fit_rms=1.0e-2)
    text = sweep._fit_rms_warning_text(summary)
    assert text  # non-empty
    assert "WARNING" in text
    assert "increase --shots" in text


def test_fit_rms_warning_silent_when_rms_below_epsilon(sweep: ModuleType) -> None:
    """A well-converged fit (RMS much smaller than epsilon) emits no warning."""
    summary = _make_summary(sweep, distance=5, p=0.002, epsilon_per_round=5.0e-3, fit_rms=1.0e-4)
    assert sweep._fit_rms_warning_text(summary) == ""


def test_fit_rms_warning_silent_for_degenerate_epsilon(sweep: ModuleType) -> None:
    """``eps==0`` and ``eps>=0.5`` are handled elsewhere and should not trigger a noise warning."""
    for eps in (0.0, 0.5):
        summary = _make_summary(sweep, distance=3, p=0.1, epsilon_per_round=eps, fit_rms=1.0)
        assert sweep._fit_rms_warning_text(summary) == ""


# ---------------------------------------------------------------------------
# Threshold-fit gating (no tautological p_th from two-point fits)
# ---------------------------------------------------------------------------


def test_fit_distance_scaling_none_with_two_distances(sweep: ModuleType) -> None:
    """Two distances trivially fit a line; p_th would be tautological so skip."""
    summaries = [
        _make_summary(sweep, distance=3, p=0.006, epsilon_per_round=6.0e-3),
        _make_summary(sweep, distance=5, p=0.006, epsilon_per_round=3.0e-3),
    ]
    assert sweep._fit_distance_scaling_at_fixed_p(summaries) is None


def test_fit_distance_scaling_ok_with_three_distances(sweep: ModuleType) -> None:
    """With three distances the fixed-p scaling fit returns a meaningful threshold."""
    summaries = [
        _make_summary(sweep, distance=3, p=0.006, epsilon_per_round=6.0e-3),
        _make_summary(sweep, distance=5, p=0.006, epsilon_per_round=3.0e-3),
        _make_summary(sweep, distance=7, p=0.006, epsilon_per_round=1.5e-3),
    ]
    fit = sweep._fit_distance_scaling_at_fixed_p(summaries)
    assert fit is not None
    assert fit.fitted_suppression_factor == pytest.approx(math.exp(math.log(2.0)))
    assert fit.fitted_threshold == pytest.approx(0.006 * 2.0)


def test_fit_global_scaling_none_with_two_points(sweep: ModuleType) -> None:
    """Two (d, p) points fit two parameters perfectly; skip to avoid tautology."""
    summaries = [
        _make_summary(sweep, distance=3, p=0.006, epsilon_per_round=6.0e-3),
        _make_summary(sweep, distance=5, p=0.006, epsilon_per_round=3.0e-3),
    ]
    assert sweep._fit_global_scaling_law(summaries) is None


def test_fit_global_scaling_ok_with_three_points(sweep: ModuleType) -> None:
    """With three or more (d, p) points the global scaling fit runs."""
    summaries = [
        _make_summary(sweep, distance=3, p=0.006, epsilon_per_round=6.0e-3),
        _make_summary(sweep, distance=5, p=0.006, epsilon_per_round=3.0e-3),
        _make_summary(sweep, distance=7, p=0.006, epsilon_per_round=1.5e-3),
    ]
    fit = sweep._fit_global_scaling_law(summaries)
    assert fit is not None
    assert fit.fitted_threshold > 0.0


# ---------------------------------------------------------------------------
# Dashboard rebuild ordering
# ---------------------------------------------------------------------------


def test_load_sweep_data_from_json_recovers_tuple_fields(sweep: ModuleType, tmp_path: Path) -> None:
    """Round-tripping a FitSummary through JSON must restore tuple-typed fields as tuples."""
    import dataclasses

    summary = _make_summary(sweep, distance=5, p=0.005, epsilon_per_round=2.0e-3)
    json_path = tmp_path / "results.json"
    json_path.write_text(json.dumps({"points": [], "fit_summaries": [dataclasses.asdict(summary)]}))

    _, summaries, _ = sweep.load_sweep_data_from_json(json_path)
    assert len(summaries) == 1
    loaded = summaries[0]
    # Tuple fields would arrive as JSON arrays without explicit reconstruction;
    # the loader must promote them back to tuples so downstream code that expects
    # tuple semantics (hashing, equality) still works.
    assert isinstance(loaded.round_values, tuple)
    assert isinstance(loaded.observed_logical_error_rates, tuple)
    assert loaded.fitted_logical_error_rate_per_round == pytest.approx(2.0e-3)


def test_render_plot_artifacts_from_loaded_json_produces_files(
    sweep: ModuleType,
    tmp_path: Path,
) -> None:
    """JSON results file should be enough to regenerate plots without rerunning the sweep."""
    import dataclasses

    points = [
        sweep.SweepPoint(
            backend="test",
            distance=d,
            basis="X",
            physical_error_rate=0.005,
            total_rounds=r,
            num_shots=1000,
            num_logical_errors=int(0.02 * 1000),
            num_raw_errors=None,
            logical_error_rate=0.02,
            raw_error_rate=None,
        )
        for d in (3, 5)
        for r in (6, 8)
    ]
    summaries = [
        _make_summary(sweep, distance=d, p=0.005, epsilon_per_round=eps, basis="X")
        for d, eps in [(3, 4.0e-3), (5, 2.0e-3)]
    ]

    json_path = tmp_path / "myrun_results.json"
    json_path.write_text(
        json.dumps(
            {
                "points": [dataclasses.asdict(point) for point in points],
                "fit_summaries": [dataclasses.asdict(summary) for summary in summaries],
            },
        ),
    )

    loaded_points, loaded_summaries, _ = sweep.load_sweep_data_from_json(json_path)
    plots = sweep.render_plot_artifacts(
        tmp_path,
        prefix="myrun",
        points=loaded_points,
        summaries=loaded_summaries,
        formats=["svg", "pdf"],
    )

    assert plots, "render_plot_artifacts should return DashboardPlot entries for SVG outputs"
    expected_files = {
        "myrun_test_per_round_overlay.svg",
        "myrun_test_per_round_overlay.pdf",
        "myrun_test_p_0p005_duration_overlay.svg",
        "myrun_test_p_0p005_duration_overlay.pdf",
        "myrun_test_x_projected_d_rounds.svg",
        "myrun_test_x_projected_d_rounds.pdf",
        "myrun_test_x_per_round.svg",
        "myrun_test_x_per_round.pdf",
    }
    actual_files = {path.name for path in tmp_path.iterdir() if path.suffix in (".svg", ".pdf")}
    assert expected_files == actual_files


def test_fit_fss_threshold_recovers_synthetic_p_th(sweep: ModuleType) -> None:
    """The Wang-Harrington-Preskill FSS fit must recover a known synthetic ``p_th``.

    Generates noise-free LER values on a grid of ``(p, d)`` points using the exact
    polynomial form the fitter targets, then checks that the returned ``p_th`` and
    ``nu`` match the generator's values within the fit's reported standard error.

    Skipped when ``pecos.analysis`` is not importable (happens in pytest because
    ``tests/pecos/`` shadows the installed ``pecos`` package).
    """
    pytest.importorskip("pecos.analysis.threshold_curve", reason="pecos.analysis shadowed by test-tree pecos/")
    true_p_th = 0.010
    true_nu = 1.3
    true_a, true_b, true_c = 0.02, 1.2, 12.0

    summaries: list[object] = []
    for d in (3, 5, 7, 9):
        for p in (0.006, 0.008, 0.009, 0.010, 0.011, 0.012, 0.014):
            x = (p - true_p_th) * (d ** (1.0 / true_nu))
            eps = true_a + true_b * x + true_c * x * x
            if eps <= 0.0:
                continue
            summaries.append(_make_summary(sweep, distance=d, p=p, epsilon_per_round=eps, basis="X"))

    fit = sweep._fit_fss_threshold(summaries, seed_threshold=0.011, seed_nu=1.0)
    assert fit is not None
    assert fit.p_th == pytest.approx(true_p_th, abs=max(3.0 * fit.p_th_std_error, 5e-4))
    assert fit.nu == pytest.approx(true_nu, abs=max(3.0 * fit.nu_std_error, 0.2))
    assert fit.num_points >= 5
    assert fit.fit_window_low > 0.0
    assert fit.fit_window_high > fit.fit_window_low


def test_fit_fss_threshold_returns_none_without_enough_data(sweep: ModuleType) -> None:
    """Too few near-threshold points -> FSS fitter returns ``None`` (caller can fall back)."""
    summaries = [
        _make_summary(sweep, distance=3, p=0.010, epsilon_per_round=1e-3),
        _make_summary(sweep, distance=5, p=0.010, epsilon_per_round=8e-4),
    ]
    fit = sweep._fit_fss_threshold(summaries, seed_threshold=0.010)
    assert fit is None


def test_power_law_fit_respects_max_physical_error_rate(sweep: ModuleType) -> None:
    """Filtering to ``p <= p_th`` must drop above-threshold points that flatten the fit.

    Simulates a clean below-threshold power law (c=3) combined with a set of
    above-threshold points where the curve visibly flattens (saturating toward
    a weaker power of p). Restricting the fit via ``max_physical_error_rate``
    must recover the true below-threshold exponent; the unrestricted fit
    gets pulled toward the flatter above-threshold slope.
    """
    true_exponent = 3.0
    prefactor = 1.0
    summaries: list[object] = []
    for p in (0.002, 0.003, 0.004, 0.005, 0.006):
        eps = prefactor * (p**true_exponent)
        summaries.append(_make_summary(sweep, distance=5, p=p, epsilon_per_round=eps, basis="X"))
    # Above-threshold points visibly saturating -- eps grows much slower with p,
    # dragging the unrestricted OLS slope below the clean c=3.
    for p, eps in ((0.010, 3.0e-7), (0.012, 3.5e-7), (0.015, 4.0e-7)):
        summaries.append(_make_summary(sweep, distance=5, p=p, epsilon_per_round=eps, basis="X"))

    unrestricted = sweep._fit_per_distance_power_law(summaries)
    restricted = sweep._fit_per_distance_power_law(summaries, max_physical_error_rate=0.007)

    assert len(unrestricted) == 1
    assert len(restricted) == 1
    # Unrestricted fit gets pulled toward the flatter above-threshold slope.
    assert unrestricted[0].fitted_exponent < true_exponent - 0.5
    # Restricted fit recovers the true exponent within tolerance.
    assert restricted[0].fitted_exponent == pytest.approx(true_exponent, abs=0.05)
    # Standard error is populated and sensible for this clean synthetic data.
    assert restricted[0].fitted_exponent_std_error >= 0.0
    assert restricted[0].fitted_exponent_std_error < 0.1


def test_estimate_threshold_uses_per_round_crossing(sweep: ModuleType) -> None:
    """Threshold estimate must cross where per-round rates match, not the d-scaled metric.

    Using ``fitted_projected_logical_error_rate_over_d_rounds`` (roughly ``d * eps``)
    makes the large-d curve overtake the small-d curve at a lower ``p`` than the true
    threshold, so the estimator would underreport. Per-round rates are the canonical
    definition.
    """
    # Below threshold (p=0.004, 0.008): d=3 per-round > d=9 per-round.
    # Above threshold (p=0.012): d=9 per-round > d=3 per-round.
    # True crossing lives between p=0.008 and p=0.012.
    summaries = [
        _make_summary(sweep, distance=3, p=0.004, epsilon_per_round=5.0e-3),
        _make_summary(sweep, distance=9, p=0.004, epsilon_per_round=1.0e-3),
        _make_summary(sweep, distance=3, p=0.008, epsilon_per_round=7.0e-3),
        _make_summary(sweep, distance=9, p=0.008, epsilon_per_round=5.0e-3),
        _make_summary(sweep, distance=3, p=0.012, epsilon_per_round=9.0e-3),
        _make_summary(sweep, distance=9, p=0.012, epsilon_per_round=1.2e-2),
    ]
    threshold = sweep._estimate_threshold(summaries)
    assert threshold is not None
    assert 0.008 < threshold < 0.012


def test_merge_sweep_shards_sums_shots_and_refits(sweep: ModuleType, tmp_path: Path) -> None:
    """Two shards with the same config must merge to combined shot counts and a consistent fit."""
    import dataclasses

    # Build one shard: three rounds at d=5, X basis, p=0.005, epsilon=2e-3.
    true_rate = 2.0e-3

    def _shard_points(num_shots: int) -> list[object]:
        return [
            sweep.SweepPoint(
                backend="test",
                distance=5,
                basis="X",
                physical_error_rate=0.005,
                total_rounds=r,
                num_shots=num_shots,
                num_logical_errors=round(sweep.ler_over_rounds(true_rate, r) * num_shots),
                num_raw_errors=None,
                logical_error_rate=sweep.ler_over_rounds(true_rate, r),
                raw_error_rate=None,
            )
            for r in (10, 12, 14)
        ]

    def _write_shard(name: str, points: list[object], shots_per_point: int) -> Path:
        payload = {
            "config": {
                "distances": [5],
                "bases": ["X"],
                "error_rates": [0.005],
                "shots": shots_per_point,
                "executed_backends": ["test"],
                "duration_rounds_by_distance": {"5": [10, 12, 14]},
            },
            "points": [dataclasses.asdict(point) for point in points],
            "fit_summaries": [],
            "timing_summary": {
                "total_wall_clock_seconds": 10.0,
                "total_shots": shots_per_point * len(points),
                "total_points": len(points),
            },
        }
        path = tmp_path / name
        path.write_text(json.dumps(payload))
        return path

    shard_a = _write_shard("a_results.json", _shard_points(2000), 2000)
    shard_b = _write_shard("b_results.json", _shard_points(3000), 3000)

    points, summaries, config, timing = sweep.merge_sweep_shards([shard_a, shard_b])

    # Three merged points (one per total_rounds value), each with summed shots.
    assert len(points) == 3
    for point in points:
        assert point.num_shots == 5000  # 2000 + 3000 under the same key
    # Re-derived fit recovers the underlying epsilon within tolerance.
    assert len(summaries) == 1
    fitted = summaries[0].fitted_logical_error_rate_per_round
    assert fitted == pytest.approx(true_rate, rel=5e-3)
    # Merged timing totals should sum across shards.
    assert timing["total_wall_clock_seconds"] == pytest.approx(20.0)
    assert timing["total_shots"] == 5000 * 3  # three points, each 5000 merged shots
    # Config provenance records the shard paths and their individual shot counts.
    assert len(config["source_shards"]) == 2
    assert {entry["shots"] for entry in config["source_shards"]} == {2000, 3000}


def test_merge_heterogeneous_shards_no_keyerror(sweep: ModuleType, tmp_path: Path) -> None:
    """Merging shards with different distances/error_rates must not KeyError on grid holes."""
    import dataclasses

    true_rate = 2.0e-3

    def _make_shard_points(distance: int, p: float, num_shots: int) -> list[object]:
        return [
            sweep.SweepPoint(
                backend="test",
                distance=distance,
                basis="X",
                physical_error_rate=p,
                total_rounds=r,
                num_shots=num_shots,
                num_logical_errors=round(sweep.ler_over_rounds(true_rate, r) * num_shots),
                num_raw_errors=None,
                logical_error_rate=sweep.ler_over_rounds(true_rate, r),
                raw_error_rate=None,
            )
            for r in (10, 12, 14)
        ]

    def _write_shard(name: str, distance: int, p: float, num_shots: int) -> Path:
        points = _make_shard_points(distance, p, num_shots)
        payload = {
            "config": {
                "distances": [distance],
                "bases": ["X"],
                "error_rates": [p],
                "shots": num_shots,
                "executed_backends": ["test"],
                "duration_rounds_by_distance": {str(distance): [10, 12, 14]},
            },
            "points": [dataclasses.asdict(pt) for pt in points],
            "fit_summaries": [],
            "timing_summary": {
                "total_wall_clock_seconds": 5.0,
                "total_shots": num_shots * len(points),
                "total_points": len(points),
            },
        }
        path = tmp_path / name
        path.write_text(json.dumps(payload))
        return path

    # Shard A has d=3, p=0.005; shard B has d=5, p=0.006.
    # The merged grid has holes: (d=3, p=0.006) and (d=5, p=0.005) are absent.
    shard_a = _write_shard("a_results.json", distance=3, p=0.005, num_shots=1000)
    shard_b = _write_shard("b_results.json", distance=5, p=0.006, num_shots=1000)

    # This must not raise KeyError despite the sparse grid.
    points, summaries, config, _timing = sweep.merge_sweep_shards([shard_a, shard_b])

    assert len(points) == 6  # 3 rounds x 2 (d, p) combos
    assert len(summaries) == 2  # one fit per (d, basis, p)
    # Config should union distances and error_rates.
    assert sorted(config["distances"]) == [3, 5]
    assert sorted(config["error_rates"]) == [0.005, 0.006]

    # Verify that suppression_summary, print_basis_table, and build_plot_figure
    # all tolerate the sparse grid without raising KeyError.
    suppression = sweep._suppression_summary(summaries)
    # With one distance per error rate, no suppression rows are produced (need >= 2 distances).
    assert isinstance(suppression, list)

    # _print_basis_table should print without error (prints to stdout).
    sweep._print_basis_table(
        summaries,
        metric="fitted_logical_error_rate_per_round",
        title="Test heterogeneous table",
    )


def test_write_pdf_report_produces_multipage_pdf(sweep: ModuleType, tmp_path: Path) -> None:
    """``write_pdf_report`` should produce a non-trivial PDF (cover + plot pages)."""
    points = [
        sweep.SweepPoint(
            backend="test",
            distance=d,
            basis="X",
            physical_error_rate=0.005,
            total_rounds=r,
            num_shots=1000,
            num_logical_errors=int(0.02 * 1000),
            num_raw_errors=None,
            logical_error_rate=0.02,
            raw_error_rate=None,
        )
        for d in (3, 5)
        for r in (6, 8)
    ]
    summaries = [
        _make_summary(sweep, distance=d, p=0.005, epsilon_per_round=eps, basis="X")
        for d, eps in [(3, 4.0e-3), (5, 2.0e-3)]
    ]
    config = {
        "distances": [3, 5],
        "error_rates": [0.005],
        "shots": 1000,
        "duration_schedule_description": "test schedule",
        "duration_rounds_by_distance": {3: [6, 8], 5: [6, 8]},
    }
    timing = {"overall_shots_per_second": 1234.5, "total_shots": 4000, "total_wall_clock_seconds": 3.24}

    report_path = tmp_path / "myrun_report.pdf"
    written = sweep.write_pdf_report(
        report_path,
        points=points,
        summaries=summaries,
        timing_summary=timing,
        config=config,
    )
    assert written == report_path
    assert report_path.is_file()
    # Sanity check: a PDF with cover + at least one plot is much larger than an empty file.
    assert report_path.stat().st_size > 4000
    # Confirm it's a real PDF by header magic, not an empty/corrupt file.
    assert report_path.read_bytes()[:4] == b"%PDF"


def test_rebuild_plot_order_matches_primary_order(tmp_path: Path) -> None:
    """Companion report script must match the primary writer's plot ordering."""
    example_dir = _repo_root() / "examples" / "surface"
    sys.path.insert(0, str(example_dir))
    try:
        import surface_sweep_report as report_mod
    finally:
        sys.path.pop(0)

    # Fake SVG artifacts matching what the primary writer produces. The primary
    # writer emits, in order: per_round_overlay (combined), duration_overlay
    # (duration), then per (backend, basis): projected_d_rounds, then per_round.
    prefix = "surface_threshold_sweep"
    backend = "native_sampler"
    filenames = [
        f"{prefix}_{backend}_per_round_overlay.svg",
        f"{prefix}_{backend}_p_0p006_duration_overlay.svg",
        f"{prefix}_{backend}_x_projected_d_rounds.svg",
        f"{prefix}_{backend}_x_per_round.svg",
        f"{prefix}_{backend}_z_projected_d_rounds.svg",
        f"{prefix}_{backend}_z_per_round.svg",
    ]
    for name in filenames:
        (tmp_path / name).write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')

    plots = report_mod._discover_dashboard_plots(tmp_path, backends=[backend])
    discovered_names = [plot.filename for plot in plots]
    assert discovered_names == filenames
