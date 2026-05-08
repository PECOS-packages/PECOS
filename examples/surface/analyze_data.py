r"""Analyze decoder performance data from JSON shards.

Reads one or more JSON shards produced by ``generate_data.py`` and
computes:
  - Decoder comparison tables (LER + timing at each operating point)
  - Threshold curves (LER vs p for each decoder, grouped by distance)
  - Per-round fitting (when multiple duration multipliers are present)

Writes an analysis JSON that ``build_report.py`` consumes.

Example:
    uv run python examples/surface/analyze_data.py results/data.json
    uv run python examples/surface/analyze_data.py shard1.json shard2.json -o analysis/
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

# -- Analysis data model ------------------------------------------------------


@dataclass
class ComparisonRow:
    """One decoder's stats at one operating point."""

    decoder: str
    distance: int
    basis: str
    physical_error_rate: float
    num_rounds: int
    num_shots: int
    num_errors: int
    logical_error_rate: float
    ci_low: float
    ci_high: float
    per_shot_mean: float
    per_shot_median: float
    per_shot_p99: float
    per_shot_max: float
    quantiles: list[float] = field(default_factory=list)


@dataclass
class ComparisonTable:
    """All decoder rows for one (distance, p, rounds) point, sorted by LER."""

    distance: int
    basis: str
    physical_error_rate: float
    num_rounds: int
    num_shots: int
    rows: list[ComparisonRow]


@dataclass
class ThresholdCurvePoint:
    """One (p, LER) point on a threshold curve."""

    physical_error_rate: float
    # LER over d rounds (either projected from fit, or raw at r=2d)
    logical_error_rate: float
    ci_low: float
    ci_high: float
    num_shots: int
    num_errors: int
    # Per-round fitted epsilon (when multiple round counts available)
    fitted_epsilon: float | None = None
    fitted_epsilon_ci_low: float | None = None
    fitted_epsilon_ci_high: float | None = None
    num_round_points: int = 1
    # Per-round LER (= fitted_epsilon when fitted, else raw_LER / num_rounds)
    per_round_ler: float | None = None
    per_round_ci_low: float | None = None
    per_round_ci_high: float | None = None


@dataclass
class ThresholdCurve:
    """LER vs p for one (decoder, distance, basis) combination.

    When multiple round counts are available, the threshold points use
    fitted per-round epsilon from p_L(r) = 0.5*(1-(1-2*eps)^r).
    """

    decoder: str
    distance: int
    basis: str
    points: list[ThresholdCurvePoint]
    uses_fitted_epsilon: bool = False


@dataclass
class DurationCurvePoint:
    """One (rounds, LER) point on a duration curve."""

    num_rounds: int
    logical_error_rate: float
    ci_low: float
    ci_high: float
    num_shots: int
    num_errors: int


@dataclass
class DurationCurve:
    """LER vs rounds for one (decoder, distance, basis, p) combination."""

    decoder: str
    distance: int
    basis: str
    physical_error_rate: float
    points: list[DurationCurvePoint]


@dataclass
class ThresholdEstimate:
    """Estimated threshold from curve crossing for one decoder.

    metric: "per_round" or "d_round" indicating which LER was used for the crossing.
    """

    decoder: str
    basis: str
    estimated_p_th: float
    d_small: int
    d_large: int
    metric: str = "per_round"  # "per_round", "d_round", or "fss"
    std_error: float | None = None


@dataclass
class AnalysisResult:
    """Full analysis output."""

    config: dict
    comparison_tables: list[ComparisonTable] = field(default_factory=list)
    threshold_curves: list[ThresholdCurve] = field(default_factory=list)
    duration_curves: list[DurationCurve] = field(default_factory=list)
    threshold_estimates: list[ThresholdEstimate] = field(default_factory=list)


# -- Wilson score interval ----------------------------------------------------


def _wilson_ci(n: int, k: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score confidence interval for binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    if p == 0:
        return 0.0, 1 - (1 - 0.95) ** (1 / n)  # Clopper-Pearson upper for 0 successes
    if p == 1:
        return (1 - 0.95) ** (1 / n), 1.0
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


# -- Per-round epsilon fitting ------------------------------------------------


def _ler_model(epsilon: float, r: int) -> float:
    """p_L(r) = 0.5 * (1 - (1 - 2*epsilon)^r)."""
    if epsilon <= 0 or epsilon >= 0.5:
        return 0.5
    return 0.5 * (1.0 - (1.0 - 2.0 * epsilon) ** r)


def _fit_epsilon(round_values: list[int], ler_values: list[float]) -> float | None:
    """Fit per-round epsilon from (rounds, LER) pairs using golden section search.

    Minimises sum of squared residuals of p_L(r) = 0.5*(1-(1-2*eps)^r).
    Returns None if fitting is not possible.
    """
    if not round_values or all(v == 0 for v in ler_values):
        return None

    def cost(eps: float) -> float:
        return sum((ler - _ler_model(eps, r)) ** 2 for r, ler in zip(round_values, ler_values, strict=True))

    # Golden section search on [1e-8, 0.499]
    a, b = 1e-8, 0.499
    gr = (math.sqrt(5) + 1) / 2
    for _ in range(80):
        c = b - (b - a) / gr
        d = a + (b - a) / gr
        if cost(c) < cost(d):
            b = d
        else:
            a = c
    return (a + b) / 2


# -- Load and merge shards ----------------------------------------------------


def _load_shards(paths: list[Path]) -> tuple[dict, list[dict]]:
    """Load JSON shards and return (merged_config, all_points)."""
    all_points = []
    config = {}
    for path in paths:
        data = json.loads(path.read_text())
        if not config:
            config = dict(data.get("config", {}))
        all_points.extend(data.get("points", []))

    # Derive config fields from actual data (shards may cover different subsets)
    all_decoders = set()
    all_distances = set()
    all_error_rates = set()
    total_shots = 0
    for pt in all_points:
        all_distances.add(pt["distance"])
        all_error_rates.add(pt["physical_error_rate"])
        total_shots = max(total_shots, pt.get("num_shots", 0))
        for ds in pt.get("decoder_stats", []):
            all_decoders.add(ds["decoder"])
    config["decoders"] = sorted(all_decoders)
    config["distances"] = sorted(all_distances)
    config["error_rates"] = sorted(all_error_rates)
    if total_shots:
        config["shots"] = total_shots
    return config, all_points


# -- Analysis -----------------------------------------------------------------


def analyze(config: dict, points: list[dict]) -> AnalysisResult:
    """Compute comparison tables and threshold curves from raw data points."""
    result = AnalysisResult(config=config)

    # Group points by (distance, basis, p, rounds)
    from collections import defaultdict

    by_cell: dict[tuple, list[dict]] = defaultdict(list)
    for pt in points:
        key = (pt["distance"], pt["basis"], pt["physical_error_rate"], pt["num_rounds"])
        by_cell[key].append(pt)

    # Comparison tables: one per (distance, p, rounds) cell
    for (d, basis, p, r), cell_points in sorted(by_cell.items()):
        # Merge decoder stats across shards for same cell
        decoder_stats: dict[str, dict] = {}
        total_shots = 0
        for pt in cell_points:
            total_shots += pt["num_shots"]
            for ds in pt.get("decoder_stats", []):
                name = ds["decoder"]
                if name not in decoder_stats:
                    decoder_stats[name] = {
                        "num_errors": 0,
                        "num_shots": 0,
                        "per_shot_mean": ds["per_shot_mean"],
                        "per_shot_median": ds["per_shot_median"],
                        "per_shot_p99": ds["per_shot_p99"],
                        "per_shot_max": ds["per_shot_max"],
                        "quantiles": ds.get("quantiles", []),
                    }
                decoder_stats[name]["num_errors"] += ds["num_errors"]
                decoder_stats[name]["num_shots"] += pt["num_shots"]

        rows = []
        for dec_name, stats in decoder_stats.items():
            n = stats["num_shots"]
            k = stats["num_errors"]
            ler = k / n if n > 0 else 0.0
            ci_lo, ci_hi = _wilson_ci(n, k)
            rows.append(
                ComparisonRow(
                    decoder=dec_name,
                    distance=d,
                    basis=basis,
                    physical_error_rate=p,
                    num_rounds=r,
                    num_shots=n,
                    num_errors=k,
                    logical_error_rate=ler,
                    ci_low=ci_lo,
                    ci_high=ci_hi,
                    per_shot_mean=stats["per_shot_mean"],
                    per_shot_median=stats["per_shot_median"],
                    per_shot_p99=stats["per_shot_p99"],
                    per_shot_max=stats["per_shot_max"],
                    quantiles=stats.get("quantiles", []),
                ),
            )

        rows.sort(key=lambda r: r.logical_error_rate)

        result.comparison_tables.append(
            ComparisonTable(
                distance=d,
                basis=basis,
                physical_error_rate=p,
                num_rounds=r,
                num_shots=total_shots,
                rows=rows,
            ),
        )

    # Threshold curves: group by (decoder, distance, basis) across all rounds.
    # For each p, if multiple round counts exist, fit per-round epsilon.
    # Otherwise, use the raw LER at the single available round count.
    tc_groups: dict[tuple, dict[float, list[ComparisonRow]]] = defaultdict(lambda: defaultdict(list))
    for table in result.comparison_tables:
        for row in table.rows:
            key = (row.decoder, row.distance, row.basis)
            tc_groups[key][row.physical_error_rate].append(row)

    for (dec, d, basis), p_to_rows in sorted(tc_groups.items()):
        curve_points = []
        has_fitting = False
        for p in sorted(p_to_rows):
            rows = p_to_rows[p]
            if len(rows) >= 2:
                # Multiple round counts: fit per-round epsilon
                round_vals = [r.num_rounds for r in rows]
                ler_vals = [r.logical_error_rate for r in rows]
                eps = _fit_epsilon(round_vals, ler_vals)
                if eps is not None:
                    has_fitting = True
                    # Project to d-round LER for display
                    projected_ler = _ler_model(eps, d)
                    # CI from fitting upper/lower LER bounds
                    ci_lo_vals = [r.ci_low for r in rows]
                    ci_hi_vals = [r.ci_high for r in rows]
                    eps_lo = _fit_epsilon(round_vals, ci_hi_vals)  # higher LER -> higher eps
                    eps_hi = _fit_epsilon(round_vals, ci_lo_vals)  # lower LER -> lower eps
                    total_shots = sum(r.num_shots for r in rows)
                    total_errors = sum(r.num_errors for r in rows)
                    proj_ci_lo = _ler_model(eps_hi, d) if eps_hi else projected_ler
                    proj_ci_hi = _ler_model(eps_lo, d) if eps_lo else projected_ler
                    curve_points.append(
                        ThresholdCurvePoint(
                            physical_error_rate=p,
                            logical_error_rate=projected_ler,
                            ci_low=proj_ci_lo,
                            ci_high=proj_ci_hi,
                            num_shots=total_shots,
                            num_errors=total_errors,
                            fitted_epsilon=eps,
                            fitted_epsilon_ci_low=eps_hi,
                            fitted_epsilon_ci_high=eps_lo,
                            num_round_points=len(rows),
                            per_round_ler=eps,
                            per_round_ci_low=eps_hi,
                            per_round_ci_high=eps_lo,
                        ),
                    )
                    continue

            # Single round count: use raw LER, estimate per-round from
            # p_L(r) = 0.5*(1-(1-2*eps)^r) inverted.
            row = rows[0]
            raw_per_round = _fit_epsilon([row.num_rounds], [row.logical_error_rate])
            raw_pr_lo = _fit_epsilon([row.num_rounds], [row.ci_high])
            raw_pr_hi = _fit_epsilon([row.num_rounds], [row.ci_low])
            curve_points.append(
                ThresholdCurvePoint(
                    physical_error_rate=row.physical_error_rate,
                    logical_error_rate=row.logical_error_rate,
                    ci_low=row.ci_low,
                    ci_high=row.ci_high,
                    num_shots=row.num_shots,
                    num_errors=row.num_errors,
                    per_round_ler=raw_per_round,
                    per_round_ci_low=raw_pr_hi,
                    per_round_ci_high=raw_pr_lo,
                ),
            )

        result.threshold_curves.append(
            ThresholdCurve(
                decoder=dec,
                distance=d,
                basis=basis,
                points=curve_points,
                uses_fitted_epsilon=has_fitting,
            ),
        )

    # Duration curves: LER vs rounds for each (decoder, distance, basis, p).
    # Only meaningful when multiple round counts exist per (d, p).
    dur_groups: dict[tuple, list[ComparisonRow]] = defaultdict(list)
    for table in result.comparison_tables:
        for row in table.rows:
            key = (row.decoder, row.distance, row.basis, row.physical_error_rate)
            dur_groups[key].append(row)

    for (dec, d, basis, p), rows in sorted(dur_groups.items()):
        if len(rows) < 2:
            continue  # need at least 2 round counts
        dur_points = [
            DurationCurvePoint(
                num_rounds=row.num_rounds,
                logical_error_rate=row.logical_error_rate,
                ci_low=row.ci_low,
                ci_high=row.ci_high,
                num_shots=row.num_shots,
                num_errors=row.num_errors,
            )
            for row in sorted(rows, key=lambda r: r.num_rounds)
        ]
        result.duration_curves.append(
            DurationCurve(
                decoder=dec,
                distance=d,
                basis=basis,
                physical_error_rate=p,
                points=dur_points,
            ),
        )

    # Threshold estimates: find where smallest/largest distance curves cross.
    # Compute for both per-round LER and d-round LER.
    import itertools as _itertools

    # Threshold estimates via FSS fit (Wang-Harrington-Preskill form).
    # Uses ALL (p, d, per_round_ler) points across ALL distances simultaneously.
    # Falls back to pairwise crossing only as a seed for the FSS fit.
    est_groups: dict[tuple, list[ThresholdCurve]] = defaultdict(list)
    for curve in result.threshold_curves:
        est_groups[(curve.decoder, curve.basis)].append(curve)

    def _crude_crossing_seed(dec_curves: list[ThresholdCurve]) -> float | None:
        """Quick pairwise crossing of smallest/largest distance for FSS seed."""
        distances = sorted({c.distance for c in dec_curves})
        if len(distances) < 2:
            return None
        small = next(c for c in dec_curves if c.distance == distances[0])
        large = next(c for c in dec_curves if c.distance == distances[-1])
        small_by_p = {
            pt.physical_error_rate: (pt.per_round_ler or pt.logical_error_rate)
            for pt in small.points
            if (pt.per_round_ler or pt.logical_error_rate) and (pt.per_round_ler or pt.logical_error_rate) > 0
        }
        large_by_p = {
            pt.physical_error_rate: (pt.per_round_ler or pt.logical_error_rate)
            for pt in large.points
            if (pt.per_round_ler or pt.logical_error_rate) and (pt.per_round_ler or pt.logical_error_rate) > 0
        }
        shared_ps = sorted(set(small_by_p) & set(large_by_p))
        diffs = [(p, large_by_p[p] - small_by_p[p]) for p in shared_ps]
        for (p0, diff0), (p1, diff1) in _itertools.pairwise(diffs):
            if diff0 == 0.0:
                return p0
            if diff0 * diff1 < 0.0:
                t = abs(diff0) / (abs(diff0) + abs(diff1))
                return math.exp((1.0 - t) * math.log(p0) + t * math.log(p1))
        return None

    for (dec, basis), dec_curves in sorted(est_groups.items()):
        distances = sorted({c.distance for c in dec_curves})
        if len(distances) < 2:
            continue

        # FSS fit for per-round LER
        plist_pr, dlist_pr, plog_pr = [], [], []
        for curve in dec_curves:
            for pt in curve.points:
                pr = pt.per_round_ler
                if pr and pr > 0:
                    plist_pr.append(pt.physical_error_rate)
                    dlist_pr.append(curve.distance)
                    plog_pr.append(pr)

        # FSS fit for d-round LER
        plist_dr, dlist_dr, plog_dr = [], [], []
        for curve in dec_curves:
            for pt in curve.points:
                lr = pt.logical_error_rate
                if lr and lr > 0:
                    plist_dr.append(pt.physical_error_rate)
                    dlist_dr.append(curve.distance)
                    plog_dr.append(lr)

        # Crude seed for FSS optimizer
        seed = _crude_crossing_seed(dec_curves)

        for metric, plist, dlist, plog in [
            ("fss_per_round", plist_pr, dlist_pr, plog_pr),
            ("fss_d_round", plist_dr, dlist_dr, plog_dr),
        ]:
            if len(plist) < 5 or len(set(dlist)) < 2:
                continue
            s = seed if seed is not None else sorted(plist)[len(plist) // 2]
            fss = _fit_fss_threshold(plist, dlist, plog, seed_threshold=s)
            if fss is not None:
                result.threshold_estimates.append(
                    ThresholdEstimate(
                        decoder=dec,
                        basis=basis,
                        estimated_p_th=fss[0],
                        d_small=min(distances),
                        d_large=max(distances),
                        metric=metric,
                        std_error=fss[1],
                    ),
                )

    return result


def _fit_fss_threshold(
    plist: list[float],
    dlist: list[int],
    plog: list[float],
    *,
    seed_threshold: float,
    seed_nu: float = 1.0,
    window_factor_low: float = 0.4,
    window_factor_high: float = 2.0,
) -> tuple[float, float] | None:
    """Fit the Wang-Harrington-Preskill FSS form using pecos.analysis.threshold_curve.

    Returns (p_th, p_th_std_error) or None if the fit fails.
    """
    try:
        from pecos.analysis.threshold_curve import func as fss_func
        from pecos.analysis.threshold_curve import threshold_fit
    except ImportError:
        return None

    # Filter to a window around the seed threshold
    low = seed_threshold * window_factor_low
    high = seed_threshold * window_factor_high
    indices = [i for i, p in enumerate(plist) if low <= p <= high and plog[i] > 0]
    if len(indices) < 5 or len({dlist[i] for i in indices}) < 2:
        return None

    import pecos as pc

    p_arr = pc.array([plist[i] for i in indices])
    d_arr = pc.array([float(dlist[i]) for i in indices])
    ler_arr = pc.array([plog[i] for i in indices])
    mean_ler = float(sum(plog[i] for i in indices) / len(indices))
    initial = [seed_threshold, seed_nu, mean_ler, 1.0, 1.0]

    try:
        popt, stdev = threshold_fit(p_arr, d_arr, ler_arr, fss_func, initial)
    except Exception:
        return None

    p_th = float(popt[0])
    p_th_se = float(stdev[0])
    if p_th <= 0:
        return None
    return (p_th, p_th_se)


# -- CLI ----------------------------------------------------------------------


def main() -> int:
    """CLI entry point for data analysis."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("shards", nargs="+", type=Path, help="JSON shard file(s) from generate_data.py")
    parser.add_argument("-o", "--output-dir", type=str, default=None)
    args = parser.parse_args()

    config, points = _load_shards(args.shards)
    print(f"Loaded {len(points)} data points from {len(args.shards)} shard(s)")

    result = analyze(config, points)
    print(f"  {len(result.comparison_tables)} comparison tables")
    print(f"  {len(result.threshold_curves)} threshold curves")

    out = Path(args.output_dir) if args.output_dir else args.shards[0].parent
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "analysis.json"
    json_path.write_text(json.dumps(asdict(result), indent=2))
    print(f"Wrote {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
