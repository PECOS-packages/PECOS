"""Compare two surface-sweep JSON artifacts.

This helper reads the JSON files emitted by ``native_dem_threshold_sweep.py``
and prints Markdown tables with matched logical-error rates, binomial
intervals, ratios, differences, and descriptive normal-approximation z-scores.
It is useful for comparing CX vs SZZ/SZZdg surface-code runs that used the same
sweep grid.

Example:
    python examples/surface/compare_surface_sweep_json.py \\
        /tmp/pecos-szz-validation/native_cx_d357_r1_5k_results.json \\
        /tmp/pecos-szz-validation/native_szz_d357_r1_5k_results.json \\
        --left-label CX --right-label SZZ
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Z_95 = 1.959963984540054
CI_METHODS = {"jeffreys", "wilson"}


@dataclass(frozen=True)
class Point:
    backend: str
    basis: str
    distance: int
    p: float
    rounds: int
    errors: int
    shots: int

    @property
    def rate(self) -> float:
        return self.errors / self.shots if self.shots else math.nan


@dataclass(frozen=True)
class Comparison:
    key: tuple[str, str, int, float, int]
    left: Point
    right: Point


def _as_point(raw: dict[str, Any]) -> Point:
    return Point(
        backend=str(raw.get("backend", "")),
        basis=str(raw["basis"]).upper(),
        distance=int(raw["distance"]),
        p=float(raw["physical_error_rate"]),
        rounds=int(raw["total_rounds"]),
        errors=int(raw["num_logical_errors"]),
        shots=int(raw["num_shots"]),
    )


def load_points(path: Path) -> dict[tuple[str, str, int, float, int], Point]:
    data = json.loads(path.read_text())
    points = data.get("points")
    if not isinstance(points, list):
        msg = f"{path} does not contain a list-valued 'points' field"
        raise TypeError(msg)

    loaded: dict[tuple[str, str, int, float, int], Point] = {}
    for raw in points:
        point = _as_point(raw)
        key = (point.backend, point.basis, point.distance, point.p, point.rounds)
        if key in loaded:
            msg = f"{path} contains duplicate point key {key}"
            raise ValueError(msg)
        loaded[key] = point
    return loaded


def wilson_interval(errors: int, shots: int, z: float = Z_95) -> tuple[float, float]:
    if shots <= 0:
        return math.nan, math.nan
    phat = errors / shots
    denom = 1.0 + z * z / shots
    center = (phat + z * z / (2.0 * shots)) / denom
    half = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * shots)) / shots) / denom
    return max(0.0, center - half), min(1.0, center + half)


def jeffreys_interval(errors: int, shots: int, confidence: float = 0.95) -> tuple[float, float]:
    """Return a Jeffreys equal-tailed interval for one binomial proportion."""
    if shots <= 0:
        return math.nan, math.nan
    from scipy.stats import beta

    alpha = (1.0 - confidence) / 2.0
    lower = 0.0 if errors == 0 else float(beta.ppf(alpha, errors + 0.5, shots - errors + 0.5))
    upper = 1.0 if errors == shots else float(beta.ppf(1.0 - alpha, errors + 0.5, shots - errors + 0.5))
    return lower, upper


def binomial_interval(errors: int, shots: int, method: str) -> tuple[float, float]:
    if method == "jeffreys":
        return jeffreys_interval(errors, shots)
    if method == "wilson":
        return wilson_interval(errors, shots)
    msg = f"unknown interval method {method!r}"
    raise ValueError(msg)


def standard_error(errors: int, shots: int) -> float:
    if shots <= 0:
        return math.nan
    rate = errors / shots
    return math.sqrt(rate * (1.0 - rate) / shots)


def z_score(left: Point, right: Point) -> float:
    denom = math.sqrt(
        standard_error(left.errors, left.shots) ** 2 + standard_error(right.errors, right.shots) ** 2,
    )
    if denom == 0.0:
        return math.nan
    return (right.rate - left.rate) / denom


def ratio(left: Point, right: Point) -> float:
    if left.rate == 0.0:
        return math.inf if right.rate > 0.0 else 1.0
    return right.rate / left.rate


def format_rate(point: Point, *, include_ci: bool, interval_method: str) -> str:
    base = f"{point.rate:.4g} ({point.errors}/{point.shots})"
    if not include_ci:
        return base
    low, high = binomial_interval(point.errors, point.shots, interval_method)
    return f"{base} [{low:.4g}, {high:.4g}]"


def format_float(value: float, precision: int = 2) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf"
    return f"{value:.{precision}f}"


def matched_comparisons(
    left: dict[tuple[str, str, int, float, int], Point],
    right: dict[tuple[str, str, int, float, int], Point],
) -> list[Comparison]:
    return [Comparison(key, left[key], right[key]) for key in sorted(set(left) & set(right))]


def aggregate_points(points: list[Point]) -> Point:
    if not points:
        msg = "cannot aggregate an empty point list"
        raise ValueError(msg)
    first = points[0]
    return Point(
        backend=first.backend,
        basis=first.basis,
        distance=first.distance,
        p=math.nan,
        rounds=first.rounds,
        errors=sum(point.errors for point in points),
        shots=sum(point.shots for point in points),
    )


def emit_point_table(
    comparisons: list[Comparison],
    *,
    left_label: str,
    right_label: str,
    include_ci: bool,
    interval_method: str,
) -> str:
    lines = [
        "## Matched Points",
        "",
        f"| backend | basis | d | rounds | p | {left_label} | {right_label} | ratio | diff | z |",
        "|---------|-------|---|--------|---|------|------|-------|------|---|",
    ]
    for comparison in comparisons:
        backend, basis, distance, p, rounds = comparison.key
        left = comparison.left
        right = comparison.right
        lines.append(
            "| "
            f"{backend} | {basis} | {distance} | {rounds} | {p:g} | "
            f"{format_rate(left, include_ci=include_ci, interval_method=interval_method)} | "
            f"{format_rate(right, include_ci=include_ci, interval_method=interval_method)} | "
            f"{format_float(ratio(left, right))} | "
            f"{right.rate - left.rate:+.4g} | "
            f"{format_float(z_score(left, right))} |",
        )
    return "\n".join(lines)


def emit_aggregate_table(
    comparisons: list[Comparison],
    *,
    left_label: str,
    right_label: str,
    include_ci: bool,
    interval_method: str,
) -> str:
    grouped: dict[tuple[str, str, int, int], list[Comparison]] = defaultdict(list)
    for comparison in comparisons:
        backend, basis, distance, _p, rounds = comparison.key
        grouped[(backend, basis, distance, rounds)].append(comparison)

    lines = [
        "## Aggregate Over Physical Error Rates",
        "",
        f"| backend | basis | d | rounds | {left_label} | {right_label} | ratio | diff | z |",
        "|---------|-------|---|--------|------|------|-------|------|---|",
    ]
    for key in sorted(grouped):
        backend, basis, distance, rounds = key
        group = grouped[key]
        left = aggregate_points([comparison.left for comparison in group])
        right = aggregate_points([comparison.right for comparison in group])
        lines.append(
            "| "
            f"{backend} | {basis} | {distance} | {rounds} | "
            f"{format_rate(left, include_ci=include_ci, interval_method=interval_method)} | "
            f"{format_rate(right, include_ci=include_ci, interval_method=interval_method)} | "
            f"{format_float(ratio(left, right))} | "
            f"{right.rate - left.rate:+.4g} | "
            f"{format_float(z_score(left, right))} |",
        )
    return "\n".join(lines)


def emit_cross_distance_pooled_table(
    comparisons: list[Comparison],
    *,
    left_label: str,
    right_label: str,
    include_ci: bool,
    interval_method: str,
) -> str:
    grouped: dict[tuple[str, str], list[Comparison]] = defaultdict(list)
    for comparison in comparisons:
        backend, basis, _distance, _p, _rounds = comparison.key
        grouped[(backend, basis)].append(comparison)

    lines = [
        "## Pooled Across Distances By Backend And Basis",
        "",
        "This table intentionally pools across distances. It is useful as a rough",
        "event-count summary, but it is dominated by lower-distance points and is",
        "not a scaling or threshold statement.",
        "",
        f"| backend | basis | {left_label} | {right_label} | ratio | diff | z |",
        "|---------|-------|------|------|-------|------|---|",
    ]
    for key in sorted(grouped):
        backend, basis = key
        group = grouped[key]
        left = aggregate_points([comparison.left for comparison in group])
        right = aggregate_points([comparison.right for comparison in group])
        lines.append(
            "| "
            f"{backend} | {basis} | "
            f"{format_rate(left, include_ci=include_ci, interval_method=interval_method)} | "
            f"{format_rate(right, include_ci=include_ci, interval_method=interval_method)} | "
            f"{format_float(ratio(left, right))} | "
            f"{right.rate - left.rate:+.4g} | "
            f"{format_float(z_score(left, right))} |",
        )
    return "\n".join(lines)


def build_report(
    left_path: Path,
    right_path: Path,
    *,
    left_label: str,
    right_label: str,
    include_ci: bool,
    interval_method: str,
    include_cross_distance_pooled: bool,
) -> str:
    left = load_points(left_path)
    right = load_points(right_path)
    comparisons = matched_comparisons(left, right)
    if not comparisons:
        msg = "No common (backend, basis, distance, p, rounds) points found"
        raise ValueError(msg)

    left_only = len(set(left) - set(right))
    right_only = len(set(right) - set(left))
    lines = [
        f"# Sweep Comparison: {left_label} vs {right_label}",
        "",
        f"- left: `{left_path}`",
        f"- right: `{right_path}`",
        f"- matched points: {len(comparisons)}",
        f"- left-only points: {left_only}",
        f"- right-only points: {right_only}",
        f"- intervals: {'none' if not include_ci else f'{interval_method} 95%'}",
        "- z-scores: descriptive unpooled Wald z-scores, uncorrected for multiple comparisons",
        "- read low-count and zero-count rows cautiously",
        "- cross-distance pooled totals are omitted by default; use "
        "`--include-cross-distance-pooled` for a rough event-count summary",
        "",
        emit_point_table(
            comparisons,
            left_label=left_label,
            right_label=right_label,
            include_ci=include_ci,
            interval_method=interval_method,
        ),
        "",
        emit_aggregate_table(
            comparisons,
            left_label=left_label,
            right_label=right_label,
            include_ci=include_ci,
            interval_method=interval_method,
        ),
    ]
    if include_cross_distance_pooled:
        lines.extend(
            [
                "",
                emit_cross_distance_pooled_table(
                    comparisons,
                    left_label=left_label,
                    right_label=right_label,
                    include_ci=include_ci,
                    interval_method=interval_method,
                ),
            ],
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path, help="First native_dem_threshold_sweep.py JSON artifact.")
    parser.add_argument("right", type=Path, help="Second native_dem_threshold_sweep.py JSON artifact.")
    parser.add_argument("--left-label", default="left", help="Label for the first artifact.")
    parser.add_argument("--right-label", default="right", help="Label for the second artifact.")
    parser.add_argument(
        "--ci",
        choices=sorted(CI_METHODS),
        default="jeffreys",
        help="Binomial interval method to report when intervals are enabled.",
    )
    parser.add_argument("--no-ci", action="store_true", help="Omit 95% binomial intervals.")
    parser.add_argument(
        "--include-cross-distance-pooled",
        action="store_true",
        help=(
            "Also emit totals pooled across all distances by backend and basis. "
            "This is not a scaling or threshold summary."
        ),
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional Markdown output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(
        args.left,
        args.right,
        left_label=args.left_label,
        right_label=args.right_label,
        include_ci=not args.no_ci,
        interval_method=args.ci,
        include_cross_distance_pooled=args.include_cross_distance_pooled,
    )
    if args.output is None:
        print(report)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n")
        print(f"Wrote comparison report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
