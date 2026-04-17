#!/usr/bin/env python3
"""Build an HTML dashboard from an existing surface sweep output directory.

By default this only re-skins the dashboard from the SVG/JSON artifacts already
present in the directory -- useful when you want to tweak the dashboard without
rerunning the simulation.

With ``--render-plots`` it instead re-renders every plot from the canonical
JSON results file (``*_results.json``), then writes the dashboard. This lets us
revisit data later: the JSON is the source of truth, so plots can be
regenerated with new styling, new formats, or after the SVGs were deleted.

Examples:
    # Just rebuild the dashboard from already-present SVGs + JSON:
    .venv/bin/python examples/surface/surface_sweep_report.py \
        --input-dir /tmp/pecos_surface_real_sweep --open

    # Re-render plots from JSON, then rebuild the dashboard:
    .venv/bin/python examples/surface/surface_sweep_report.py \
        --input-dir /tmp/pecos_surface_real_sweep --render-plots --open

    # Re-render in both SVG and PDF:
    .venv/bin/python examples/surface/surface_sweep_report.py \
        --input-dir /tmp/pecos_surface_real_sweep --render-plots --formats svg pdf
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from native_dem_threshold_sweep import (
    DashboardPlot,
    FitSummary,
    _maybe_open_html_dashboard,
    _write_html_dashboard,
    merge_sweep_shards,
    render_plot_artifacts,
    write_pdf_report,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing existing surface sweep SVG and optional JSON artifacts.",
    )
    parser.add_argument(
        "--json-file",
        "--json-files",
        dest="json_files",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "Sweep JSON results file(s). When multiple files are given, they are "
            "merged by SweepPoint key -- shot counts accumulate and fit summaries "
            "are re-derived from the merged points. Omit to auto-discover a single "
            "``*_results.json`` inside ``--input-dir``."
        ),
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=None,
        help="Optional explicit path for the generated dashboard HTML.",
    )
    parser.add_argument(
        "--render-plots",
        action="store_true",
        help=(
            "Re-render every plot from the canonical JSON results file, "
            "overwriting any existing plot files in the input directory. "
            "Requires that a ``*_results.json`` exists (or is given via --json-file)."
        ),
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["svg"],
        choices=["svg", "pdf"],
        help="Plot file formats to write when --render-plots is set. Default: svg.",
    )
    parser.add_argument(
        "--report-pdf",
        action="store_true",
        help=(
            "Also write a multi-page PDF report (cover + every plot) alongside the HTML "
            "dashboard. Requires that JSON results are loadable (sweep was run with --save-json)."
        ),
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the generated dashboard after writing it.",
    )
    return parser.parse_args()


def _resolve_json_paths(input_dir: Path, json_files: list[Path] | None) -> list[Path]:
    """Resolve CLI JSON paths, falling back to one shard in ``input_dir`` when unset."""
    if json_files:
        resolved = [path.expanduser().resolve() for path in json_files]
        for path in resolved:
            if not path.is_file():
                msg = f"JSON results file does not exist: {path}"
                raise ValueError(msg)
        return resolved

    candidates = sorted(input_dir.glob("*_results.json"))
    if not candidates:
        return []
    return [candidates[0]]


def _load_json_payload(json_path: Path | None) -> dict[str, Any] | None:
    """Read a single shard's raw JSON payload (for dashboard meta display only)."""
    if json_path is None:
        return None
    return json.loads(json_path.read_text())


def _infer_output_html(input_dir: Path, json_path: Path | None, explicit_output: Path | None) -> Path:
    if explicit_output is not None:
        return explicit_output
    if json_path is not None and json_path.name.endswith("_results.json"):
        return input_dir / json_path.name.replace("_results.json", "_dashboard.html")
    return input_dir / "surface_sweep_dashboard.html"


def _extract_backend(prefix_text: str, backends: list[str]) -> str:
    for backend in sorted(backends, key=len, reverse=True):
        if prefix_text.endswith(f"_{backend}"):
            return backend
    return prefix_text


def _maybe_parse_rate(rate_text: str) -> float | None:
    try:
        return float(rate_text.replace("p", ".", 1))
    except ValueError:
        return None


# Matches the order in which the primary sweep script's _write_artifacts appends
# plots: combined -> duration -> basis (projected_d_rounds before per_round).
_SECTION_ORDER = {"combined": 0, "duration": 1, "basis": 2}
_BASIS_METRIC_ORDER = {"_projected_d_rounds": 0, "_per_round": 1}


def _plot_sort_key(plot: DashboardPlot) -> tuple:
    section = _SECTION_ORDER.get(plot.section, 99)
    backend = plot.backend
    if plot.section == "combined":
        return (section, backend)
    if plot.section == "duration":
        return (section, backend, plot.physical_error_rate if plot.physical_error_rate is not None else 0.0)
    if plot.section == "basis":
        # Read metric from filename suffix to preserve projected-before-per-round ordering.
        stem = Path(plot.filename).stem
        metric_rank = next(
            (rank for suffix, rank in _BASIS_METRIC_ORDER.items() if stem.endswith(suffix)),
            99,
        )
        return (section, backend, plot.basis or "", metric_rank)
    return (section, backend)


def _discover_dashboard_plots(input_dir: Path, *, backends: list[str]) -> list[DashboardPlot]:
    plots: list[DashboardPlot] = []
    for svg_path in sorted(input_dir.glob("*.svg")):
        stem = svg_path.stem

        if stem.endswith("_per_round_overlay"):
            prefix_text = stem[: -len("_per_round_overlay")]
            backend = _extract_backend(prefix_text, backends)
            plots.append(
                DashboardPlot(
                    section="combined",
                    title=f"Per-round logical error rate vs p ({backend})",
                    filename=svg_path.name,
                    backend=backend,
                ),
            )
            continue

        duration_match = re.match(r"^(?P<prefix>.+)_p_(?P<rate>.+)_duration_overlay$", stem)
        if duration_match is not None:
            prefix_text = duration_match.group("prefix")
            backend = _extract_backend(prefix_text, backends)
            rate = _maybe_parse_rate(duration_match.group("rate"))
            rate_label = duration_match.group("rate").replace("p", ".", 1)
            plots.append(
                DashboardPlot(
                    section="duration",
                    title=f"Logical memory error vs duration ({backend}, p={rate_label})",
                    filename=svg_path.name,
                    backend=backend,
                    physical_error_rate=rate,
                ),
            )
            continue

        basis_match = re.match(r"^(?P<prefix>.+)_(?P<basis>x|z)_(?P<metric>per_round|projected_d_rounds)$", stem)
        if basis_match is not None:
            prefix_text = basis_match.group("prefix")
            backend = _extract_backend(prefix_text, backends)
            basis = basis_match.group("basis").upper()
            metric = basis_match.group("metric")
            title = (
                f"{basis}-Basis Fitted Logical Error Rate Per Round ({backend})"
                if metric == "per_round"
                else f"{basis}-Basis Fitted Logical Error Rate Over d Rounds ({backend})"
            )
            plots.append(
                DashboardPlot(
                    section="basis",
                    title=title,
                    filename=svg_path.name,
                    backend=backend,
                    basis=basis,
                ),
            )

    plots.sort(key=_plot_sort_key)
    return plots


def _dashboard_args(payload: dict[str, Any] | None) -> argparse.Namespace:
    config = {} if payload is None else dict(payload.get("config", {}))
    return argparse.Namespace(
        distances=config.get("distances", []),
        duration_multipliers=config.get("duration_multipliers", []),
        duration_schedule_description=config.get("duration_schedule_description"),
        duration_rounds_by_distance={
            int(distance): tuple(values) for distance, values in config.get("duration_rounds_by_distance", {}).items()
        },
        error_rates=config.get("error_rates", []),
        shots=config.get("shots", "?"),
    )


def _dashboard_summaries(payload: dict[str, Any] | None) -> list[FitSummary]:
    if payload is None:
        return []
    return [FitSummary(**row) for row in payload.get("fit_summaries", [])]


def _dashboard_timing_summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {"overall_shots_per_second": 0.0}
    return dict(payload.get("timing_summary", {"overall_shots_per_second": 0.0}))


def _prefix_from_json_path(json_path: Path) -> str:
    """Recover the sweep ``output_prefix`` from the JSON results filename."""
    stem = json_path.name
    if stem.endswith("_results.json"):
        return stem[: -len("_results.json")]
    return json_path.stem


def _merged_config_as_dashboard_args(config: dict[str, Any]) -> argparse.Namespace:
    """Adapt a merged config dict to the ``argparse.Namespace`` the HTML writer expects."""
    return argparse.Namespace(
        distances=config.get("distances", []),
        duration_multipliers=config.get("duration_multipliers", []),
        duration_schedule_description=config.get("duration_schedule_description"),
        duration_rounds_by_distance={
            int(distance): tuple(values) for distance, values in config.get("duration_rounds_by_distance", {}).items()
        },
        error_rates=config.get("error_rates", []),
        shots=config.get("shots", "?"),
    )


def main() -> int:
    """Build a dashboard (and optional PDF report) from one or more sweep shards."""
    args = _parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        msg = f"Input directory does not exist: {input_dir}"
        raise ValueError(msg)

    json_paths = _resolve_json_paths(input_dir, args.json_files)
    primary_json_path = json_paths[0] if json_paths else None
    output_html = _infer_output_html(input_dir, primary_json_path, args.output_html)

    # Merge (or single-shard load) when JSON is available so downstream code
    # always sees the same shape of merged data regardless of shard count.
    merged_points: list[Any] = []
    merged_summaries: list[FitSummary] = []
    merged_config: dict[str, Any] = {}
    merged_timing: dict[str, Any] = {}
    if json_paths:
        merged_points, merged_summaries, merged_config, merged_timing = merge_sweep_shards(json_paths)
        if len(json_paths) > 1:
            print(f"Merged {len(json_paths)} shards: {[str(p) for p in json_paths]}")

    if args.render_plots or len(json_paths) > 1:
        # Multi-shard merges always re-render; single-shard --render-plots
        # also regenerates plot files before the dashboard assembles them.
        if not json_paths:
            msg = f"--render-plots requires a JSON results file in {input_dir} (or --json-file)"
            raise ValueError(msg)
        plots = render_plot_artifacts(
            input_dir,
            prefix=_prefix_from_json_path(primary_json_path),
            points=merged_points,
            summaries=merged_summaries,
            formats=args.formats,
        )
        if not plots:
            msg = "Plot rendering produced no SVG entries; ensure 'svg' is in --formats."
            raise ValueError(msg)
    else:
        backends = list(merged_config.get("executed_backends", []))
        plots = _discover_dashboard_plots(input_dir, backends=backends)
        if not plots:
            msg = f"No SVG plots found in {input_dir}; pass --render-plots to regenerate from JSON."
            raise ValueError(msg)

    # Dashboard meta uses merged data so the HTML card counts reflect the
    # combined run. When only one shard is loaded, this is identical to the
    # old single-shard payload path.
    raw_payload_for_meta = _load_json_payload(primary_json_path) if len(json_paths) == 1 else None
    _write_html_dashboard(
        output_html,
        args=_merged_config_as_dashboard_args(merged_config) if json_paths else _dashboard_args(None),
        summaries=(merged_summaries if json_paths else _dashboard_summaries(raw_payload_for_meta)),
        timing_summary=(merged_timing if json_paths else _dashboard_timing_summary(raw_payload_for_meta)),
        plots=plots,
        json_filename=primary_json_path.name if primary_json_path is not None else None,
    )
    print(f"Wrote HTML dashboard to {output_html}")

    if args.report_pdf:
        if not json_paths:
            msg = f"--report-pdf requires a JSON results file in {input_dir} (or --json-file)"
            raise ValueError(msg)
        report_pdf_path = output_html.with_name(output_html.name.replace("_dashboard.html", "_report.pdf"))
        if report_pdf_path == output_html:
            report_pdf_path = output_html.with_suffix(".pdf")
        write_pdf_report(
            report_pdf_path,
            points=merged_points,
            summaries=merged_summaries,
            timing_summary=merged_timing,
            config=merged_config,
        )
        print(f"Wrote PDF report to {report_pdf_path}")

    if args.open:
        _maybe_open_html_dashboard(output_html)
        print(f"Opened HTML dashboard at {output_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
