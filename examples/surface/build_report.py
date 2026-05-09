r"""Build HTML and/or PDF reports from analysis JSON.

Reads the analysis JSON produced by ``analyze_data.py`` and renders:
  - HTML report with comparison tables + threshold curve plots (inline SVG)
  - PDF report with the same content via matplotlib

Example:
    uv run python examples/surface/build_report.py results/analysis.json --html --pdf
    uv run python examples/surface/build_report.py results/analysis.json --html --open
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
import math
from collections import defaultdict
from pathlib import Path
from textwrap import dedent

# -- Load analysis ------------------------------------------------------------


def _load_analysis(path: Path) -> dict:
    return json.loads(path.read_text())


# -- Threshold curve SVG (inline) ---------------------------------------------


def _sci(v: float) -> str:
    """Format a value in clean scientific notation for axis labels."""
    if v == 0:
        return "0"
    return f"{v:.1e}"


_COLORS = [
    "#2563eb",  # blue
    "#dc2626",  # red
    "#16a34a",  # green
    "#9333ea",  # purple
    "#ea580c",  # orange
    "#0891b2",  # cyan
    "#be185d",  # pink
    "#854d0e",  # brown
]

_DASH_PATTERNS = [
    "",  # solid
    "6,3",  # dashed
    "2,3",  # dotted
    "8,3,2,3",  # dash-dot
]


def _build_threshold_svg(
    curves: list[dict],
    title: str,
    width: int = 560,
    height: int = 380,
    color_by: str = "decoder",
    threshold_p: float | None = None,
    y_field: str = "logical_error_rate",
    y_label: str = "Logical error rate",
    decoder_color_map: dict[str, str] | None = None,
    distance_color_map: dict[int, str] | None = None,
) -> str:
    """Build an inline SVG plot of LER vs p for multiple (decoder, distance) curves.

    decoder_color_map / distance_color_map: fixed color assignments for
    consistency across plots. When provided, overrides index-based coloring.
    y_field: which field on each point to use for y-axis values.
             Also uses "{y_field}" with "ci_low"/"ci_high" replaced accordingly.
    threshold_p: if provided, draw a vertical dashed line at this p value.
    color_by: "decoder" assigns colors by decoder (dashes by distance),
              "distance" assigns colors by distance (dashes by decoder).
    """
    margin = {"top": 45, "right": 160, "bottom": 55, "left": 70}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]

    # Resolve CI field names based on y_field
    if y_field == "per_round_ler":
        ci_lo_field, ci_hi_field = "per_round_ci_low", "per_round_ci_high"
    else:
        ci_lo_field, ci_hi_field = "ci_low", "ci_high"

    def _y(pt: dict) -> float:
        v = pt.get(y_field)
        return v if v is not None and v > 0 else 0.0

    # Collect all p and y values for axis scaling
    all_p = []
    all_ler = []
    for curve in curves:
        for pt in curve["points"]:
            all_p.append(pt["physical_error_rate"])
            yv = _y(pt)
            if yv > 0:
                all_ler.append(yv)

    if not all_p or not all_ler:
        return f'<svg width="{width}" height="{height}"><text x="50%" y="50%">No data</text></svg>'

    p_vals_pos = [p for p in all_p if p > 0]
    p_min = min(p_vals_pos) if p_vals_pos else 1e-4
    p_max = max(p_vals_pos) if p_vals_pos else 1.0
    ler_min = max(1e-5, min(all_ler) * 0.5)
    ler_max = min(1.0, max(all_ler) * 2.0)

    # Log scale for both axes
    log_p_min = math.log10(p_min * 0.8)
    log_p_max = math.log10(p_max * 1.2)
    log_ler_min = math.log10(ler_min)
    log_ler_max = math.log10(ler_max)

    def x_of(p: float) -> float:
        if p <= 0:
            return margin["left"]
        log_p = math.log10(p)
        frac = (log_p - log_p_min) / (log_p_max - log_p_min) if log_p_max != log_p_min else 0.5
        return margin["left"] + frac * plot_w

    def y_of(ler: float) -> float:
        if ler <= 0:
            return margin["top"] + plot_h
        log_val = math.log10(ler)
        frac = (log_val - log_ler_min) / (log_ler_max - log_ler_min) if log_ler_max != log_ler_min else 0.5
        return margin["top"] + (1 - frac) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'style="font-family:ui-sans-serif,sans-serif;font-size:11px;">',
        # Background
        f'<rect width="{width}" height="{height}" fill="var(--card-bg, white)" rx="8"/>',
        # Title
        f'<text x="{width // 2}" y="24" text-anchor="middle" font-size="13" '
        f'font-weight="600" fill="var(--fg, #0f172a)">{html_mod.escape(title)}</text>',
    ]

    # Grid lines (horizontal, log scale for y)
    for exp in range(math.floor(log_ler_min), math.ceil(log_ler_max) + 1):
        y = y_of(10**exp)
        if margin["top"] <= y <= margin["top"] + plot_h:
            parts.append(
                f'<line x1="{margin["left"]}" y1="{y:.1f}" '
                f'x2="{margin["left"] + plot_w}" y2="{y:.1f}" '
                f'stroke="var(--table-border, #e2e8f0)" stroke-width="0.5"/>',
            )
            parts.append(
                f'<text x="{margin["left"] - 8}" y="{y + 4:.1f}" text-anchor="end" '
                f'fill="var(--muted, #475569)" font-size="10">1e{exp}</text>',
            )

    # Grid lines (vertical, log scale for x)
    for exp in range(math.floor(log_p_min), math.ceil(log_p_max) + 1):
        x = x_of(10**exp)
        if margin["left"] <= x <= margin["left"] + plot_w:
            parts.append(
                f'<line x1="{x:.1f}" y1="{margin["top"]}" '
                f'x2="{x:.1f}" y2="{margin["top"] + plot_h}" '
                f'stroke="var(--table-border, #e2e8f0)" stroke-width="0.5"/>',
            )

    # Axis labels
    parts.append(
        f'<text x="{margin["left"] + plot_w // 2}" y="{height - 8}" text-anchor="middle" '
        f'fill="var(--muted, #475569)" font-size="11">Physical error rate (p)</text>',
    )
    parts.append(
        f'<text x="14" y="{margin["top"] + plot_h // 2}" text-anchor="middle" '
        f'fill="var(--muted, #475569)" font-size="11" '
        f'transform="rotate(-90, 14, {margin["top"] + plot_h // 2})">{html_mod.escape(y_label)}</text>',
    )

    # X-axis tick labels (log-spaced)
    # Show ticks at 1, 2, 5 * 10^n (standard log-scale subdivisions)
    x_ticks = set()
    for exp in range(math.floor(log_p_min) - 1, math.ceil(log_p_max) + 1):
        for mult in [1.0, 2.0, 5.0]:
            x_ticks.add(mult * 10.0**exp)
    # Filter to visible range and limit density
    visible_ticks = sorted(p for p in x_ticks if p > 0 and log_p_min <= math.log10(p) <= log_p_max)
    # If too many ticks, keep only powers of 10
    if len(visible_ticks) > 8:
        visible_ticks = sorted(p for p in visible_ticks if p == 10.0 ** round(math.log10(p)))
    for p in visible_ticks:
        x = x_of(p)
        parts.append(
            f'<text x="{x:.1f}" y="{margin["top"] + plot_h + 18}" text-anchor="middle" '
            f'fill="var(--muted, #475569)" font-size="10">{_sci(p)}</text>',
        )

    # Plot area border
    parts.append(
        f'<rect x="{margin["left"]}" y="{margin["top"]}" '
        f'width="{plot_w}" height="{plot_h}" '
        f'fill="none" stroke="var(--table-border, #e2e8f0)"/>',
    )

    # Threshold vertical line
    if threshold_p is not None and p_min <= threshold_p <= p_max:
        tx = x_of(threshold_p)
        parts.append(
            f'<line x1="{tx:.1f}" y1="{margin["top"]}" '
            f'x2="{tx:.1f}" y2="{margin["top"] + plot_h}" '
            f'stroke="var(--muted, #334155)" stroke-width="1.5" stroke-dasharray="4,3" opacity="0.7"/>',
        )
        parts.append(
            f'<text x="{tx + 4:.1f}" y="{margin["top"] + 12}" '
            f'fill="var(--muted, #334155)" font-size="9" opacity="0.8">p_th~{_sci(threshold_p)}</text>',
        )

    # Draw curves
    decoder_names = list(dict.fromkeys(c["decoder"] for c in curves))
    distances = sorted({c["distance"] for c in curves})

    legend_y = margin["top"] + 10
    for curve in sorted(curves, key=lambda c: (decoder_names.index(c["decoder"]), c["distance"])):
        dec_idx = decoder_names.index(curve["decoder"])
        dist_idx = distances.index(curve["distance"])
        if color_by == "distance":
            color = (distance_color_map or {}).get(curve["distance"], _COLORS[dist_idx % len(_COLORS)])
            dash = _DASH_PATTERNS[dec_idx % len(_DASH_PATTERNS)]
        else:
            color = (decoder_color_map or {}).get(curve["decoder"], _COLORS[dec_idx % len(_COLORS)])
            dash = _DASH_PATTERNS[dist_idx % len(_DASH_PATTERNS)]

        # CI shaded band (draw first so it's behind everything)
        band_upper = []
        band_lower = []
        for pt in curve["points"]:
            yv = _y(pt)
            if yv <= 0:
                continue
            ci_lo = pt.get(ci_lo_field) or 0
            ci_hi = pt.get(ci_hi_field) or 0
            if ci_lo > 0 and ci_hi > 0:
                x = x_of(pt["physical_error_rate"])
                band_upper.append((x, y_of(ci_hi)))
                band_lower.append((x, y_of(ci_lo)))

        if len(band_upper) >= 2:
            band_d = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(band_upper))
            for x, y in reversed(band_lower):
                band_d += f" L{x:.1f},{y:.1f}"
            band_d += " Z"
            parts.append(
                f'<path d="{band_d}" fill="{color}" opacity="0.2" stroke="none"/>',
            )

        # Center line
        pts = [(x_of(pt["physical_error_rate"]), y_of(_y(pt))) for pt in curve["points"] if _y(pt) > 0]

        if len(pts) >= 2:
            path_d = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(pts))
            dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
            parts.append(
                f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="2"{dash_attr}/>',
            )

        # Data points with CI whiskers
        for pt in curve["points"]:
            yv = _y(pt)
            if yv <= 0:
                continue
            x = x_of(pt["physical_error_rate"])
            y = y_of(yv)
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"/>')
            # CI whiskers
            ci_lo = pt.get(ci_lo_field) or 0
            ci_hi = pt.get(ci_hi_field) or 0
            if ci_lo > 0 and ci_hi > 0:
                y_lo = y_of(ci_lo)
                y_hi = y_of(ci_hi)
                parts.append(
                    f'<line x1="{x:.1f}" y1="{y_lo:.1f}" x2="{x:.1f}" y2="{y_hi:.1f}" '
                    f'stroke="{color}" stroke-width="1.5" opacity="0.7"/>',
                )

        # Legend entry
        label = f"{curve['decoder']} d={curve['distance']}"
        lx = margin["left"] + plot_w + 12
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(
            f'<line x1="{lx}" y1="{legend_y}" x2="{lx + 18}" y2="{legend_y}" '
            f'stroke="{color}" stroke-width="2"{dash_attr}/>',
        )
        parts.append(
            f'<text x="{lx + 24}" y="{legend_y + 4}" fill="var(--fg, #0f172a)" '
            f'font-size="10">{html_mod.escape(label)}</text>',
        )
        legend_y += 16

    parts.append("</svg>")
    return "\n".join(parts)


def _build_duration_svg(
    curves: list[dict],
    title: str,
    width: int = 560,
    height: int = 380,
) -> str:
    """Build an inline SVG of LER vs rounds for multiple (decoder, distance) curves at fixed p."""
    margin = {"top": 45, "right": 160, "bottom": 55, "left": 70}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]

    all_r = []
    all_ler = []
    for curve in curves:
        for pt in curve["points"]:
            all_r.append(pt["num_rounds"])
            if pt["logical_error_rate"] > 0:
                all_ler.append(pt["logical_error_rate"])

    if not all_r or not all_ler:
        return f'<svg width="{width}" height="{height}"><text x="50%" y="50%">No data</text></svg>'

    r_min, r_max = min(all_r), max(all_r)
    ler_min = max(1e-5, min(all_ler) * 0.5)
    ler_max = min(1.0, max(all_ler) * 2.0)
    log_ler_min = math.log10(ler_min)
    log_ler_max = math.log10(ler_max)

    def x_of(r: float) -> float:
        if r_max == r_min:
            return margin["left"] + plot_w / 2
        return margin["left"] + (r - r_min) / (r_max - r_min) * plot_w

    def y_of(ler: float) -> float:
        if ler <= 0:
            return margin["top"] + plot_h
        log_val = math.log10(ler)
        frac = (log_val - log_ler_min) / (log_ler_max - log_ler_min) if log_ler_max != log_ler_min else 0.5
        return margin["top"] + (1 - frac) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'style="font-family:ui-sans-serif,sans-serif;font-size:11px;">',
        f'<rect width="{width}" height="{height}" fill="var(--card-bg, white)" rx="8"/>',
        f'<text x="{width // 2}" y="24" text-anchor="middle" font-size="13" '
        f'font-weight="600" fill="var(--fg, #0f172a)">{html_mod.escape(title)}</text>',
    ]

    # Grid + axis labels
    for exp in range(math.floor(log_ler_min), math.ceil(log_ler_max) + 1):
        y = y_of(10**exp)
        if margin["top"] <= y <= margin["top"] + plot_h:
            parts.append(
                f'<line x1="{margin["left"]}" y1="{y:.1f}" '
                f'x2="{margin["left"] + plot_w}" y2="{y:.1f}" '
                f'stroke="var(--table-border, #e2e8f0)" stroke-width="0.5"/>',
            )
            parts.append(
                f'<text x="{margin["left"] - 8}" y="{y + 4:.1f}" text-anchor="end" '
                f'fill="var(--muted, #475569)" font-size="10">1e{exp}</text>',
            )

    parts.append(
        f'<text x="{margin["left"] + plot_w // 2}" y="{height - 8}" text-anchor="middle" '
        f'fill="var(--muted, #475569)" font-size="11">Rounds</text>',
    )
    parts.append(
        f'<text x="14" y="{margin["top"] + plot_h // 2}" text-anchor="middle" '
        f'fill="var(--muted, #475569)" font-size="11" '
        f'transform="rotate(-90, 14, {margin["top"] + plot_h // 2})">Logical error rate</text>',
    )

    for r in sorted(set(all_r)):
        x = x_of(r)
        parts.append(
            f'<text x="{x:.1f}" y="{margin["top"] + plot_h + 18}" text-anchor="middle" '
            f'fill="var(--muted, #475569)" font-size="10">{r}</text>',
        )

    parts.append(
        f'<rect x="{margin["left"]}" y="{margin["top"]}" '
        f'width="{plot_w}" height="{plot_h}" '
        f'fill="none" stroke="var(--table-border, #e2e8f0)"/>',
    )

    # Draw curves
    legend_y = margin["top"] + 10
    for curve_idx, curve in enumerate(curves):
        color = _COLORS[curve_idx % len(_COLORS)]
        dash = _DASH_PATTERNS[curve_idx // len(_COLORS) % len(_DASH_PATTERNS)]

        pts = [
            (x_of(pt["num_rounds"]), y_of(pt["logical_error_rate"]))
            for pt in curve["points"]
            if pt["logical_error_rate"] > 0
        ]

        if len(pts) >= 2:
            path_d = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(pts))
            dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
            parts.append(
                f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="2"{dash_attr}/>',
            )

        for pt in curve["points"]:
            if pt["logical_error_rate"] <= 0:
                continue
            x = x_of(pt["num_rounds"])
            y = y_of(pt["logical_error_rate"])
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"/>')
            if pt["ci_low"] > 0 and pt["ci_high"] > 0:
                y_lo = y_of(pt["ci_low"])
                y_hi = y_of(pt["ci_high"])
                parts.append(
                    f'<line x1="{x:.1f}" y1="{y_lo:.1f}" x2="{x:.1f}" y2="{y_hi:.1f}" '
                    f'stroke="{color}" stroke-width="1.5" opacity="0.7"/>',
                )

        label = f"{curve['decoder']} d={curve['distance']}"
        lx = margin["left"] + plot_w + 12
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(
            f'<line x1="{lx}" y1="{legend_y}" x2="{lx + 18}" y2="{legend_y}" '
            f'stroke="{color}" stroke-width="2"{dash_attr}/>',
        )
        parts.append(
            f'<text x="{lx + 24}" y="{legend_y + 4}" fill="var(--fg, #0f172a)" '
            f'font-size="10">{html_mod.escape(label)}</text>',
        )
        legend_y += 16

    parts.append("</svg>")
    return "\n".join(parts)


def _build_timing_svg(
    tables: list[dict],
    title: str,
    width: int = 700,
    height: int = 400,
) -> str:
    """Build a violin plot SVG showing decode time distributions per decoder.

    Uses quantile data from DecodeStats (21 percentiles at 0%, 5%, ..., 100%)
    to draw symmetric violins on a log-scale horizontal axis.
    Falls back to box-style whiskers if quantiles are unavailable.
    """
    margin = {"top": 45, "right": 30, "bottom": 55, "left": 120}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]

    # Collect all quantile arrays per decoder (one per operating point).
    # We'll merge by taking the element-wise geometric mean.
    decoder_quantiles: dict[str, list[list[float]]] = {}
    for table in tables:
        for row in table["rows"]:
            dec = row["decoder"]
            q = row.get("quantiles", [])
            if q and any(v > 0 for v in q):
                decoder_quantiles.setdefault(dec, []).append(q)

    # Fallback: synthesize quantiles from summary stats
    if not decoder_quantiles:
        for table in tables:
            for row in table["rows"]:
                dec = row["decoder"]
                med = row["per_shot_median"]
                p99 = row["per_shot_p99"]
                mx = row["per_shot_max"]
                if med > 0:
                    q = [med * 0.5] + [med] * 9 + [med] + [med] * 5 + [p99] * 3 + [mx, mx]
                    decoder_quantiles.setdefault(dec, []).append(q)

    if not decoder_quantiles:
        return ""

    # Merge quantiles per decoder: geometric mean of each percentile position
    merged: dict[str, list[float]] = {}
    for dec, q_list in sorted(decoder_quantiles.items()):
        n_q = len(q_list[0])
        result = []
        for i in range(n_q):
            vals = [q[i] for q in q_list if i < len(q) and q[i] > 0]
            if vals:
                geo_mean = math.exp(sum(math.log(v) for v in vals) / len(vals))
                result.append(geo_mean)
            else:
                result.append(0.0)
        merged[dec] = result

    decoders = list(merged.keys())
    all_vals = [v for qs in merged.values() for v in qs if v > 0]
    if not all_vals:
        return ""

    val_min = min(all_vals) * 0.3
    val_max = max(all_vals) * 3.0
    log_min = math.log10(val_min)
    log_max = math.log10(val_max)

    def x_of(v: float) -> float:
        if v <= 0:
            return margin["left"]
        frac = (math.log10(v) - log_min) / (log_max - log_min) if log_max != log_min else 0.5
        return margin["left"] + frac * plot_w

    violin_h = min(60, plot_h / len(decoders) * 0.75)
    group_h = plot_h / len(decoders)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'style="font-family:ui-sans-serif,sans-serif;font-size:11px;">',
        f'<rect width="{width}" height="{height}" fill="var(--card-bg, white)" rx="8"/>',
        f'<text x="{width // 2}" y="24" text-anchor="middle" font-size="13" '
        f'font-weight="600" fill="var(--fg, #0f172a)">{html_mod.escape(title)}</text>',
    ]

    # Grid lines (log scale)
    for exp in range(math.floor(log_min), math.ceil(log_max) + 1):
        x = x_of(10**exp)
        if margin["left"] <= x <= margin["left"] + plot_w:
            parts.append(
                f'<line x1="{x:.1f}" y1="{margin["top"]}" '
                f'x2="{x:.1f}" y2="{margin["top"] + plot_h}" '
                f'stroke="var(--table-border, #e2e8f0)" stroke-width="0.5"/>',
            )
            parts.append(
                f'<text x="{x:.1f}" y="{margin["top"] + plot_h + 16}" text-anchor="middle" '
                f'fill="var(--muted, #475569)" font-size="10">1e{exp}s</text>',
            )

    parts.append(
        f'<text x="{margin["left"] + plot_w // 2}" y="{height - 8}" text-anchor="middle" '
        f'fill="var(--muted, #475569)" font-size="11">Decode time per shot (seconds, log scale)</text>',
    )

    # Draw violins
    for di, dec in enumerate(decoders):
        qs = merged[dec]
        cy = margin["top"] + di * group_h + group_h / 2
        color = _COLORS[di % len(_COLORS)]

        # Decoder label
        parts.append(
            f'<text x="{margin["left"] - 8}" y="{cy + 4:.1f}" '
            f'text-anchor="end" fill="var(--fg, #0f172a)" font-size="11">{html_mod.escape(dec)}</text>',
        )

        # Build violin shape from quantiles.
        # The "width" at each quantile represents density: wider near the median,
        # narrower at tails. We use a triangular kernel approximation where
        # density ~ 1 / (gap between adjacent quantiles in log space).
        n = len(qs)
        if n < 3:
            continue

        # Compute density proxy at each quantile
        log_qs = [math.log10(max(v, 1e-12)) for v in qs]
        densities = []
        for i in range(n):
            left = log_qs[i] - log_qs[max(0, i - 1)]
            right = log_qs[min(n - 1, i + 1)] - log_qs[i]
            gap = left + right
            densities.append(1.0 / max(gap, 0.01))

        max_density = max(densities) if densities else 1.0
        half_h = violin_h / 2

        # Build SVG path: top half then bottom half (mirror)
        top_points = []
        bot_points = []
        for i in range(n):
            if qs[i] <= 0:
                continue
            x = x_of(qs[i])
            dy = (densities[i] / max_density) * half_h
            top_points.append((x, cy - dy))
            bot_points.append((x, cy + dy))

        if len(top_points) < 2:
            continue

        # Build path: top left-to-right, then bottom right-to-left
        path_parts = [f"M{top_points[0][0]:.1f},{top_points[0][1]:.1f}"]
        for x, y in top_points[1:]:
            path_parts.append(f"L{x:.1f},{y:.1f}")
        for x, y in reversed(bot_points):
            path_parts.append(f"L{x:.1f},{y:.1f}")
        path_parts.append("Z")

        parts.append(
            f'<path d="{" ".join(path_parts)}" fill="{color}" opacity="0.3" stroke="{color}" stroke-width="1"/>',
        )

        # Median line
        med_idx = n // 2
        if qs[med_idx] > 0:
            mx = x_of(qs[med_idx])
            dy = (densities[med_idx] / max_density) * half_h
            parts.append(
                f'<line x1="{mx:.1f}" y1="{cy - dy:.1f}" '
                f'x2="{mx:.1f}" y2="{cy + dy:.1f}" '
                f'stroke="{color}" stroke-width="2"/>',
            )
            parts.append(
                f'<text x="{mx:.1f}" y="{cy - dy - 4:.1f}" text-anchor="middle" '
                f'fill="var(--fg, #0f172a)" font-size="9">'
                f"{qs[med_idx]:.1e}s</text>",
            )

        # p99 marker (index 19 of 21 = 95th percentile is close, use index ~20*0.99=19.8)
        p99_idx = min(n - 2, int(n * 0.99))
        if p99_idx < n and qs[p99_idx] > 0:
            px = x_of(qs[p99_idx])
            parts.append(
                f'<line x1="{px:.1f}" y1="{cy - 3:.1f}" '
                f'x2="{px:.1f}" y2="{cy + 3:.1f}" '
                f'stroke="{color}" stroke-width="1.5"/>',
            )

    parts.append("</svg>")
    return "\n".join(parts)


# -- HTML report --------------------------------------------------------------


def _shots_summary(tables: list[dict]) -> str:
    """Summarise shot counts across all comparison table rows."""
    all_counts = set()
    for table in tables:
        for row in table["rows"]:
            all_counts.add(row["num_shots"])
    if not all_counts:
        return "N/A"
    lo, hi = min(all_counts), max(all_counts)
    if lo == hi:
        return str(lo)
    return f"{lo} -- {hi} per point"


def _rounds_summary(tables: list[dict]) -> str:
    """Summarise round counts per distance, e.g. 'd=3: r=[6,7,9]; d=5: r=[10,12,15]'."""
    from collections import defaultdict

    rounds_by_d: dict[int, set[int]] = defaultdict(set)
    for table in tables:
        rounds_by_d[table["distance"]].add(table["num_rounds"])
    if not rounds_by_d:
        return "N/A"
    lines = []
    for d in sorted(rounds_by_d):
        rs = sorted(rounds_by_d[d])
        lines.append(html_mod.escape(f"d={d}: r=[{', '.join(str(r) for r in rs)}]"))
    return "<br>".join(lines)


def _build_html(analysis: dict) -> str:
    """Build the full HTML report string."""
    config = analysis.get("config", {})
    tables = analysis.get("comparison_tables", [])
    curves = analysis.get("threshold_curves", [])

    style = dedent("""
        :root {
          color-scheme: light dark;
          --bg: #f8fafc; --fg: #0f172a;
          --hero-bg: linear-gradient(135deg, #e0f2fe, #f8fafc 55%, #dcfce7);
          --card-bg: white; --card-border: #dbeafe; --card-shadow: rgba(15,23,42,0.05);
          --meta-bg: rgba(255,255,255,0.82);
          --muted: #475569; --link: #2563eb;
          --table-stripe: #f1f5f9; --table-border: #e2e8f0;
        }
        [data-theme="dark"] {
          --bg: #0f172a; --fg: #e2e8f0;
          --hero-bg: linear-gradient(135deg, #1e293b, #0f172a 55%, #1a2e1a);
          --card-bg: #1e293b; --card-border: #334155; --card-shadow: rgba(0,0,0,0.3);
          --meta-bg: rgba(30,41,59,0.82);
          --muted: #94a3b8; --link: #60a5fa;
          --table-stripe: #0f172a; --table-border: #334155;
        }
        @media (prefers-color-scheme: dark) {
          :root:not([data-theme="light"]) {
            --bg: #0f172a; --fg: #e2e8f0;
            --hero-bg: linear-gradient(135deg, #1e293b, #0f172a 55%, #1a2e1a);
            --card-bg: #1e293b; --card-border: #334155; --card-shadow: rgba(0,0,0,0.3);
            --meta-bg: rgba(30,41,59,0.82);
            --muted: #94a3b8; --link: #60a5fa;
            --table-stripe: #0f172a; --table-border: #334155;
          }
        }
        body {
          margin: 0;
          font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, sans-serif;
          background: var(--bg); color: var(--fg);
        }
        main { max-width: 1400px; margin: 0 auto; padding: 32px 24px 56px; }
        h1, h2, h3, p { margin-top: 0; }
        .theme-toggle {
          position: fixed; top: 16px; right: 16px; z-index: 100;
          background: var(--card-bg); border: 1px solid var(--card-border);
          border-radius: 8px; padding: 6px 12px; cursor: pointer;
          color: var(--fg); font-size: 0.85rem; font-weight: 600;
        }
        .theme-toggle:hover { opacity: 0.8; }
        .hero {
          background: var(--hero-bg);
          border: 1px solid var(--card-border);
          border-radius: 20px;
          padding: 24px;
          margin-bottom: 24px;
        }
        .meta {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 12px;
          margin-top: 18px;
        }
        .meta-card {
          background: var(--meta-bg);
          border: 1px solid var(--card-border);
          border-radius: 14px;
          padding: 14px 16px;
        }
        .meta-card strong {
          display: block; font-size: 0.82rem; text-transform: uppercase;
          letter-spacing: 0.04em; color: var(--muted); margin-bottom: 6px;
        }
        .section {
          background: var(--card-bg);
          border: 1px solid var(--card-border);
          border-radius: 18px;
          padding: 20px 22px;
          margin-top: 20px;
          box-shadow: 0 10px 24px var(--card-shadow);
          overflow-x: auto;
        }
        .plots {
          display: flex;
          flex-wrap: wrap;
          gap: 16px;
          justify-content: center;
        }
        table { border-collapse: collapse; width: 100%; margin: 12px 0 0; }
        th, td { padding: 10px 14px; text-align: right; border-bottom: 1px solid var(--table-border); }
        th {
          font-size: 0.82rem; text-transform: uppercase;
          letter-spacing: 0.03em; color: var(--muted); border-bottom-width: 2px;
        }
        td:first-child, th:first-child { text-align: left; }
        tr:nth-child(even) td { background: var(--table-stripe); }
        code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
        details.collapsible { margin-top: 20px; }
        details.collapsible > summary {
          cursor: pointer; font-size: 1.1rem; font-weight: 600;
          padding: 14px 22px;
          background: var(--card-bg); border: 1px solid var(--card-border);
          border-radius: 18px;
          box-shadow: 0 10px 24px var(--card-shadow);
          list-style: none;
        }
        details.collapsible > summary::before { content: "\\25B6  "; font-size: 0.8em; }
        details.collapsible[open] > summary::before { content: "\\25BC  "; }
        details.collapsible > .section { margin-top: 8px; }
    """).strip()

    def meta_card(label: str, value: str, *, raw: bool = False) -> str:
        val = value if raw else html_mod.escape(value)
        return f'      <div class="meta-card"><strong>{html_mod.escape(label)}</strong>{val}</div>'

    decoders = config.get("decoders", [])
    p1s = config.get("p1_scale", 1 / 30)
    pms = config.get("p_meas_scale", 1 / 3)
    pps = config.get("p_prep_scale", 1 / 3)
    noise_str = f"Depolarizing: p1={p1s:.4g}*p, p2=p, p_meas={pms:.4g}*p, p_prep={pps:.4g}*p"

    # Build global color maps for consistency across all plots
    all_decoders_sorted = sorted({c["decoder"] for c in curves}) if curves else decoders
    all_distances_sorted = sorted({c["distance"] for c in curves}) if curves else []
    dec_colors = {d: _COLORS[i % len(_COLORS)] for i, d in enumerate(all_decoders_sorted)}
    dist_colors = {d: _COLORS[i % len(_COLORS)] for i, d in enumerate(all_distances_sorted)}

    parts = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '  <meta charset="utf-8" />',
        '  <meta name="viewport" content="width=device-width, initial-scale=1" />',
        "  <title>PECOS Decoder Performance Report</title>",
        f"  <style>{style}</style>",
        "</head>",
        "<body>",
        '<button id="theme-toggle" class="theme-toggle">Light / Dark</button>',
        "<main>",
        '  <section class="hero">',
        "    <h1>PECOS Decoder Performance Report</h1>",
        "    <p>Same samples decoded by multiple decoders. LER differences reflect "
        "decoder quality, not sampling noise.</p>",
        '    <div class="meta">',
        meta_card("Decoders", ", ".join(decoders)),
        meta_card("Distances", ", ".join(str(d) for d in config.get("distances", []))),
        meta_card("Basis", config.get("basis", "Z")),
        meta_card("Shots", _shots_summary(tables)),
        meta_card("Noise Model", noise_str),
        meta_card("Error Rates (p)", ", ".join(f"{p:.4g}" for p in config.get("error_rates", []))),
        "    </div>",
        '    <div class="meta" style="grid-template-columns: 1fr;">',
        meta_card("Rounds", _rounds_summary(tables), raw=True),
        "    </div>",
        "  </section>",
    ]

    # -- Threshold curves section --
    if curves:
        # Group curves by distance and by decoder
        curves_by_distance: dict[int, list[dict]] = defaultdict(list)
        curves_by_decoder: dict[str, list[dict]] = defaultdict(list)
        for curve in curves:
            curves_by_distance[curve["distance"]].append(curve)
            curves_by_decoder[curve["decoder"]].append(curve)

        threshold_estimates = analysis.get("threshold_estimates", [])
        fss_per_round_est = {e["decoder"]: e for e in threshold_estimates if e.get("metric") == "fss_per_round"}
        fss_d_round_est = {e["decoder"]: e for e in threshold_estimates if e.get("metric") == "fss_d_round"}

        def _threshold_table(estimates: dict) -> list[str]:
            if not estimates:
                return []
            lines = []
            lines.append("    <table>")
            lines.append(
                "      <tr><th>Decoder</th><th>Threshold (FSS fit)</th><th>Distances</th></tr>",
            )
            for est in sorted(estimates.values(), key=lambda e: e.get("estimated_p_th", 0), reverse=True):
                p_th = est["estimated_p_th"]
                se = est.get("std_error")
                th_str = f"{_sci(p_th)} +/- {_sci(se)}" if se else f"{_sci(p_th)}"
                lines.append(
                    f"      <tr><td>{html_mod.escape(est['decoder'])}</td>"
                    f"<td>{th_str}</td>"
                    f"<td>d={est['d_small']} -- d={est['d_large']}</td></tr>",
                )
            lines.append("    </table>")
            return lines

        # === ALWAYS VISIBLE: Per-round LER ===

        # Per-round by distance
        parts.append('  <div class="section">')
        parts.append("    <h2>Per-Round Logical Error Rate -- by Distance</h2>")
        parts.append(
            "    <p>All decoders compared at each distance. "
            "Fitted from multiple syndrome extraction rounds per point.</p>",
        )
        parts.append('    <div class="plots">')
        for d in sorted(curves_by_distance):
            svg = _build_threshold_svg(
                curves_by_distance[d],
                title=f"d = {d}",
                y_field="per_round_ler",
                y_label="LER per round",
                decoder_color_map=dec_colors,
            )
            parts.append(f"      {svg}")
        parts.append("    </div>")
        parts.append("  </div>")

        # Per-round by decoder (threshold view)
        parts.append('  <div class="section">')
        parts.append("    <h2>Per-Round Logical Error Rate -- by Decoder</h2>")
        parts.append(
            "    <p>Distance scaling for each decoder. "
            "Curves crossing = threshold (per-round LER is distance-independent at threshold).</p>",
        )
        parts.extend(_threshold_table(fss_per_round_est))
        parts.append('    <div class="plots">')
        for dec in sorted(curves_by_decoder):
            est = fss_per_round_est.get(dec)
            title = dec
            p_th = None
            if est:
                p_th = est["estimated_p_th"]
                title = f"{dec} (p_th ~ {p_th:.4f})"
            svg = _build_threshold_svg(
                curves_by_decoder[dec],
                title=title,
                color_by="distance",
                threshold_p=p_th,
                y_field="per_round_ler",
                y_label="LER per round",
                distance_color_map=dist_colors,
            )
            parts.append(f"      {svg}")
        parts.append("    </div>")
        parts.append("  </div>")

        # === COLLAPSIBLE: d-round LER ===
        parts.append('  <details class="collapsible">')
        parts.append("    <summary>d-Round Logical Error Rate (click to expand)</summary>")

        # d-round by distance
        parts.append('    <div class="section">')
        parts.append("      <h3>d-Round LER -- by Distance</h3>")
        parts.append("      <p>Logical error rate over d rounds of syndrome extraction.</p>")
        parts.append('      <div class="plots">')
        for d in sorted(curves_by_distance):
            svg = _build_threshold_svg(
                curves_by_distance[d],
                title=f"d = {d}",
                y_label="LER (d rounds)",
                decoder_color_map=dec_colors,
            )
            parts.append(f"        {svg}")
        parts.append("      </div>")
        parts.append("    </div>")

        # d-round by decoder
        parts.append('    <div class="section">')
        parts.append("      <h3>d-Round LER -- by Decoder</h3>")
        parts.extend(_threshold_table(fss_d_round_est))
        parts.append('      <div class="plots">')
        for dec in sorted(curves_by_decoder):
            est = fss_d_round_est.get(dec)
            title = dec
            p_th = None
            if est:
                p_th = est["estimated_p_th"]
                title = f"{dec} (p_th ~ {p_th:.4f})"
            svg = _build_threshold_svg(
                curves_by_decoder[dec],
                title=title,
                color_by="distance",
                threshold_p=p_th,
                y_label="LER (d rounds)",
                distance_color_map=dist_colors,
            )
            parts.append(f"        {svg}")
        parts.append("      </div>")
        parts.append("    </div>")
        parts.append("  </details>")

    # -- Duration curves (collapsible) --
    duration_curves = analysis.get("duration_curves", [])
    if duration_curves:
        # Group by physical_error_rate
        dur_by_p: dict[float, list[dict]] = defaultdict(list)
        for dc in duration_curves:
            dur_by_p[dc["physical_error_rate"]].append(dc)

        parts.append('  <details class="collapsible">')
        parts.append("    <summary>Duration Curves -- LER vs Rounds (click to expand)</summary>")
        parts.append('    <div class="section">')
        parts.append(
            "      <p>Logical error rate vs number of rounds at fixed physical error rate. "
            "Shows how LER grows with memory duration.</p>",
        )
        parts.append('      <div class="plots">')
        for p in sorted(dur_by_p):
            svg = _build_duration_svg(dur_by_p[p], title=f"p = {p:.4g}")
            parts.append(f"        {svg}")
        parts.append("      </div>")
        parts.append("    </div>")
        parts.append("  </details>")

    # -- Timing comparison section --
    if tables:
        err_rates = sorted({t["physical_error_rate"] for t in tables})
        distances_available = sorted({t["distance"] for t in tables})

        def _violin_plots_for_p(p_val: float) -> list[str]:
            """Generate violin SVGs for each distance at a given error rate."""
            svgs = []
            for d in distances_available:
                candidates = [t for t in tables if t["distance"] == d and t["physical_error_rate"] == p_val]
                if not candidates:
                    continue
                target_r = 2 * d
                best = min(candidates, key=lambda t: abs(t["num_rounds"] - target_r))
                svg = _build_timing_svg(
                    [best],
                    title=f"d = {d}, r = {best['num_rounds']}",
                )
                svgs.append(svg)
            return svgs

        # Always-visible: lowest error rate
        if err_rates:
            lowest_p = err_rates[0]
            svgs = _violin_plots_for_p(lowest_p)
            if svgs:
                parts.append('  <div class="section">')
                parts.append(f"    <h2>Decode Speed (p = {lowest_p:.4g})</h2>")
                parts.append("    <p>Per-shot decode time distribution for each decoder and distance.</p>")
                parts.append('    <div class="plots">')
                parts.extend(f"      {svg}" for svg in svgs)
                parts.append("    </div>")
                parts.append("  </div>")

        # Collapsible: all other error rates
        other_rates = [p for p in err_rates if p != err_rates[0]]
        if other_rates:
            parts.append('  <details class="collapsible">')
            parts.append("    <summary>Decode Speed at Other Error Rates (click to expand)</summary>")
            for p_val in other_rates:
                svgs = _violin_plots_for_p(p_val)
                if svgs:
                    parts.append('    <div class="section">')
                    parts.append(f"      <h3>p = {p_val:.4g}</h3>")
                    parts.append('      <div class="plots">')
                    parts.extend(f"        {svg}" for svg in svgs)
                    parts.append("      </div>")
                    parts.append("    </div>")
            parts.append("  </details>")

    # -- Comparison tables (collapsible) --
    if tables:
        parts.append('  <details class="collapsible">')
        parts.append("    <summary>Detailed Comparison Tables (click to expand)</summary>")
        for table in tables:
            d = table["distance"]
            p = table["physical_error_rate"]
            r = table["num_rounds"]
            n = table["num_shots"]

            parts.append('    <div class="section">')
            parts.append(f"      <h3>d={d}, p={p:.4g}, rounds={r} ({n} shots)</h3>")
            parts.append("      <table>")
            parts.append(
                "        <tr><th>Decoder</th><th>LER (95% CI)</th>"
                "<th>Median</th><th>p99</th><th>Max</th><th>Throughput</th></tr>",
            )

            for row in table["rows"]:
                ler_str = f"{row['logical_error_rate']:.4f} ({row['ci_low']:.4f} - {row['ci_high']:.4f})"
                sps = row["num_shots"] / row["per_shot_median"] if row["per_shot_median"] > 0 else float("inf")
                parts.append(
                    f"        <tr>"
                    f"<td>{html_mod.escape(row['decoder'])}</td>"
                    f"<td>{ler_str}</td>"
                    f"<td>{row['per_shot_median']:.1e} s</td>"
                    f"<td>{row['per_shot_p99']:.1e} s</td>"
                    f"<td>{row['per_shot_max']:.1e} s</td>"
                    f"<td>{sps:.1e} shots/s</td>"
                    f"</tr>",
                )

            parts.append("      </table>")
            parts.append("    </div>")
        parts.append("  </details>")

    parts.extend(
        [
            "</main>",
            dedent("""
        <script>
        (function() {
          var html = document.documentElement;
          var btn = document.getElementById('theme-toggle');
          var stored = localStorage.getItem('pecos-theme');
          if (stored) html.setAttribute('data-theme', stored);
          btn.addEventListener('click', function() {
            var current = html.getAttribute('data-theme');
            var next = current === 'dark' ? 'light' : 'dark';
            if (!current) next = 'dark';
            html.setAttribute('data-theme', next);
            localStorage.setItem('pecos-theme', next);
          });
        })();
        </script>
        """).strip(),
            "</body>",
            "</html>",
        ],
    )

    return "\n".join(parts)


# -- PDF report (matplotlib) --------------------------------------------------


def _build_pdf(analysis: dict, output_path: Path) -> None:
    """Build a multi-page PDF report using matplotlib."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    config = analysis.get("config", {})
    tables = analysis.get("comparison_tables", [])
    curves = analysis.get("threshold_curves", [])

    page_size = (11, 8.5)

    with PdfPages(output_path) as pdf:
        # -- Cover page --
        fig, ax = plt.subplots(figsize=page_size)
        ax.axis("off")
        ax.text(
            0.5,
            0.65,
            "PECOS Decoder Performance Report",
            transform=ax.transAxes,
            fontsize=24,
            ha="center",
            va="center",
            fontweight="bold",
        )

        info_lines = [
            f"Decoders: {', '.join(config.get('decoders', []))}",
            f"Distances: {config.get('distances', [])}",
            f"Error rates: {config.get('error_rates', [])}",
            f"Shots: {config.get('shots', 'N/A')}",
            f"Basis: {config.get('basis', 'Z')}",
        ]
        ax.text(
            0.5,
            0.40,
            "\n".join(info_lines),
            transform=ax.transAxes,
            fontsize=12,
            ha="center",
            va="center",
            family="monospace",
        )
        pdf.savefig(fig)
        plt.close(fig)

        # -- Threshold curve plots --
        if curves:
            curves_by_distance: dict[int, list[dict]] = defaultdict(list)
            curves_by_decoder: dict[str, list[dict]] = defaultdict(list)
            for curve in curves:
                curves_by_distance[curve["distance"]].append(curve)
                curves_by_decoder[curve["decoder"]].append(curve)

            def _plot_curves(
                ax: plt.Axes,
                curve_list: list[dict],
                title: str,
                color_by: str = "decoder",
            ) -> None:
                ax.set_title(title, fontsize=14, fontweight="bold")
                ax.set_xlabel("Physical error rate (p)")
                ax.set_ylabel("Logical error rate")
                ax.set_yscale("log")
                ax.grid(visible=True, alpha=0.3)
                decoder_names = list(dict.fromkeys(c["decoder"] for c in curve_list))
                dists = sorted({c["distance"] for c in curve_list})
                linestyles = ["-", "--", ":", "-."]
                for curve in sorted(curve_list, key=lambda c: (decoder_names.index(c["decoder"]), c["distance"])):
                    pts = [pt for pt in curve["points"] if pt["logical_error_rate"] > 0]
                    if not pts:
                        continue
                    ps = [pt["physical_error_rate"] for pt in pts]
                    lers = [pt["logical_error_rate"] for pt in pts]
                    ci_lows = [pt["ci_low"] for pt in pts]
                    ci_highs = [pt["ci_high"] for pt in pts]
                    dec_idx = decoder_names.index(curve["decoder"])
                    dist_idx = dists.index(curve["distance"])
                    if color_by == "distance":
                        color = _COLORS[dist_idx % len(_COLORS)]
                        ls = linestyles[dec_idx % len(linestyles)]
                    else:
                        color = _COLORS[dec_idx % len(_COLORS)]
                        ls = linestyles[dist_idx % len(linestyles)]
                    label = f"{curve['decoder']} d={curve['distance']}"
                    ax.plot(ps, lers, marker="o", linestyle=ls, color=color, label=label, markersize=4)
                    ax.fill_between(ps, ci_lows, ci_highs, color=color, alpha=0.15)
                ax.legend(fontsize=9, loc="best")

            # Per-distance plots (all decoders)
            for d in sorted(curves_by_distance):
                fig, ax = plt.subplots(figsize=page_size)
                _plot_curves(ax, curves_by_distance[d], f"By Distance -- d = {d}")
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

            # Per-decoder plots (all distances -- color by distance, threshold line)
            threshold_estimates = analysis.get("threshold_estimates", [])
            est_by_dec = {e["decoder"]: e for e in threshold_estimates}
            for dec in sorted(curves_by_decoder):
                fig, ax = plt.subplots(figsize=page_size)
                est = est_by_dec.get(dec)
                title = f"By Decoder -- {dec}"
                if est:
                    title += f" (p_th ~ {est['estimated_p_th']:.4f})"
                _plot_curves(ax, curves_by_decoder[dec], title, color_by="distance")
                if est:
                    ax.axvline(
                        est["estimated_p_th"],
                        color="#334155",
                        linestyle=":",
                        linewidth=1.8,
                        alpha=0.7,
                        zorder=0,
                    )
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

        # -- Duration curve plots --
        duration_curves = analysis.get("duration_curves", [])
        if duration_curves:
            dur_by_p: dict[float, list[dict]] = defaultdict(list)
            for dc in duration_curves:
                dur_by_p[dc["physical_error_rate"]].append(dc)

            for p_val in sorted(dur_by_p):
                fig, ax = plt.subplots(figsize=page_size)
                ax.set_title(f"Duration -- p = {p_val:.4g}", fontsize=14, fontweight="bold")
                ax.set_xlabel("Rounds")
                ax.set_ylabel("Logical error rate")
                ax.set_yscale("log")
                ax.grid(visible=True, alpha=0.3)

                for ci, dc in enumerate(dur_by_p[p_val]):
                    pts = [pt for pt in dc["points"] if pt["logical_error_rate"] > 0]
                    if not pts:
                        continue
                    rs = [pt["num_rounds"] for pt in pts]
                    lers = [pt["logical_error_rate"] for pt in pts]
                    ci_lows = [pt["ci_low"] for pt in pts]
                    ci_highs = [pt["ci_high"] for pt in pts]
                    color = _COLORS[ci % len(_COLORS)]
                    label = f"{dc['decoder']} d={dc['distance']}"
                    ax.plot(rs, lers, "o-", color=color, label=label, markersize=4)
                    ax.fill_between(rs, ci_lows, ci_highs, color=color, alpha=0.15)

                ax.legend(fontsize=9, loc="best")
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

        # -- Comparison tables as figures --
        for table in tables:
            fig, ax = plt.subplots(figsize=page_size)
            ax.axis("off")
            ax.set_title(
                f"d={table['distance']}, p={table['physical_error_rate']:.4g}, "
                f"rounds={table['num_rounds']} ({table['num_shots']} shots)",
                fontsize=14,
                fontweight="bold",
                pad=20,
            )

            col_labels = ["Decoder", "LER", "95% CI", "Median", "p99", "Max"]
            cell_data = [
                [
                    row["decoder"],
                    f"{row['logical_error_rate']:.4f}",
                    f"{row['ci_low']:.4f} - {row['ci_high']:.4f}",
                    f"{row['per_shot_median']:.1e} s",
                    f"{row['per_shot_p99']:.1e} s",
                    f"{row['per_shot_max']:.1e} s",
                ]
                for row in table["rows"]
            ]

            if cell_data:
                tbl = ax.table(
                    cellText=cell_data,
                    colLabels=col_labels,
                    loc="center",
                    cellLoc="center",
                )
                tbl.auto_set_font_size(value=False)
                tbl.set_fontsize(10)
                tbl.scale(1.0, 1.8)

                # Style header row
                for j in range(len(col_labels)):
                    tbl[0, j].set_facecolor("#e2e8f0")
                    tbl[0, j].set_text_props(fontweight="bold")

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


# -- Markdown report (Obsidian-compatible) ------------------------------------


def _build_markdown(analysis: dict, plots_dir: Path) -> str:
    """Build an Obsidian-compatible Markdown report with standalone SVG plots."""
    config = analysis.get("config", {})
    tables = analysis.get("comparison_tables", [])
    curves = analysis.get("threshold_curves", [])

    decoders = config.get("decoders", [])
    p1s = config.get("p1_scale", 1 / 30)
    pms = config.get("p_meas_scale", 1 / 3)
    pps = config.get("p_prep_scale", 1 / 3)
    noise_str = f"Depolarizing: p1={p1s:.4g}*p, p2=p, p_meas={pms:.4g}*p, p_prep={pps:.4g}*p"

    # Global color maps
    all_decoders_sorted = sorted({c["decoder"] for c in curves}) if curves else decoders
    all_distances_sorted = sorted({c["distance"] for c in curves}) if curves else []
    dec_colors = {d: _COLORS[i % len(_COLORS)] for i, d in enumerate(all_decoders_sorted)}
    dist_colors = {d: _COLORS[i % len(_COLORS)] for i, d in enumerate(all_distances_sorted)}

    # Rounds summary
    rounds_by_d: dict[int, set[int]] = defaultdict(set)
    for table in tables:
        rounds_by_d[table["distance"]].add(table["num_rounds"])

    def _save_svg(svg_content: str, name: str) -> str:
        """Save SVG to plots_dir and return relative path."""
        filename = f"{name}.svg"
        (plots_dir / filename).write_text(svg_content)
        return f"plots/{filename}"

    lines = []

    # -- Frontmatter --
    lines.append("---")
    lines.append("title: PECOS Decoder Performance Report")
    lines.append("tags: [report, surface-code, decoders]")
    lines.append(f"decoders: [{', '.join(decoders)}]")
    lines.append(f"distances: [{', '.join(str(d) for d in config.get('distances', []))}]")
    lines.append(f"date: {__import__('datetime').date.today().isoformat()}")
    lines.append("---")
    lines.append("")

    # -- Header --
    lines.append("# PECOS Decoder Performance Report")
    lines.append("")
    lines.append("> [!info] Configuration")
    lines.append(f"> **Decoders:** {', '.join(decoders)}")
    lines.append(f"> **Distances:** {', '.join(str(d) for d in config.get('distances', []))}")
    lines.append(f"> **Error Rates (p):** {', '.join(f'{p:.4g}' for p in config.get('error_rates', []))}")
    lines.append(f"> **Shots:** {_shots_summary(tables)}")
    lines.append(f"> **Noise Model:** {noise_str}")
    lines.append(f"> **Basis:** {config.get('basis', 'Z')}")
    for d in sorted(rounds_by_d):
        rs = sorted(rounds_by_d[d])
        lines.append(f"> **d={d}:** r=[{', '.join(str(r) for r in rs)}]")
    lines.append("")

    if curves:
        curves_by_distance: dict[int, list[dict]] = defaultdict(list)
        curves_by_decoder: dict[str, list[dict]] = defaultdict(list)
        for curve in curves:
            curves_by_distance[curve["distance"]].append(curve)
            curves_by_decoder[curve["decoder"]].append(curve)

        threshold_estimates = analysis.get("threshold_estimates", [])
        fss_per_round_est = {e["decoder"]: e for e in threshold_estimates if e.get("metric") == "fss_per_round"}
        fss_d_round_est = {e["decoder"]: e for e in threshold_estimates if e.get("metric") == "fss_d_round"}

        # -- Per-round by distance --
        lines.append("## Per-Round LER -- by Distance")
        lines.append("")
        for d in sorted(curves_by_distance):
            svg = _build_threshold_svg(
                curves_by_distance[d],
                title=f"d = {d}",
                y_field="per_round_ler",
                y_label="LER per round",
                decoder_color_map=dec_colors,
            )
            path = _save_svg(svg, f"per_round_by_dist_d{d}")
            lines.append(f"![d={d}]({path})")
            lines.append("")

        # -- Per-round by decoder with thresholds --
        lines.append("## Per-Round LER -- by Decoder")
        lines.append("")

        if fss_per_round_est:
            lines.append("### Threshold Estimates (per-round)")
            lines.append("")
            lines.append("| Decoder | Threshold | Distances |")
            lines.append("|---------|-----------|-----------|")
            lines.extend(
                f"| {est['decoder']} | {est['estimated_p_th']:.4f} "
                f"| d={est['d_small']} / d={est['d_large']} crossing |"
                for est in sorted(fss_per_round_est.values(), key=lambda e: e["estimated_p_th"], reverse=True)
            )
            lines.append("")

        for dec in sorted(curves_by_decoder):
            est = fss_per_round_est.get(dec)
            title = dec
            p_th = None
            if est:
                p_th = est["estimated_p_th"]
                title = f"{dec} (p_th ~ {p_th:.4f})"
            svg = _build_threshold_svg(
                curves_by_decoder[dec],
                title=title,
                color_by="distance",
                threshold_p=p_th,
                y_field="per_round_ler",
                y_label="LER per round",
                distance_color_map=dist_colors,
            )
            path = _save_svg(svg, f"per_round_by_dec_{dec.replace(':', '_')}")
            lines.append(f"![{dec}]({path})")
            lines.append("")

        # -- d-round LER (collapsible) --
        lines.append("> [!note]- d-Round Logical Error Rate (click to expand)")
        lines.append(">")
        lines.append("> ### d-Round LER -- by Distance")
        lines.append(">")
        for d in sorted(curves_by_distance):
            svg = _build_threshold_svg(
                curves_by_distance[d],
                title=f"d = {d}",
                y_label="LER (d rounds)",
                decoder_color_map=dec_colors,
            )
            path = _save_svg(svg, f"d_round_by_dist_d{d}")
            lines.append(f"> ![d={d}]({path})")
            lines.append(">")

        if fss_d_round_est:
            lines.append("> ### Threshold Estimates (d-round)")
            lines.append(">")
            lines.append("> | Decoder | Threshold | Distances |")
            lines.append("> |---------|-----------|-----------|")
            lines.extend(
                f"> | {est['decoder']} | {est['estimated_p_th']:.4f} "
                f"| d={est['d_small']} / d={est['d_large']} crossing |"
                for est in sorted(fss_d_round_est.values(), key=lambda e: e["estimated_p_th"], reverse=True)
            )
            lines.append(">")

        for dec in sorted(curves_by_decoder):
            est = fss_d_round_est.get(dec)
            title = dec
            p_th = None
            if est:
                p_th = est["estimated_p_th"]
                title = f"{dec} (p_th ~ {p_th:.4f})"
            svg = _build_threshold_svg(
                curves_by_decoder[dec],
                title=title,
                color_by="distance",
                threshold_p=p_th,
                y_label="LER (d rounds)",
                distance_color_map=dist_colors,
            )
            path = _save_svg(svg, f"d_round_by_dec_{dec.replace(':', '_')}")
            lines.append(f"> ![{dec}]({path})")
            lines.append(">")
        lines.append("")

    # -- Duration curves (collapsible) --
    duration_curves = analysis.get("duration_curves", [])
    if duration_curves:
        dur_by_p: dict[float, list[dict]] = defaultdict(list)
        for dc in duration_curves:
            dur_by_p[dc["physical_error_rate"]].append(dc)

        lines.append("> [!note]- Duration Curves -- LER vs Rounds (click to expand)")
        lines.append(">")
        for p in sorted(dur_by_p):
            svg = _build_duration_svg(dur_by_p[p], title=f"p = {p:.4g}")
            path = _save_svg(svg, f"duration_p{p:.4g}".replace(".", "_"))
            lines.append(f"> ![p={p:.4g}]({path})")
            lines.append(">")
        lines.append("")

    # -- Decode speed --
    if tables:
        err_rates = sorted({t["physical_error_rate"] for t in tables})
        distances_available = sorted({t["distance"] for t in tables})

        if err_rates:
            lowest_p = err_rates[0]
            lines.append(f"## Decode Speed (p = {lowest_p:.4g})")
            lines.append("")
            for d in distances_available:
                candidates = [t for t in tables if t["distance"] == d and t["physical_error_rate"] == lowest_p]
                if not candidates:
                    continue
                target_r = 2 * d
                best = min(candidates, key=lambda t: abs(t["num_rounds"] - target_r))
                svg = _build_timing_svg([best], title=f"d = {d}, r = {best['num_rounds']}")
                path = _save_svg(svg, f"timing_d{d}_p{lowest_p:.4g}".replace(".", "_"))
                lines.append(f"![d={d}]({path})")
                lines.append("")

        # Other error rates collapsible
        other_rates = [p for p in err_rates if p != err_rates[0]] if err_rates else []
        if other_rates:
            lines.append("> [!note]- Decode Speed at Other Error Rates (click to expand)")
            lines.append(">")
            for p_val in other_rates:
                lines.append(f"> **p = {p_val:.4g}**")
                lines.append(">")
                for d in distances_available:
                    candidates = [t for t in tables if t["distance"] == d and t["physical_error_rate"] == p_val]
                    if not candidates:
                        continue
                    target_r = 2 * d
                    best = min(candidates, key=lambda t: abs(t["num_rounds"] - target_r))
                    svg = _build_timing_svg([best], title=f"d = {d}, r = {best['num_rounds']}")
                    path = _save_svg(svg, f"timing_d{d}_p{p_val:.4g}".replace(".", "_"))
                    lines.append(f"> ![d={d}]({path})")
                    lines.append(">")
            lines.append("")

    # -- Comparison tables (collapsible) --
    if tables:
        lines.append("> [!note]- Detailed Comparison Tables (click to expand)")
        lines.append(">")
        for table in tables:
            d = table["distance"]
            p = table["physical_error_rate"]
            r = table["num_rounds"]
            n = table["num_shots"]
            lines.append(f"> ### d={d}, p={p:.4g}, rounds={r} ({n} shots)")
            lines.append(">")
            lines.append("> | Decoder | LER (95% CI) | Median | p99 | Max | Throughput |")
            lines.append("> |---------|-------------|--------|-----|-----|------------|")
            for row in table["rows"]:
                ler_str = f"{row['logical_error_rate']:.4f} ({row['ci_low']:.4f} - {row['ci_high']:.4f})"
                sps = row["num_shots"] / row["per_shot_median"] if row["per_shot_median"] > 0 else float("inf")
                lines.append(
                    f"> | {row['decoder']} | {ler_str} "
                    f"| {row['per_shot_median']:.1e} s "
                    f"| {row['per_shot_p99']:.1e} s "
                    f"| {row['per_shot_max']:.1e} s "
                    f"| {sps:.1e} shots/s |",
                )
            lines.append(">")
        lines.append("")

    return "\n".join(lines)


# -- CLI ----------------------------------------------------------------------


def main() -> int:
    """CLI entry point for report generation."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("analysis", type=Path, help="Analysis JSON from analyze_data.py")
    parser.add_argument("--html", action="store_true", help="Generate HTML report")
    parser.add_argument("--pdf", action="store_true", help="Generate PDF report (requires matplotlib)")
    parser.add_argument("--markdown", action="store_true", help="Generate Obsidian-compatible Markdown report")
    parser.add_argument("--open", action="store_true", help="Open HTML report in browser")
    parser.add_argument("-o", "--output-dir", type=str, default=None)
    args = parser.parse_args()

    if not args.html and not args.pdf and not args.markdown:
        args.html = True  # default to HTML

    analysis = _load_analysis(args.analysis)

    out = Path(args.output_dir) if args.output_dir else args.analysis.parent
    out.mkdir(parents=True, exist_ok=True)

    if args.html:
        html_path = out / "report.html"
        html_path.write_text(_build_html(analysis))
        print(f"Wrote {html_path}")

        if args.open:
            import webbrowser

            webbrowser.open(html_path.as_uri())
            print(f"Opened {html_path}")

    if args.pdf:
        pdf_path = out / "report.pdf"
        _build_pdf(analysis, pdf_path)
        print(f"Wrote {pdf_path}")

    if args.markdown:
        plots_dir = out / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        md_path = out / "report.md"
        md_path.write_text(_build_markdown(analysis, plots_dir))
        n_plots = len(list(plots_dir.glob("*.svg")))
        print(f"Wrote {md_path} ({n_plots} SVG plots in {plots_dir}/)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
