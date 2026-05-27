r"""Decoder comparison: same samples, multiple decoders, one table.

Generates DEM samples once per (distance, error_rate, rounds) point and
decodes them with every requested decoder. Produces an HTML report with
comparison tables showing logical error rates and decode throughput.

This is complementary to ``native_dem_threshold_sweep.py`` which produces
threshold curves. This script answers "which decoder is best at a given
operating point?" with a controlled experiment (identical samples).

Example:
    python examples/surface/decoder_comparison.py

    python examples/surface/decoder_comparison.py \\
        --distances 3 5 7 \\
        --error-rates 0.004 0.008 \\
        --shots 2000 \\
        --decoders pymatching tesseract bp_osd \\
        --open-html
"""

from __future__ import annotations

import argparse
import html
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pecos.qec.surface import NoiseModel


@dataclass
class DecoderResult:
    """Result for one decoder at one operating point."""

    decoder: str
    distance: int
    basis: str
    physical_error_rate: float
    num_rounds: int
    num_shots: int
    num_errors: int
    logical_error_rate: float
    decode_seconds: float
    shots_per_second: float
    per_shot_median: float = 0.0
    per_shot_p99: float = 0.0
    per_shot_max: float = 0.0


@dataclass
class ComparisonPoint:
    """All decoder results for one (distance, p, rounds) point."""

    distance: int
    basis: str
    physical_error_rate: float
    num_rounds: int
    num_shots: int
    sample_seconds: float
    results: list[DecoderResult]


def _build_sampler(
    distance: int,
    num_rounds: int,
    noise: NoiseModel,
    basis: str,
    circuit_source: str,
) -> tuple:
    """Build the native sampler and get DEM strings."""
    from pecos.qec.surface import SurfacePatch, build_native_sampler
    from pecos.qec.surface.decode import SurfaceDecoder, generate_circuit_level_dem_from_builder

    patch = SurfacePatch.create(distance=distance)
    sampler = build_native_sampler(
        patch,
        num_rounds,
        noise,
        basis=basis,
        circuit_source=circuit_source,
    )

    # Decomposed DEM for MWPM decoders
    dec = SurfaceDecoder(
        patch,
        num_rounds=num_rounds,
        noise=noise,
        decoder_type="pymatching",
        use_circuit_level_dem=True,
        circuit_level_dem_mode="native_decomposed",
        circuit_level_dem_source=circuit_source,
    )
    dem_decomp = dec.get_dem(basis.upper(), circuit_level=True)
    dem_decomp = "\n".join(line for line in dem_decomp.split("\n") if not line.startswith("logical_observable"))

    # Full DEM for non-MWPM decoders
    dem_full = generate_circuit_level_dem_from_builder(
        patch,
        num_rounds,
        noise,
        basis=basis,
        decompose_errors=False,
        circuit_source=circuit_source,
    )
    dem_full = "\n".join(line for line in dem_full.split("\n") if not line.startswith("logical_observable"))

    return sampler, dem_decomp, dem_full


# Decoders that need decomposed (graphlike) DEMs
# MWPM decoders get decomposed DEMs. DemMatchingGraph handles fault-ID-aware
# merging with first-observable-wins (matching PyMatching's INDEPENDENT strategy).
_MWPM_DECODERS = {
    "pymatching",
    "pymatching_uncorrelated",
    "fusion_blossom",
    "fusion_blossom_serial",
    "fusion_blossom_parallel",
    "pecos_uf",
    "pecos_uf_correlated",
    "pecos_uf:balanced",
    "pecos_uf:fast",
    "pecos_uf:bp",
    "windowed",
}

# All supported decoders
_ALL_DECODERS = [
    "pymatching",
    "pymatching_uncorrelated",
    "fusion_blossom",
    "fusion_blossom_serial",
    "fusion_blossom_parallel",
    "tesseract",
    "mwpf",
    "bp_osd",
    "union_find",
    "min_sum_bp",
    "relay_bp",
    "pecos_uf",
    "pecos_uf:balanced",
    "pecos_uf:bp",
    "pecos_uf_correlated",  # legacy alias for pecos_uf:balanced
]

# Decoders where parallel decoding helps
_SLOW_DECODERS = {"tesseract", "mwpf", "bp_osd", "relay_bp"}


def _decoder_base_name(name: str) -> str:
    """Extract base decoder name, stripping config suffix (e.g. 'mwpf:c=30' -> 'mwpf')."""
    return name.split(":", maxsplit=1)[0]


def run_comparison(
    *,
    distances: list[int],
    error_rates: list[float],
    decoders: list[str],
    basis: str,
    shots: int,
    seed: int,
    circuit_source: str,
    p1_scale: float,
    p_meas_scale: float,
    p_prep_scale: float,
) -> list[ComparisonPoint]:
    """Run the full comparison and return results."""
    from pecos.qec.surface import NoiseModel

    points: list[ComparisonPoint] = []
    total_configs = len(distances) * len(error_rates)
    config_idx = 0

    for distance in distances:
        num_rounds = 2 * distance
        for p in error_rates:
            config_idx += 1
            noise = NoiseModel(
                p1=p * p1_scale,
                p2=p,
                p_meas=p * p_meas_scale,
                p_prep=p * p_prep_scale,
            )

            print(f"[{config_idx}/{total_configs}] d={distance} p={p:.4g} r={num_rounds} ...")

            sampler, dem_decomp, dem_full = _build_sampler(
                distance,
                num_rounds,
                noise,
                basis,
                circuit_source,
            )

            # Generate samples once
            t0 = time.perf_counter()
            sample_batch = sampler.sampler.generate_samples(shots, seed=seed + config_idx)
            sample_seconds = time.perf_counter() - t0

            results: list[DecoderResult] = []
            for decoder_name in decoders:
                base = _decoder_base_name(decoder_name)
                # Ensemble uses decomposed DEMs (all ensemble members are matching-graph decoders)
                dem = dem_decomp if base in _MWPM_DECODERS or base == "ensemble" else dem_full

                if base in _SLOW_DECODERS:
                    stats = sample_batch.decode_stats_parallel(dem, decoder_name)
                else:
                    stats = sample_batch.decode_stats(dem, decoder_name)

                results.append(
                    DecoderResult(
                        decoder=decoder_name,
                        distance=distance,
                        basis=basis.upper(),
                        physical_error_rate=p,
                        num_rounds=num_rounds,
                        num_shots=shots,
                        num_errors=stats.num_errors,
                        logical_error_rate=stats.logical_error_rate,
                        decode_seconds=stats.total_seconds,
                        shots_per_second=shots / stats.total_seconds if stats.total_seconds > 0 else float("inf"),
                        per_shot_median=stats.per_shot_median,
                        per_shot_p99=stats.per_shot_p99,
                        per_shot_max=stats.per_shot_max,
                    ),
                )
                print(
                    f"    {decoder_name:14s}: {stats.num_errors:>4d}/{shots}  "
                    f"LER={stats.logical_error_rate:.4f}  "
                    f"mean={stats.per_shot_mean:.1e}s  "
                    f"median={stats.per_shot_median:.1e}s  "
                    f"p99={stats.per_shot_p99:.1e}s  "
                    f"max={stats.per_shot_max:.1e}s",
                )

            points.append(
                ComparisonPoint(
                    distance=distance,
                    basis=basis.upper(),
                    physical_error_rate=p,
                    num_rounds=num_rounds,
                    num_shots=shots,
                    sample_seconds=sample_seconds,
                    results=results,
                ),
            )

    return points


def write_json(path: Path, points: list[ComparisonPoint], config: dict) -> None:
    """Write results as JSON."""
    data = {
        "config": config,
        "points": [asdict(p) for p in points],
    }
    path.write_text(json.dumps(data, indent=2))
    print(f"Wrote JSON to {path}")


def write_html(path: Path, points: list[ComparisonPoint], config: dict) -> None:
    """Write an HTML report with comparison tables."""
    style = dedent(
        """
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
        table { border-collapse: collapse; width: 100%; margin: 12px 0 0; }
        th, td { padding: 10px 14px; text-align: right; border-bottom: 1px solid var(--table-border); }
        th {
          font-size: 0.82rem; text-transform: uppercase;
          letter-spacing: 0.03em; color: var(--muted); border-bottom-width: 2px;
        }
        td:first-child, th:first-child { text-align: left; }
        tr:nth-child(even) td { background: var(--table-stripe); }
        code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    """,
    ).strip()

    def meta_card(label: str, value: str) -> str:
        return f'      <div class="meta-card"><strong>{html.escape(label)}</strong>{html.escape(value)}</div>'

    decoders = config.get("decoders", [])
    p1s = config.get("p1_scale", 1 / 30)
    pms = config.get("p_meas_scale", 1 / 3)
    pps = config.get("p_prep_scale", 1 / 3)
    noise_str = f"p1={p1s:.4g}*p, p2=p, p_meas={pms:.4g}*p, p_prep={pps:.4g}*p"
    parts = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '  <meta charset="utf-8" />',
        '  <meta name="viewport" content="width=device-width, initial-scale=1" />',
        "  <title>PECOS Decoder Comparison</title>",
        f"  <style>{style}</style>",
        "</head>",
        "<body>",
        '<button id="theme-toggle" class="theme-toggle">Light / Dark</button>',
        "<main>",
        '  <section class="hero">',
        "    <h1>PECOS Decoder Comparison</h1>",
        "    <p>Same samples decoded by multiple decoders. LER differences reflect "
        "decoder quality, not sampling noise.</p>",
        '    <div class="meta">',
        meta_card("Decoders", ", ".join(decoders)),
        meta_card("Distances", ", ".join(str(d) for d in config.get("distances", []))),
        meta_card("Error Rates", ", ".join(f"{p:.4g}" for p in config.get("error_rates", []))),
        meta_card("Shots", str(config.get("shots", 0))),
        meta_card("Basis", config.get("basis", "Z")),
        meta_card("Noise Model", noise_str),
        meta_card("Circuit Source", config.get("circuit_source", "traced_qis")),
        "    </div>",
        "  </section>",
    ]

    # Group by (distance, p)
    for point in points:
        d = point.distance
        p = point.physical_error_rate
        r = point.num_rounds
        parts.append('  <div class="section">')
        parts.append(f"    <h2>d={d}, p={p:.4g}, rounds={r} ({point.num_shots} shots)</h2>")

        parts.append("    <table>")
        parts.append(
            "      <tr><th>Decoder</th><th>LER (95% CI)</th><th>Mean</th>"
            "<th>Median</th><th>p99</th><th>Max</th><th>Throughput</th></tr>",
        )
        for res in sorted(point.results, key=lambda r: r.logical_error_rate):
            mean_s = res.decode_seconds / res.num_shots if res.num_shots > 0 else 0
            sps = f"{res.shots_per_second:.1e}"
            ler = res.logical_error_rate
            n = res.num_shots
            z = 1.96
            if n > 0 and 0 < ler < 1:
                denom = 1 + z * z / n
                center = (ler + z * z / (2 * n)) / denom
                half = z * (ler * (1 - ler) / n + z * z / (4 * n * n)) ** 0.5 / denom
                ci_lo, ci_hi = max(0, center - half), min(1, center + half)
            elif n > 0:
                ci_lo, ci_hi = ler, ler
            else:
                ci_lo, ci_hi = 0, 0
            ler_str = f"{ler:.4f} ({ci_lo:.4f} - {ci_hi:.4f})"
            parts.append(
                f"      <tr>"
                f"<td>{html.escape(res.decoder)}</td>"
                f"<td>{ler_str}</td>"
                f"<td>{mean_s:.1e} s</td>"
                f"<td>{res.per_shot_median:.1e} s</td>"
                f"<td>{res.per_shot_p99:.1e} s</td>"
                f"<td>{res.per_shot_max:.1e} s</td>"
                f"<td>{sps} shots/s</td>"
                f"</tr>",
            )
        parts.append("    </table>")
        parts.append("  </div>")

    parts.extend(
        [
            "</main>",
            dedent(
                """
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
        """,
            ).strip(),
            "</body>",
            "</html>",
        ],
    )

    path.write_text("\n".join(parts))
    print(f"Wrote HTML to {path}")


def main() -> int:
    """CLI entry point for decoder comparison."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--distances", nargs="+", type=int, default=[3, 5])
    parser.add_argument("--error-rates", nargs="+", type=float, default=[0.004, 0.008])
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--basis", default="Z")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--decoders",
        nargs="+",
        default=["pymatching", "tesseract", "bp_osd"],
        help=f"Decoders to compare. Available: {', '.join(_ALL_DECODERS)}",
    )
    parser.add_argument("--circuit-source", default="traced_qis", choices=["traced_qis", "abstract"])
    parser.add_argument("--p1-scale", type=float, default=1.0 / 30.0)
    parser.add_argument("--p-meas-scale", type=float, default=1.0 / 3.0)
    parser.add_argument("--p-prep-scale", type=float, default=1.0 / 3.0)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--open-html", action="store_true")
    args = parser.parse_args()

    config = {
        "distances": sorted(args.distances),
        "error_rates": sorted(args.error_rates),
        "shots": args.shots,
        "basis": args.basis.upper(),
        "decoders": args.decoders,
        "circuit_source": args.circuit_source,
        "p1_scale": args.p1_scale,
        "p_meas_scale": args.p_meas_scale,
        "p_prep_scale": args.p_prep_scale,
    }

    print("PECOS Decoder Comparison")
    print("=" * 40)
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()

    t0 = time.perf_counter()
    points = run_comparison(
        distances=sorted(args.distances),
        error_rates=sorted(args.error_rates),
        decoders=args.decoders,
        basis=args.basis,
        shots=args.shots,
        seed=args.seed,
        circuit_source=args.circuit_source,
        p1_scale=args.p1_scale,
        p_meas_scale=args.p_meas_scale,
        p_prep_scale=args.p_prep_scale,
    )
    elapsed = time.perf_counter() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    if args.output_dir:
        out = Path(args.output_dir)
    else:
        import tempfile

        out = Path(tempfile.mkdtemp(prefix="pecos_decoder_comparison_"))

    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "decoder_comparison.json", points, config)
    html_path = out / "decoder_comparison.html"
    write_html(html_path, points, config)

    if args.open_html:
        import webbrowser

        webbrowser.open(html_path.as_uri())
        print(f"Opened {html_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
