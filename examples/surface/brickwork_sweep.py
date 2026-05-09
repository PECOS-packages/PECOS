r"""Brickwork circuit sweep: transversal gate performance across widths, depths, distances.

Generates mirrored brickwork circuits (H + CX) at various configurations,
decodes with multiple decoders, and writes JSON results that can be fed
to ``build_report.py`` for HTML/PDF reports.

The mirrored brickwork guarantees output = |0...0>, so logical errors
are unambiguous.

Decoder names:
    observable_subgraph:INNER  -- OSD with inner decoder (pymatching, pecos_uf:fast, etc.)
    logical_circuit:BUDGET:INNER  -- LogicalCircuitDecoder (unlimited, windowed, 10ms, etc.)
    logical_algorithm:INNER  -- LogicalAlgorithmDecoder (full-circuit, no budget)

Example:
    uv run python examples/surface/brickwork_sweep.py \
        --distances 3 5 --widths 2 3 4 --depths 1 2 3 \
        --error-rates 0.001 0.002 \
        --decoders observable_subgraph:pymatching \
        --shots 5000 --output-dir /tmp/brickwork_sweep

    uv run python examples/surface/brickwork_sweep.py \
        --distances 3 5 --widths 2 3 --depths 1 2 \
        --error-rates 0.001 \
        --decoders observable_subgraph:pymatching logical_circuit:windowed:pymatching \
        --shots 2000 --save-html --open
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python" / "quantum-pecos" / "src"))


# -- Data model ---------------------------------------------------------------


@dataclass
class DecoderResult:
    decoder: str
    num_errors: int
    logical_error_rate: float
    decode_seconds: float


@dataclass
class BrickworkPoint:
    distance: int
    width: int
    depth: int
    physical_error_rate: float
    num_shots: int
    seed: int
    sample_seconds: float
    decoder_results: list[DecoderResult] = field(default_factory=list)


@dataclass
class BrickworkShard:
    config: dict
    points: list[BrickworkPoint] = field(default_factory=list)
    total_seconds: float = 0.0


# -- Circuit builder -----------------------------------------------------------


def build_mirrored_brickwork(width, depth, seed, patch, rounds=2):
    """Build a mirrored brickwork circuit (identity, output = |0...0>)."""
    from pecos.qec.surface import LogicalCircuitBuilder

    nq = patch.geometry.num_data + patch.geometry.num_ancilla
    rng = random.Random(seed)

    b = LogicalCircuitBuilder()
    labels = [f"Q{i}" for i in range(width)]
    for i, label in enumerate(labels):
        b.add_patch(patch, label, qubit_offset=i * nq)

    eff = dict.fromkeys(labels, "Z")
    b.add_memory(labels, rounds, "Z")

    ops_forward = []
    for layer in range(depth):
        layer_ops = []
        for label in labels:
            if rng.random() < 0.5:
                b.add_transversal_h(label)
                eff[label] = "X" if eff[label] == "Z" else "Z"
                layer_ops.append(("H", label))
        b.add_memory(labels, rounds, basis={label: eff[label] for label in labels})

        offset = layer % 2
        cx_applied = []
        for i in range(offset, width - 1, 2):
            ctrl, tgt = labels[i], labels[i + 1]
            if eff[ctrl] == eff[tgt]:
                b.add_transversal_cx(ctrl, tgt)
                cx_applied.append((ctrl, tgt))
        if cx_applied:
            b.add_memory(labels, rounds, basis={label: eff[label] for label in labels})
            layer_ops.append(("CX", cx_applied))
        ops_forward.append(layer_ops)

    # Mirror
    for layer_ops in reversed(ops_forward):
        for op_type, *args in reversed(layer_ops):
            if op_type == "CX":
                for ctrl, tgt in reversed(args[0]):
                    if eff[ctrl] == eff[tgt]:
                        b.add_transversal_cx(ctrl, tgt)
                b.add_memory(labels, rounds, basis={label: eff[label] for label in labels})
        for op_type, *args in reversed(layer_ops):
            if op_type == "H":
                label = args[0]
                b.add_transversal_h(label)
                eff[label] = "X" if eff[label] == "Z" else "Z"
        b.add_memory(labels, rounds, basis={label: eff[label] for label in labels})

    return b


def build_t_injection_circuit(distance, _seed, patch, rounds_per_layer):
    """Build a T-gate injection circuit: memory + T injection + memory."""
    from pecos.qec.surface import LogicalCircuitBuilder

    nq = patch.geometry.num_data + patch.geometry.num_ancilla
    b = LogicalCircuitBuilder()
    b.add_patch(patch, "D", qubit_offset=0)
    b.add_patch(patch, "A", qubit_offset=nq)

    # Memory → T injection → Memory
    rounds = max(rounds_per_layer, distance)
    b.add_memory(["D", "A"], rounds=rounds, basis="Z")
    b.add_t_via_injection("D", "A", rounds_before=rounds, rounds_after=rounds)
    return b


# -- Sweep --------------------------------------------------------------------


def run_sweep(
    *,
    distances: list[int],
    widths: list[int],
    depths: list[int],
    error_rates: list[float],
    decoders: list[str],
    shots: int,
    circuit_seed: int,
    rounds_per_layer: int,
) -> BrickworkShard:
    """Run the full brickwork sweep."""
    from pecos.qec.surface import SurfacePatch
    from pecos_rslib.qec import ObservableSubgraphDecoder, ParsedDem

    config = {
        "distances": distances,
        "widths": widths,
        "depths": depths,
        "error_rates": error_rates,
        "decoders": decoders,
        "shots": shots,
        "circuit_seed": circuit_seed,
        "rounds_per_layer": rounds_per_layer,
    }

    shard = BrickworkShard(config=config)
    t_total = time.perf_counter()

    total_cells = len(distances) * len(widths) * len(depths) * len(error_rates)
    cell_idx = 0

    for d in distances:
        patch = SurfacePatch.create(distance=d)
        for w in widths:
            for depth in depths:
                # Build circuit once per (d, w, depth) — reuse across error rates
                b = build_mirrored_brickwork(w, depth, circuit_seed, patch, rounds_per_layer)
                sc = b.stab_coords()

                for p in error_rates:
                    cell_idx += 1
                    print(f"[{cell_idx}/{total_cells}] d={d} w={w} depth={depth} p={p:.4g} ...", end=" ", flush=True)

                    dem_str = b.build_dem(p1=p, p2=p, p_meas=p, p_prep=p)

                    # Sample
                    t0 = time.perf_counter()
                    parsed = ParsedDem.from_string(dem_str)
                    rust_sampler = parsed.to_dem_sampler()
                    batch = rust_sampler.generate_samples(shots, seed=circuit_seed + cell_idx)
                    sample_sec = time.perf_counter() - t0

                    point = BrickworkPoint(
                        distance=d,
                        width=w,
                        depth=depth,
                        physical_error_rate=p,
                        num_shots=shots,
                        seed=circuit_seed,
                        sample_seconds=sample_sec,
                    )

                    # Decode with each decoder
                    for decoder_name in decoders:
                        t0 = time.perf_counter()
                        if decoder_name.startswith("logical_circuit"):
                            from pecos_rslib.qec import LogicalCircuitDecoder

                            # Format: logical_circuit:budget:inner
                            parts = decoder_name.split(":")
                            budget = parts[1] if len(parts) > 1 else "offline"
                            inner = parts[2] if len(parts) > 2 else "pymatching"
                            desc = b.build_algorithm_descriptor(p1=p, p2=p, p_meas=p, p_prep=p)
                            dec = LogicalCircuitDecoder(desc, budget, inner)
                            errors = dec.decode_count(batch)
                        elif decoder_name.startswith("logical_algorithm"):
                            from pecos_rslib.qec import LogicalAlgorithmDecoder

                            parts = decoder_name.split(":", 1)
                            inner = parts[1] if len(parts) > 1 else "pymatching"
                            desc = b.build_algorithm_descriptor(p1=p, p2=p, p_meas=p, p_prep=p)
                            algo = LogicalAlgorithmDecoder(desc, inner)
                            errors = algo.decode_count(batch)
                        elif decoder_name.startswith("observable_subgraph"):
                            parts = decoder_name.split(":", 1)
                            inner = parts[1] if len(parts) > 1 else "pymatching"
                            osd = ObservableSubgraphDecoder(dem_str, sc, inner)
                            # Use parallel decode for large shot counts
                            if shots >= 5000:
                                errors = osd.decode_count_parallel(batch, dem_str, sc, inner)
                            else:
                                errors = osd.decode_count(batch)
                        else:
                            msg = (
                                f"Unknown decoder: '{decoder_name}'. "
                                f"Supported: observable_subgraph:INNER, "
                                f"logical_circuit:BUDGET:INNER, "
                                f"logical_algorithm:INNER"
                            )
                            raise ValueError(msg)
                        dec_sec = time.perf_counter() - t0
                        ler = errors / shots

                        point.decoder_results.append(
                            DecoderResult(
                                decoder=decoder_name,
                                num_errors=errors,
                                logical_error_rate=ler,
                                decode_seconds=dec_sec,
                            ),
                        )

                    shard.points.append(point)
                    lers = {r.decoder: f"{r.logical_error_rate:.5f}" for r in point.decoder_results}
                    print(f"LER={lers}")

    shard.total_seconds = time.perf_counter() - t_total
    return shard


# -- HTML report ---------------------------------------------------------------


def write_html_report(shard: BrickworkShard, path: Path, coherent_results=None) -> None:
    """Write an HTML report from the sweep results."""
    from collections import defaultdict

    # Group by (width, depth) for each distance
    tables = defaultdict(list)
    for pt in shard.points:
        tables[(pt.distance, pt.physical_error_rate)].append(pt)

    style = """
        :root {
          color-scheme: light dark;
          --bg: #f8fafc; --fg: #0f172a;
          --hero-bg: linear-gradient(135deg, #e0f2fe, #f8fafc 55%, #dcfce7);
          --card-bg: white; --card-border: #dbeafe; --card-shadow: rgba(15,23,42,0.05);
          --meta-bg: rgba(255,255,255,0.82);
          --muted: #475569; --link: #2563eb;
          --table-stripe: #f1f5f9; --table-border: #e2e8f0;
          --good: #16a34a; --warn: #ea580c; --bad: #dc2626;
        }
        @media (prefers-color-scheme: dark) {
          :root {
            --bg: #0f172a; --fg: #e2e8f0;
            --hero-bg: linear-gradient(135deg, #1e293b, #0f172a 55%, #1a2e1a);
            --card-bg: #1e293b; --card-border: #334155; --card-shadow: rgba(0,0,0,0.3);
            --meta-bg: rgba(30,41,59,0.82);
            --muted: #94a3b8; --link: #60a5fa;
            --table-stripe: #0f172a; --table-border: #334155;
            --good: #4ade80; --warn: #fb923c; --bad: #f87171;
          }
        }
        body {
          margin: 0;
          font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, sans-serif;
          background: var(--bg); color: var(--fg);
        }
        main { max-width: 1100px; margin: 0 auto; padding: 32px 24px 56px; }
        h1, h2, h3, p { margin-top: 0; }
        .hero {
          background: var(--hero-bg);
          border: 1px solid var(--card-border);
          border-radius: 20px;
          padding: 28px 32px;
          margin-bottom: 24px;
        }
        .hero h1 { font-size: 1.6rem; margin-bottom: 0.3em; }
        .hero p { color: var(--muted); margin: 0; }
        .meta {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
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
          display: block; font-size: 0.78rem; text-transform: uppercase;
          letter-spacing: 0.04em; color: var(--muted); margin-bottom: 4px;
        }
        .section {
          background: var(--card-bg);
          border: 1px solid var(--card-border);
          border-radius: 18px;
          padding: 22px 26px;
          margin-top: 20px;
          box-shadow: 0 10px 24px var(--card-shadow);
          overflow-x: auto;
        }
        .section h2 { font-size: 1.15rem; margin-bottom: 0.3em; }
        .section h3 { font-size: 0.95rem; color: var(--muted); }
        .section p { font-size: 0.9rem; color: var(--muted); }
        table { border-collapse: collapse; width: 100%; margin: 12px 0 0; }
        th, td { padding: 10px 14px; text-align: right; border-bottom: 1px solid var(--table-border); }
        th {
          font-size: 0.78rem; text-transform: uppercase;
          letter-spacing: 0.03em; color: var(--muted); border-bottom-width: 2px;
        }
        td:first-child, th:first-child { text-align: left; }
        tr:nth-child(even) td { background: var(--table-stripe); }
        .good { color: var(--good); font-weight: 600; }
        .warn { color: var(--warn); font-weight: 600; }
        .bad { color: var(--bad); font-weight: 600; }
        code {
          font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
          font-size: 0.85em; background: var(--table-stripe);
          padding: 0.15em 0.4em; border-radius: 3px;
        }
        details.info {
          background: var(--card-bg);
          border: 1px solid var(--card-border);
          border-radius: 18px;
          margin-top: 20px;
          box-shadow: 0 10px 24px var(--card-shadow);
        }
        details.info > summary {
          cursor: pointer; font-size: 0.95rem; font-weight: 600;
          padding: 16px 26px; list-style: none;
        }
        details.info > summary::before { content: "\\25B6\\00a0\\00a0"; font-size: 0.75em; }
        details.info[open] > summary::before { content: "\\25BC\\00a0\\00a0"; }
        details.info .content {
          padding: 0 26px 22px; line-height: 1.7; font-size: 0.9rem;
        }
        details.info .content h4 { margin: 1em 0 0.3em; }
    """

    # Build meta cards
    distances = sorted({pt.distance for pt in shard.points}) if shard.points else []
    decoders_used = sorted({r.decoder for pt in shard.points for r in pt.decoder_results})

    meta_cards = []
    if distances:
        meta_cards.append(
            f'<div class="meta-card"><strong>Distances</strong>{", ".join(str(d) for d in distances)}</div>',
        )
    if decoders_used:
        meta_cards.append(f'<div class="meta-card"><strong>Decoders</strong>{len(decoders_used)}</div>')
    if shard.points:
        meta_cards.append(f'<div class="meta-card"><strong>Shots</strong>{shard.points[0].num_shots:,}</div>')
    meta_cards.append(f'<div class="meta-card"><strong>Time</strong>{shard.total_seconds:.1f}s</div>')

    html_parts = [
        "<!doctype html>",
        '<html lang="en"><head>',
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>PECOS Surface Code Report</title>",
        f"<style>{style}</style>",
        "</head><body><main>",
        '<section class="hero">',
        "<h1>PECOS Surface Code Report</h1>",
        "<p>Mirrored brickwork circuits, T-gate injection, and coherent noise analysis</p>",
        f'<div class="meta">{" ".join(meta_cards)}</div>',
        "</section>",
        "",
        '<details class="info"><summary>Decoding Concepts: Throughput, Reaction Time, and Budgets</summary>',
        '<div class="content">',
        "<h4>Throughput (avoiding backlog)</h4>",
        "<p>The decoder must process syndrome data at least as fast as the hardware "
        "generates it. If syndrome extraction takes T<sub>cycle</sub> per round, the "
        "decoder must process each round in &le; T<sub>cycle</sub> on average. If it "
        "falls behind, a <em>backlog</em> accumulates, causing exponential slowdown of "
        "the quantum computation.</p>",
        "",
        "<h4>Reaction time</h4>",
        "<p>At feed-forward decision points (T-gate injection, magic state consumption), "
        "the physical hardware waits for the decoder to produce a correction. The "
        "<em>reaction time</em> is the time available between the last syndrome arriving "
        "and the correction being applied. For <strong>Clifford-only circuits</strong> "
        "(like these brickwork circuits), there are no mid-circuit decisions &mdash; the "
        "Pauli frame is metadata applied at the final measurement. The reaction time is "
        "effectively unlimited.</p>",
        "",
        "<h4>Budget strategies</h4>",
        "<p>The decoder framework selects a strategy based on the user-specified reaction time budget:</p>",
        "<ul>",
        "<li><code>unlimited</code> &mdash; Full-circuit OSD. Maximum accuracy. "
        "Appropriate for Clifford circuits or offline analysis.</li>",
        "<li><code>windowed</code> / <code>10ms</code> &mdash; Windowed OSD with "
        "overlap buffers inside each per-observable subgraph. Bounded latency, full "
        "accuracy with sufficient overlap.</li>",
        "<li><code>100us</code> / <code>1us</code> &mdash; Tight budget. Windowed "
        "without overlap. Accuracy degrades; requires advanced techniques (ghost "
        "protocol) for improvement.</li>",
        "</ul>",
        "",
        "<h4>Compute hardware</h4>",
        "<p>The budget is a <em>constraint</em>, not a measurement. A decode that takes "
        "300&mu;s on a CPU might take 30&mu;s on an FPGA. The strategy doesn't change "
        "&mdash; only whether it fits within the budget on a given compute platform. "
        "Profiling on the target hardware determines feasibility.</p>",
        "",
        "<h4>References</h4>",
        "<ul>",
        "<li>Riverlane + Rigetti (arXiv:2410.05202): 9.6&mu;s response time, backlog definition</li>",
        "<li>Cain et al. (arXiv:2505.13587): software commitment, &lt;100&mu;s at d=25</li>",
        "<li>Turner et al. (arXiv:2505.23567): ghost protocol for scalable windowed decoding</li>",
        "<li>Serra-Peralta et al. (arXiv:2505.13599): per-observable subgraph MWPM</li>",
        "</ul>",
        "</div></details>",
        "",
    ]

    # Separate brickwork and T-injection points
    brickwork_tables = defaultdict(list)
    t_injection_tables = defaultdict(list)
    for pt in shard.points:
        if pt.depth == 0:  # T-injection marker
            t_injection_tables[(pt.distance, pt.physical_error_rate)].append(pt)
        else:
            brickwork_tables[(pt.distance, pt.physical_error_rate)].append(pt)

    if brickwork_tables:
        html_parts.append('<section class="section">')
        html_parts.append("<h2>Brickwork Circuits (Clifford)</h2>")
        html_parts.append(
            "<p>Mirrored random gate sequences (identity operation). "
            "LER from stochastic depolarizing noise only.</p>",
        )

    for (d, p), points in sorted(brickwork_tables.items()):
        html_parts.append(f"<h3>d={d}, p={p}</h3>")

        decoders = sorted({r.decoder for pt in points for r in pt.decoder_results})
        html_parts.append("<table><tr><th>Width</th><th>Depth</th>")
        html_parts.extend(f"<th>{dec}</th>" for dec in decoders)
        html_parts.append("</tr>")

        for pt in sorted(points, key=lambda x: (x.width, x.depth)):
            html_parts.append(f"<tr><td>{pt.width}</td><td>{pt.depth}</td>")
            for dec in decoders:
                r = next((r for r in pt.decoder_results if r.decoder == dec), None)
                if r:
                    cls = "good" if r.logical_error_rate < 0.01 else "warn" if r.logical_error_rate < 0.05 else "bad"
                    html_parts.append(
                        f'<td class="{cls}">{r.logical_error_rate:.5f} ({r.decode_seconds:.2f}s)</td>',
                    )
                else:
                    html_parts.append("<td>-</td>")
            html_parts.append("</tr>")
        html_parts.append("</table>")

    if brickwork_tables:
        html_parts.append("</section>")

    if t_injection_tables:
        html_parts.append('<section class="section">')
        html_parts.append("<h2>T-Gate Injection (Non-Clifford)</h2>")
        html_parts.append(
            "<p>T gate via magic state teleportation: "
            "|T&rang; ancilla + CX + measure + conditional S. "
            "Feed-forward decision point for the decoder.</p>",
        )

    for (d, p), points in sorted(t_injection_tables.items()):
        decoders = sorted({r.decoder for pt in points for r in pt.decoder_results})
        html_parts.append(f"<h3>d={d}, p={p}</h3>")
        html_parts.append("<table><tr><th>Circuit</th>")
        html_parts.extend(f"<th>{dec}</th>" for dec in decoders)
        html_parts.append("</tr>")

        for pt in points:
            html_parts.append("<tr><td>T-injection</td>")
            for dec in decoders:
                r = next((r for r in pt.decoder_results if r.decoder == dec), None)
                if r:
                    cls = "good" if r.logical_error_rate < 0.01 else "warn" if r.logical_error_rate < 0.05 else "bad"
                    html_parts.append(
                        f'<td class="{cls}">{r.logical_error_rate:.5f} ({r.decode_seconds:.2f}s)</td>',
                    )
                else:
                    html_parts.append("<td>-</td>")
            html_parts.append("</tr>")
        html_parts.append("</table>")

    if t_injection_tables:
        html_parts.append("</section>")

    # Coherent noise section
    if coherent_results:
        html_parts.append('<section class="section">')
        html_parts.append("<h2>Coherent Idle Noise (X-basis Memory)</h2>")
        html_parts.append(
            "<p>RZ(&theta;) rotation on both qubits after each CX gate models "
            "uncompensated phase accumulation during idle time. Unlike stochastic "
            "Z errors, coherent rotations accumulate constructively &mdash; the LER "
            "far exceeds the Pauli-twirled equivalent sin&sup2;(&theta;/2). "
            "Decoder uses a stochastic-only DEM. Simulated with StateVec.</p>",
        )

        for (d, p), result in sorted(coherent_results.items()):
            html_parts.append(f"<h3>d={d}, p_depol={p}</h3>")
            html_parts.append(
                "<table><tr>"
                "<th style='text-align:left'>&theta; (rad)</th>"
                "<th>sin&sup2;(&theta;/2)</th>"
                "<th>LER</th>"
                "<th>&plusmn; SE</th>"
                "<th>Errors</th>"
                "<th>vs baseline</th>"
                "</tr>",
            )
            for pt in result.points:
                cls = "good" if pt.ler < 0.02 else "warn" if pt.ler < 0.05 else "bad"
                amplification = (
                    f"{pt.ler / result.points[0].ler:.1f}x" if result.points[0].ler > 0 and pt.p_idle > 0 else ""
                )
                html_parts.append(
                    f'<tr><td style="text-align:left">{pt.p_idle:.3f}</td>'
                    f"<td>{pt.p_twirled:.5f}</td>"
                    f'<td class="{cls}">{pt.ler:.4f}</td>'
                    f"<td>{pt.standard_error:.4f}</td>"
                    f"<td>{pt.errors}/{pt.shots}</td>"
                    f"<td>{amplification}</td></tr>",
                )
            html_parts.append("</table>")

        html_parts.append("</section>")

    html_parts.append("</main></body></html>")
    path.write_text("\n".join(html_parts))
    print(f"Report written to {path}")


# -- CLI -----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--distances", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--widths", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--scaled-depth",
        action="store_true",
        help="Override --depths: set depth=2^((d+1)/2) per distance",
    )
    parser.add_argument("--error-rates", type=float, nargs="+", default=[0.001])
    parser.add_argument("--decoders", nargs="+", default=["observable_subgraph:pymatching"])
    parser.add_argument("--shots", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds-per-layer", type=int, default=2)
    parser.add_argument(
        "--include-t-injection",
        action="store_true",
        help="Include T-gate injection circuits (non-Clifford)",
    )
    parser.add_argument(
        "--t-injection-only",
        action="store_true",
        help="Only T-gate injection circuits (skip brickwork)",
    )
    parser.add_argument(
        "--include-coherent-noise",
        action="store_true",
        help="Include coherent idle noise sweep (RZ after CX)",
    )
    parser.add_argument(
        "--coherent-p-idle",
        type=float,
        nargs="+",
        default=[0.0, 0.01, 0.03, 0.05, 0.07, 0.1],
        help="Coherent idle RZ angles to sweep",
    )
    parser.add_argument("--coherent-shots", type=int, default=None, help="Shots for coherent noise (default: --shots)")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/brickwork_sweep"))
    parser.add_argument("--save-json", action="store_true")
    parser.add_argument("--save-html", action="store_true")
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()

    if args.t_injection_only:
        shard = BrickworkShard(
            config={
                "distances": args.distances,
                "error_rates": args.error_rates,
                "decoders": args.decoders,
                "shots": args.shots,
                "t_injection_only": True,
            },
        )
    elif args.scaled_depth:
        # Per-distance depth: 2^((d+1)/2) — challenges the decoder proportionally
        # d=3→4, d=5→8, d=7→16, d=9→32
        depths = [int(2 ** ((d + 1) / 2)) for d in args.distances]
        print(f"Scaled depths: {dict(zip(args.distances, depths, strict=False))}")

        all_points = []
        for d, depth in zip(args.distances, depths, strict=False):
            partial = run_sweep(
                distances=[d],
                widths=args.widths,
                depths=[depth],
                error_rates=args.error_rates,
                decoders=args.decoders,
                shots=args.shots,
                circuit_seed=args.seed,
                rounds_per_layer=args.rounds_per_layer,
            )
            all_points.extend(partial.points)
        shard = BrickworkShard(
            config={
                "distances": args.distances,
                "widths": args.widths,
                "scaled_depths": dict(zip(args.distances, depths, strict=False)),
                "error_rates": args.error_rates,
                "decoders": args.decoders,
                "shots": args.shots,
            },
            points=all_points,
        )
    else:
        shard = run_sweep(
            distances=args.distances,
            widths=args.widths,
            depths=args.depths,
            error_rates=args.error_rates,
            decoders=args.decoders,
            shots=args.shots,
            circuit_seed=args.seed,
            rounds_per_layer=args.rounds_per_layer,
        )

    # T-injection circuits
    if args.include_t_injection or args.t_injection_only:
        from pecos.qec.surface import SurfacePatch
        from pecos_rslib.qec import ObservableSubgraphDecoder, ParsedDem

        print("\n--- T-Injection Circuits ---")
        for d in args.distances:
            patch = SurfacePatch.create(distance=d)
            for p in args.error_rates:
                b = build_t_injection_circuit(d, args.seed, patch, args.rounds_per_layer)
                sc = b.stab_coords()
                dem_str = b.build_dem(p1=p, p2=p, p_meas=p, p_prep=p)
                parsed = ParsedDem.from_string(dem_str)
                batch = parsed.to_dem_sampler().generate_samples(args.shots, seed=args.seed)

                point = BrickworkPoint(
                    distance=d,
                    width=2,
                    depth=0,  # depth=0 signals T-injection
                    physical_error_rate=p,
                    num_shots=args.shots,
                    seed=args.seed,
                    sample_seconds=0,
                )

                for decoder_name in args.decoders:
                    t0 = time.perf_counter()
                    if decoder_name.startswith("logical_circuit"):
                        from pecos_rslib.qec import LogicalCircuitDecoder

                        parts = decoder_name.split(":")
                        budget = parts[1] if len(parts) > 1 else "unlimited"
                        inner = parts[2] if len(parts) > 2 else "pymatching"
                        desc = b.build_algorithm_descriptor(p1=p, p2=p, p_meas=p, p_prep=p)
                        dec = LogicalCircuitDecoder(desc, budget, inner)
                        errors = dec.decode_count(batch)
                    elif decoder_name.startswith("logical_algorithm"):
                        from pecos_rslib.qec import LogicalAlgorithmDecoder

                        parts = decoder_name.split(":", 1)
                        inner = parts[1] if len(parts) > 1 else "pymatching"
                        desc = b.build_algorithm_descriptor(p1=p, p2=p, p_meas=p, p_prep=p)
                        algo = LogicalAlgorithmDecoder(desc, inner)
                        errors = algo.decode_count(batch)
                    elif decoder_name.startswith("observable_subgraph"):
                        parts = decoder_name.split(":", 1)
                        inner = parts[1] if len(parts) > 1 else "pymatching"
                        osd = ObservableSubgraphDecoder(dem_str, sc, inner)
                        errors = osd.decode_count(batch)
                    else:
                        msg = (
                            f"Unknown decoder: '{decoder_name}'. "
                            f"Supported: observable_subgraph:INNER, "
                            f"logical_circuit:BUDGET:INNER, "
                            f"logical_algorithm:INNER"
                        )
                        raise ValueError(msg)
                    dec_sec = time.perf_counter() - t0
                    ler = errors / args.shots

                    point.decoder_results.append(
                        DecoderResult(
                            decoder=decoder_name,
                            num_errors=errors,
                            logical_error_rate=ler,
                            decode_seconds=dec_sec,
                        ),
                    )

                shard.points.append(point)
                lers = {r.decoder: f"{r.logical_error_rate:.5f}" for r in point.decoder_results}
                print(f"d={d} T-injection p={p:.4g} ... LER={lers}")

    # Coherent noise sweep
    coherent_results = None
    if args.include_coherent_noise:
        from coherent_noise_sweep import run_sweep as run_coherent_sweep

        coherent_shots = args.coherent_shots or args.shots
        # StateVec is limited to d=3 (17 qubits). Skip larger distances.
        coherent_distances = [d for d in args.distances if d <= 3]
        if not coherent_distances:
            print("\n--- Coherent Idle Noise: skipped (StateVec limited to d<=3) ---")
        else:
            print("\n--- Coherent Idle Noise ---")
            coherent_results = {}
            for d in coherent_distances:
                for p in args.error_rates:
                    print(f"d={d} p={p:.4g}:")
                    result = run_coherent_sweep(
                        distance=d,
                        rounds=d,
                        basis="X",
                        p_depol=p,
                        p_idle_values=args.coherent_p_idle,
                        shots=coherent_shots,
                        seed=args.seed,
                        backend="statevec",
                        lazy_measure=True,
                        max_bond_dim=128,
                    )
                    coherent_results[(d, p)] = result

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.save_json:
        json_path = args.output_dir / "brickwork_sweep_results.json"
        json_path.write_text(json.dumps(asdict(shard), indent=2))
        print(f"JSON written to {json_path}")

    if args.save_html or args.open:
        html_path = args.output_dir / "brickwork_sweep_report.html"
        write_html_report(shard, html_path, coherent_results=coherent_results)
        if args.open:
            import webbrowser

            webbrowser.open(str(html_path))


if __name__ == "__main__":
    main()
