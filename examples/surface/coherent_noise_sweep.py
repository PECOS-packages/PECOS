r"""Coherent idle noise sweep: surface code memory with RZ phase accumulation.

Studies the impact of coherent Z-phase noise (RZ rotation after each CX gate)
on surface code memory experiments. Uses sim_neo for Rust-native simulation
with composable noise model (depolarizing + coherent idle RZ).

The coherent noise models uncompensated phase accumulation during idle time
between gates. Unlike stochastic Z errors (which scale as p_z per gate),
coherent RZ rotations accumulate constructively, causing error rates far
higher than the Pauli-twirled equivalent sin²(θ/2).

Example:
    uv run python examples/surface/coherent_noise_sweep.py \
        --distance 3 --rounds 3 --basis X \
        --p-depol 0.003 \
        --p-idle 0.0 0.01 0.03 0.05 0.07 0.1 \
        --shots 10000 --save-html --open
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python" / "quantum-pecos" / "src"))


@dataclass
class CoherentNoisePoint:
    p_idle: float
    p_twirled: float
    ler: float
    errors: int
    shots: int
    standard_error: float
    sim_seconds: float
    decode_seconds: float


@dataclass
class CoherentNoiseSweep:
    distance: int
    rounds: int
    basis: str
    p_depol: float
    backend: str
    points: list[CoherentNoisePoint] = field(default_factory=list)
    total_seconds: float = 0.0


def run_sweep(
    *,
    distance: int,
    rounds: int,
    basis: str,
    p_depol: float,
    p_idle_values: list[float],
    shots: int,
    seed: int,
    backend: str,
    lazy_measure: bool,
    max_bond_dim: int,
) -> CoherentNoiseSweep:
    """Run a coherent noise sweep using sim_neo."""
    from pecos.qec.surface import LogicalCircuitBuilder, SurfacePatch
    from pecos_rslib.qec import ObservableSubgraphDecoder
    from pecos_rslib_exp import depolarizing, sim_neo, stab_mps, statevec

    patch = SurfacePatch.create(distance=distance)
    b = LogicalCircuitBuilder()
    b.add_patch(patch, "Q0")
    b.add_memory("Q0", rounds=rounds, basis=basis)
    tc = b.to_tick_circuit()

    det_json = json.loads(tc.get_meta("detectors"))
    obs_json = json.loads(tc.get_meta("observables"))
    num_meas = int(tc.get_meta("num_measurements"))
    dem_str = b.build_dem(p1=p_depol, p2=p_depol, p_meas=p_depol, p_prep=p_depol)
    sc = b.stab_coords()
    osd = ObservableSubgraphDecoder(dem_str, sc, "pymatching")

    sweep = CoherentNoiseSweep(
        distance=distance,
        rounds=rounds,
        basis=basis,
        p_depol=p_depol,
        backend=backend,
    )
    t_total = time.perf_counter()

    for p_idle in p_idle_values:
        # Simulate
        t0 = time.perf_counter()
        noise = depolarizing().p1(p_depol).p2(p_depol).p_meas(p_depol).p_prep(p_depol).idle_rz(p_idle)
        builder = sim_neo(tc).noise(noise).shots(shots).seed(seed)
        if backend == "stabmps":
            builder = builder.quantum(
                (
                    stab_mps().lazy_measure().max_bond_dim(max_bond_dim)
                    if lazy_measure
                    else stab_mps().max_bond_dim(max_bond_dim)
                ),
            )
        else:
            builder = builder.quantum(statevec())
        results = builder.run()
        sim_time = time.perf_counter() - t0

        # Decode
        t0 = time.perf_counter()
        errors = 0
        for r in results:
            meas = list(r)
            det_events = []
            for det in det_json:
                val = 0
                for rec in det["records"]:
                    idx = num_meas + rec
                    if 0 <= idx < len(meas):
                        val ^= meas[idx]
                det_events.append(val)
            obs_mask = 0
            for obs in obs_json:
                val = 0
                for rec in obs["records"]:
                    idx = num_meas + rec
                    if 0 <= idx < len(meas):
                        val ^= meas[idx]
                if val:
                    obs_mask |= 1 << obs["id"]
            pred = osd.decode([int(x) for x in det_events])
            if pred != obs_mask:
                errors += 1
        decode_time = time.perf_counter() - t0

        ler = errors / shots
        se = math.sqrt(ler * (1 - ler) / shots) if shots > 0 else 0
        p_twirled = math.sin(p_idle / 2) ** 2

        point = CoherentNoisePoint(
            p_idle=p_idle,
            p_twirled=p_twirled,
            ler=ler,
            errors=errors,
            shots=shots,
            standard_error=se,
            sim_seconds=sim_time,
            decode_seconds=decode_time,
        )
        sweep.points.append(point)
        print(
            f"  p_idle={p_idle:.3f}  sin²(θ/2)={p_twirled:.5f}  "
            f"LER={ler:.4f} ± {se:.4f}  ({errors}/{shots})  "
            f"sim={sim_time:.1f}s  decode={decode_time:.1f}s",
            flush=True,
        )

    sweep.total_seconds = time.perf_counter() - t_total
    return sweep


def write_html_report(sweep: CoherentNoiseSweep, path: Path) -> None:
    """Write an HTML report from the sweep results."""
    html_parts = [
        "<!DOCTYPE html><html><head>",
        "<title>Coherent Idle Noise Sweep</title>",
        "<style>",
        "body { font-family: sans-serif; margin: 2em; max-width: 900px; }",
        "table { border-collapse: collapse; margin: 1em 0; }",
        "th, td { border: 1px solid #ccc; padding: 6px 12px; text-align: right; }",
        "th { background: #f5f5f5; }",
        ".good { color: green; } .bad { color: red; }",
        "details { background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px;"
        " padding: 0.5em 1em; margin: 1em 0; }",
        "details summary { cursor: pointer; font-weight: bold; padding: 0.5em 0; }",
        "details .content { padding: 0.5em 0; line-height: 1.6; }",
        "</style></head><body>",
        f"<h1>Coherent Idle Noise: {sweep.basis}-basis Memory d={sweep.distance}</h1>",
        f"<p>Depolarizing: p={sweep.p_depol}, rounds={sweep.rounds}, backend={sweep.backend}</p>",
        f"<p>Total time: {sweep.total_seconds:.1f}s</p>",
        "",
        "<details><summary>About Coherent Idle Noise</summary>",
        '<div class="content">',
        "<p>After each two-qubit gate (CX), an RZ(&theta;) rotation is applied to both "
        "qubits, modeling uncompensated phase accumulation during idle time. Unlike "
        "stochastic Z errors, coherent rotations accumulate constructively across gates, "
        "causing logical error rates far higher than the Pauli-twirled equivalent "
        "sin&sup2;(&theta;/2).</p>",
        "<p>The decoder uses a stochastic-only DEM (no knowledge of coherent noise). "
        "The gap between the coherent LER and the twirled LER measures how much the "
        "decoder is mismatched to the actual noise.</p>",
        "</div></details>",
        "",
        "<h2>Results</h2>",
        "<table><tr>",
        "<th>&theta; (rad)</th>",
        "<th>sin&sup2;(&theta;/2)</th>",
        "<th>LER</th>",
        "<th>&plusmn; SE</th>",
        "<th>Errors</th>",
        "<th>Shots</th>",
        "</tr>",
    ]

    for pt in sweep.points:
        cls = "good" if pt.ler < 0.05 else "bad"
        html_parts.append(
            f"<tr><td>{pt.p_idle:.3f}</td>"
            f"<td>{pt.p_twirled:.5f}</td>"
            f'<td class="{cls}">{pt.ler:.4f}</td>'
            f"<td>{pt.standard_error:.4f}</td>"
            f"<td>{pt.errors}</td>"
            f"<td>{pt.shots}</td></tr>",
        )

    html_parts.append("</table>")

    # Amplification factor
    if len(sweep.points) >= 2 and sweep.points[0].ler > 0:
        baseline = sweep.points[0].ler
        html_parts.append("<h2>Coherent Amplification</h2>")
        html_parts.append("<table><tr><th>&theta;</th><th>LER / baseline</th><th>LER / twirled</th></tr>")
        for pt in sweep.points[1:]:
            ratio = pt.ler / baseline
            twirl_ratio = pt.ler / pt.p_twirled if pt.p_twirled > 0 else 0
            html_parts.append(
                f"<tr><td>{pt.p_idle:.3f}</td><td>{ratio:.1f}x</td><td>{twirl_ratio:.0f}x</td></tr>",
            )
        html_parts.append("</table>")

    html_parts.append("</body></html>")
    path.write_text("\n".join(html_parts))
    print(f"Report written to {path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--distance", "-d", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=None, help="Syndrome rounds (default: distance)")
    parser.add_argument(
        "--basis",
        choices=["X", "Z"],
        default="X",
        help="Memory basis (default: X, where RZ noise is visible)",
    )
    parser.add_argument("--p-depol", type=float, default=0.003)
    parser.add_argument("--p-idle", type=float, nargs="+", default=[0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1])
    parser.add_argument("--shots", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", choices=["statevec", "stabmps"], default="statevec")
    parser.add_argument("--lazy-measure", action="store_true", default=True)
    parser.add_argument("--max-bond-dim", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/coherent_noise"))
    parser.add_argument("--save-json", action="store_true")
    parser.add_argument("--save-html", action="store_true")
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()

    if args.rounds is None:
        args.rounds = args.distance

    print(
        f"Coherent idle noise sweep: {args.basis}-basis memory d={args.distance}, "
        f"rounds={args.rounds}, p_depol={args.p_depol}",
        flush=True,
    )
    print(
        f"Backend: {args.backend}, shots={args.shots}, seed={args.seed}",
        flush=True,
    )
    print(flush=True)

    sweep = run_sweep(
        distance=args.distance,
        rounds=args.rounds,
        basis=args.basis,
        p_depol=args.p_depol,
        p_idle_values=args.p_idle,
        shots=args.shots,
        seed=args.seed,
        backend=args.backend,
        lazy_measure=args.lazy_measure,
        max_bond_dim=args.max_bond_dim,
    )

    print(f"\nTotal: {sweep.total_seconds:.1f}s", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.save_json:
        json_path = args.output_dir / "coherent_noise_results.json"
        json_path.write_text(json.dumps(asdict(sweep), indent=2))
        print(f"JSON written to {json_path}")

    if args.save_html or args.open:
        html_path = args.output_dir / "coherent_noise_report.html"
        write_html_report(sweep, html_path)
        if args.open:
            import webbrowser

            webbrowser.open(str(html_path))


if __name__ == "__main__":
    main()
