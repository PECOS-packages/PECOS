r"""Compare all forward EEG formula variants against backward Heisenberg.

Tests 6 forward EEG configurations:
  1. Taylor + BCH1 (default)
  2. SinSquared + BCH1
  3. ExactCommuting + BCH1
  4. Taylor + BCH2
  5. SinSquared + BCH2
  6. ExactCommuting + BCH2

Against:
  7. Backward Heisenberg (exact)
  8. StateVec simulation (ground truth, optional)

Example:
    uv run python examples/surface/eeg_formula_comparison.py
    uv run python examples/surface/eeg_formula_comparison.py -d 3 --theta 0.05 --shots 50000
    uv run python examples/surface/eeg_formula_comparison.py -d 2 3 --no-statevec
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python" / "quantum-pecos" / "src"))

CONFIGS = [
    ("Taylor", "taylor", 1),
    ("SinSq", "sin_squared", 1),
    ("ExCom", "exact_commuting", 1),
    ("ExSubset", "exact_subset", 1),
]


def marginals_from_events(events, num_dets):
    rates = [0.0] * num_dets
    for prob, det_ids, _obs_ids in events:
        for d in det_ids:
            if d < num_dets:
                rates[d] += prob
    return rates


def run(*, distance, rounds, basis, theta_values, shots, seed, run_statevec):
    from pecos.qec.surface import LogicalCircuitBuilder, SurfacePatch
    from pecos_rslib_exp import eeg_per_detector, exact_detection_rates, perturbative_dem_events

    patch = SurfacePatch.create(distance=distance)
    b = LogicalCircuitBuilder()
    b.add_patch(patch, "Q0")
    b.add_memory("Q0", rounds=rounds, basis=basis)
    tc = b.to_tick_circuit()

    det_json = json.loads(tc.get_meta("detectors"))
    num_meas = int(tc.get_meta("num_measurements"))
    num_dets = len(det_json)

    print(f"\n{'='*80}")
    print(f"d={distance} {basis}-basis memory, {rounds} rounds, {num_dets} detectors")
    print(f"{'='*80}")

    for theta in theta_values:
        print(f"\ntheta = {theta:.4f}")

        # Forward EEG: all 6 configs
        fwd_marginals = {}
        for name, h_formula, bch in CONFIGS:
            t0 = time.perf_counter()
            events = perturbative_dem_events(tc, idle_rz=theta, h_formula=h_formula, bch_order=bch)
            dt = time.perf_counter() - t0
            fwd_marginals[name] = (marginals_from_events(events, num_dets), dt)

        # Per-detector computation (cross-event beta)
        per_det_marginals = {}
        for name, h_formula, _ in [
            ("PD-Taylor", "taylor", 0),
            ("PD-SinSq", "sin_squared", 0),
            ("PD-ExCom", "exact_commuting", 0),
        ]:
            t0 = time.perf_counter()
            pd_results = eeg_per_detector(tc, idle_rz=theta, h_formula=h_formula)
            dt = time.perf_counter() - t0
            rates = [0.0] * num_dets
            for det_id, prob in pd_results:
                if det_id < num_dets:
                    rates[det_id] = prob
            per_det_marginals[name] = (rates, dt)

        # Backward Heisenberg (exact)
        t0 = time.perf_counter()
        heis_results = exact_detection_rates(tc, idle_rz=theta)
        heis_time = time.perf_counter() - t0
        heis = [0.0] * num_dets
        for det_id, prob in heis_results:
            if det_id < num_dets:
                heis[det_id] = prob

        # StateVec (optional ground truth)
        sv_rate = None
        if run_statevec:
            from pecos_rslib_exp import depolarizing, sim_neo, statevec

            t0 = time.perf_counter()
            noise = depolarizing().idle_rz(theta)
            results = sim_neo(tc).quantum(statevec()).noise(noise).shots(shots).seed(seed).run()
            sv_time = time.perf_counter() - t0

            sv_det_count = [0] * num_dets
            for r in results:
                meas = list(r)
                for i, det in enumerate(det_json):
                    val = 0
                    for rec in det["records"]:
                        idx = num_meas + rec
                        if 0 <= idx < len(meas):
                            val ^= meas[idx]
                    if val:
                        sv_det_count[i] += 1
            sv_rate = [c / shots for c in sv_det_count]

        # Summary table: max relative error vs Heisenberg for each config
        print(f"\n  {'Config':<14} {'Time':>8} {'MaxRelErr':>10} {'MeanRelErr':>11} {'MaxAbsErr':>10}")
        print(f"  {'-'*14} {'-'*8} {'-'*10} {'-'*11} {'-'*10}")

        all_configs = [(name, fwd_marginals[name]) for name, _, _ in CONFIGS]
        all_configs += [(name, per_det_marginals[name]) for name in ["PD-Taylor", "PD-SinSq", "PD-ExCom"]]

        for name, (rates, dt) in all_configs:
            max_rel = 0.0
            sum_rel = 0.0
            max_abs = 0.0
            count = 0
            for d in range(num_dets):
                if heis[d] > 1e-6:
                    rel = abs(rates[d] - heis[d]) / heis[d]
                    max_rel = max(max_rel, rel)
                    sum_rel += rel
                    count += 1
                max_abs = max(max_abs, abs(rates[d] - heis[d]))
            mean_rel = sum_rel / count if count > 0 else 0
            print(f"  {name:<14} {dt*1000:>7.1f}ms {max_rel*100:>9.1f}% {mean_rel*100:>10.1f}% {max_abs:>10.6f}")

        print(f"  {'Heisenberg':<14} {heis_time*1000:>7.1f}ms {'(exact)':>10}")

        if sv_rate is not None:
            # Also compare Heisenberg vs StateVec
            max_rel_h = 0.0
            for d in range(num_dets):
                if sv_rate[d] > 0.01:
                    max_rel_h = max(max_rel_h, abs(heis[d] - sv_rate[d]) / sv_rate[d])
            print(f"  {'StateVec':<14} {sv_time:>6.1f}s   H/SV max err: {max_rel_h*100:.1f}%")

        # Per-detector detail (compact — only show non-zero detectors, skip redundant DEM configs)
        if num_dets <= 40:
            # Show: Heisenberg, Taylor (DEM), PD-Taylor, PD-ExCom, SV
            show_configs = [
                ("Taylor", lambda fwd_marginals=fwd_marginals: fwd_marginals["Taylor"]),
                ("ExSubset", lambda fwd_marginals=fwd_marginals: fwd_marginals["ExSubset"]),
                ("PD-Tayl", lambda per_det_marginals=per_det_marginals: per_det_marginals["PD-Taylor"]),
            ]
            cols = ["Det", "Heisen"] + [n for n, _ in show_configs]
            if sv_rate is not None:
                cols.append("SV")
            header = f"  {cols[0]:>4} {cols[1]:>10}"
            for c in cols[2:]:
                header += f" {c:>10}"
            print(f"\n{header}")

            for d in range(num_dets):
                if heis[d] < 1e-8 and all(fn()[0][d] < 1e-8 for _, fn in show_configs):
                    continue
                line = f"  D{d:>2} {heis[d]:>10.6f}"
                for _, fn in show_configs:
                    rates, _ = fn()
                    line += f" {rates[d]:>10.6f}"
                if sv_rate is not None:
                    line += f" {sv_rate[d]:>10.6f}"
                print(line)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--distance", "-d", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--basis", choices=["X", "Z"], nargs="+", default=["Z"])
    parser.add_argument("--theta", type=float, nargs="+", default=[0.01, 0.05, 0.1])
    parser.add_argument("--shots", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-statevec", action="store_true")
    args = parser.parse_args()

    for dist in args.distance:
        for basis in args.basis:
            rds = args.rounds if args.rounds is not None else dist
            run(
                distance=dist,
                rounds=rds,
                basis=basis,
                theta_values=args.theta,
                shots=args.shots,
                seed=args.seed,
                run_statevec=not args.no_statevec,
            )


if __name__ == "__main__":
    main()
