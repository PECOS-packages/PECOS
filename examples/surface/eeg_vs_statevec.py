r"""Compare EEG analytical DEM vs StateVec empirical detection rates.

Three approaches compared:
  1. Forward EEG (perturbative Taylor/SinSquared formula) — fast, approximate
  2. Backward Heisenberg (exact coherent + stochastic) — slower build, exact
  3. StateVec simulation (ground truth) — slow, limited by qubit count

Both EEG approaches produce Stim-format DEM strings that can be sampled
at ~15M shots/sec via DemSampler.from_dem_string().

Example:
    uv run python examples/surface/eeg_vs_statevec.py
    uv run python examples/surface/eeg_vs_statevec.py --distance 3 --basis X --shots 20000
    uv run python examples/surface/eeg_vs_statevec.py --distance 2 --basis Z --shots 50000
    uv run python examples/surface/eeg_vs_statevec.py --distance 5 --basis Z --dem-sample
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python" / "quantum-pecos" / "src"))


def run_comparison(
    *,
    distance: int,
    rounds: int,
    basis: str,
    theta_values: list[float],
    shots: int,
    seed: int,
    dem_sample: bool,
):
    from pecos.qec.surface import LogicalCircuitBuilder, SurfacePatch
    from pecos_rslib_exp import (
        coherent_dem_exact,
        depolarizing,
        exact_detection_rates,
        perturbative_dem,
        perturbative_dem_events,
        sim_neo,
        statevec,
    )

    patch = SurfacePatch.create(distance=distance)
    b = LogicalCircuitBuilder()
    b.add_patch(patch, "Q0")
    b.add_memory("Q0", rounds=rounds, basis=basis)
    tc = b.to_tick_circuit()

    det_json = json.loads(tc.get_meta("detectors"))
    num_meas = int(tc.get_meta("num_measurements"))
    num_dets = len(det_json)

    print(f"\n{'='*70}")
    print(f"d={distance} {basis}-basis memory, {rounds} rounds, {num_dets} detectors, {num_meas} measurements")
    print(f"{'='*70}")

    for theta in theta_values:
        print(f"\ntheta = {theta:.4f}")

        # --- Forward EEG DEM (Taylor) ---
        t0 = time.perf_counter()
        eeg_events = perturbative_dem_events(tc, idle_rz=theta, h_formula="taylor")
        eeg_time = time.perf_counter() - t0

        eeg_taylor = [0.0] * num_dets
        for prob, det_ids, _obs_ids in eeg_events:
            for d in det_ids:
                if d < num_dets:
                    eeg_taylor[d] += prob

        # --- Heisenberg backward propagation ---
        t0 = time.perf_counter()
        heis_results = exact_detection_rates(tc, idle_rz=theta)
        heis_time = time.perf_counter() - t0
        heis = [0.0] * num_dets
        for det_id, prob in heis_results:
            if det_id < num_dets:
                heis[det_id] = prob

        # --- DEM sampling path (two-stage: build DEM once, sample fast) ---
        if dem_sample:
            from pecos_rslib.qec import DemSampler

            # Forward EEG DEM → sampler
            t0 = time.perf_counter()
            dem_taylor_str = perturbative_dem(tc, idle_rz=theta)
            sampler_taylor = DemSampler.from_dem_string(dem_taylor_str)
            batch_taylor = sampler_taylor.generate_samples(num_shots=shots, seed=seed)
            taylor_sample_time = time.perf_counter() - t0

            # Heisenberg DEM → sampler
            t0 = time.perf_counter()
            dem_heis_str = coherent_dem_exact(tc, idle_rz=theta)
            sampler_heis = DemSampler.from_dem_string(dem_heis_str)
            batch_heis = sampler_heis.generate_samples(num_shots=shots, seed=seed)
            heis_sample_time = time.perf_counter() - t0

            # Compute per-detector rates from DEM samples
            taylor_dem_rates = [0.0] * num_dets
            heis_dem_rates = [0.0] * num_dets
            for i in range(shots):
                syn_t = batch_taylor.get_syndrome(i)
                syn_h = batch_heis.get_syndrome(i)
                for d in range(min(num_dets, len(syn_t))):
                    if syn_t[d]:
                        taylor_dem_rates[d] += 1.0 / shots
                    if syn_h[d]:
                        heis_dem_rates[d] += 1.0 / shots
        else:
            taylor_dem_rates = None
            heis_dem_rates = None
            taylor_sample_time = 0
            heis_sample_time = 0

        # --- StateVec simulation (ground truth) ---
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

        sv_det_rate = [c / shots for c in sv_det_count]

        # --- Compare ---
        print(f"  EEG: {eeg_time*1000:.1f}ms, Heisenberg: {heis_time*1000:.1f}ms")
        print(f"  StateVec: {sv_time:.1f}s ({shots} shots)")
        if dem_sample:
            print(f"  DEM sample: Taylor {taylor_sample_time:.2f}s, Heisenberg {heis_sample_time:.2f}s ({shots} shots)")

        if dem_sample:
            print(
                f"  {'Det':>4} {'Taylor':>10} {'Heisen':>10} {'T(DEM)':>10} "
                f"{'H(DEM)':>10} {'StateVec':>10} {'SV_se':>8} {'T/SV':>7} {'H/SV':>7}",
            )
        else:
            print(f"  {'Det':>4} {'Taylor':>10} {'Heisen':>10} {'StateVec':>10} {'SV_se':>8} {'T/SV':>7} {'H/SV':>7}")

        max_rel_taylor = 0.0
        max_rel_heis = 0.0
        for d in range(num_dets):
            tp = eeg_taylor[d]
            hp = heis[d]
            sv_r = sv_det_rate[d]
            sv_se = math.sqrt(sv_r * (1 - sv_r) / shots) if shots > 0 else 0

            if sv_r > 0.0001:
                t_ratio = tp / sv_r
                h_ratio = hp / sv_r
                max_rel_taylor = max(max_rel_taylor, abs(tp - sv_r) / sv_r)
                max_rel_heis = max(max_rel_heis, abs(hp - sv_r) / sv_r)
            else:
                t_ratio = float("nan")
                h_ratio = float("nan")

            if dem_sample:
                td = taylor_dem_rates[d] if taylor_dem_rates else 0
                hd = heis_dem_rates[d] if heis_dem_rates else 0
                print(
                    f"  D{d:>3} {tp:>10.6f} {hp:>10.6f} {td:>10.6f} {hd:>10.6f} "
                    f"{sv_r:>10.6f} {sv_se:>8.6f} {t_ratio:>7.3f} {h_ratio:>7.3f}",
                )
            else:
                print(
                    f"  D{d:>3} {tp:>10.6f} {hp:>10.6f} {sv_r:>10.6f} "
                    f"{sv_se:>8.6f} {t_ratio:>7.3f} {h_ratio:>7.3f}",
                )

        print(f"  Max rel err: Taylor={max_rel_taylor*100:.1f}%, Heisenberg={max_rel_heis*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--distance", "-d", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--basis", choices=["X", "Z"], nargs="+", default=["X", "Z"])
    parser.add_argument("--theta", type=float, nargs="+", default=[0.01, 0.03, 0.05, 0.1])
    parser.add_argument("--shots", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dem-sample", action="store_true", help="Also sample from both DEMs and compare rates")
    args = parser.parse_args()

    for dist in args.distance:
        for basis in args.basis:
            rds = args.rounds if args.rounds is not None else dist
            run_comparison(
                distance=dist,
                rounds=rds,
                basis=basis,
                theta_values=args.theta,
                shots=args.shots,
                seed=args.seed,
                dem_sample=args.dem_sample,
            )


if __name__ == "__main__":
    main()
