r"""Compare non-EEG DEM detection rates against stabilizer() ground truth.

Tests per-detector firing rates from:
  1. DemBuilder analytical marginals (sum of DEM event probabilities)
  2. DemSampler.from_circuit sampled rates
  3. stabilizer() simulation (SparseStab ground truth)

Pure depolarizing noise only (no coherent idle_rz).

Example:
    uv run python examples/surface/dem_vs_stabilizer.py
    uv run python examples/surface/dem_vs_stabilizer.py -d 2 3 --p 0.005
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python" / "quantum-pecos" / "src"))


def run_comparison(*, distance, rounds, basis, p, shots, seed):
    from pecos.qec.surface import LogicalCircuitBuilder, SurfacePatch
    from pecos_rslib.qec import DagFaultAnalyzer, DemBuilder, DemSampler
    from pecos_rslib_exp import depolarizing, sim_neo, stabilizer

    patch = SurfacePatch.create(distance=distance)
    b = LogicalCircuitBuilder()
    b.add_patch(patch, "Q0")
    b.add_memory("Q0", rounds=rounds, basis=basis)
    tc = b.to_tick_circuit()
    dag = tc.to_dag_circuit()

    det_json = json.loads(tc.get_meta("detectors"))
    num_meas = int(tc.get_meta("num_measurements"))
    num_dets = len(det_json)

    print(f"\n{'='*70}")
    print(f"d={distance} {basis}-basis, {rounds} rounds, {num_dets} dets, p={p}")
    print(f"{'='*70}")

    # 1. DemBuilder analytical marginals
    analyzer = DagFaultAnalyzer(dag)
    influence = analyzer.build_influence_map()
    dem_obj = (
        DemBuilder(influence)
        .with_noise(p1=p, p2=p, p_meas=p, p_prep=p)
        .with_detectors_json(tc.get_meta("detectors"))
        .with_observables_json(tc.get_meta("observables"))
        .with_num_measurements(num_meas)
        .build()
    )
    dem_str = dem_obj.to_string()
    analytical = [0.0] * num_dets
    for line in dem_str.strip().split("\n"):
        if line.startswith("error("):
            prob = float(line.split("(")[1].split(")")[0])
            for x in line.split():
                if x.startswith("D"):
                    d_id = int(x[1:])
                    if d_id < num_dets:
                        analytical[d_id] += prob

    # 2. DemSampler.from_circuit
    t0 = time.perf_counter()
    sampler_fc = DemSampler.from_circuit(dag, p1=p, p2=p, p_meas=p, p_prep=p)
    batch_fc = sampler_fc.generate_samples(num_shots=shots, seed=seed)
    dem_fc = [0.0] * num_dets
    for i in range(shots):
        syn = batch_fc.get_syndrome(i)
        for dd in range(min(num_dets, len(syn))):
            if syn[dd]:
                dem_fc[dd] += 1.0 / shots
    fc_time = time.perf_counter() - t0

    # 3. stabilizer() ground truth
    t0 = time.perf_counter()
    noise = depolarizing().p1(p).p2(p).p_meas(p).p_prep(p)
    results = sim_neo(tc).quantum(stabilizer()).noise(noise).shots(shots).seed(seed).run()
    sim = [0.0] * num_dets
    for r in results:
        meas = list(r)
        for i, det in enumerate(det_json):
            val = 0
            for rec in det["records"]:
                idx = num_meas + rec
                if 0 <= idx < len(meas):
                    val ^= meas[idx]
            if val:
                sim[i] += 1.0 / shots
    sim_time = time.perf_counter() - t0

    print(f"  DEM sample: {fc_time:.2f}s, Stabilizer: {sim_time:.2f}s ({shots} shots)")
    print(
        f"  {'Det':>4} {'Analytical':>11} {'FromCirc':>10} {'Stabiliz':>10}"
        f" {'SV_se':>8} {'A/Sim':>7} {'FC/Sim':>7}",
    )

    max_a_err = 0.0
    max_fc_err = 0.0
    flagged = 0
    for dd in range(num_dets):
        sv_r = sim[dd]
        se = math.sqrt(sv_r * (1 - sv_r) / shots) if sv_r > 0 else 0

        if sv_r > 0.002:
            ra = analytical[dd] / sv_r
            rf = dem_fc[dd] / sv_r
            max_a_err = max(max_a_err, abs(1 - ra))
            max_fc_err = max(max_fc_err, abs(1 - rf))
        else:
            ra = float("nan")
            rf = float("nan")

        flag = ""
        if sv_r > 0.002 and (abs(1 - ra) > 0.15 or abs(1 - rf) > 0.15):
            flag = " ***"
            flagged += 1

        print(
            f"  D{dd:>2} {analytical[dd]:>11.6f} {dem_fc[dd]:>10.6f} {sv_r:>10.6f}"
            f" {se:>8.5f} {ra:>7.3f} {rf:>7.3f}{flag}",
        )

    print(
        f"  Max deviation: Analytical={max_a_err*100:.1f}%, FromCircuit={max_fc_err*100:.1f}%, flagged={flagged}",
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--distance", "-d", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--basis", choices=["X", "Z"], nargs="+", default=["X", "Z"])
    parser.add_argument("--p", type=float, nargs="+", default=[0.005])
    parser.add_argument("--shots", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for dist in args.distance:
        for basis in args.basis:
            for p_val in args.p:
                rds = args.rounds if args.rounds is not None else dist
                run_comparison(
                    distance=dist,
                    rounds=rds,
                    basis=basis,
                    p=p_val,
                    shots=args.shots,
                    seed=args.seed,
                )


if __name__ == "__main__":
    main()
