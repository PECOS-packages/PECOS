r"""Compare ALL DEM generation methods against simulation ground truth.

Compares per-detector firing rates from:
  1. Non-EEG DemBuilder (backward Pauli propagation)
  2. DemSampler.from_circuit (separate DEM path)
  3. Backward Heisenberg EEG (exact for coherent, handles depol via attenuation)
  4. stabilizer() simulation (SparseStab, exact for depol, any distance)
  5. statevec() simulation (exact for everything, limited to small circuits)

Example:
    uv run python examples/surface/dem_comparison.py
    uv run python examples/surface/dem_comparison.py -d 2 3 --p 0.005 --shots 20000
    uv run python examples/surface/dem_comparison.py -d 5 --no-statevec
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python" / "quantum-pecos" / "src"))


def run(*, distance, rounds, basis, p, shots, seed, run_statevec):
    from pecos.qec.surface import LogicalCircuitBuilder, SurfacePatch
    from pecos_rslib.qec import DagFaultAnalyzer, DemBuilder, DemSampler
    from pecos_rslib_exp import (
        depolarizing,
        exact_detection_rates,
        sim_neo,
        stabilizer,
        statevec,
    )

    patch = SurfacePatch.create(distance=distance)
    b = LogicalCircuitBuilder()
    b.add_patch(patch, "Q0")
    b.add_memory("Q0", rounds=rounds, basis=basis)
    tc = b.to_tick_circuit()
    dag = tc.to_dag_circuit()

    det_json = json.loads(tc.get_meta("detectors"))
    num_meas = int(tc.get_meta("num_measurements"))
    num_dets = len(det_json)

    print(f"\n{'='*80}")
    print(f"d={distance} {basis}-basis, {rounds} rounds, {num_dets} dets, p={p} (depol only)")
    print(f"{'='*80}")

    def extract_det_rates(results):
        rates = [0.0] * num_dets
        for r in results:
            meas = list(r)
            for i, det in enumerate(det_json):
                val = 0
                for rec in det["records"]:
                    idx = num_meas + rec
                    if 0 <= idx < len(meas):
                        val ^= meas[idx]
                if val:
                    rates[i] += 1.0 / len(results)
        return rates

    # 1. Non-EEG DemBuilder analytical marginals
    t0 = time.perf_counter()
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
    dem_analytical = [0.0] * num_dets
    for line in dem_str.strip().split("\n"):
        if line.startswith("error("):
            prob = float(line.split("(")[1].split(")")[0])
            for x in line.split():
                if x.startswith("D"):
                    d_id = int(x[1:])
                    if d_id < num_dets:
                        dem_analytical[d_id] += prob
    dem_build_time = time.perf_counter() - t0

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

    # 3. Backward Heisenberg
    t0 = time.perf_counter()
    heis_results = exact_detection_rates(tc, p1=p, p2=p, p_meas=p, p_prep=p)
    heis = [0.0] * num_dets
    for det_id, prob in heis_results:
        if det_id < num_dets:
            heis[det_id] = prob
    heis_time = time.perf_counter() - t0

    # 4. stabilizer() ground truth
    t0 = time.perf_counter()
    noise = depolarizing().p1(p).p2(p).p_meas(p).p_prep(p)
    stab_results = sim_neo(tc).quantum(stabilizer()).noise(noise).shots(shots).seed(seed).run()
    stab = extract_det_rates(stab_results)
    stab_time = time.perf_counter() - t0

    # 5. statevec() (optional, small circuits only)
    sv = None
    sv_time = 0
    if run_statevec:
        t0 = time.perf_counter()
        sv_results = sim_neo(tc).quantum(statevec()).noise(noise).shots(shots).seed(seed + 1).run()
        sv = extract_det_rates(sv_results)
        sv_time = time.perf_counter() - t0

    # Print timing
    print(
        f"  DemBuilder: {dem_build_time*1000:.0f}ms, FromCircuit: {fc_time:.2f}s,"
        f" Heisenberg: {heis_time*1000:.0f}ms, Stabilizer: {stab_time:.2f}s"
        + (f", StateVec: {sv_time:.1f}s" if sv else ""),
    )

    # Header
    cols = ["Det", "DemBuild", "FromCirc", "Heisen", "Stabiliz"]
    if sv:
        cols.append("StateVec")
    cols += ["DB/Stab", "FC/Stab", "H/Stab"]
    hdr = f"  {cols[0]:>4}"
    for c in cols[1:]:
        hdr += f" {c:>10}"
    print(hdr)

    for dd in range(num_dets):
        s = stab[dd]
        if s < 0.001:
            continue

        math.sqrt(s * (1 - s) / shots)
        r_db = dem_analytical[dd] / s
        r_fc = dem_fc[dd] / s
        r_h = heis[dd] / s

        line = f"  D{dd:>2} {dem_analytical[dd]:>10.6f} {dem_fc[dd]:>10.6f} {heis[dd]:>10.6f} {s:>10.6f}"
        if sv:
            line += f" {sv[dd]:>10.6f}"
        line += f" {r_db:>10.3f} {r_fc:>10.3f} {r_h:>10.3f}"

        flags = []
        if abs(1 - r_db) > 0.15:
            flags.append("DB")
        if abs(1 - r_fc) > 0.15:
            flags.append("FC")
        if abs(1 - r_h) > 0.15:
            flags.append("H")
        if flags:
            line += f"  *** {','.join(flags)}"
        print(line)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--distance", "-d", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--basis", choices=["X", "Z"], nargs="+", default=["Z"])
    parser.add_argument("--p", type=float, nargs="+", default=[0.005])
    parser.add_argument("--shots", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-statevec", action="store_true")
    args = parser.parse_args()

    for dist in args.distance:
        for basis in args.basis:
            for p_val in args.p:
                rds = args.rounds if args.rounds is not None else dist
                run(
                    distance=dist,
                    rounds=rds,
                    basis=basis,
                    p=p_val,
                    shots=args.shots,
                    seed=args.seed,
                    run_statevec=not args.no_statevec and dist <= 3,
                )


if __name__ == "__main__":
    main()
