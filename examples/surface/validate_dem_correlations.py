# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Validate DEM detector correlations against simulation ground truth.

Computes per-round detector flip frequency matrices from both simulation
and DEM sampling, then compares them element-wise. The matrix diagonal
gives marginal detection rates; off-diagonal elements give half the
joint detection probability, capturing the correlated error structure.

Usage:
    uv run python examples/surface/validate_dem_correlations.py
    uv run python examples/surface/validate_dem_correlations.py -d 3 5 --shots 100000
    uv run python examples/surface/validate_dem_correlations.py --circuit-source traced_qis
    uv run python examples/surface/validate_dem_correlations.py --show-matrices
"""

from __future__ import annotations

import argparse
import json
import sys
import time


def build_circuit(distance, rounds, basis, circuit_source):
    from pecos.qec.surface import SurfacePatch

    patch = SurfacePatch.create(distance=distance)

    if circuit_source == "traced_qis":
        from pecos.qec.surface.decode import _build_surface_tick_circuit_for_native_model

        tc = _build_surface_tick_circuit_for_native_model(
            patch,
            rounds,
            basis,
            circuit_source="traced_qis",
        )
        tc.lower_clifford_rotations()
        tc.assign_missing_meas_ids()
    else:
        from pecos.qec.surface import LogicalCircuitBuilder

        b = LogicalCircuitBuilder()
        b.add_patch(patch, "Q0")
        b.add_memory("Q0", rounds=rounds, basis=basis)
        tc = b.to_tick_circuit()

    return tc, patch


def simulate_detector_events(tc, noise_kw, shots, seed):
    """Run simulation and extract per-shot detector event lists."""
    from pecos_rslib_exp import depolarizing, sim_neo, stabilizer

    noise = depolarizing()
    for k, v in noise_kw.items():
        if v > 0:
            noise = getattr(noise, k)(v)

    det_json = json.loads(tc.get_meta("detectors"))
    num_meas = int(tc.get_meta("num_measurements"))
    num_dets = len(det_json)

    results = sim_neo(tc).quantum(stabilizer()).noise(noise).shots(shots).seed(seed).run()

    events = []
    for r in results:
        meas = list(r)
        fired = []
        for i, det in enumerate(det_json):
            val = 0
            for rec in det["records"]:
                idx = num_meas + rec
                if 0 <= idx < len(meas):
                    val ^= meas[idx]
            if val:
                fired.append(i)
        events.append(fired)

    return events, num_dets


def dem_detector_events(tc, noise_kw, shots, seed):
    """Sample from DEM and extract per-shot detector event lists."""
    from pecos_rslib.qec import DemSampler

    full_kw = {k: noise_kw.get(k, 0.0) for k in ["p1", "p2", "p_meas", "p_prep"]}
    sampler = DemSampler.from_circuit(tc, **full_kw)
    batch = sampler.generate_samples(num_shots=shots, seed=seed)
    num_dets = len(json.loads(tc.get_meta("detectors")))

    events = []
    for i in range(shots):
        syn = batch.get_syndrome(i)
        fired = [d for d in range(min(num_dets, len(syn))) if syn[d]]
        events.append(fired)

    return events, num_dets


def format_matrix(matrix, width=8, precision=5):
    """Format a matrix as an aligned string."""
    lines = []
    for row in matrix:
        cells = [f"{v:{width}.{precision}f}" for v in row]
        lines.append("[" + " ".join(cells) + "]")
    return "\n".join(lines)


def run_validation(
    *,
    distances,
    bases,
    rounds_per_d,
    shots,
    seed,
    circuit_sources,
    noise_configs,
    threshold,
    show_matrices,
    max_order,
):
    from pecos.qec.analysis import (
        compare_flip_matrices,
        compare_k_body_rates,
        detector_flip_matrices_by_round,
        detector_k_body_rates_by_round,
    )

    total_pass = 0
    total_fail = 0
    failures = []

    for distance in distances:
        rounds = rounds_per_d if rounds_per_d else distance
        for basis in bases:
            for source in circuit_sources:
                tc, patch = build_circuit(distance, rounds, basis, source)
                num_dets = len(json.loads(tc.get_meta("detectors")))
                n_ancilla = len(patch.x_stabilizers) + len(patch.z_stabilizers)

                src_label = f" [{source}]" if len(circuit_sources) > 1 else ""
                print(f"\n{'=' * 72}")
                print(
                    f"d={distance} {basis}-basis{src_label}, {num_dets} detectors, "
                    f"{n_ancilla} per round, {rounds} rounds",
                )
                print(f"{'=' * 72}")

                for noise_label, noise_kw in noise_configs:
                    t0 = time.perf_counter()
                    sim_events, nd = simulate_detector_events(tc, noise_kw, shots, seed)
                    sim_time = time.perf_counter() - t0

                    dem_events, _ = dem_detector_events(tc, noise_kw, shots, seed)

                    # --- Pairwise flip matrices ---
                    sim_mats = detector_flip_matrices_by_round(sim_events, nd, n_ancilla)
                    dem_mats = detector_flip_matrices_by_round(dem_events, nd, n_ancilla)

                    all_pass = True
                    round_results = []
                    for r_idx, (sm, dm) in enumerate(zip(sim_mats, dem_mats, strict=False)):
                        max_err, frob_err, worst = compare_flip_matrices(sm, dm)
                        ok = max_err <= threshold
                        if not ok:
                            all_pass = False
                        round_results.append((r_idx, max_err, frob_err, worst, ok))

                    # --- Higher-order correlations ---
                    sim_kbody = detector_k_body_rates_by_round(
                        sim_events,
                        nd,
                        n_ancilla,
                        max_order=max_order,
                    )
                    dem_kbody = detector_k_body_rates_by_round(
                        dem_events,
                        nd,
                        n_ancilla,
                        max_order=max_order,
                    )

                    kbody_results = []  # (round, order, max_err, rms_err, worst, ok)
                    for r_idx, (sr, dr) in enumerate(zip(sim_kbody, dem_kbody, strict=False)):
                        order_stats = compare_k_body_rates(sr, dr, max_order=max_order)
                        for order, (me, rms, worst_ev) in order_stats.items():
                            ok = me <= threshold
                            if not ok:
                                all_pass = False
                            kbody_results.append((r_idx, order, me, rms, worst_ev, ok))

                    # Aggregate
                    worst_round_err = max(rr[1] for rr in round_results)
                    status = "PASS" if all_pass else "FAIL"
                    if all_pass:
                        total_pass += 1
                    else:
                        total_fail += 1
                        failures.append(
                            f"d={distance} {basis} {source} {noise_label}: {worst_round_err * 100:.0f}%",
                        )

                    print(f"\n  {noise_label} (sim: {sim_time:.2f}s)  {status}")

                    # Pairwise per-round summary
                    print("    Pairwise (flip matrices):")
                    for r_idx, max_err, frob_err, worst, ok in round_results:
                        flag = "" if ok else " <-- FAIL"
                        print(
                            f"      Round {r_idx}: max_rel={max_err * 100:5.1f}%  "
                            f"frob_rel={frob_err * 100:5.1f}%  "
                            f"worst={worst}{flag}",
                        )

                    # Higher-order per-round summary
                    for order in range(1, max_order + 1):
                        order_entries = [(r, me, rms, w, ok) for r, o, me, rms, w, ok in kbody_results if o == order]
                        if not order_entries:
                            continue
                        worst_me = max(e[1] for e in order_entries)
                        avg_rms = sum(e[2] for e in order_entries) / len(order_entries)
                        label = {
                            1: "1-body (marginals)",
                            2: "2-body (pairs)",
                            3: "3-body (triples)",
                            4: "4-body (quads)",
                        }.get(order, f"{order}-body")
                        any_fail = any(not e[4] for e in order_entries)
                        flag = " <-- FAIL" if any_fail else ""
                        print(
                            f"    {label}: worst_max_rel={worst_me * 100:5.1f}%  "
                            f"avg_rms_rel={avg_rms * 100:5.1f}%{flag}",
                        )
                        if any_fail:
                            for r, me, _rms, w, ok in order_entries:
                                if not ok:
                                    print(f"      Round {r}: max_rel={me * 100:.1f}% worst={w}")

                    if show_matrices and not all_pass:
                        for r_idx, _max_err, _, _, ok in round_results:
                            if not ok:
                                print(f"\n    Round {r_idx} sim:")
                                print("    " + format_matrix(sim_mats[r_idx]).replace("\n", "\n    "))
                                print(f"    Round {r_idx} dem:")
                                print("    " + format_matrix(dem_mats[r_idx]).replace("\n", "\n    "))

    print(f"\n{'=' * 72}")
    print(f"SUMMARY: {total_pass}/{total_pass + total_fail} passed (threshold: {threshold * 100:.0f}%)")
    if failures:
        print("Failures:")
        for f in failures:
            print(f"  {f}")
    print(f"{'=' * 72}")

    return total_fail == 0


def main():
    parser = argparse.ArgumentParser(
        description="Validate DEM detector correlations against simulation.",
    )
    parser.add_argument("--distance", "-d", type=int, nargs="+", default=[2, 3], help="Code distances (default: 2 3)")
    parser.add_argument("--basis", type=str, nargs="+", default=["Z"], choices=["Z", "X"], help="Bases (default: Z)")
    parser.add_argument("--rounds", type=int, default=None, help="Syndrome rounds (default: same as distance)")
    parser.add_argument("--shots", type=int, default=100000, help="Shots per test (default: 100000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--circuit-source",
        choices=["abstract", "traced_qis", "both"],
        default="both",
        help="Circuit pipeline (default: both)",
    )
    parser.add_argument("--threshold", type=float, default=0.20, help="Max relative error threshold (default: 0.20)")
    parser.add_argument("--max-order", type=int, default=3, help="Max correlation order (default: 3)")
    parser.add_argument("--show-matrices", action="store_true", help="Print matrices for failing rounds")

    args = parser.parse_args()

    sources = ["abstract", "traced_qis"] if args.circuit_source == "both" else [args.circuit_source]

    noise_configs = [
        ("p_meas=0.01", {"p1": 0.0, "p2": 0.0, "p_meas": 0.01, "p_prep": 0.0}),
        ("p2=0.01", {"p1": 0.0, "p2": 0.01, "p_meas": 0.0, "p_prep": 0.0}),
        ("depol", {"p1": 0.001, "p2": 0.01, "p_meas": 0.01, "p_prep": 0.01}),
        ("strong_depol", {"p1": 0.005, "p2": 0.05, "p_meas": 0.05, "p_prep": 0.05}),
    ]

    ok = run_validation(
        distances=args.distance,
        bases=args.basis,
        rounds_per_d=args.rounds,
        shots=args.shots,
        seed=args.seed,
        circuit_sources=sources,
        noise_configs=noise_configs,
        threshold=args.threshold,
        show_matrices=args.show_matrices,
        max_order=args.max_order,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
