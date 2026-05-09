#!/usr/bin/env python3
# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0
"""Compare meas_sampling pipeline against native DEM sampler on surface-code memory.

Uses the Guppy → traced QIS → TickCircuit pipeline (lowered to Clifford gates).
Decodes with PyMatching. Compares logical error rates and timing.

Usage:
    .venv/bin/python scripts/compare_meas_sampling_pipeline.py
    .venv/bin/python scripts/compare_meas_sampling_pipeline.py --distances 3 5 7 --shots 10000
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pymatching
import stim
from pecos.qec.surface import SurfacePatch
from pecos.qec.surface.circuit_builder import tick_circuit_to_stim
from pecos.qec.surface.decode import _build_surface_tick_circuit_for_native_model
from pecos_rslib.qec import DemSampler
from pecos_rslib_exp import depolarizing, meas_sampling, sim_neo


def build_circuit(distance, rounds, basis="Z"):
    """Build a traced-QIS surface-code TickCircuit, lowered to Clifford gates."""
    patch = SurfacePatch.create(distance=distance)
    tc = _build_surface_tick_circuit_for_native_model(
        patch,
        rounds,
        basis,
        circuit_source="traced_qis",
    )
    # Lower R1XY/RZ rotations to standard Clifford gates (H, SZ, SZdg, etc.)
    tc.lower_clifford_rotations()
    return tc


def get_pymatching_decoder(tc, noise_args):
    """Build a PyMatching decoder from a circuit's Stim DEM."""
    stim_str = tick_circuit_to_stim(tc, **noise_args)
    dem = stim.Circuit(stim_str).detector_error_model(decompose_errors=True)
    return pymatching.Matching.from_detector_error_model(dem)


def run_meas_sampling(tc, noise_args, shots, seed):
    """Sample raw measurements via meas_sampling."""
    depol = (
        depolarizing()
        .p1(noise_args["p1"])
        .p2(noise_args["p2"])
        .p_meas(
            noise_args["p_meas"],
        )
        .p_prep(noise_args["p_prep"])
    )

    t0 = time.perf_counter()
    result = sim_neo(tc).quantum(meas_sampling()).noise(depol).shots(shots).seed(seed).run()
    t_sample = time.perf_counter() - t0
    return result, t_sample


def extract_and_decode(result, tc, matching, shots):
    """Extract detection events from raw measurements and decode with PyMatching."""
    det_json = json.loads(tc.get_meta("detectors"))
    obs_json_str = tc.get_meta("observables")
    obs_json = json.loads(obs_json_str) if obs_json_str else []
    num_meas = int(tc.get_meta("num_measurements"))
    num_dets = len(det_json)

    t0 = time.perf_counter()

    errors = 0
    syndrome = np.zeros(num_dets, dtype=np.uint8)

    for shot_idx in range(shots):
        row = result[shot_idx]

        # Extract detector events
        syndrome.fill(0)
        for i, det in enumerate(det_json):
            val = 0
            for rec in det["records"]:
                idx = num_meas + rec
                if 0 <= idx < len(row):
                    val ^= row[idx]
            syndrome[i] = val

        # Extract observable flips
        actual_mask = 0
        for i, obs in enumerate(obs_json):
            val = 0
            for rec in obs["records"]:
                idx = num_meas + rec
                if 0 <= idx < len(row):
                    val ^= row[idx]
            if val:
                actual_mask |= 1 << i

        # Decode
        predicted = matching.decode(syndrome)
        pred_mask = sum(int(v) << j for j, v in enumerate(predicted))
        if pred_mask != actual_mask:
            errors += 1

    t_decode = time.perf_counter() - t0
    return errors, t_decode


def run_native_sampler(tc, noise_args, matching, shots, seed):
    """Sample + decode via the native DEM sampler path."""
    sampler = DemSampler.from_circuit(tc, **noise_args)
    num_dets = sampler.num_detectors

    t0 = time.perf_counter()
    batch = sampler.generate_samples(shots, seed=seed)
    t_sample = time.perf_counter() - t0

    t0 = time.perf_counter()
    errors = 0
    syndrome = np.zeros(num_dets, dtype=np.uint8)

    for i in range(shots):
        syn = batch.get_syndrome(i)
        for j in range(num_dets):
            syndrome[j] = syn[j]

        predicted = matching.decode(syndrome)
        pred_mask = sum(int(v) << j for j, v in enumerate(predicted))
        if pred_mask != batch.get_observable_mask(i):
            errors += 1

    t_decode = time.perf_counter() - t0
    return errors, t_sample, t_decode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distances", type=int, nargs="+", default=[3, 5])
    parser.add_argument(
        "--rounds-per-d",
        type=int,
        default=3,
        help="Syndrome rounds = distance * rounds_per_d",
    )
    parser.add_argument("--shots", type=int, default=5000)
    parser.add_argument("--error-rate", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    p = args.error_rate
    noise_args = {"p1": p * 0.1, "p2": p, "p_meas": p * 0.5, "p_prep": p * 0.5}

    print("=" * 80)
    print("meas_sampling vs native DEM sampler: traced-QIS surface-code memory + PyMatching")
    print("=" * 80)
    print(f"  shots={args.shots}, p={p}")
    print(
        f"  noise: p1={noise_args['p1']:.1e} p2={noise_args['p2']:.1e} "
        f"p_meas={noise_args['p_meas']:.1e} p_prep={noise_args['p_prep']:.1e}",
    )
    print("  circuit: Guppy -> traced QIS -> lower_clifford_rotations()")
    print()

    header = (
        f"{'d':>3} {'rounds':>6} | {'backend':>15} | {'sample':>8} | "
        f"{'decode':>8} | {'total':>8} | {'LER':>8} | {'errors':>10}"
    )
    print(header)
    print("-" * len(header))

    for d in args.distances:
        rounds = d * args.rounds_per_d
        tc = build_circuit(d, rounds)

        matching = get_pymatching_decoder(tc, noise_args)

        # --- meas_sampling ---
        result, t_sample_ms = run_meas_sampling(tc, noise_args, args.shots, args.seed)
        errors_ms, t_decode_ms = extract_and_decode(result, tc, matching, args.shots)
        ler_ms = errors_ms / args.shots
        t_total_ms = t_sample_ms + t_decode_ms

        print(
            f"d={d:>1} {rounds:>6} | {'meas_sampling':>15} | {t_sample_ms*1000:>7.0f}ms | "
            f"{t_decode_ms*1000:>7.0f}ms | {t_total_ms*1000:>7.0f}ms | "
            f"{ler_ms:>7.4f} | {errors_ms:>5}/{args.shots}",
        )

        # --- native DEM sampler ---
        errors_ns, t_sample_ns, t_decode_ns = run_native_sampler(
            tc,
            noise_args,
            matching,
            args.shots,
            args.seed,
        )
        ler_ns = errors_ns / args.shots
        t_total_ns = t_sample_ns + t_decode_ns

        print(
            f"    {' ':>6} | {'native_sampler':>15} | {t_sample_ns*1000:>7.0f}ms | "
            f"{t_decode_ns*1000:>7.0f}ms | {t_total_ns*1000:>7.0f}ms | "
            f"{ler_ns:>7.4f} | {errors_ns:>5}/{args.shots}",
        )
        print()

    print("Notes:")
    print("  Circuit: Guppy surface code -> traced QIS -> lower_clifford_rotations()")
    print("  Decoder: PyMatching (stim DEM, decompose_errors=True)")
    print("  meas_sampling: geometric raw measurement DEM sampler + Python extraction")
    print("  native_sampler: DemSampler.generate_samples (detector events directly)")
    print("  LER differences from different RNG streams, not systematic bias.")


if __name__ == "__main__":
    main()
