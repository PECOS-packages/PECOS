#!/usr/bin/env python3
# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0
"""Benchmark: raw measurement sampling / detector DEM vs stabilizer simulation.

Compares detector DEM (generate_samples), raw meas_sampling, and stabilizer.
Quick smoke test by default (~10s). Use --full for stable headline numbers.

Usage:
    uv run python scripts/bench_raw_meas_sampling.py          # quick
    .venv/bin/python scripts/bench_raw_meas_sampling.py --full # full (release)
"""

import sys
import time

from pecos.qec.surface import SurfacePatch
from pecos.qec.surface.decode import _build_surface_tick_circuit_for_native_model
from pecos_rslib.qec import DemSampler
from pecos_rslib_exp import depolarizing, meas_sampling, sim_neo, stabilizer

FULL = "--full" in sys.argv


def build(d):
    patch = SurfacePatch.create(distance=d)
    return _build_surface_tick_circuit_for_native_model(patch, 6, "Z", circuit_source="abstract")


def main():
    noise_args = {"p1": 0.005, "p2": 0.005, "p_meas": 0.005, "p_prep": 0.005}
    depol = depolarizing().p1(0.005).p2(0.005).p_meas(0.005).p_prep(0.005)
    mode = "full" if FULL else "quick"

    print(f"Raw measurement / detector DEM benchmark ({mode}): surface code, 6 rounds, p=0.005")
    print("=" * 78)

    # ---- Section 1: Generation only ----
    gen_configs = (
        [(3, [100_000, 1_000_000]), (5, [100_000, 1_000_000]), (7, [100_000, 1_000_000])]
        if FULL
        else [(3, [10_000, 100_000]), (5, [10_000, 100_000])]
    )
    stab_limit = 100_000 if FULL else 10_000

    print()
    print("1. Generation only (no decoding):")
    print(f"{'d':>3} {'shots':>10} | {'Det DEM':>9} | {'Raw meas':>9} | {'Stab sim':>9} | {'stab/det':>9}")
    print("-" * 72)

    for d, shot_list in gen_configs:
        tc = build(d)
        sampler = DemSampler.from_circuit(tc, **noise_args)

        for shots in shot_list:
            t0 = time.perf_counter()
            _ = sampler.generate_samples(shots, seed=42)
            t_det = time.perf_counter() - t0

            t0 = time.perf_counter()
            _ = sim_neo(tc).quantum(meas_sampling()).noise(depol).shots(shots).seed(42).run()
            t_raw = time.perf_counter() - t0

            if shots <= stab_limit:
                t0 = time.perf_counter()
                _ = sim_neo(tc).quantum(stabilizer()).noise(depol).shots(shots).seed(42).run()
                t_stab = time.perf_counter() - t0
                stab_s = f"{t_stab * 1000:>8.0f}ms"
                ratio_s = f"{t_stab / t_det:>8.0f}x"
            else:
                stab_s = ratio_s = f"{'--':>9}"

            label = f"d={d}" if shots == shot_list[0] else ""
            print(
                f"{label:>3} {shots:>10,} | {t_det * 1000:>8.1f}ms | {t_raw * 1000:>8.1f}ms | {stab_s} | {ratio_s}",
            )
        print()

    # ---- Section 2: Generate + decode end-to-end ----
    dec_configs = [(3, [10_000, 100_000]), (5, [10_000, 100_000])] if FULL else [(3, [1_000, 10_000]), (5, [1_000])]

    print("2. Detector DEM generate + pymatching decode:")
    print(f"{'d':>3} {'shots':>10} | {'generate':>9} | {'decode':>9} | {'total':>9} | {'gen%':>6}")
    print("-" * 62)

    import stim
    from pecos.qec.surface.circuit_builder import tick_circuit_to_stim

    for d, shot_list in dec_configs:
        tc = build(d)
        sampler = DemSampler.from_circuit(tc, **noise_args)
        stim_str = tick_circuit_to_stim(tc, **noise_args)
        dem_str = str(stim.Circuit(stim_str).detector_error_model(decompose_errors=True))

        for shots in shot_list:
            t0 = time.perf_counter()
            batch = sampler.generate_samples(shots, seed=42)
            t_gen = time.perf_counter() - t0

            t0 = time.perf_counter()
            _ = batch.decode_count(dem_str, "pymatching")
            t_dec = time.perf_counter() - t0

            t_total = t_gen + t_dec
            gen_pct = t_gen / t_total * 100

            label = f"d={d}" if shots == shot_list[0] else ""
            print(
                f"{label:>3} {shots:>10,} | {t_gen * 1000:>8.1f}ms | "
                f"{t_dec * 1000:>8.1f}ms | {t_total * 1000:>8.1f}ms | {gen_pct:>5.1f}%",
            )
        print()

    print("Notes:")
    print("  generate_samples is 3-6x faster after columnar SampleBatch.")
    print("  End-to-end generate+decode is decode-dominated (<1% generation).")
    if not FULL:
        print("  Use --full for larger shot counts and stable headline numbers.")


if __name__ == "__main__":
    main()
