"""Maximum likelihood lookup decoder from simulation samples.

For small codes (d=3: 256 syndromes), precomputes the optimal correction
for every possible syndrome by counting simulation outcomes.  This is the
provably optimal decoder — useful as a gold standard for validation.

Example:
    uv run python examples/surface/ml_lookup_decoder.py
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python" / "quantum-pecos" / "src"))


def build_lookup_table(batch, num_detectors: int) -> dict[tuple[int, ...], int]:
    """Build a syndrome -> most_likely_observable_mask lookup table.

    For each unique syndrome observed in the batch, count how often each
    observable mask occurs. The most common mask is the ML prediction.
    """
    # syndrome (as tuple of fired detector indices) -> {obs_mask: count}
    syndrome_counts: dict[tuple[int, ...], dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for i in range(batch.num_shots):
        syn = batch.get_syndrome(i)
        obs = batch.get_observable_mask(i)

        # Convert syndrome to tuple of fired detector indices
        fired = tuple(d for d in range(min(num_detectors, len(syn))) if syn[d])
        syndrome_counts[fired][obs] += 1

    # For each syndrome, pick the most likely observable mask
    table: dict[tuple[int, ...], int] = {}
    for syndrome, counts in syndrome_counts.items():
        best_mask = max(counts, key=counts.get)
        table[syndrome] = best_mask

    return table


def decode_with_lookup(batch, table: dict, num_detectors: int) -> tuple[int, int]:
    """Decode a batch using the lookup table. Returns (num_errors, num_shots)."""
    errors = 0
    for i in range(batch.num_shots):
        syn = batch.get_syndrome(i)
        obs_true = batch.get_observable_mask(i)

        fired = tuple(d for d in range(min(num_detectors, len(syn))) if syn[d])
        predicted = table.get(fired, 0)  # default: no correction

        if predicted != obs_true:
            errors += 1

    return errors, batch.num_shots


def main():
    parser = argparse.ArgumentParser(description="ML lookup decoder from simulation")
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--shots", type=int, default=50_000)
    parser.add_argument("--basis", default="Z")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--p2", type=float, default=0.005)
    parser.add_argument("--idle-rz", type=float, default=0.0)
    parser.add_argument(
        "--sample-backend",
        default="stabilizer",
        choices=["stabilizer", "statevec", "native"],
    )
    args = parser.parse_args()

    import json

    from pecos.qec.surface import SurfacePatch
    from pecos.qec.surface.decode import _build_surface_tick_circuit_for_native_model

    patch = SurfacePatch.create(distance=args.distance)
    num_rounds = 2 * args.distance
    tc = _build_surface_tick_circuit_for_native_model(
        patch,
        num_rounds,
        args.basis,
        circuit_source="abstract",
    )

    num_dets = len(json.loads(tc.get_meta("detectors")))
    print(f"d={args.distance}, {num_rounds} rounds, {num_dets} detectors, {2**num_dets} possible syndromes")

    noise_params = {
        "p1": args.p2 / 10,
        "p2": args.p2,
        "p_meas": args.p2,
        "p_prep": args.p2,
        "idle_rz": args.idle_rz,
    }

    # Generate training samples
    print(f"Generating {args.shots} training samples ({args.sample_backend})...")
    t0 = time.perf_counter()

    if args.sample_backend in ("stabilizer", "statevec"):
        from pecos_rslib.qec import SampleBatch
        from pecos_rslib_exp import depolarizing, sim_neo, stabilizer, statevec

        noise = (
            depolarizing()
            .p1(noise_params["p1"])
            .p2(noise_params["p2"])
            .p_meas(noise_params["p_meas"])
            .p_prep(noise_params["p_prep"])
        )
        if args.idle_rz > 0:
            noise = noise.idle_rz(args.idle_rz)

        backend = stabilizer() if args.sample_backend == "stabilizer" else statevec()
        results = sim_neo(tc).quantum(backend).noise(noise).shots(args.shots).seed(args.seed).run()

        det_json = json.loads(tc.get_meta("detectors"))
        obs_json = json.loads(tc.get_meta("observables"))
        num_meas = int(tc.get_meta("num_measurements"))

        # Convert to SampleBatch
        detection_events = []
        observable_masks = []
        for r in results:
            meas = list(r)
            syn = [0] * num_dets
            for i, det in enumerate(det_json):
                val = 0
                for rec in det["records"]:
                    idx = num_meas + rec
                    if 0 <= idx < len(meas):
                        val ^= meas[idx]
                syn[i] = val
            detection_events.append(syn)
            obs_mask = 0
            for j, ob in enumerate(obs_json):
                val = 0
                for rec in ob["records"]:
                    idx = num_meas + rec
                    if 0 <= idx < len(meas):
                        val ^= meas[idx]
                if val:
                    obs_mask |= 1 << j
            observable_masks.append(obs_mask)
        train_batch = SampleBatch(detection_events, observable_masks)
    else:
        from pecos_rslib.qec import DemSampler

        sampler_params = {k: v for k, v in noise_params.items() if k in ("p1", "p2", "p_meas", "p_prep")}
        sampler = DemSampler.from_circuit(tc, **sampler_params)
        train_batch = sampler.generate_samples(args.shots, seed=args.seed)

    t_sample = time.perf_counter() - t0
    print(f"  Sampled in {t_sample:.2f}s")

    # Build lookup table
    t0 = time.perf_counter()
    table = build_lookup_table(train_batch, num_dets)
    t_build = time.perf_counter() - t0
    print(f"  Lookup table: {len(table)} unique syndromes seen (of {2**num_dets} possible)")
    print(f"  Built in {t_build:.4f}s")

    # Test on separate samples
    test_shots = args.shots
    print(f"\nGenerating {test_shots} test samples...")
    if args.sample_backend in ("stabilizer", "statevec"):
        results2 = sim_neo(tc).quantum(backend).noise(noise).shots(test_shots).seed(args.seed + 1000).run()
        detection_events2 = []
        observable_masks2 = []
        for r in results2:
            meas = list(r)
            syn = [0] * num_dets
            for i, det in enumerate(det_json):
                val = 0
                for rec in det["records"]:
                    idx = num_meas + rec
                    if 0 <= idx < len(meas):
                        val ^= meas[idx]
                syn[i] = val
            detection_events2.append(syn)
            obs_mask = 0
            for j, ob in enumerate(obs_json):
                val = 0
                for rec in ob["records"]:
                    idx = num_meas + rec
                    if 0 <= idx < len(meas):
                        val ^= meas[idx]
                if val:
                    obs_mask |= 1 << j
            observable_masks2.append(obs_mask)
        test_batch = SampleBatch(detection_events2, observable_masks2)
    else:
        test_batch = sampler.generate_samples(test_shots, seed=args.seed + 1000)

    # Decode with lookup
    errors_lookup, n = decode_with_lookup(test_batch, table, num_dets)
    ler_lookup = errors_lookup / n

    # Compare with pymatching
    from pecos.qec.surface import NoiseModel
    from pecos.qec.surface.decode import generate_circuit_level_dem_from_builder

    noise_obj = NoiseModel(
        p1=noise_params["p1"],
        p2=noise_params["p2"],
        p_meas=noise_params["p_meas"],
        p_prep=noise_params["p_prep"],
    )
    dem_decomp = generate_circuit_level_dem_from_builder(
        patch,
        num_rounds,
        noise_obj,
        basis=args.basis,
        decompose_errors=True,
        circuit_source="abstract",
    )
    dem_clean = "\n".join(line for line in dem_decomp.split("\n") if not line.startswith("logical_observable"))
    stats_pm = test_batch.decode_stats(dem_clean, "pymatching")

    # Compare with coherent_dem_decomposed if available
    try:
        from pecos_rslib_exp import coherent_dem_decomposed

        _, coherent_decomp = coherent_dem_decomposed(tc, **noise_params)
        coherent_clean = "\n".join(
            line for line in coherent_decomp.split("\n") if not line.startswith("logical_observable")
        )
        stats_coherent = test_batch.decode_stats(coherent_clean, "pymatching")
        ler_coherent = stats_coherent.logical_error_rate
    except Exception:
        ler_coherent = None

    print(f"\n{'='*60}")
    print(f"Results (d={args.distance}, p2={args.p2}, irz={args.idle_rz}):")
    print(f"{'='*60}")
    print(f"  ML Lookup:                  LER = {ler_lookup:.6f}  ({errors_lookup}/{n})")
    print(f"  PyMatching (from_circuit):  LER = {stats_pm.logical_error_rate:.6f}  ({stats_pm.num_errors}/{n})")
    if ler_coherent is not None:
        print(f"  PyMatching (coherent):      LER = {ler_coherent:.6f}")
    if stats_pm.logical_error_rate > 0:
        improvement = (stats_pm.logical_error_rate - ler_lookup) / stats_pm.logical_error_rate * 100
        print(f"\n  ML Lookup vs PyMatching:    {improvement:+.1f}%")


if __name__ == "__main__":
    main()
