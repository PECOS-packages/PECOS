r"""DEM method x decoder LER comparison on traced_qis circuits.

Generates a traced_qis surface code circuit, builds DEMs from multiple
methods, samples once, and decodes with multiple decoders.  Reports LER
for each (DEM method, decoder) combination.

DEM methods:
  1. from_circuit   — non-EEG backward Pauli propagation (stochastic only)
  2. coherent_exact — EEG backward Heisenberg + L-BFGS fit
  3. noise_char     — EEG unified (correlations + mechanisms + DEM)
  4. perturbative   — EEG forward pass (fast, approximate)

Each method produces a raw DEM string.  MWPM decoders (pymatching,
fusion_blossom) use the standard decomposed DEM from ``from_circuit``
since they cannot handle hyperedges.  Non-MWPM decoders (tesseract,
bp_osd) use the raw DEM from each method.

Example:
    uv run python examples/surface/dem_method_ler_comparison.py
    uv run python examples/surface/dem_method_ler_comparison.py \
        --distances 3 5 --shots 50000 --decoders pymatching tesseract
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python" / "quantum-pecos" / "src"))

# Slow decoders that benefit from parallel decoding
SLOW_DECODERS = {"tesseract", "bp_osd"}


def _decoder_requires_graphlike(decoder: str) -> bool:
    """Check if a decoder requires graphlike (decomposed) DEMs."""
    from pecos_rslib.qec import decoder_dem_requirement

    base = decoder.split(":", maxsplit=1)[0]
    return decoder_dem_requirement(base) == "graphlike"


def sim_results_to_sample_batch(results, det_json, obs_json, num_meas):
    """Convert sim_neo() results to a SampleBatch.

    Computes detection events from detector record XOR definitions,
    and observable flips from observable record XOR definitions.
    """
    import json

    from pecos_rslib.qec import SampleBatch

    dets = det_json if isinstance(det_json, list) else json.loads(det_json)
    obs = obs_json if isinstance(obs_json, list) else json.loads(obs_json)
    num_dets = len(dets)
    len(obs)

    detection_events = []
    observable_masks = []

    for r in results:
        meas = list(r)

        # Detection events: XOR of records per detector
        syn = [0] * num_dets
        for i, det in enumerate(dets):
            val = 0
            for rec in det["records"]:
                idx = num_meas + rec
                if 0 <= idx < len(meas):
                    val ^= meas[idx]
            syn[i] = val
        detection_events.append(syn)

        # Observable flips: XOR of records per observable
        obs_mask = 0
        for j, ob in enumerate(obs):
            val = 0
            for rec in ob["records"]:
                idx = num_meas + rec
                if 0 <= idx < len(meas):
                    val ^= meas[idx]
            if val:
                obs_mask |= 1 << j
        observable_masks.append(obs_mask)

    return SampleBatch(detection_events, observable_masks)


def build_tick_circuits(distance: int, num_rounds: int, basis: str):
    """Build both abstract and traced_qis TickCircuits for a surface code.

    Returns (patch, abstract_tc, traced_tc).
    EEG methods need the abstract circuit (CX/H gate set).
    Non-EEG DemBuilder uses the traced circuit (physical gates).
    """
    from pecos.qec.surface import SurfacePatch
    from pecos.qec.surface.decode import _build_surface_tick_circuit_for_native_model

    patch = SurfacePatch.create(distance=distance)
    abstract_tc = _build_surface_tick_circuit_for_native_model(
        patch,
        num_rounds,
        basis,
        circuit_source="abstract",
    )
    traced_tc = _build_surface_tick_circuit_for_native_model(
        patch,
        num_rounds,
        basis,
        circuit_source="traced_qis",
    )
    return patch, abstract_tc, traced_tc


def generate_dems(
    abstract_tc,
    _traced_tc,
    patch,
    num_rounds,
    noise_params: dict,
    basis: str,
) -> list[tuple[str, str, str | None]]:
    """Generate DEM strings from all methods.

    EEG methods use the abstract circuit (CX/H gate set).
    Non-EEG DemBuilder uses the traced circuit (physical Selene gates).

    Returns list of (method_name, raw_dem, decomposed_dem_or_None).
    decomposed_dem is None when the method cannot produce a graphlike DEM.
    """
    from pecos.qec.surface import NoiseModel
    from pecos.qec.surface.decode import generate_circuit_level_dem_from_builder

    results = []

    noise = NoiseModel(
        p1=noise_params.get("p1", 0.0),
        p2=noise_params.get("p2", 0.0),
        p_meas=noise_params.get("p_meas", 0.0),
        p_prep=noise_params.get("p_prep", 0.0),
    )

    # 1. from_circuit on traced (non-EEG, stochastic, physical gates)
    try:
        raw = generate_circuit_level_dem_from_builder(
            patch,
            num_rounds,
            noise,
            basis=basis,
            decompose_errors=False,
            circuit_source="traced_qis",
        )
        decomp = generate_circuit_level_dem_from_builder(
            patch,
            num_rounds,
            noise,
            basis=basis,
            decompose_errors=True,
            circuit_source="traced_qis",
        )
        results.append(("from_circuit_traced", raw, decomp))
    except Exception as e:
        print(f"  WARN: from_circuit_traced failed: {e}")

    # 1b. from_circuit on abstract (non-EEG, stochastic, logical gates)
    try:
        raw = generate_circuit_level_dem_from_builder(
            patch,
            num_rounds,
            noise,
            basis=basis,
            decompose_errors=False,
            circuit_source="abstract",
        )
        decomp = generate_circuit_level_dem_from_builder(
            patch,
            num_rounds,
            noise,
            basis=basis,
            decompose_errors=True,
            circuit_source="abstract",
        )
        results.append(("from_circuit_abstract", raw, decomp))
    except Exception as e:
        print(f"  WARN: from_circuit_abstract failed: {e}")

    # EEG methods use the abstract circuit (CX/H gate set)
    # 2. coherent_dem_decomposed (EEG, X/Z Pauli-aware decomposition)
    try:
        from pecos_rslib_exp import coherent_dem_decomposed

        raw_dem, decomp_dem = coherent_dem_decomposed(abstract_tc, **noise_params)
        if raw_dem.strip():
            results.append(("coherent_decomp", raw_dem, decomp_dem))
    except Exception as e:
        print(f"  WARN: coherent_dem_decomposed failed: {e}")

    # 3. noise_characterization (EEG, unified)
    try:
        from pecos_rslib_exp import noise_characterization

        _json_str, dem_raw, dem_decomp = noise_characterization(abstract_tc, **noise_params)
        if dem_raw.strip():
            results.append(("noise_char", dem_raw, dem_decomp))
    except Exception as e:
        print(f"  WARN: noise_characterization failed: {e}")

    # 4. perturbative_dem (EEG, forward)
    try:
        from pecos_rslib_exp import perturbative_dem

        dem_raw, dem_decomp = perturbative_dem(abstract_tc, **noise_params)
        if dem_raw.strip():
            results.append(("perturbative", dem_raw, dem_decomp))
    except Exception as e:
        print(f"  WARN: perturbative_dem failed: {e}")

    return results


def _sample_from_sim(tc, noise_params, shots, seed, backend="statevec"):
    """Sample using sim_neo (captures actual noise including coherent).

    backend: "statevec" (exact, small circuits),
             "stabilizer" (exact for depol, fast — no coherent idle_rz),
             "stab_mps" (handles coherent noise, any distance, approximate).
    """
    import json

    from pecos_rslib_exp import depolarizing, sim_neo, stab_mps, stabilizer, statevec

    p1 = noise_params.get("p1", 0.0)
    p2 = noise_params.get("p2", 0.0)
    p_meas = noise_params.get("p_meas", 0.0)
    p_prep = noise_params.get("p_prep", 0.0)
    irz = noise_params.get("idle_rz", 0.0)

    noise = depolarizing().p1(p1).p2(p2).p_meas(p_meas).p_prep(p_prep)
    if irz > 0:
        noise = noise.idle_rz(irz)

    if backend == "stabilizer":
        quantum_backend = stabilizer()
    elif backend == "stab_mps":
        quantum_backend = stab_mps()
    else:
        quantum_backend = statevec()

    results = sim_neo(tc).quantum(quantum_backend).noise(noise).shots(shots).seed(seed).run()

    det_json = json.loads(tc.get_meta("detectors"))
    obs_json = json.loads(tc.get_meta("observables"))
    num_meas = int(tc.get_meta("num_measurements"))

    return sim_results_to_sample_batch(results, det_json, obs_json, num_meas)


def strip_logical_observable_lines(dem_str: str) -> str:
    """Remove logical_observable lines that some decoders choke on."""
    return "\n".join(line for line in dem_str.split("\n") if not line.startswith("logical_observable"))


def run_comparison(
    *,
    distances: list[int],
    noise_configs: list[tuple[str, dict]],
    decoders: list[str],
    basis: str,
    shots: int,
    seed: int,
    sample_backend: str,
):
    """Run the full DEM method x decoder comparison."""
    all_results = []

    for distance in distances:
        num_rounds = 2 * distance
        for noise_label, noise_params in noise_configs:
            print(f"\n{'='*72}")
            print(f"d={distance} {basis}-basis, {num_rounds} rounds, {noise_label}")
            print(f"  sample_backend={sample_backend}")
            print(f"{'='*72}")

            # Build both abstract and traced circuits
            t0 = time.perf_counter()
            patch, abstract_tc, traced_tc = build_tick_circuits(distance, num_rounds, basis)
            t_circuit = time.perf_counter() - t0
            print(f"  Circuits built in {t_circuit:.2f}s")

            sampler_params = {k: v for k, v in noise_params.items() if k in ("p1", "p2", "p_meas", "p_prep")}
            idle_rz = noise_params.get("idle_rz", 0.0)

            # Generate samples
            t0 = time.perf_counter()
            if sample_backend in ("statevec", "stabilizer", "stab_mps"):
                # Simulator-based sampling
                batch = _sample_from_sim(
                    abstract_tc,
                    noise_params,
                    shots,
                    seed,
                    backend=sample_backend,
                )
            else:
                # DemSampler: fast, stochastic-only sampling
                from pecos_rslib.qec import DemSampler

                sampler = DemSampler.from_circuit(
                    traced_tc,
                    **sampler_params,
                    idle_rz=idle_rz if idle_rz > 0 else None,
                )
                batch = sampler.generate_samples(shots, seed=seed)
            t_sample = time.perf_counter() - t0
            print(f"  Sampled {shots} shots in {t_sample:.2f}s")

            # Generate DEMs from all methods
            t0 = time.perf_counter()
            dems = generate_dems(abstract_tc, traced_tc, patch, num_rounds, noise_params, basis)
            t_dems = time.perf_counter() - t0
            print(f"  Generated {len(dems)} DEMs in {t_dems:.2f}s")
            for name, dem_str, _decomp in dems:
                n_lines = len([line for line in dem_str.strip().split("\n") if line.strip()])
                print(f"    {name}: {n_lines} lines")

            # Build column headers: for raw-capable decoders, show both raw and decomposed
            columns = []
            for dec in decoders:
                if _decoder_requires_graphlike(dec):
                    columns.append((dec, "decomp"))
                else:
                    columns.append((dec, "raw"))
                    columns.append((dec, "decomp"))

            # Print header
            print(f"\n  {'DEM Method':<22s}", end="")
            for dec, dem_type in columns:
                label = f"{dec}({dem_type[0]})" if dem_type == "raw" else f"{dec}(d)"
                print(f" | {label:>16s}", end="")
            print()
            print(f"  {'-'*22}", end="")
            for _ in columns:
                print(f" | {'-'*16}", end="")
            print()

            for dem_name, dem_raw, dem_decomp in dems:
                dem_raw_clean = strip_logical_observable_lines(dem_raw)
                dem_decomp_clean = strip_logical_observable_lines(dem_decomp) if dem_decomp else None
                print(f"  {dem_name:<22s}", end="", flush=True)

                for decoder, dem_type in columns:
                    base = decoder.split(":")[0]

                    if dem_type == "decomp" and dem_decomp_clean is None:
                        # No decomposed DEM available for this method
                        print(f" | {'N/A':>16s}", end="")
                        continue

                    dem = dem_raw_clean if dem_type == "raw" else dem_decomp_clean

                    try:
                        if base in SLOW_DECODERS:
                            stats = batch.decode_stats_parallel(dem, decoder)
                        else:
                            stats = batch.decode_stats(dem, decoder)
                        ler = stats.logical_error_rate
                        print(f" | {ler:>16.4f}", end="")
                        all_results.append(
                            {
                                "distance": distance,
                                "noise": noise_label,
                                "dem_method": dem_name,
                                "decoder": decoder,
                                "dem_type": dem_type,
                                "num_shots": shots,
                                "num_errors": stats.num_errors,
                                "ler": ler,
                                "decode_s": stats.total_seconds,
                            },
                        )
                    except Exception:
                        # Decoder can't handle this DEM (e.g., hyperedges in graphlike DEM)
                        print(f" | {'N/A':>16s}", end="")

                print()

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="DEM method x decoder LER comparison on traced_qis circuits",
    )
    parser.add_argument("--distances", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--shots", type=int, default=10_000)
    parser.add_argument("--basis", default="Z")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--decoders",
        nargs="+",
        default=["pymatching", "fusion_blossom", "tesseract"],
    )
    parser.add_argument("--p2", type=float, default=0.005)
    parser.add_argument("--idle-rz", type=float, default=0.05)
    parser.add_argument(
        "--noise",
        nargs="+",
        default=["depol", "depol+irz"],
        choices=["depol", "depol+irz"],
    )
    parser.add_argument(
        "--sample-backend",
        default="native",
        choices=["native", "statevec", "stabilizer", "stab_mps"],
        help="'native' uses DemSampler (fast, stochastic). "
        "'statevec' uses exact state vector sim (slow, captures coherent). "
        "'stabilizer' uses stabilizer sim (fast, exact for depolarizing). "
        "'stab_mps' uses tensor network sim (handles coherent, any distance).",
    )
    args = parser.parse_args()

    p2 = args.p2
    noise_configs = []
    if "depol" in args.noise:
        noise_configs.append(
            (
                f"depol(p2={p2})",
                {"p1": p2 / 10, "p2": p2, "p_meas": p2, "p_prep": p2, "idle_rz": 0.0},
            ),
        )
    if "depol+irz" in args.noise:
        noise_configs.append(
            (
                f"depol+irz(p2={p2},irz={args.idle_rz})",
                {"p1": p2 / 10, "p2": p2, "p_meas": p2, "p_prep": p2, "idle_rz": args.idle_rz},
            ),
        )

    results = run_comparison(
        distances=args.distances,
        noise_configs=noise_configs,
        decoders=args.decoders,
        basis=args.basis,
        shots=args.shots,
        seed=args.seed,
        sample_backend=args.sample_backend,
    )

    print(f"\n\nTotal results: {len(results)}")


if __name__ == "__main__":
    main()
