r"""Validate ALL DEM generators against stabilizer() ground truth.

Systematically tests each DEM generation path across:
- Multiple distances (d=2, 3)
- Both bases (Z, X)
- Each noise component independently + combined
- All DEM generators (DemBuilder, from_circuit, exact_detection_rates, perturbative_dem)

Uses sim_neo().quantum(stabilizer()) as ground truth for depolarizing noise.

Example:
    uv run python examples/surface/validate_dem_generators.py
    uv run python examples/surface/validate_dem_generators.py --shots 50000
    uv run python examples/surface/validate_dem_generators.py -d 2 --verbose
    uv run python examples/surface/validate_dem_generators.py --circuit-source both
    uv run python examples/surface/validate_dem_generators.py --circuit-source traced_qis
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python" / "quantum-pecos" / "src"))

# Noise configurations: each component independently + combined.
# "ground_truth" specifies which simulator to use:
#   "stabilizer" for depolarizing (exact, any distance)
#   "statevec" for coherent noise (exact, limited to small circuits)
NOISE_CONFIGS = [
    # Depolarizing components (ground truth: stabilizer)
    ("p_meas only", {"p_meas": 0.01}, "stabilizer"),
    ("p_prep only", {"p_prep": 0.01}, "stabilizer"),
    ("p1 only", {"p1": 0.01}, "stabilizer"),
    ("p2 only", {"p2": 0.01}, "stabilizer"),
    ("depol all", {"p1": 0.005, "p2": 0.005, "p_meas": 0.005, "p_prep": 0.005}, "stabilizer"),
    ("depol strong", {"p1": 0.01, "p2": 0.01, "p_meas": 0.01, "p_prep": 0.01}, "stabilizer"),
]

# Coherent noise configs (ground truth: statevec, small circuits only)
COHERENT_CONFIGS = [
    ("idle_rz only", {"idle_rz": 0.05}, "statevec"),
    ("rz+depol", {"idle_rz": 0.05, "p1": 0.005, "p2": 0.005, "p_meas": 0.005, "p_prep": 0.005}, "statevec"),
]

# Threshold for pass/fail (relative error vs stabilizer)
THRESHOLD = 0.15  # 15% — accounts for statistical noise at moderate shot counts


def build_circuit(distance, rounds, basis, circuit_source="abstract", *, fill_idle=False):
    """Build surface code TickCircuit.

    circuit_source:
        "abstract" — LogicalCircuitBuilder (direct gate construction)
        "traced_qis" — Guppy → Selene → QIS trace → TickCircuit
    fill_idle:
        Insert Idle(1) gates on inactive qubits each tick. Needed for
        realistic idle_rz noise (RZ applied when Idle gate is seen).
    """
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
        # Compilation passes for traced QIS circuits:
        tc.lower_clifford_rotations()  # RZ(pi/2) -> SZ, etc.
        tc.assign_missing_meas_ids()  # Stamp MeasId on MZ gates
    else:
        from pecos.qec.surface import LogicalCircuitBuilder

        b = LogicalCircuitBuilder()
        b.add_patch(patch, "Q0")
        b.add_memory("Q0", rounds=rounds, basis=basis)
        tc = b.to_tick_circuit()

    # Optional passes applied to all circuits:
    if fill_idle:
        # Insert Idle(1) after 2q gates (for idle_rz noise modeling)
        tc.insert_idle_after_two_qubit_gates(1.0)
        # Fill remaining inactive qubits with Idle gates
        tc.fill_idle_gates()

    return tc


def ground_truth_rates(tc, noise_kw, shots, seed, det_json, num_meas, num_dets, simulator="stabilizer"):
    """Ground truth: simulation with detector extraction."""
    from pecos_rslib_exp import depolarizing, sim_neo, stabilizer, statevec

    noise = depolarizing()
    for k, v in noise_kw.items():
        noise = getattr(noise, k)(v)

    backend = stabilizer() if simulator == "stabilizer" else statevec()
    results = sim_neo(tc).quantum(backend).noise(noise).shots(shots).seed(seed).run()
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
                rates[i] += 1.0 / shots
    return rates


def full_noise_kw(noise_kw):
    """Ensure all noise params are present (default 0 for missing)."""
    result = {
        "p1": noise_kw.get("p1", 0.0),
        "p2": noise_kw.get("p2", 0.0),
        "p_meas": noise_kw.get("p_meas", 0.0),
        "p_prep": noise_kw.get("p_prep", 0.0),
    }
    if "idle_rz" in noise_kw:
        result["idle_rz"] = noise_kw["idle_rz"]
    return result


def dem_sampler_rates(tc, noise_kw, shots, seed, num_dets):
    """DemSampler.from_circuit path."""
    from pecos_rslib.qec import DemSampler

    sampler = DemSampler.from_circuit(tc, **full_noise_kw(noise_kw))
    batch = sampler.generate_samples(num_shots=shots, seed=seed)
    rates = [0.0] * num_dets
    for i in range(shots):
        syn = batch.get_syndrome(i)
        for d in range(min(num_dets, len(syn))):
            if syn[d]:
                rates[d] += 1.0 / shots
    return rates


def dem_builder_rates(tc, noise_kw, shots, seed, num_dets):
    """DemBuilder path (explicit, uses .to_sampler())."""
    from pecos_rslib.qec import DagFaultAnalyzer, DemBuilder

    dag = tc.to_dag_circuit()
    analyzer = DagFaultAnalyzer(dag)
    influence = analyzer.build_influence_map()
    dem = (
        DemBuilder(influence)
        .with_noise(**full_noise_kw(noise_kw))
        .with_detectors_json(tc.get_meta("detectors"))
        .with_observables_json(tc.get_meta("observables"))
        .with_num_measurements(int(tc.get_meta("num_measurements")))
        .build()
    )
    sampler = dem.to_sampler()
    batch = sampler.generate_samples(num_shots=shots, seed=seed)
    rates = [0.0] * num_dets
    for i in range(shots):
        syn = batch.get_syndrome(i)
        for d in range(min(num_dets, len(syn))):
            if syn[d]:
                rates[d] += 1.0 / shots
    return rates


def heisenberg_rates(tc, noise_kw, num_dets):
    """Backward Heisenberg exact detection rates."""
    from pecos_rslib_exp import exact_detection_rates

    results = exact_detection_rates(tc, **full_noise_kw(noise_kw))
    rates = [0.0] * num_dets
    for det_id, prob in results:
        if det_id < num_dets:
            rates[det_id] = prob
    return rates


def perturbative_rates(tc, noise_kw, num_dets):
    """Forward EEG perturbative marginals."""
    from pecos_rslib_exp import perturbative_dem_events

    events = perturbative_dem_events(tc, **full_noise_kw(noise_kw))
    rates = [0.0] * num_dets
    for prob, dets, _obs in events:
        for d in dets:
            if d < num_dets:
                rates[d] += prob
    return rates


def max_rel_error(test_rates, ref_rates, min_rate=0.003):
    """Max relative error across detectors with rate > min_rate."""
    max_err = 0.0
    for t, r in zip(test_rates, ref_rates, strict=False):
        if r > min_rate:
            max_err = max(max_err, abs(t / r - 1))
    return max_err


def run_validation(*, distances, bases, shots, seed, verbose, circuit_sources, fill_idle):
    total_tests = 0
    total_pass = 0
    total_fail = 0
    failures = []

    # Generators: (name, func, supports_coherent)
    # from_circuit and DemBuilder only support depolarizing (no idle_rz)
    generators = [
        ("from_circuit", dem_sampler_rates, False),
        ("DemBuilder", dem_builder_rates, False),
        ("Heisenberg", None, True),
        ("Perturbative", None, True),
    ]

    for distance in distances:
        for basis in bases:
            for circuit_source in circuit_sources:
                try:
                    tc = build_circuit(distance, distance, basis, circuit_source, fill_idle=fill_idle)
                except Exception as e:
                    print(f"\n  d={distance} {basis} [{circuit_source}]: SKIP ({e})")
                    continue

                det_json = json.loads(tc.get_meta("detectors"))
                num_meas = int(tc.get_meta("num_measurements"))
                num_dets = len(det_json)

                src_label = f" [{circuit_source}]" if len(circuit_sources) > 1 else ""
                print(f"\n{'='*72}")
                print(f"d={distance} {basis}-basis{src_label}, {num_dets} detectors, {num_meas} measurements")
                print(f"{'='*72}")

                # Combine depolarizing + coherent configs
                all_configs = list(NOISE_CONFIGS) + list(COHERENT_CONFIGS)

                for noise_label, noise_kw, gt_simulator in all_configs:
                    is_coherent = "idle_rz" in noise_kw

                    # Ground truth
                    t0 = time.perf_counter()
                    try:
                        ref = ground_truth_rates(
                            tc,
                            noise_kw,
                            shots,
                            seed,
                            det_json,
                            num_meas,
                            num_dets,
                            simulator=gt_simulator,
                        )
                    except BaseException as e:
                        if verbose:
                            print(f"\n  {noise_label}: SKIP ground truth ({type(e).__name__}: {e})")
                        continue
                    ref_time = time.perf_counter() - t0

                    if verbose:
                        print(f"\n  {noise_label} ({gt_simulator}: {ref_time:.2f}s)")

                    for gen_name, gen_func, supports_coherent in generators:
                        # Skip non-EEG generators for coherent noise
                        if is_coherent and not supports_coherent:
                            if verbose:
                                print(f"    {gen_name:<14}          (skipped: no coherent support)")
                            continue
                        t0 = time.perf_counter()
                        try:
                            if gen_name == "Heisenberg":
                                test = heisenberg_rates(tc, noise_kw, num_dets)
                            elif gen_name == "Perturbative":
                                test = perturbative_rates(tc, noise_kw, num_dets)
                            else:
                                test = gen_func(tc, noise_kw, shots, seed, num_dets)
                            dt = time.perf_counter() - t0

                            err = max_rel_error(test, ref)
                            ok = err < THRESHOLD
                            total_tests += 1
                            if ok:
                                total_pass += 1
                            else:
                                total_fail += 1
                                failures.append(
                                    f"d={distance} {basis} {noise_label} {gen_name}: {err*100:.0f}%",
                                )

                            status = "PASS" if ok else f"FAIL({err*100:.0f}%)"
                            if verbose:
                                print(f"    {gen_name:<14} {dt*1000:>7.0f}ms  {status}")
                            elif not ok:
                                print(f"  {noise_label:<14} {gen_name:<14} {status}")

                        except Exception as e:
                            total_tests += 1
                            total_fail += 1
                            failures.append(
                                f"d={distance} {basis} {noise_label} {gen_name}: ERROR {e}",
                            )
                            if verbose:
                                print(f"    {gen_name:<14}  ERROR: {e}")

                if not verbose:
                    # Print summary for this config
                    pass

    # Final summary
    print(f"\n{'='*72}")
    print(f"VALIDATION SUMMARY: {total_pass}/{total_tests} passed, {total_fail} failed")
    print(f"Threshold: {THRESHOLD*100:.0f}% relative error ({shots} shots)")
    if failures:
        print("\nFailures:")
        for f in failures:
            print(f"  {f}")
    else:
        print("\nAll tests passed.")
    print(f"{'='*72}")

    return total_fail == 0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--distance",
        "-d",
        type=int,
        nargs="+",
        default=[2, 3],
        help="Code distances to test (default: 2 3)",
    )
    parser.add_argument(
        "--basis",
        choices=["X", "Z"],
        nargs="+",
        default=["Z", "X"],
        help="Bases to test (default: Z X)",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=20000,
        help="Shots per test (default: 20000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show per-generator timing and results",
    )
    parser.add_argument(
        "--circuit-source",
        choices=["abstract", "traced_qis", "both"],
        default="abstract",
        help="Circuit construction pipeline (default: abstract)",
    )
    parser.add_argument(
        "--fill-idle",
        action="store_true",
        help="Insert Idle(1) gates on inactive qubits (needed for idle_rz noise)",
    )
    args = parser.parse_args()

    sources = ["abstract", "traced_qis"] if args.circuit_source == "both" else [args.circuit_source]

    ok = run_validation(
        distances=args.distance,
        bases=args.basis,
        shots=args.shots,
        seed=args.seed,
        verbose=args.verbose,
        circuit_sources=sources,
        fill_idle=args.fill_idle,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
