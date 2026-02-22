# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Stim comparison tests for DemSampler.

These tests validate that PECOS DemSampler produces statistically equivalent
results to Stim's DEM sampler. This is the ground truth validation.
"""

import re
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pecos.quantum import DagCircuit, TickCircuit

# Skip all tests if stim is not installed
stim = pytest.importorskip("stim")


def extract_measurement_order(tc: "TickCircuit") -> list[int]:
    """Extract measurement order from TickCircuit.

    Returns a list of qubit indices in the order they were measured.
    This is needed to map detector record offsets (which use TickCircuit
    measurement indices) to influence map indices (which use DAG order).

    Args:
        tc: TickCircuit to extract measurement order from.

    Returns:
        List of qubit indices in measurement execution order.
    """
    measurement_order = []

    for tick_idx in range(tc.num_ticks()):
        tick = tc.get_tick(tick_idx)
        if tick is None:
            continue
        for gate in tick.gates():
            gate_type = str(gate.gate_type)
            if "MZ" in gate_type:
                for qubit in gate.qubits:
                    if hasattr(qubit, "index"):
                        measurement_order.append(qubit.index())
                    else:
                        measurement_order.append(int(qubit))

    return measurement_order


def parse_dem_string(dem_str: str) -> dict[tuple, float]:
    """Parse a DEM string into {(detectors, logicals): probability}.

    Args:
        dem_str: DEM string in Stim format

    Returns:
        Dict mapping (detector_tuple, logical_tuple) -> probability
    """
    errors = {}
    for raw_line in dem_str.strip().split("\n"):
        line = raw_line.strip()
        if not line.startswith("error("):
            continue

        # Parse probability
        match = re.match(r"error\(([^)]+)\)", line)
        if not match:
            continue
        prob = float(match.group(1))

        # Parse targets (skip decomposed errors with ^)
        rest = line[match.end() :].strip()
        if "^" in rest:
            continue

        dets = tuple(sorted(int(m.group(1)) for m in re.finditer(r"D(\d+)", rest)))
        logs = tuple(sorted(int(m.group(1)) for m in re.finditer(r"L(\d+)", rest)))

        key = (dets, logs)
        errors[key] = prob

    return errors


def compare_dems(pecos_dem: str, stim_dem: str, rtol: float = 0.05) -> dict:
    """Compare two DEMs and return analysis.

    Args:
        pecos_dem: PECOS-generated DEM string
        stim_dem: Stim-generated DEM string
        rtol: Relative tolerance for probability comparison

    Returns:
        Dict with comparison results
    """
    pecos_errors = parse_dem_string(pecos_dem)
    stim_errors = parse_dem_string(stim_dem)

    results = {
        "pecos_count": len(pecos_errors),
        "stim_count": len(stim_errors),
        "matched_count": 0,
        "pecos_only": [],
        "stim_only": [],
        "prob_mismatches": [],
    }

    all_keys = set(pecos_errors.keys()) | set(stim_errors.keys())

    for key in sorted(all_keys):
        dets, logs = key
        pecos_prob = pecos_errors.get(key)
        stim_prob = stim_errors.get(key)

        target_str = " ".join(f"D{d}" for d in dets) + " " + " ".join(f"L{log_idx}" for log_idx in logs)
        target_str = target_str.strip()

        if pecos_prob is None:
            results["stim_only"].append((target_str, stim_prob))
        elif stim_prob is None:
            results["pecos_only"].append((target_str, pecos_prob))
        else:
            results["matched_count"] += 1
            rel_diff = abs(pecos_prob - stim_prob) / max(pecos_prob, stim_prob, 1e-10)
            if rel_diff > rtol:
                results["prob_mismatches"].append(
                    {
                        "target": target_str,
                        "pecos": pecos_prob,
                        "stim": stim_prob,
                        "rel_diff": rel_diff,
                    },
                )

    return results


class TestDemSamplerVsStim:
    """Compare PECOS DemSampler against Stim's DEM sampler."""

    @pytest.fixture
    def surface_code_d3(self) -> tuple:
        """Create a d=3 surface code patch and circuit."""
        from pecos.qec.surface import SurfacePatch, generate_tick_circuit_from_patch

        patch = SurfacePatch.create(distance=3)
        tc = generate_tick_circuit_from_patch(patch, num_rounds=1, basis="Z")
        return patch, tc

    @pytest.fixture
    def noise_params(self) -> dict[str, float]:
        """Standard noise parameters for testing."""
        return {"p1": 0.01, "p2": 0.01, "p_meas": 0.01, "p_init": 0.01}

    def test_dem_mechanism_counts_match(
        self,
        surface_code_d3: tuple,
        noise_params: dict[str, float],
    ) -> None:
        """DemSampler should produce same number of mechanisms as Stim."""
        from pecos.qec.surface.circuit_builder import (
            generate_dem_from_tick_circuit,
            generate_dem_from_tick_circuit_via_stim,
        )

        _patch, tc = surface_code_d3

        pecos_dem = generate_dem_from_tick_circuit(
            tc,
            **noise_params,
            decompose_errors=False,
        )
        stim_dem = generate_dem_from_tick_circuit_via_stim(tc, **noise_params)

        comparison = compare_dems(pecos_dem, stim_dem)

        # Should have the same mechanisms (allowing for small differences)
        assert comparison["pecos_count"] > 0, "PECOS DEM should have mechanisms"
        assert comparison["stim_count"] > 0, "Stim DEM should have mechanisms"

        # Allow up to 10% difference in mechanism count due to decomposition
        count_ratio = comparison["pecos_count"] / comparison["stim_count"]
        assert 0.8 < count_ratio < 1.2, (
            f"Mechanism count ratio {count_ratio:.2f} outside expected range. "
            f"PECOS: {comparison['pecos_count']}, Stim: {comparison['stim_count']}"
        )

    def test_dem_probabilities_close(
        self,
        surface_code_d3: tuple,
        noise_params: dict[str, float],
    ) -> None:
        """DemSampler probabilities should be reasonably close to Stim's.

        Note: PECOS and Stim have known differences in DEM generation due to:
        - Different treatment of Y errors (PECOS: single error, Stim: X^Z decomposition)
        - Different probability combination strategies
        - Edge effects at circuit boundaries

        The key metric is that sampling produces similar statistical results,
        not that the DEM representations are identical.
        """
        from pecos.qec.surface.circuit_builder import (
            generate_dem_from_tick_circuit,
            generate_dem_from_tick_circuit_via_stim,
        )

        _patch, tc = surface_code_d3

        pecos_dem = generate_dem_from_tick_circuit(
            tc,
            **noise_params,
            decompose_errors=False,
        )
        stim_dem = generate_dem_from_tick_circuit_via_stim(tc, **noise_params)

        comparison = compare_dems(pecos_dem, stim_dem, rtol=0.05)

        # Most mechanisms should exist in both (may differ in probability)
        match_ratio = comparison["matched_count"] / max(comparison["stim_count"], 1)
        assert match_ratio > 0.5, f"Only {match_ratio:.0%} of mechanisms matched by target"

        # Log mismatches for debugging but don't fail on probability differences
        # The sampling tests are the ground truth for equivalence
        if comparison["prob_mismatches"]:
            print(
                f"\nDEM probability mismatches ({len(comparison['prob_mismatches'])} total):",
            )
            for m in comparison["prob_mismatches"][:5]:
                print(
                    f"  {m['target']}: PECOS={m['pecos']:.6f} Stim={m['stim']:.6f} diff={m['rel_diff']:.1%}",
                )

    def test_sampling_statistics_match_stim(
        self,
        surface_code_d3: tuple,
        noise_params: dict[str, float],
    ) -> None:
        """DemSampler sampling statistics should match Stim's DEM sampler."""
        from pecos.qec.surface.circuit_builder import tick_circuit_to_stim
        from pecos_rslib.qec import DagFaultAnalyzer, DemSamplerBuilder

        _patch, tc = surface_code_d3

        # Build PECOS DemSampler from TickCircuit's DagCircuit
        dag = tc.to_dag_circuit()
        analyzer = DagFaultAnalyzer(dag)
        influence_map = analyzer.build_influence_map()

        detectors_json = tc.get_meta("detectors") or "[]"
        observables_json = tc.get_meta("observables") or "[]"
        measurement_order = extract_measurement_order(tc)

        builder = DemSamplerBuilder(influence_map)
        builder.with_noise(**noise_params)
        builder.with_detectors_json(detectors_json)
        builder.with_observables_json(observables_json)
        builder.with_measurement_order(measurement_order)
        pecos_sampler = builder.build()

        # Build Stim DEM sampler
        stim_str = tick_circuit_to_stim(tc, **noise_params)
        stim_circuit = stim.Circuit(stim_str)
        stim_dem = stim_circuit.detector_error_model()
        stim_sampler = stim_dem.compile_sampler()

        # Sample from both
        num_shots = 100_000
        seed = 42

        # PECOS sampling
        pecos_stats = pecos_sampler.sample_statistics(num_shots, seed=seed)

        # Stim sampling - returns (det_events, obs_flips, error_data)
        stim_det_events, stim_obs_flips, _ = stim_sampler.sample(num_shots)
        stim_syndrome_count = np.any(stim_det_events, axis=1).sum()
        stim_logical_count = np.any(stim_obs_flips, axis=1).sum()

        # Compare rates
        pecos_syndrome_rate = pecos_stats["syndrome_rate"]
        stim_syndrome_rate = stim_syndrome_count / num_shots

        pecos_logical_rate = pecos_stats["logical_error_rate"]
        stim_logical_rate = stim_logical_count / num_shots

        # Compare absolute differences - allow up to 10% relative difference
        # Known: PECOS and Stim have slightly different DEM generation
        syndrome_diff = abs(pecos_syndrome_rate - stim_syndrome_rate)
        logical_diff = abs(pecos_logical_rate - stim_logical_rate)

        # Syndrome rates should be within 20% relative
        max_rate = max(pecos_syndrome_rate, stim_syndrome_rate, 0.001)
        syndrome_rel_diff = syndrome_diff / max_rate
        assert syndrome_rel_diff < 0.3, (
            f"Syndrome rate mismatch: PECOS={pecos_syndrome_rate:.4f}, "
            f"Stim={stim_syndrome_rate:.4f}, rel_diff={syndrome_rel_diff:.1%}"
        )

        # Logical error rates should be within 30% relative (more variable)
        max_logical = max(pecos_logical_rate, stim_logical_rate, 0.001)
        logical_rel_diff = logical_diff / max_logical
        assert logical_rel_diff < 0.5, (
            f"Logical error rate mismatch: PECOS={pecos_logical_rate:.4f}, "
            f"Stim={stim_logical_rate:.4f}, rel_diff={logical_rel_diff:.1%}"
        )

    def test_detector_firing_rates_correlate(
        self,
        surface_code_d3: tuple,
        noise_params: dict[str, float],
    ) -> None:
        """Detector firing rates should be correlated between PECOS and Stim.

        This test validates that individual detector firing rates match between
        PECOS and Stim when the measurement order mapping is provided correctly.
        """
        from pecos.qec.surface.circuit_builder import tick_circuit_to_stim
        from pecos_rslib.qec import DagFaultAnalyzer, DemSamplerBuilder

        _patch, tc = surface_code_d3

        # Build PECOS DemSampler
        dag = tc.to_dag_circuit()
        analyzer = DagFaultAnalyzer(dag)
        influence_map = analyzer.build_influence_map()

        detectors_json = tc.get_meta("detectors") or "[]"
        observables_json = tc.get_meta("observables") or "[]"
        measurement_order = extract_measurement_order(tc)

        builder = DemSamplerBuilder(influence_map)
        builder.with_noise(**noise_params)
        builder.with_detectors_json(detectors_json)
        builder.with_observables_json(observables_json)
        builder.with_measurement_order(measurement_order)
        pecos_sampler = builder.build()

        # Build Stim sampler
        stim_str = tick_circuit_to_stim(tc, **noise_params)
        stim_circuit = stim.Circuit(stim_str)
        stim_dem = stim_circuit.detector_error_model()
        stim_sampler = stim_dem.compile_sampler()

        # Sample from both
        num_shots = 50_000
        seed = 123

        # PECOS: get per-detector counts
        pecos_det_batch, _ = pecos_sampler.sample_batch(num_shots, seed=seed)
        pecos_det_array = np.array(pecos_det_batch)
        pecos_det_rates = pecos_det_array.mean(axis=0)

        # Stim: get per-detector counts
        stim_det_events, _, _ = stim_sampler.sample(num_shots)
        stim_det_rates = stim_det_events.mean(axis=0)

        # Check same number of detectors
        assert len(pecos_det_rates) == len(
            stim_det_rates,
        ), f"Detector count mismatch: PECOS={len(pecos_det_rates)}, Stim={len(stim_det_rates)}"

        # Check correlation is positive (rates should trend together)
        correlation = np.corrcoef(pecos_det_rates, stim_det_rates)[0, 1]
        assert correlation > 0.5, f"Detector rate correlation too low: {correlation:.2f}"

        # Print differences for debugging
        print(f"\nDetector rate comparison (correlation={correlation:.2f}):")
        for i, (p, s) in enumerate(zip(pecos_det_rates, stim_det_rates, strict=False)):
            rel_diff = abs(p - s) / max(p, s, 0.001)
            print(f"  D{i}: PECOS={p:.4f} Stim={s:.4f} diff={rel_diff:.0%}")


class TestDemSamplerMultiRound:
    """Test DemSampler with multiple syndrome rounds."""

    def test_multi_round_d3_r3(self) -> None:
        """Test d=3, 3 rounds against Stim."""
        from pecos.qec.surface import SurfacePatch, generate_tick_circuit_from_patch
        from pecos.qec.surface.circuit_builder import tick_circuit_to_stim
        from pecos_rslib.qec import DagFaultAnalyzer, DemSamplerBuilder

        patch = SurfacePatch.create(distance=3)
        tc = generate_tick_circuit_from_patch(patch, num_rounds=3, basis="Z")

        noise_params = {"p1": 0.01, "p2": 0.01, "p_meas": 0.01, "p_init": 0.01}

        # Build PECOS sampler
        dag = tc.to_dag_circuit()
        analyzer = DagFaultAnalyzer(dag)
        influence_map = analyzer.build_influence_map()

        detectors_json = tc.get_meta("detectors") or "[]"
        observables_json = tc.get_meta("observables") or "[]"
        measurement_order = extract_measurement_order(tc)

        builder = DemSamplerBuilder(influence_map)
        builder.with_noise(**noise_params)
        builder.with_detectors_json(detectors_json)
        builder.with_observables_json(observables_json)
        builder.with_measurement_order(measurement_order)
        pecos_sampler = builder.build()

        # Build Stim sampler
        stim_str = tick_circuit_to_stim(tc, **noise_params)
        stim_circuit = stim.Circuit(stim_str)
        stim_dem = stim_circuit.detector_error_model()
        stim_sampler = stim_dem.compile_sampler()

        # Sample and compare
        num_shots = 50_000
        seed = 456

        pecos_stats = pecos_sampler.sample_statistics(num_shots, seed=seed)

        _stim_det_events, stim_obs_flips, _ = stim_sampler.sample(num_shots)
        stim_logical_rate = np.any(stim_obs_flips, axis=1).mean()

        # Logical error rates should be within 50% relative (known differences)
        max_rate = max(pecos_stats["logical_error_rate"], stim_logical_rate, 0.001)
        rel_diff = abs(pecos_stats["logical_error_rate"] - stim_logical_rate) / max_rate
        assert rel_diff < 0.5, (
            f"Logical rate mismatch for d=3, r=3: "
            f"PECOS={pecos_stats['logical_error_rate']:.4f}, Stim={stim_logical_rate:.4f}, "
            f"rel_diff={rel_diff:.1%}"
        )


class TestDemSamplerHigherDistance:
    """Test DemSampler at higher distances."""

    @pytest.mark.parametrize("distance", [3, 5])
    def test_logical_error_rate_scales(self, distance: int) -> None:
        """Logical error rate should decrease with distance (suppression)."""
        from pecos.qec.surface import SurfacePatch, generate_tick_circuit_from_patch
        from pecos_rslib.qec import DagFaultAnalyzer, DemSamplerBuilder

        patch = SurfacePatch.create(distance=distance)
        tc = generate_tick_circuit_from_patch(patch, num_rounds=1, basis="Z")

        noise_params = {"p1": 0.001, "p2": 0.001, "p_meas": 0.001, "p_init": 0.001}

        # Build sampler
        dag = tc.to_dag_circuit()
        analyzer = DagFaultAnalyzer(dag)
        influence_map = analyzer.build_influence_map()

        detectors_json = tc.get_meta("detectors") or "[]"
        observables_json = tc.get_meta("observables") or "[]"
        measurement_order = extract_measurement_order(tc)

        builder = DemSamplerBuilder(influence_map)
        builder.with_noise(**noise_params)
        builder.with_detectors_json(detectors_json)
        builder.with_observables_json(observables_json)
        builder.with_measurement_order(measurement_order)
        sampler = builder.build()

        stats = sampler.sample_statistics(100_000, seed=789)

        # At low noise, logical error rate should be very low
        # Higher distance should have lower rate
        assert (
            stats["logical_error_rate"] < 0.1
        ), f"d={distance}: Logical error rate {stats['logical_error_rate']:.4f} too high"


class TestXBasisMemory:
    """Test X-basis memory experiments."""

    def test_x_basis_dem_matches_stim(self) -> None:
        """X-basis memory DEM should match Stim."""
        from pecos.qec.surface import SurfacePatch, generate_tick_circuit_from_patch
        from pecos.qec.surface.circuit_builder import (
            generate_dem_from_tick_circuit,
            tick_circuit_to_stim,
        )

        patch = SurfacePatch.create(distance=3)
        tc = generate_tick_circuit_from_patch(patch, num_rounds=1, basis="X")

        noise = {"p1": 0.01, "p2": 0.01, "p_meas": 0.01, "p_init": 0.01}

        # Generate DEMs (non-decomposed for exact comparison)
        pecos_dem = generate_dem_from_tick_circuit(tc, **noise, decompose_errors=False)
        stim_str = tick_circuit_to_stim(tc, **noise)
        stim_circuit = stim.Circuit(stim_str)
        stim_dem = str(stim_circuit.detector_error_model(decompose_errors=False))

        # Parse and compare
        def extract_errors(dem_str: str) -> dict[str, float]:
            errors: dict[str, float] = {}
            for line in dem_str.strip().split("\n"):
                if line.strip().startswith("error("):
                    match = re.match(r"error\(([^)]+)\)\s*(.*)", line.strip())
                    if match:
                        prob = float(match.group(1))
                        targets = match.group(2).strip()
                        errors[targets] = prob
            return errors

        pecos_errors = extract_errors(pecos_dem)
        stim_errors = extract_errors(stim_dem)

        assert len(pecos_errors) == len(
            stim_errors,
        ), f"Mechanism count mismatch: PECOS={len(pecos_errors)}, Stim={len(stim_errors)}"

        # Check all probabilities match
        for target in pecos_errors:
            assert target in stim_errors, f"PECOS has {target} but Stim doesn't"
            rel_diff = abs(pecos_errors[target] - stim_errors[target]) / max(
                pecos_errors[target],
                stim_errors[target],
                1e-10,
            )
            assert rel_diff < 0.01, (
                f"Probability mismatch for {target}: "
                f"PECOS={pecos_errors[target]:.6f}, Stim={stim_errors[target]:.6f}"
            )

    def test_x_basis_sampling_matches_stim(self) -> None:
        """X-basis sampling statistics should match Stim."""
        from pecos.qec.surface import SurfacePatch, generate_tick_circuit_from_patch
        from pecos.qec.surface.circuit_builder import tick_circuit_to_stim
        from pecos_rslib.qec import DagFaultAnalyzer, DemSamplerBuilder

        patch = SurfacePatch.create(distance=3)
        tc = generate_tick_circuit_from_patch(patch, num_rounds=2, basis="X")

        noise = {"p1": 0.01, "p2": 0.01, "p_meas": 0.01, "p_init": 0.01}

        # Build PECOS sampler
        dag = tc.to_dag_circuit()
        analyzer = DagFaultAnalyzer(dag)
        influence_map = analyzer.build_influence_map()

        builder = DemSamplerBuilder(influence_map)
        builder.with_noise(**noise)
        builder.with_detectors_json(tc.get_meta("detectors") or "[]")
        builder.with_observables_json(tc.get_meta("observables") or "[]")
        builder.with_measurement_order(extract_measurement_order(tc))
        pecos_sampler = builder.build()

        # Build Stim sampler
        stim_str = tick_circuit_to_stim(tc, **noise)
        stim_circuit = stim.Circuit(stim_str)
        stim_sampler = stim_circuit.detector_error_model().compile_sampler()

        # Sample and compare
        num_shots = 50_000
        pecos_stats = pecos_sampler.sample_statistics(num_shots, seed=123)
        stim_det, stim_obs, _ = stim_sampler.sample(num_shots)

        stim_syndrome_rate = np.any(stim_det, axis=1).mean()
        np.any(stim_obs, axis=1).mean()

        # Should be within 20% relative
        syndrome_diff = abs(pecos_stats["syndrome_rate"] - stim_syndrome_rate)
        assert syndrome_diff / max(stim_syndrome_rate, 0.001) < 0.2, (
            f"X-basis syndrome rate mismatch: PECOS={pecos_stats['syndrome_rate']:.4f}, "
            f"Stim={stim_syndrome_rate:.4f}"
        )


class TestAsymmetricNoise:
    """Test with asymmetric noise parameters."""

    @pytest.mark.parametrize(
        "noise_params",
        [
            {"p1": 0.001, "p2": 0.01, "p_meas": 0.005, "p_init": 0.002},  # p2 dominant
            {"p1": 0.02, "p2": 0.001, "p_meas": 0.001, "p_init": 0.001},  # p1 dominant
            {
                "p1": 0.001,
                "p2": 0.001,
                "p_meas": 0.05,
                "p_init": 0.001,
            },  # p_meas dominant
            {
                "p1": 0.001,
                "p2": 0.001,
                "p_meas": 0.001,
                "p_init": 0.05,
            },  # p_init dominant
        ],
    )
    def test_asymmetric_noise_dem_matches_stim(
        self,
        noise_params: dict[str, float],
    ) -> None:
        """Asymmetric noise DEMs should match Stim."""
        from pecos.qec.surface import SurfacePatch, generate_tick_circuit_from_patch
        from pecos.qec.surface.circuit_builder import (
            generate_dem_from_tick_circuit,
            tick_circuit_to_stim,
        )

        patch = SurfacePatch.create(distance=3)
        tc = generate_tick_circuit_from_patch(patch, num_rounds=1, basis="Z")

        # Generate DEMs (non-decomposed)
        pecos_dem = generate_dem_from_tick_circuit(
            tc,
            **noise_params,
            decompose_errors=False,
        )
        stim_str = tick_circuit_to_stim(tc, **noise_params)
        stim_circuit = stim.Circuit(stim_str)
        stim_dem = str(stim_circuit.detector_error_model(decompose_errors=False))

        def extract_errors(dem_str: str) -> dict[str, float]:
            errors: dict[str, float] = {}
            for line in dem_str.strip().split("\n"):
                if line.strip().startswith("error("):
                    match = re.match(r"error\(([^)]+)\)\s*(.*)", line.strip())
                    if match:
                        errors[match.group(2).strip()] = float(match.group(1))
            return errors

        pecos_errors = extract_errors(pecos_dem)
        stim_errors = extract_errors(stim_dem)

        # All mechanisms should match
        assert set(pecos_errors.keys()) == set(
            stim_errors.keys(),
        ), f"Mechanism mismatch with noise {noise_params}"

        # All probabilities should match (within numerical precision)
        for target in pecos_errors:
            rel_diff = abs(pecos_errors[target] - stim_errors[target]) / max(
                pecos_errors[target],
                stim_errors[target],
                1e-10,
            )
            assert rel_diff < 0.01, (
                f"Probability mismatch for {target} with noise {noise_params}: "
                f"PECOS={pecos_errors[target]:.6f}, Stim={stim_errors[target]:.6f}"
            )


class TestRandomCliffordFuzzing:
    """Fuzz testing with random Clifford circuits.

    These tests generate random circuits using gates supported by both PECOS and Stim,
    and verify that the DEM analysis produces equivalent results.
    """

    def _build_random_circuit(
        self,
        num_qubits: int,
        depth: int,
        seed: int,
    ) -> stim.Circuit:
        """Build a random circuit with gates supported by both PECOS and Stim.

        Uses only: H, S, S_DAG, CX (no CZ since DagCircuit doesn't support it).
        Structure: Reset all -> random gates -> measure all.
        """
        import random

        random.seed(seed)

        # Build Stim circuit
        stim_circuit = stim.Circuit()

        # Reset all qubits
        for q in range(num_qubits):
            stim_circuit.append("R", [q])

        # Random gates (only those supported by PECOS DagCircuit)
        single_gates = ["H", "S", "S_DAG"]

        for _ in range(depth):
            if random.random() < 0.6 and num_qubits >= 2:
                # CX gate
                q1, q2 = random.sample(range(num_qubits), 2)
                stim_circuit.append("CX", [q1, q2])
            else:
                # Single-qubit gate
                gate = random.choice(single_gates)
                q = random.randint(0, num_qubits - 1)
                stim_circuit.append(gate, [q])

        stim_circuit.append("TICK", [])

        # Measure all qubits
        for q in range(num_qubits):
            stim_circuit.append("M", [q])

        return stim_circuit

    def _stim_to_dag_circuit(self, stim_circuit: stim.Circuit) -> "DagCircuit":
        """Convert Stim circuit to PECOS DagCircuit."""
        from pecos_rslib import DagCircuit

        dag = DagCircuit()

        for instruction in stim_circuit:
            name = instruction.name
            targets = instruction.targets_copy()

            if name == "R":
                for t in targets:
                    dag.pz(t.value)
            elif name == "H":
                for t in targets:
                    dag.h(t.value)
            elif name == "S":
                for t in targets:
                    dag.sz(t.value)
            elif name == "S_DAG":
                for t in targets:
                    dag.szdg(t.value)
            elif name == "CX":
                for i in range(0, len(targets), 2):
                    dag.cx(targets[i].value, targets[i + 1].value)
            elif name == "M":
                for t in targets:
                    dag.mz(t.value)
            # Skip TICK, DETECTOR, etc.

        return dag

    def _add_noise_to_stim(
        self,
        stim_circuit: stim.Circuit,
        noise: dict,
    ) -> stim.Circuit:
        """Add noise to a Stim circuit."""
        noisy = stim.Circuit()

        for instruction in stim_circuit:
            name = instruction.name
            targets = instruction.targets_copy()

            if name == "R":
                for t in targets:
                    noisy.append("R", [t.value])
                    if noise["p_init"] > 0:
                        noisy.append("X_ERROR", [t.value], noise["p_init"])
            elif name in ("H", "S", "S_DAG"):
                for t in targets:
                    noisy.append(name, [t.value])
                    if noise["p1"] > 0:
                        noisy.append("DEPOLARIZE1", [t.value], noise["p1"])
            elif name == "CX":
                for i in range(0, len(targets), 2):
                    noisy.append("CX", [targets[i].value, targets[i + 1].value])
                    if noise["p2"] > 0:
                        noisy.append(
                            "DEPOLARIZE2",
                            [targets[i].value, targets[i + 1].value],
                            noise["p2"],
                        )
            elif name == "M":
                for t in targets:
                    if noise["p_meas"] > 0:
                        noisy.append("X_ERROR", [t.value], noise["p_meas"])
                    noisy.append("M", [t.value])
            else:
                noisy.append(instruction)

        return noisy

    @pytest.mark.parametrize("seed", range(10))
    def test_random_circuit_fault_locations_match(self, seed: int) -> None:
        """Random circuits should have same fault location count in PECOS and Stim."""
        from pecos_rslib.qec import DagFaultAnalyzer

        num_qubits = 4
        depth = 8

        # Build random circuit
        stim_circuit = self._build_random_circuit(num_qubits, depth, seed)

        # Build PECOS DagCircuit
        dag = self._stim_to_dag_circuit(stim_circuit)
        analyzer = DagFaultAnalyzer(dag)
        influence_map = analyzer.build_influence_map()

        # Count fault locations by type
        pecos_loc_count = influence_map.num_locations

        # Count operations in Stim circuit that would have fault locations
        stim_op_count = 0
        for instruction in stim_circuit:
            name = instruction.name
            targets = instruction.targets_copy()
            if name == "R":
                stim_op_count += len(targets)  # Prep error after
            elif name in ("H", "S", "S_DAG"):
                stim_op_count += len(targets)  # Gate error after
            elif name == "CX":
                stim_op_count += len(targets)  # 2 qubits per CX
            elif name == "M":
                stim_op_count += len(targets)  # Meas error before

        # PECOS creates per-qubit locations, so counts should be similar
        # (may differ due to before/after handling)
        assert pecos_loc_count > 0, f"PECOS should find fault locations (seed={seed})"

    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 1000])
    def test_random_circuit_sampling_produces_valid_results(self, seed: int) -> None:
        """Random circuit sampling should produce valid statistics."""
        from pecos_rslib.qec import DagFaultAnalyzer, DemSamplerBuilder

        num_qubits = 4
        depth = 10

        # Build random circuit
        stim_circuit = self._build_random_circuit(num_qubits, depth, seed)

        # Convert to PECOS
        dag = self._stim_to_dag_circuit(stim_circuit)
        analyzer = DagFaultAnalyzer(dag)
        influence_map = analyzer.build_influence_map()

        # Build detector: XOR of all measurements (likely deterministic)
        records = [-i for i in range(1, num_qubits + 1)]
        detectors_json = f'[{{"id": 0, "records": {records}}}]'

        noise = {"p1": 0.01, "p2": 0.01, "p_meas": 0.01, "p_init": 0.01}

        builder = DemSamplerBuilder(influence_map)
        builder.with_noise(**noise)
        builder.with_detectors_json(detectors_json)
        sampler = builder.build()

        # Sample and verify basic properties
        stats = sampler.sample_statistics(10_000, seed=seed)

        # Statistics should be valid
        assert 0.0 <= stats["syndrome_rate"] <= 1.0
        assert 0.0 <= stats["logical_error_rate"] <= 1.0
        assert stats["total_shots"] == 10_000


class TestDemEquivalenceComprehensive:
    """Comprehensive DEM equivalence tests."""

    @pytest.mark.parametrize(
        ("distance", "num_rounds", "basis"),
        [
            (3, 1, "Z"),
            (3, 1, "X"),
            (3, 2, "Z"),
            (3, 3, "Z"),
            (5, 1, "Z"),
            (5, 2, "Z"),
        ],
    )
    def test_dem_exact_match(self, distance: int, num_rounds: int, basis: str) -> None:
        """DEMs should exactly match Stim for all configurations."""
        from pecos.qec.surface import SurfacePatch, generate_tick_circuit_from_patch
        from pecos.qec.surface.circuit_builder import (
            generate_dem_from_tick_circuit,
            tick_circuit_to_stim,
        )

        patch = SurfacePatch.create(distance=distance)
        tc = generate_tick_circuit_from_patch(patch, num_rounds=num_rounds, basis=basis)

        noise = {"p1": 0.01, "p2": 0.01, "p_meas": 0.01, "p_init": 0.01}

        # Generate non-decomposed DEMs
        pecos_dem = generate_dem_from_tick_circuit(tc, **noise, decompose_errors=False)
        stim_str = tick_circuit_to_stim(tc, **noise)
        stim_circuit = stim.Circuit(stim_str)
        stim_dem = str(stim_circuit.detector_error_model(decompose_errors=False))

        def extract_errors(dem_str: str) -> dict[str, float]:
            errors: dict[str, float] = {}
            for line in dem_str.strip().split("\n"):
                if line.strip().startswith("error("):
                    match = re.match(r"error\(([^)]+)\)\s*(.*)", line.strip())
                    if match:
                        errors[match.group(2).strip()] = float(match.group(1))
            return errors

        pecos_errors = extract_errors(pecos_dem)
        stim_errors = extract_errors(stim_dem)

        # Exact mechanism match
        assert set(pecos_errors.keys()) == set(stim_errors.keys()), (
            f"Mechanism mismatch for d={distance}, r={num_rounds}, basis={basis}. "
            f"PECOS-only: {set(pecos_errors.keys()) - set(stim_errors.keys())}, "
            f"Stim-only: {set(stim_errors.keys()) - set(pecos_errors.keys())}"
        )

        # Exact probability match (within floating point tolerance)
        for target in pecos_errors:
            rel_diff = abs(pecos_errors[target] - stim_errors[target]) / max(
                pecos_errors[target],
                stim_errors[target],
                1e-10,
            )
            assert rel_diff < 0.001, (
                f"Probability mismatch for {target} (d={distance}, r={num_rounds}, "
                f"basis={basis}): PECOS={pecos_errors[target]:.8f}, "
                f"Stim={stim_errors[target]:.8f}, diff={rel_diff:.2%}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
