# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Tests for DEM expression equivalence validation (using Rust implementation)."""

import numpy as np
import pytest
from pecos_rslib.qec import (
    ParsedDem,
    assert_dems_equivalent,
    compare_dems_exact,
    compare_dems_statistical,
    verify_dem_equivalence,
)


class TestFaultMechanismParsing:
    """Test parsing of error mechanisms."""

    def test_parse_simple_mechanism(self) -> None:
        """Parse a simple error mechanism."""
        dem_str = "error(0.01) D0 D1"
        dem = ParsedDem.from_string(dem_str)

        assert dem.num_mechanisms == 1
        assert dem.num_detectors == 2
        assert dem.num_observables == 0

    def test_parse_mechanism_with_observable(self) -> None:
        """Parse mechanism with observable."""
        dem_str = "error(0.02) D0 L0"
        dem = ParsedDem.from_string(dem_str)

        assert dem.num_mechanisms == 1
        assert dem.num_detectors == 1
        assert dem.num_observables == 1

    def test_parse_decomposed_mechanism(self) -> None:
        """Parse a decomposed mechanism (XOR chain)."""
        dem_str = "error(0.01) D0 ^ D1 D2"
        dem = ParsedDem.from_string(dem_str)

        assert dem.num_mechanisms == 1
        # Decomposed mechanism has detectors from all components
        assert dem.num_detectors == 3

    def test_parse_multiple_mechanisms(self) -> None:
        """Parse multiple mechanisms."""
        dem_str = """
error(0.01) D0
error(0.02) D1 D2
error(0.03) D0 D1 L0
"""
        dem = ParsedDem.from_string(dem_str)

        assert dem.num_mechanisms == 3
        assert dem.num_detectors == 3
        assert dem.num_observables == 1

    def test_parse_detector_declarations(self) -> None:
        """Parse detector declarations."""
        dem_str = """
detector(0, 0, 0) D0
detector(1, 0, 0) D1
error(0.01) D0 D1
"""
        dem = ParsedDem.from_string(dem_str)

        assert dem.num_detectors == 2
        assert dem.num_mechanisms == 1

    def test_skip_comments(self) -> None:
        """Comments should be skipped."""
        dem_str = """
# This is a comment
error(0.01) D0
# Another comment
error(0.02) D1
"""
        dem = ParsedDem.from_string(dem_str)
        assert dem.num_mechanisms == 2


class TestSampling:
    """Test DEM sampling."""

    def test_sample_simple_dem(self) -> None:
        """Sample from a simple DEM."""
        dem_str = "error(0.5) D0"
        dem = ParsedDem.from_string(dem_str)

        det_events, obs_flips = dem.sample_batch(10000, seed=42)

        assert len(det_events) == 10000
        assert len(obs_flips) == 10000

        # With p=0.5, detector should fire ~50% of the time
        det_array = np.array(det_events)
        rate = det_array[:, 0].mean()
        assert 0.45 < rate < 0.55

    def test_sample_decomposed_dem(self) -> None:
        """Sample from a decomposed DEM."""
        dem_str = "error(0.1) D0 ^ D1"
        dem = ParsedDem.from_string(dem_str)

        det_events, _obs_flips = dem.sample_batch(50000, seed=42)

        det_array = np.array(det_events)
        # Each sub-mechanism fires independently at p=0.1
        d0_rate = det_array[:, 0].mean()
        d1_rate = det_array[:, 1].mean()

        assert 0.08 < d0_rate < 0.12
        assert 0.08 < d1_rate < 0.12

    def test_sample_deterministic(self) -> None:
        """Sampling should be deterministic with same seed."""
        dem_str = "error(0.5) D0 D1"
        dem = ParsedDem.from_string(dem_str)

        det1, obs1 = dem.sample_batch(1000, seed=123)
        det2, obs2 = dem.sample_batch(1000, seed=123)

        assert det1 == det2
        assert obs1 == obs2


class TestAggregation:
    """Test mechanism aggregation."""

    def test_aggregate_same_effect(self) -> None:
        """Mechanisms with same effect should be aggregated."""
        dem_str = """
error(0.1) D0
error(0.2) D0
"""
        dem = ParsedDem.from_string(dem_str)
        agg = dem.aggregate()

        # Key: ((0,), ())
        key = ((0,), ())
        assert key in agg

        # Combined probability: 0.1*(1-0.2) + 0.2*(1-0.1) = 0.08 + 0.18 = 0.26
        assert agg[key] == pytest.approx(0.26)

    def test_aggregate_different_effects(self) -> None:
        """Mechanisms with different effects stay separate."""
        dem_str = """
error(0.1) D0
error(0.2) D1
"""
        dem = ParsedDem.from_string(dem_str)
        agg = dem.aggregate()

        assert len(agg) == 2
        assert ((0,), ()) in agg
        assert ((1,), ()) in agg


class TestExactComparison:
    """Test exact DEM comparison."""

    def test_identical_dems(self) -> None:
        """Identical DEMs should be equivalent."""
        dem_str = """
error(0.01) D0 D1
error(0.02) D1 D2
"""
        result = compare_dems_exact(dem_str, dem_str)

        assert result.equivalent
        assert result.max_rate_difference == pytest.approx(0.0)

    def test_different_probabilities(self) -> None:
        """DEMs with different probabilities should not be equivalent."""
        dem1 = "error(0.01) D0"
        dem2 = "error(0.02) D0"

        result = compare_dems_exact(dem1, dem2, prob_tolerance=0.001)

        assert not result.equivalent
        assert result.max_rate_difference == pytest.approx(0.01)

    def test_different_mechanisms(self) -> None:
        """DEMs with different mechanisms should not be equivalent."""
        dem1 = "error(0.01) D0 D1"
        dem2 = "error(0.01) D0 D2"

        result = compare_dems_exact(dem1, dem2)

        assert not result.equivalent
        assert len(result.only_in_dem1) == 1
        assert len(result.only_in_dem2) == 1

    def test_aggregated_equivalence(self) -> None:
        """Two mechanisms that aggregate to same effect should match."""
        # Two 0.1 errors on D0 combine to 0.1*(1-0.1) + 0.1*(1-0.1) = 0.18
        dem1 = """
error(0.1) D0
error(0.1) D0
"""
        dem2 = "error(0.18) D0"

        result = compare_dems_exact(dem1, dem2, prob_tolerance=0.001)

        assert result.equivalent


class TestStatisticalComparison:
    """Test statistical DEM comparison."""

    def test_identical_dems_statistical(self) -> None:
        """Identical DEMs should be statistically equivalent."""
        dem_str = """
error(0.01) D0 D1
error(0.02) D1 D2
"""
        result = compare_dems_statistical(dem_str, dem_str, num_shots=50000)

        assert result.equivalent
        assert result.correlation > 0.9

    def test_similar_dems_statistical(self) -> None:
        """Similar DEMs should be statistically equivalent within tolerance."""
        dem1 = "error(0.10) D0"
        dem2 = "error(0.11) D0"  # 10% difference

        # Use larger tolerance and more shots for this edge case
        result = compare_dems_statistical(dem1, dem2, num_shots=100000, tolerance=0.10)

        # Should be equivalent within 10% tolerance
        assert result.equivalent

    def test_decomposition_equivalence(self) -> None:
        """Decomposed (^) and non-decomposed with same effect ARE equivalent.

        In Stim DEM format, `error(p) D0 ^ D1` is a decomposition representation
        of a 2-detector error for MWPM decoders. It does NOT mean D0 and D1 flip
        independently - both still flip together with probability p.

        The ^ syntax means XOR: when the error fires, D0 XOR true and D1 XOR true
        are applied, which is the same effect as `D0 D1`.
        """
        # D0 and D1 flip together (direct form)
        dem1 = "error(0.1) D0 D1"

        # D0 and D1 flip together (decomposed form - same semantics)
        dem2 = "error(0.1) D0 ^ D1"

        result = compare_dems_statistical(dem1, dem2, num_shots=50000, tolerance=0.05)

        # These SHOULD be equivalent - both represent the same error mechanism
        assert result.equivalent

    def test_independent_vs_correlated_not_equivalent(self) -> None:
        """Independent detector errors should NOT be equivalent to correlated errors.

        This test verifies that:
        - error(0.1) D0 D1: Both flip together with probability 0.1
        - error(0.1) D0 + error(0.1) D1: Each flips independently with probability 0.1

        Expected probabilities for independent case:
        - P(D0=0, D1=0) = 0.9 * 0.9 = 0.81
        - P(D0=1, D1=0) = 0.1 * 0.9 = 0.09
        - P(D0=0, D1=1) = 0.9 * 0.1 = 0.09
        - P(D0=1, D1=1) = 0.1 * 0.1 = 0.01

        Expected probabilities for correlated case:
        - P(D0=0, D1=0) = 0.9
        - P(D0=1, D1=1) = 0.1
        """
        # D0 and D1 flip together
        dem_correlated = "error(0.1) D0 D1"

        # D0 and D1 flip independently (separate error lines)
        dem_independent = """error(0.1) D0
error(0.1) D1"""

        result = compare_dems_statistical(
            dem_correlated,
            dem_independent,
            num_shots=50000,
            tolerance=0.05,
        )

        # These should NOT be equivalent - very different joint distributions
        assert not result.equivalent


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_verify_dem_equivalence(self) -> None:
        """Test verify_dem_equivalence function."""
        dem_str = "error(0.01) D0 D1"

        assert verify_dem_equivalence(dem_str, dem_str, method="exact")
        assert verify_dem_equivalence(
            dem_str,
            dem_str,
            method="statistical",
            num_shots=50000,
        )

    def test_assert_dems_equivalent_pass(self) -> None:
        """assert_dems_equivalent should pass for equivalent DEMs."""
        dem_str = "error(0.01) D0 D1"

        # Should not raise
        assert_dems_equivalent(dem_str, dem_str, method="exact")

    def test_assert_dems_equivalent_fail(self) -> None:
        """assert_dems_equivalent should fail for non-equivalent DEMs."""
        dem1 = "error(0.01) D0"
        dem2 = "error(0.01) D1"

        with pytest.raises(AssertionError, match="DEMs are not equivalent"):
            assert_dems_equivalent(dem1, dem2, method="exact")


class TestIntegrationWithPecos:
    """Integration tests with PECOS DEM generation."""

    @pytest.fixture
    def surface_code_dem(self) -> tuple[str, str]:
        """Generate a surface code DEM pair (PECOS and Stim)."""
        pytest.importorskip("stim")

        from pecos.qec.surface import SurfacePatch, generate_tick_circuit_from_patch
        from pecos.qec.surface.circuit_builder import (
            generate_dem_from_tick_circuit,
            tick_circuit_to_stim,
        )

        patch = SurfacePatch.create(distance=3)
        tc = generate_tick_circuit_from_patch(patch, num_rounds=1, basis="Z")

        noise = {"p1": 0.01, "p2": 0.01, "p_meas": 0.01, "p_init": 0.01}

        pecos_dem = generate_dem_from_tick_circuit(tc, **noise, decompose_errors=False)

        import stim

        stim_str = tick_circuit_to_stim(tc, **noise)
        stim_circuit = stim.Circuit(stim_str)
        stim_dem = str(stim_circuit.detector_error_model(decompose_errors=False))

        return pecos_dem, stim_dem

    def test_pecos_stim_exact_equivalence(
        self,
        surface_code_dem: tuple[str, str],
    ) -> None:
        """PECOS and Stim non-decomposed DEMs should be exactly equivalent."""
        pecos_dem, stim_dem = surface_code_dem

        result = compare_dems_exact(pecos_dem, stim_dem, prob_tolerance=0.001)

        assert result.equivalent, (
            f"PECOS and Stim DEMs not equivalent: "
            f"only in PECOS: {result.only_in_dem1}, "
            f"only in Stim: {result.only_in_dem2}"
        )

    def test_pecos_stim_statistical_equivalence(
        self,
        surface_code_dem: tuple[str, str],
    ) -> None:
        """PECOS and Stim DEMs should be statistically equivalent."""
        pecos_dem, stim_dem = surface_code_dem

        result = compare_dems_statistical(
            pecos_dem,
            stim_dem,
            num_shots=100000,
            tolerance=0.05,
        )

        assert result.equivalent, (
            f"PECOS and Stim DEMs not statistically equivalent: "
            f"max diff={result.max_rate_difference:.4f}, "
            f"correlation={result.correlation:.4f}"
        )


class TestPecosDecompositionEquivalence:
    """Test that PECOS raw and decomposed DEMs are statistically equivalent.

    This validates that the decomposition logic in DemBuilder produces DEMs
    that sample identically to the raw (non-decomposed) DEMs.
    """

    @pytest.fixture
    def surface_code_dem_pair(self) -> tuple[str, str]:
        """Generate both raw and decomposed DEMs from PECOS."""
        pytest.importorskip("stim")

        from pecos.qec.surface import SurfacePatch, generate_tick_circuit_from_patch
        from pecos.qec.surface.circuit_builder import generate_dem_from_tick_circuit

        patch = SurfacePatch.create(distance=3)
        tc = generate_tick_circuit_from_patch(patch, num_rounds=2, basis="Z")

        noise = {"p1": 0.01, "p2": 0.01, "p_meas": 0.01, "p_init": 0.01}

        raw_dem = generate_dem_from_tick_circuit(tc, **noise, decompose_errors=False)
        decomposed_dem = generate_dem_from_tick_circuit(
            tc,
            **noise,
            decompose_errors=True,
        )

        return raw_dem, decomposed_dem

    def test_raw_decomposed_syndrome_rates_match(
        self,
        surface_code_dem_pair: tuple[str, str],
    ) -> None:
        """Raw and decomposed DEMs should have same syndrome rates."""
        raw_dem_str, decomposed_dem_str = surface_code_dem_pair

        raw_dem = ParsedDem.from_string(raw_dem_str)
        decomposed_dem = ParsedDem.from_string(decomposed_dem_str)

        num_shots = 100_000
        seed = 42

        raw_dets, _raw_obs = raw_dem.sample_batch(num_shots, seed=seed)
        decomp_dets, _decomp_obs = decomposed_dem.sample_batch(num_shots, seed=seed)

        raw_array = np.array(raw_dets)
        decomp_array = np.array(decomp_dets)

        raw_syndrome_rate = np.any(raw_array, axis=1).mean()
        decomp_syndrome_rate = np.any(decomp_array, axis=1).mean()

        # 3-sigma statistical tolerance
        tolerance = 3.0 / np.sqrt(num_shots)
        diff = abs(raw_syndrome_rate - decomp_syndrome_rate)

        assert diff < tolerance + 0.01, (
            f"Syndrome rate mismatch: raw={raw_syndrome_rate:.4f}, "
            f"decomposed={decomp_syndrome_rate:.4f}, diff={diff:.4f}, "
            f"tolerance={tolerance:.4f}"
        )

    def test_raw_decomposed_per_detector_rates_match(
        self,
        surface_code_dem_pair: tuple[str, str],
    ) -> None:
        """Raw and decomposed DEMs should have similar per-detector rates."""
        raw_dem_str, decomposed_dem_str = surface_code_dem_pair

        raw_dem = ParsedDem.from_string(raw_dem_str)
        decomposed_dem = ParsedDem.from_string(decomposed_dem_str)

        num_shots = 100_000
        seed = 42

        raw_dets, _ = raw_dem.sample_batch(num_shots, seed=seed)
        decomp_dets, _ = decomposed_dem.sample_batch(num_shots, seed=seed)

        raw_array = np.array(raw_dets)
        decomp_array = np.array(decomp_dets)

        raw_rates = raw_array.mean(axis=0)
        decomp_rates = decomp_array.mean(axis=0)

        # Per-detector rates should match within statistical tolerance
        max_diff = np.abs(raw_rates - decomp_rates).max()
        tolerance = 3.0 / np.sqrt(num_shots) + 0.01

        assert (
            max_diff < tolerance
        ), f"Max per-detector rate difference {max_diff:.4f} exceeds tolerance {tolerance:.4f}"

    def test_raw_decomposed_logical_rates_match(
        self,
        surface_code_dem_pair: tuple[str, str],
    ) -> None:
        """Raw and decomposed DEMs should have same logical error rates."""
        raw_dem_str, decomposed_dem_str = surface_code_dem_pair

        raw_dem = ParsedDem.from_string(raw_dem_str)
        decomposed_dem = ParsedDem.from_string(decomposed_dem_str)

        num_shots = 100_000
        seed = 42

        _, raw_obs = raw_dem.sample_batch(num_shots, seed=seed)
        _, decomp_obs = decomposed_dem.sample_batch(num_shots, seed=seed)

        raw_obs_array = np.array(raw_obs)
        decomp_obs_array = np.array(decomp_obs)

        raw_logical_rate = np.any(raw_obs_array, axis=1).mean()
        decomp_logical_rate = np.any(decomp_obs_array, axis=1).mean()

        # 3-sigma statistical tolerance
        tolerance = 3.0 / np.sqrt(num_shots)
        diff = abs(raw_logical_rate - decomp_logical_rate)

        assert diff < tolerance + 0.01, (
            f"Logical error rate mismatch: raw={raw_logical_rate:.4f}, "
            f"decomposed={decomp_logical_rate:.4f}, diff={diff:.4f}"
        )

    @pytest.mark.parametrize("distance", [3, 5])
    @pytest.mark.parametrize("num_rounds", [1, 2])
    def test_decomposition_equivalence_various_sizes(
        self,
        distance: int,
        num_rounds: int,
    ) -> None:
        """Decomposition should be equivalent for various code sizes."""
        pytest.importorskip("stim")

        from pecos.qec.surface import SurfacePatch, generate_tick_circuit_from_patch
        from pecos.qec.surface.circuit_builder import generate_dem_from_tick_circuit

        patch = SurfacePatch.create(distance=distance)
        tc = generate_tick_circuit_from_patch(patch, num_rounds=num_rounds, basis="Z")

        noise = {"p1": 0.01, "p2": 0.01, "p_meas": 0.01, "p_init": 0.01}

        raw_dem_str = generate_dem_from_tick_circuit(
            tc,
            **noise,
            decompose_errors=False,
        )
        decomposed_dem_str = generate_dem_from_tick_circuit(
            tc,
            **noise,
            decompose_errors=True,
        )

        raw_dem = ParsedDem.from_string(raw_dem_str)
        decomposed_dem = ParsedDem.from_string(decomposed_dem_str)

        num_shots = 50_000
        seed = 123

        raw_dets, _ = raw_dem.sample_batch(num_shots, seed=seed)
        decomp_dets, _ = decomposed_dem.sample_batch(num_shots, seed=seed)

        raw_array = np.array(raw_dets)
        decomp_array = np.array(decomp_dets)

        raw_rate = np.any(raw_array, axis=1).mean()
        decomp_rate = np.any(decomp_array, axis=1).mean()

        tolerance = 3.0 / np.sqrt(num_shots) + 0.01
        diff = abs(raw_rate - decomp_rate)

        assert (
            diff < tolerance
        ), f"d={distance}, r={num_rounds}: syndrome rate mismatch raw={raw_rate:.4f}, decomposed={decomp_rate:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
