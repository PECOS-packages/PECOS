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


class TestErrorMechanismParsing:
    """Test parsing of error mechanisms."""

    def test_parse_simple_mechanism(self):
        """Parse a simple error mechanism."""
        dem_str = "error(0.01) D0 D1"
        dem = ParsedDem.from_string(dem_str)

        assert dem.num_mechanisms == 1
        assert dem.num_detectors == 2
        assert dem.num_observables == 0

    def test_parse_mechanism_with_observable(self):
        """Parse mechanism with observable."""
        dem_str = "error(0.02) D0 L0"
        dem = ParsedDem.from_string(dem_str)

        assert dem.num_mechanisms == 1
        assert dem.num_detectors == 1
        assert dem.num_observables == 1

    def test_parse_decomposed_mechanism(self):
        """Parse a decomposed mechanism (XOR chain)."""
        dem_str = "error(0.01) D0 ^ D1 D2"
        dem = ParsedDem.from_string(dem_str)

        assert dem.num_mechanisms == 1
        # Decomposed mechanism has detectors from all components
        assert dem.num_detectors == 3

    def test_parse_multiple_mechanisms(self):
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

    def test_parse_detector_declarations(self):
        """Parse detector declarations."""
        dem_str = """
detector(0, 0, 0) D0
detector(1, 0, 0) D1
error(0.01) D0 D1
"""
        dem = ParsedDem.from_string(dem_str)

        assert dem.num_detectors == 2
        assert dem.num_mechanisms == 1

    def test_skip_comments(self):
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

    def test_sample_simple_dem(self):
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

    def test_sample_decomposed_dem(self):
        """Sample from a decomposed DEM."""
        dem_str = "error(0.1) D0 ^ D1"
        dem = ParsedDem.from_string(dem_str)

        det_events, obs_flips = dem.sample_batch(50000, seed=42)

        det_array = np.array(det_events)
        # Each sub-mechanism fires independently at p=0.1
        d0_rate = det_array[:, 0].mean()
        d1_rate = det_array[:, 1].mean()

        assert 0.08 < d0_rate < 0.12
        assert 0.08 < d1_rate < 0.12

    def test_sample_deterministic(self):
        """Sampling should be deterministic with same seed."""
        dem_str = "error(0.5) D0 D1"
        dem = ParsedDem.from_string(dem_str)

        det1, obs1 = dem.sample_batch(1000, seed=123)
        det2, obs2 = dem.sample_batch(1000, seed=123)

        assert det1 == det2
        assert obs1 == obs2


class TestAggregation:
    """Test mechanism aggregation."""

    def test_aggregate_same_effect(self):
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

    def test_aggregate_different_effects(self):
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

    def test_identical_dems(self):
        """Identical DEMs should be equivalent."""
        dem_str = """
error(0.01) D0 D1
error(0.02) D1 D2
"""
        result = compare_dems_exact(dem_str, dem_str)

        assert result.equivalent
        assert result.max_rate_difference == pytest.approx(0.0)

    def test_different_probabilities(self):
        """DEMs with different probabilities should not be equivalent."""
        dem1 = "error(0.01) D0"
        dem2 = "error(0.02) D0"

        result = compare_dems_exact(dem1, dem2, prob_tolerance=0.001)

        assert not result.equivalent
        assert result.max_rate_difference == pytest.approx(0.01)

    def test_different_mechanisms(self):
        """DEMs with different mechanisms should not be equivalent."""
        dem1 = "error(0.01) D0 D1"
        dem2 = "error(0.01) D0 D2"

        result = compare_dems_exact(dem1, dem2)

        assert not result.equivalent
        assert len(result.only_in_dem1) == 1
        assert len(result.only_in_dem2) == 1

    def test_aggregated_equivalence(self):
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

    def test_identical_dems_statistical(self):
        """Identical DEMs should be statistically equivalent."""
        dem_str = """
error(0.01) D0 D1
error(0.02) D1 D2
"""
        result = compare_dems_statistical(dem_str, dem_str, num_shots=50000)

        assert result.equivalent
        assert result.correlation > 0.9

    def test_similar_dems_statistical(self):
        """Similar DEMs should be statistically equivalent within tolerance."""
        dem1 = "error(0.10) D0"
        dem2 = "error(0.11) D0"  # 10% difference

        # Use larger tolerance and more shots for this edge case
        result = compare_dems_statistical(dem1, dem2, num_shots=100000, tolerance=0.10)

        # Should be equivalent within 10% tolerance
        assert result.equivalent

    def test_decomposition_equivalence(self):
        """Decomposed and non-decomposed with same effect should NOT be equivalent.

        This is a key test: error(p) D0 D1 means both flip together,
        while error(p) D0 ^ D1 means they flip independently.
        These are NOT equivalent for the same p!
        """
        # D0 and D1 flip together
        dem1 = "error(0.1) D0 D1"

        # D0 and D1 flip independently (each with p=0.1)
        dem2 = "error(0.1) D0 ^ D1"

        result = compare_dems_statistical(dem1, dem2, num_shots=50000, tolerance=0.05)

        # These should NOT be equivalent
        # In dem1: P(D0=1, D1=1) = 0.1, P(D0=1, D1=0) = 0
        # In dem2: P(D0=1, D1=1) = 0.01, P(D0=1, D1=0) = 0.09
        assert not result.equivalent


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_verify_dem_equivalence(self):
        """Test verify_dem_equivalence function."""
        dem_str = "error(0.01) D0 D1"

        assert verify_dem_equivalence(dem_str, dem_str, method="exact")
        assert verify_dem_equivalence(dem_str, dem_str, method="statistical", num_shots=50000)

    def test_assert_dems_equivalent_pass(self):
        """assert_dems_equivalent should pass for equivalent DEMs."""
        dem_str = "error(0.01) D0 D1"

        # Should not raise
        assert_dems_equivalent(dem_str, dem_str, method="exact")

    def test_assert_dems_equivalent_fail(self):
        """assert_dems_equivalent should fail for non-equivalent DEMs."""
        dem1 = "error(0.01) D0"
        dem2 = "error(0.01) D1"

        with pytest.raises(AssertionError, match="DEMs are not equivalent"):
            assert_dems_equivalent(dem1, dem2, method="exact")


class TestIntegrationWithPecos:
    """Integration tests with PECOS DEM generation."""

    @pytest.fixture
    def surface_code_dem(self):
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

    def test_pecos_stim_exact_equivalence(self, surface_code_dem):
        """PECOS and Stim non-decomposed DEMs should be exactly equivalent."""
        pecos_dem, stim_dem = surface_code_dem

        result = compare_dems_exact(pecos_dem, stim_dem, prob_tolerance=0.001)

        assert result.equivalent, (
            f"PECOS and Stim DEMs not equivalent: "
            f"only in PECOS: {result.only_in_dem1}, "
            f"only in Stim: {result.only_in_dem2}"
        )

    def test_pecos_stim_statistical_equivalence(self, surface_code_dem):
        """PECOS and Stim DEMs should be statistically equivalent."""
        pecos_dem, stim_dem = surface_code_dem

        result = compare_dems_statistical(
            pecos_dem, stim_dem, num_shots=100000, tolerance=0.05
        )

        assert result.equivalent, (
            f"PECOS and Stim DEMs not statistically equivalent: "
            f"max diff={result.max_rate_difference:.4f}, "
            f"correlation={result.correlation:.4f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
