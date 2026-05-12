# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Tests for ParsedDem and its optimized sampler conversion.

These tests verify that:
1. ParsedDem correctly parses Stim DEM format
2. ParsedDem.to_dem_sampler() produces statistically equivalent sampling
3. Both naive (sample_batch) and optimized (to_dem_sampler) paths match Stim
"""

import numpy as np
import pytest

# Skip all tests if stim is not installed
stim = pytest.importorskip("stim")


class TestParsedDemBasics:
    """Basic ParsedDem functionality tests."""

    def test_parse_simple_dem(self) -> None:
        """ParsedDem should correctly parse simple DEM strings."""
        from pecos_rslib.qec import ParsedDem

        dem_str = "error(0.01) D0 D1"
        parsed = ParsedDem.from_string(dem_str)

        assert parsed.num_mechanisms == 1
        assert parsed.num_detectors == 2

    def test_parse_with_observable(self) -> None:
        """ParsedDem should correctly parse DEMs with observables."""
        from pecos_rslib.qec import ParsedDem

        dem_str = "error(0.05) D0 L0"
        parsed = ParsedDem.from_string(dem_str)

        assert parsed.num_mechanisms == 1
        assert parsed.num_detectors == 1
        assert parsed.num_observables == 1

    def test_parse_decomposed_dem(self) -> None:
        """ParsedDem should correctly parse decomposed DEMs."""
        from pecos_rslib.qec import ParsedDem

        # Decomposed error - all components fire together
        dem_str = "error(0.1) D0 ^ D1 ^ D2"
        parsed = ParsedDem.from_string(dem_str)

        assert parsed.num_mechanisms == 1
        assert parsed.num_detectors == 3

    def test_parse_stim_surface_code_dem(self) -> None:
        """ParsedDem should correctly parse Stim-generated DEMs."""
        from pecos_rslib.qec import ParsedDem

        # Generate a small surface code DEM from Stim
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=3,
            rounds=1,
            after_clifford_depolarization=0.01,
        )
        dem = circuit.detector_error_model(decompose_errors=True)
        dem_str = str(dem)

        parsed = ParsedDem.from_string(dem_str)

        assert parsed.num_mechanisms > 0
        assert parsed.num_detectors == dem.num_detectors


class TestParsedDemDecomposedSemantics:
    """Test that decomposed errors are handled correctly (Stim semantics)."""

    def test_decomposed_all_fire_together(self) -> None:
        """Decomposed errors should fire all components together."""
        from pecos_rslib.qec import ParsedDem

        # error(0.5) D0 ^ D1 means: with p=0.5, BOTH D0 and D1 flip together
        dem_str = "error(0.5) D0 ^ D1"
        parsed = ParsedDem.from_string(dem_str)

        # Sample with naive sampler
        dets, _ = parsed.sample_batch(10000, seed=42)
        dets = np.array(dets)

        # D0 and D1 should always fire together
        both_fire = (dets[:, 0] & dets[:, 1]).mean()
        exactly_one = (dets[:, 0] ^ dets[:, 1]).mean()

        assert abs(both_fire - 0.5) < 0.05, "Both should fire ~50% of the time"
        assert exactly_one < 0.01, "Exactly one should almost never fire"

    def test_xor_cancellation(self) -> None:
        """error(p) D0 ^ D0 should result in no effect (XOR cancellation)."""
        from pecos_rslib.qec import ParsedDem

        dem_str = "error(0.5) D0 ^ D0"
        parsed = ParsedDem.from_string(dem_str)

        # Sample - D0 should never fire due to XOR cancellation
        dets, _ = parsed.sample_batch(10000, seed=42)
        d0_fires = sum(1 for d in dets if d and d[0])

        assert d0_fires == 0, "D0 should never fire due to XOR cancellation"

    def test_decomposed_matches_stim_semantics(self) -> None:
        """Decomposed error sampling should match Stim's semantics."""
        # Generate identical samples from Stim and PECOS
        from pecos_rslib.qec import ParsedDem

        dem_str = "error(0.5) D0 ^ D1"

        # Stim
        stim_dem = stim.DetectorErrorModel(dem_str)
        stim_sampler = stim_dem.compile_sampler(seed=42)
        stim_det, _, _ = stim_sampler.sample(10000)

        # PECOS
        parsed = ParsedDem.from_string(dem_str)
        pecos_det, _ = parsed.sample_batch(10000, seed=42)
        pecos_det = np.array(pecos_det)

        # Compare statistics
        stim_both = (stim_det[:, 0] & stim_det[:, 1]).mean()
        pecos_both = (pecos_det[:, 0] & pecos_det[:, 1]).mean()

        assert (
            abs(stim_both - pecos_both) < 0.05
        ), f"Stim and PECOS should match: Stim={stim_both:.4f}, PECOS={pecos_both:.4f}"


class TestParsedDemOptimizedSampler:
    """Test ParsedDem.to_dem_sampler() optimized path."""

    def test_optimized_sampler_creation(self) -> None:
        """to_dem_sampler() should create a valid DemSampler."""
        from pecos_rslib.qec import ParsedDem

        dem_str = "error(0.1) D0 D1"
        parsed = ParsedDem.from_string(dem_str)
        sampler = parsed.to_dem_sampler()

        assert sampler.num_mechanisms == 1
        assert sampler.num_detectors == 2

    def test_optimized_sampler_projects_tracked_paulis_but_fails_direct_sampling(self) -> None:
        """Parsed PECOS DEM samplers preserve tracked-Pauli IDs but do not sample them directly."""
        from pecos_rslib.qec import ParsedDem

        parsed = ParsedDem.from_string("error(0.1) D0 TP1")
        sampler = parsed.to_dem_sampler()

        assert parsed.num_tracked_paulis == 2
        assert sampler.num_detectors == 1
        assert sampler.num_dem_outputs == 0
        assert sampler.num_tracked_paulis == 2

        detectors, dem_outputs = sampler.sample(seed=11)
        assert len(detectors) == 1
        assert isinstance(detectors[0], bool)
        assert dem_outputs == []

        with pytest.raises(RuntimeError, match="cannot directly sample tracked Pauli flips"):
            sampler.sample_tracked_paulis(seed=11)

    def test_parser_rejects_legacy_tracked_metadata_extension(self) -> None:
        """The PECOS DEM parser should not accept old tracked-op extension lines."""
        from pecos_rslib.qec import ParsedDem

        with pytest.raises(ValueError, match="unsupported PECOS DEM extension line"):
            ParsedDem.from_string('pecos_tracked_op {"id":0,"pauli":"+X0"}')

    def test_optimized_matches_naive_sampler(self) -> None:
        """Optimized sampler should produce same statistics as naive sampler."""
        from pecos_rslib.qec import ParsedDem

        dem_str = """
error(0.1) D0
error(0.05) D0 D1
error(0.02) D1
"""
        parsed = ParsedDem.from_string(dem_str)
        sampler = parsed.to_dem_sampler()

        # Naive sampling
        dets_naive, _ = parsed.sample_batch(50000, seed=42)
        naive_rate = sum(1 for d in dets_naive if any(d)) / len(dets_naive)

        # Optimized sampling
        stats = sampler.sample_statistics(50000, seed=42)
        opt_rate = stats["syndrome_rate"]

        # Should be within statistical tolerance
        assert (
            abs(naive_rate - opt_rate) < 0.02
        ), f"Naive and optimized should match: naive={naive_rate:.4f}, opt={opt_rate:.4f}"

    def test_optimized_matches_stim_surface_code(self) -> None:
        """Optimized sampler should match Stim for surface code DEMs."""
        from pecos_rslib.qec import ParsedDem

        # Generate surface code DEM from Stim
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=5,
            rounds=5,
            after_clifford_depolarization=0.001,
            after_reset_flip_probability=0.001,
            before_measure_flip_probability=0.001,
        )
        stim_dem = circuit.detector_error_model(decompose_errors=True)
        dem_str = str(stim_dem)

        # PECOS optimized sampler
        parsed = ParsedDem.from_string(dem_str)
        pecos_sampler = parsed.to_dem_sampler()

        # Sample from both
        num_shots = 100_000

        # Stim
        stim_sampler = stim_dem.compile_sampler(seed=42)
        stim_det, _, _ = stim_sampler.sample(num_shots)
        stim_syndrome_rate = np.any(stim_det, axis=1).mean()

        # PECOS
        pecos_stats = pecos_sampler.sample_statistics(num_shots, seed=42)
        pecos_syndrome_rate = pecos_stats["syndrome_rate"]

        # Should be within 3-sigma statistical tolerance
        tolerance = 3.0 / np.sqrt(num_shots)
        assert abs(stim_syndrome_rate - pecos_syndrome_rate) < tolerance + 0.01, (
            f"PECOS should match Stim: Stim={stim_syndrome_rate:.4f}, "
            f"PECOS={pecos_syndrome_rate:.4f}, tolerance={tolerance:.4f}"
        )


class TestParsedDemVsStimComprehensive:
    """Comprehensive tests comparing PECOS ParsedDem to Stim."""

    @pytest.mark.parametrize("distance", [3, 5])
    @pytest.mark.parametrize("rounds", [1, 3])
    @pytest.mark.parametrize("p", [0.001, 0.01])
    def test_syndrome_rate_matches_stim(
        self,
        distance: int,
        rounds: int,
        p: float,
    ) -> None:
        """Syndrome rate should match Stim for various configurations."""
        from pecos_rslib.qec import ParsedDem

        # Generate DEM from Stim
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=distance,
            rounds=rounds,
            after_clifford_depolarization=p,
            after_reset_flip_probability=p,
            before_measure_flip_probability=p,
        )
        stim_dem = circuit.detector_error_model(decompose_errors=True)
        dem_str = str(stim_dem)

        # PECOS sampling
        parsed = ParsedDem.from_string(dem_str)
        sampler = parsed.to_dem_sampler()

        num_shots = 50_000

        # Stim
        stim_sampler = stim_dem.compile_sampler(seed=123)
        stim_det, _, _ = stim_sampler.sample(num_shots)
        stim_rate = np.any(stim_det, axis=1).mean()

        # PECOS
        pecos_stats = sampler.sample_statistics(num_shots, seed=123)
        pecos_rate = pecos_stats["syndrome_rate"]

        # Should be within statistical tolerance
        tolerance = 3.0 / np.sqrt(num_shots) + 0.01
        diff = abs(stim_rate - pecos_rate)
        assert (
            diff < tolerance
        ), f"d={distance}, r={rounds}, p={p}: Stim={stim_rate:.4f}, PECOS={pecos_rate:.4f}, diff={diff:.4f}"

    def test_high_error_rate_matching(self) -> None:
        """PECOS should match Stim even at high error rates."""
        from pecos_rslib.qec import ParsedDem

        # High error rate
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=3,
            rounds=3,
            after_clifford_depolarization=0.1,
            after_reset_flip_probability=0.1,
            before_measure_flip_probability=0.1,
        )
        stim_dem = circuit.detector_error_model(decompose_errors=True)
        dem_str = str(stim_dem)

        parsed = ParsedDem.from_string(dem_str)
        sampler = parsed.to_dem_sampler()

        num_shots = 50_000

        # Stim
        stim_sampler = stim_dem.compile_sampler(seed=456)
        stim_det, _, _ = stim_sampler.sample(num_shots)
        stim_rate = np.any(stim_det, axis=1).mean()

        # PECOS
        pecos_stats = sampler.sample_statistics(num_shots, seed=456)
        pecos_rate = pecos_stats["syndrome_rate"]

        # At high error rates, syndrome rate should be ~1.0 for both
        assert abs(stim_rate - pecos_rate) < 0.02, f"High error rate: Stim={stim_rate:.4f}, PECOS={pecos_rate:.4f}"


class TestParsedDemPerformance:
    """Verify the optimized sampler is actually faster."""

    def test_optimized_faster_than_naive(self) -> None:
        """Optimized sampler should be faster than naive for large shots."""
        import time

        from pecos_rslib.qec import ParsedDem

        # Generate a reasonably sized DEM
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=5,
            rounds=5,
            after_clifford_depolarization=0.001,
        )
        dem_str = str(circuit.detector_error_model(decompose_errors=True))

        parsed = ParsedDem.from_string(dem_str)
        sampler = parsed.to_dem_sampler()

        num_shots = 10_000

        # Naive timing
        start = time.perf_counter()
        _ = parsed.sample_batch(num_shots, seed=42)
        naive_time = time.perf_counter() - start

        # Optimized timing
        start = time.perf_counter()
        _ = sampler.sample_statistics(num_shots, seed=42)
        opt_time = time.perf_counter() - start

        # Optimized should be significantly faster
        assert opt_time < naive_time, f"Optimized ({opt_time:.3f}s) should be faster than naive ({naive_time:.3f}s)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
