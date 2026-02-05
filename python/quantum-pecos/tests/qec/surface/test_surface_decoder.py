# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Tests for surface code decoder and noisy simulation.

These tests verify:
1. SurfaceDecoder initialization and basic decoding
2. DEM generation
3. Syndrome-to-detection-event conversion
4. Integration with noisy simulation (requires selene_sim)
"""

import numpy as np
import pytest
from pecos.qec.surface import (
    NoiseModel,
    SurfaceDecoder,
    SurfacePatch,
    generate_surface_code_dem,
    syndromes_to_detection_events,
)


class TestNoiseModel:
    """Tests for NoiseModel dataclass."""

    def test_default_values(self) -> None:
        """Default noise model should have zero error rates."""
        noise = NoiseModel()
        assert noise.p1 == 0.0
        assert noise.p2 == 0.0
        assert noise.p_meas == 0.0
        assert noise.p_init == 0.0

    def test_is_noiseless(self) -> None:
        """Test is_noiseless property."""
        assert NoiseModel().is_noiseless
        assert not NoiseModel(p1=0.01).is_noiseless
        assert not NoiseModel(p2=0.01).is_noiseless
        assert not NoiseModel(p_meas=0.01).is_noiseless
        assert not NoiseModel(p_init=0.01).is_noiseless

    def test_physical_error_rate(self) -> None:
        """Test physical_error_rate property."""
        noise = NoiseModel(p1=0.001, p2=0.01, p_meas=0.005, p_init=0.002)
        assert noise.physical_error_rate == 0.01  # max of all rates


class TestSyndromeConversion:
    """Tests for syndrome to detection event conversion."""

    def test_single_round_trivial(self) -> None:
        """Single round trivial syndrome."""
        syndromes = np.array([[0, 0, 0, 0]], dtype=np.uint8)
        events = syndromes_to_detection_events(
            syndromes,
            num_rounds=1,
            num_detectors_per_round=4,
        )
        np.testing.assert_array_equal(events, [[0, 0, 0, 0]])

    def test_single_round_with_errors(self) -> None:
        """Single round with syndrome bits set."""
        syndromes = np.array([[1, 0, 1, 0]], dtype=np.uint8)
        events = syndromes_to_detection_events(
            syndromes,
            num_rounds=1,
            num_detectors_per_round=4,
        )
        np.testing.assert_array_equal(events, [[1, 0, 1, 0]])

    def test_multi_round_xor(self) -> None:
        """Multi-round syndrome should XOR consecutive rounds."""
        syndromes = np.array(
            [
                [1, 0, 0, 0],  # Round 0
                [1, 1, 0, 0],  # Round 1
                [0, 1, 0, 0],  # Round 2
            ],
            dtype=np.uint8,
        )
        events = syndromes_to_detection_events(
            syndromes,
            num_rounds=3,
            num_detectors_per_round=4,
        )

        # Round 0: same as syndrome (compare to zero)
        # Round 1: XOR with round 0 -> [0, 1, 0, 0]
        # Round 2: XOR with round 1 -> [1, 0, 0, 0]
        expected = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [1, 0, 0, 0],
            ],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(events, expected)

    def test_flat_input(self) -> None:
        """Test with flat (1D) input array."""
        syndromes = np.array([1, 0, 0, 1, 1, 0], dtype=np.uint8)
        events = syndromes_to_detection_events(
            syndromes,
            num_rounds=2,
            num_detectors_per_round=3,
        )

        # Round 0: [1, 0, 0] (same as first round syndrome)
        # Round 1: [1, 1, 0] XOR [1, 0, 0] = [0, 1, 0]
        expected = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
            ],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(events, expected)


class TestSurfaceDecoder:
    """Tests for SurfaceDecoder class."""

    def test_create_decoder_d3(self) -> None:
        """Create decoder for distance-3 patch."""
        patch = SurfacePatch.create(distance=3)
        noise = NoiseModel(p2=0.01, p_meas=0.01)
        decoder = SurfaceDecoder(patch, num_rounds=1, noise=noise)

        assert decoder.patch == patch
        assert decoder.num_rounds == 1
        assert decoder.decoder_type.value == "pymatching"

    def test_create_decoder_d5(self) -> None:
        """Create decoder for distance-5 patch."""
        patch = SurfacePatch.create(distance=5)
        noise = NoiseModel(p2=0.01, p_meas=0.01)
        decoder = SurfaceDecoder(patch, num_rounds=3, noise=noise)

        assert decoder.patch == patch
        assert decoder.num_rounds == 3

    def test_decoder_types(self) -> None:
        """Test different decoder type options."""
        patch = SurfacePatch.create(distance=3)
        noise = NoiseModel(p2=0.01, p_meas=0.01)

        # PyMatching (default)
        d1 = SurfaceDecoder(patch, decoder_type="pymatching", noise=noise)
        assert d1.decoder_type.value == "pymatching"

        # FusionBlossom
        d2 = SurfaceDecoder(patch, decoder_type="fusion_blossom", noise=noise)
        assert d2.decoder_type.value == "fusion_blossom"

        # BP+OSD
        d3 = SurfaceDecoder(patch, decoder_type="bp_osd", noise=noise)
        assert d3.decoder_type.value == "bp_osd"

    def test_get_dem(self) -> None:
        """Test DEM generation via decoder."""
        patch = SurfacePatch.create(distance=3)
        noise = NoiseModel(p2=0.01, p_meas=0.01)
        decoder = SurfaceDecoder(patch, num_rounds=3, noise=noise)

        # Test circuit-level DEM (default)
        dem_z = decoder.get_dem(basis="Z")
        dem_x = decoder.get_dem(basis="X")

        assert isinstance(dem_z, str)
        assert isinstance(dem_x, str)
        assert "error" in dem_z

        # Test phenomenological DEM
        pheno_dem = decoder.get_dem(basis="Z", circuit_level=False)
        assert isinstance(pheno_dem, str)
        assert "error" in pheno_dem
        assert "detector" in pheno_dem

    def test_decode_trivial_syndrome_z(self) -> None:
        """Decode trivial Z syndrome (no errors)."""
        patch = SurfacePatch.create(distance=3)
        noise = NoiseModel(p2=0.01, p_meas=0.01)
        decoder = SurfaceDecoder(patch, num_rounds=1, noise=noise)

        # All-zero syndrome
        num_x_stab = len(patch.geometry.x_stabilizers)
        num_z_stab = len(patch.geometry.z_stabilizers)

        synx_list = [np.zeros(num_x_stab, dtype=np.uint8)]
        synz_list = [np.zeros(num_z_stab, dtype=np.uint8)]
        final = np.zeros(patch.num_data, dtype=np.uint8)

        is_error, _result = decoder.decode_memory_z(synx_list, synz_list, final)

        # No errors should be detected
        assert not is_error

    def test_decode_trivial_syndrome_x(self) -> None:
        """Decode trivial X syndrome (no errors)."""
        patch = SurfacePatch.create(distance=3)
        noise = NoiseModel(p2=0.01, p_meas=0.01)
        decoder = SurfaceDecoder(patch, num_rounds=1, noise=noise)

        num_x_stab = len(patch.geometry.x_stabilizers)
        num_z_stab = len(patch.geometry.z_stabilizers)

        synx_list = [np.zeros(num_x_stab, dtype=np.uint8)]
        synz_list = [np.zeros(num_z_stab, dtype=np.uint8)]
        final = np.zeros(patch.num_data, dtype=np.uint8)

        is_error, _result = decoder.decode_memory_x(synx_list, synz_list, final)

        # No errors should be detected
        assert not is_error


class TestDemGeneration:
    """Tests for DEM generation functions."""

    def test_generate_surface_code_dem_z(self) -> None:
        """Generate Z-stabilizer DEM."""
        patch = SurfacePatch.create(distance=3)
        noise = NoiseModel(p2=0.01, p_meas=0.01)

        dem = generate_surface_code_dem(patch, num_rounds=3, noise=noise, stab_type="Z")

        assert isinstance(dem, str)
        assert "error" in dem
        assert "detector" in dem
        assert "logical_observable" in dem

    def test_generate_surface_code_dem_x(self) -> None:
        """Generate X-stabilizer DEM."""
        patch = SurfacePatch.create(distance=3)
        noise = NoiseModel(p2=0.01, p_meas=0.01)

        dem = generate_surface_code_dem(patch, num_rounds=3, noise=noise, stab_type="X")

        assert isinstance(dem, str)
        assert "error" in dem

    def test_dem_detector_count(self) -> None:
        """DEM should have correct number of detectors."""
        patch = SurfacePatch.create(distance=3)
        noise = NoiseModel(p2=0.01, p_meas=0.01)
        num_rounds = 3

        dem = generate_surface_code_dem(
            patch,
            num_rounds=num_rounds,
            noise=noise,
            stab_type="Z",
        )

        # Count detector declarations
        num_detectors = dem.count("detector(")
        num_z_stab = len(patch.geometry.z_stabilizers)
        expected = num_z_stab * num_rounds

        assert num_detectors == expected

    def test_dem_single_round(self) -> None:
        """DEM with single round should have boundary measurement errors."""
        patch = SurfacePatch.create(distance=3)
        noise = NoiseModel(p2=0.01, p_meas=0.01)

        dem = generate_surface_code_dem(patch, num_rounds=1, noise=noise, stab_type="Z")

        # Single round has measurement error as single-detector events
        assert "error" in dem


class TestNoisySimulation:
    """Integration tests for noisy simulation (requires selene_sim)."""

    @pytest.fixture
    def check_selene(self) -> bool | None:
        """Check if selene_sim is available."""
        try:
            import selene_sim

            return True
        except ImportError:
            pytest.skip("selene_sim not available")
            return False

    def test_run_noisy_memory_experiment_import(
        self,
        check_selene: bool,
    ) -> None:
        """Test that run_noisy_memory_experiment can be imported."""
        _ = check_selene  # Fixture triggers skip if unavailable
        from pecos.qec.surface import run_noisy_memory_experiment

        assert callable(run_noisy_memory_experiment)

    def test_simulation_result_dataclass(self) -> None:
        """Test SimulationResult dataclass."""
        from pecos.qec.surface import SimulationResult

        result = SimulationResult(
            distance=3,
            num_shots=100,
            num_rounds=1,
            basis="Z",
            num_logical_errors=5,
            num_raw_errors=10,
            logical_error_rate=0.05,
            raw_error_rate=0.10,
            decoded=True,
            decoder_type="pymatching",
        )

        assert result.distance == 3
        assert result.num_shots == 100
        assert result.logical_error_rate == 0.05

    def test_noiseless_simulation(self, check_selene: bool) -> None:
        """Noiseless simulation should have zero logical error rate."""
        _ = check_selene  # Fixture triggers skip if unavailable
        from pecos.compilation_pipeline import compile_guppy_to_hugr
        from pecos.guppy.surface import get_num_qubits, make_surface_code
        from pecos.qec.surface import SurfacePatch
        from selene_sim import IdealErrorModel, SimpleRuntime, Stim, build

        distance = 3
        num_rounds = 1
        num_shots = 10
        basis = "Z"

        patch = SurfacePatch.create(distance=distance)
        logical_qubits = patch.geometry.logical_z.data_qubits

        num_qubits = get_num_qubits(distance)
        prog = make_surface_code(distance=distance, num_rounds=num_rounds, basis=basis)
        hugr_bytes = compile_guppy_to_hugr(prog)
        instance = build(hugr_bytes, name=f"surface_d{distance}")

        num_errors = 0
        for shot_results in instance.run_shots(
            simulator=Stim(),
            n_qubits=num_qubits,
            n_shots=num_shots,
            error_model=IdealErrorModel(),
            runtime=SimpleRuntime(),
            n_processes=1,
        ):
            final = None
            for name, values in shot_results:
                if name == "final":
                    final = list(values)

            if final is not None:
                parity = sum(final[q] for q in logical_qubits) % 2
                if parity != 0:
                    num_errors += 1

        # Noiseless should have no errors
        assert num_errors == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
