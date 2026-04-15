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
    generate_dem_from_tick_circuit,
    generate_surface_code_dem,
    generate_tick_circuit_from_patch,
    syndromes_to_detection_events,
)


def _require_selene_runtime() -> None:
    """Eagerly instantiate the Selene engine to fail fast if it is unavailable.

    The PECOS test environment is expected to have the Selene runtime installed
    (see ``pecos setup``). A failure here means the environment is broken, not
    that the test should be skipped.
    """
    import pecos

    pecos.selene_engine()


def _count_singleton_error_parts(dem: str) -> int:
    """Count decomposed error parts that touch exactly one detector."""
    count = 0
    for line in dem.splitlines():
        stripped = line.strip()
        if not stripped.startswith("error("):
            continue
        payload = stripped.split(")", 1)[1]
        for part in payload.split("^"):
            detectors = [token for token in part.split() if token.startswith("D")]
            if len(detectors) == 1:
                count += 1
    return count


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

    def test_get_dem_caches_circuit_level_dem(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Repeated circuit-level DEM requests should reuse the decoder-local cache."""
        import pecos.qec.surface.decode as decode_module

        patch = SurfacePatch.create(distance=3)
        noise = NoiseModel(p1=0.001, p2=0.01, p_meas=0.01, p_init=0.001)
        decoder = SurfaceDecoder(
            patch,
            num_rounds=3,
            noise=noise,
            circuit_level_dem_mode="native_decomposed",
        )

        real_generate = decode_module.generate_circuit_level_dem_from_builder
        calls = 0

        def wrapped_generate(*args: object, **kwargs: object) -> str:
            nonlocal calls
            calls += 1
            return real_generate(*args, **kwargs)

        monkeypatch.setattr(decode_module, "generate_circuit_level_dem_from_builder", wrapped_generate)

        dem_1 = decoder.get_dem("Z", circuit_level=True)
        dem_2 = decoder.get_dem("Z", circuit_level=True)

        assert dem_1 == dem_2
        assert calls == 1

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

    def test_generate_dem_from_patch_can_skip_stim_decomposition(self) -> None:
        """Stim patch DEM helper should support raw vs decomposed DEM output."""
        from pecos.qec.surface.decode import generate_dem_from_patch

        patch = SurfacePatch.create(distance=3)
        noise = NoiseModel(p1=0.001, p2=0.01, p_meas=0.01, p_init=0.001)

        full_dem = generate_dem_from_patch(patch, num_rounds=4, noise=noise, basis="X", decompose_errors=False)
        decomposed_dem = generate_dem_from_patch(patch, num_rounds=4, noise=noise, basis="X", decompose_errors=True)

        assert "^" not in full_dem
        assert "^" in decomposed_dem

    def test_generate_dem_from_tick_circuit_supports_raw_and_decomposed_output(self) -> None:
        """Native TickCircuit DEM helper should preserve both public output forms."""
        patch = SurfacePatch.create(distance=3)
        tc = generate_tick_circuit_from_patch(patch, num_rounds=4, basis="X")
        params = {"p1": 0.001, "p2": 0.01, "p_meas": 0.01, "p_init": 0.001}

        raw_dem = generate_dem_from_tick_circuit(tc, **params, decompose_errors=False)
        decomposed_dem = generate_dem_from_tick_circuit(tc, **params, decompose_errors=True)

        assert raw_dem != decomposed_dem
        assert "^" not in raw_dem
        assert "^" in decomposed_dem

    def test_native_circuit_level_dem_threads_ancilla_budget(self) -> None:
        """Native DEM helpers should use the requested ancilla-budgeted circuit family."""
        from pecos.qec.surface.decode import generate_circuit_level_dem_from_builder

        patch = SurfacePatch.create(distance=3)
        noise = NoiseModel(p1=0.001, p2=0.01, p_meas=0.01, p_init=0.001)
        params = {"p1": noise.p1, "p2": noise.p2, "p_meas": noise.p_meas, "p_init": noise.p_init}

        full_tc = generate_tick_circuit_from_patch(patch, num_rounds=2, basis="X")
        batched_tc = generate_tick_circuit_from_patch(
            patch,
            num_rounds=2,
            basis="X",
            ancilla_budget=2,
        )

        full_dem = generate_circuit_level_dem_from_builder(patch, num_rounds=2, noise=noise, basis="X")
        batched_dem = generate_circuit_level_dem_from_builder(
            patch,
            num_rounds=2,
            noise=noise,
            basis="X",
            ancilla_budget=2,
        )

        assert full_dem == generate_dem_from_tick_circuit(full_tc, **params, decompose_errors=False)
        assert batched_dem == generate_dem_from_tick_circuit(batched_tc, **params, decompose_errors=False)
        assert batched_dem != full_dem

        decoder = SurfaceDecoder(
            patch,
            num_rounds=2,
            noise=noise,
            ancilla_budget=2,
            circuit_level_dem_mode="native_full",
        )
        assert decoder.get_dem("X", circuit_level=True) == batched_dem

    def test_native_circuit_level_dem_cache_respects_patch_geometry(self) -> None:
        """Shared native DEM caching should preserve asymmetric patch geometry."""
        from pecos.qec.surface.circuit_builder import generate_dem_from_tick_circuit, generate_tick_circuit_from_patch
        from pecos.qec.surface.decode import generate_circuit_level_dem_from_builder

        patch = SurfacePatch.create(dx=3, dz=5)
        noise = NoiseModel(p1=0.001, p2=0.01, p_meas=0.01, p_init=0.001)
        params = {"p1": noise.p1, "p2": noise.p2, "p_meas": noise.p_meas, "p_init": noise.p_init}

        tc = generate_tick_circuit_from_patch(patch, num_rounds=2, basis="X")
        expected_dem = generate_dem_from_tick_circuit(tc, **params, decompose_errors=False)
        cached_dem = generate_circuit_level_dem_from_builder(
            patch,
            num_rounds=2,
            noise=noise,
            basis="X",
        )

        assert cached_dem == expected_dem

    def test_traced_qis_native_dem_and_sampler_build(self) -> None:
        """The traced-QIS circuit source should build DEMs and samplers end-to-end."""
        from pecos.qec.surface import build_native_sampler
        from pecos.qec.surface.decode import generate_circuit_level_dem_from_builder

        _require_selene_runtime()

        patch = SurfacePatch.create(distance=3)
        noise = NoiseModel(p1=0.001, p2=0.001, p_meas=0.001, p_init=0.001)

        dem = generate_circuit_level_dem_from_builder(
            patch,
            num_rounds=2,
            noise=noise,
            basis="Z",
            decompose_errors=True,
            circuit_source="traced_qis",
        )
        assert "error(" in dem

        sampler = build_native_sampler(
            patch,
            num_rounds=2,
            noise=noise,
            basis="Z",
            circuit_source="traced_qis",
        )
        det_events, obs_flips = sampler.sample(4, seed=7)
        assert det_events.shape == (4, sampler.num_detectors)
        assert obs_flips.shape == (4, sampler.num_observables)
        assert sampler.sampling_model == "dem"

        mnm_sampler = build_native_sampler(
            patch,
            num_rounds=2,
            noise=noise,
            basis="Z",
            circuit_source="traced_qis",
            sampling_model="mnm",
        )
        mnm_det_events, mnm_obs_flips = mnm_sampler.sample(4, seed=7)
        assert mnm_det_events.shape == (4, mnm_sampler.num_detectors)
        assert mnm_obs_flips.shape == (4, mnm_sampler.num_observables)
        assert mnm_sampler.sampling_model == "mnm"

        influence_sampler = build_native_sampler(
            patch,
            num_rounds=2,
            noise=noise,
            basis="Z",
            circuit_source="traced_qis",
            sampling_model="influence_dem",
        )
        influence_det_events, influence_obs_flips = influence_sampler.sample(4, seed=7)
        assert influence_det_events.shape == (4, influence_sampler.num_detectors)
        assert influence_obs_flips.shape == (4, influence_sampler.num_observables)
        assert influence_sampler.sampling_model == "influence_dem"

        decoder = SurfaceDecoder(
            patch,
            num_rounds=2,
            noise=noise,
            circuit_level_dem_mode="native_decomposed",
            circuit_level_dem_source="traced_qis",
        )
        decoder_dem = decoder.get_dem("Z", circuit_level=True)
        assert "error(" in decoder_dem
        assert decoder_dem.count("detector(") == dem.count("detector(")
        assert decoder_dem.count("logical_observable") == dem.count("logical_observable")

    def test_traced_qis_native_dem_matches_stim_dem(self) -> None:
        """The traced-QIS PECOS DEM should exactly match the traced-QIS Stim DEM."""
        import re

        stim = pytest.importorskip("stim")

        from pecos.qec.surface.circuit_builder import (
            generate_dem_from_tick_circuit,
            tick_circuit_to_stim,
        )
        from pecos.qec.surface.decode import _build_surface_tick_circuit_for_native_model

        _require_selene_runtime()

        def extract_errors(dem_str: str) -> dict[str, float]:
            errors: dict[str, float] = {}
            for line in dem_str.strip().splitlines():
                match = re.match(r"error\(([^)]+)\)\s*(.*)", line.strip())
                if match:
                    errors[match.group(2).strip()] = float(match.group(1))
            return errors

        patch = SurfacePatch.create(distance=3)
        noise = NoiseModel(p1=0.003, p2=0.003, p_meas=0.003, p_init=0.003)

        for basis in ("X", "Z"):
            tc = _build_surface_tick_circuit_for_native_model(
                patch,
                num_rounds=6,
                basis=basis,
                circuit_source="traced_qis",
            )
            pecos_dem = generate_dem_from_tick_circuit(
                tc,
                p1=noise.p1,
                p2=noise.p2,
                p_meas=noise.p_meas,
                p_init=noise.p_init,
                decompose_errors=False,
            )
            stim_dem = str(
                stim.Circuit(
                    tick_circuit_to_stim(
                        tc,
                        p1=noise.p1,
                        p2=noise.p2,
                        p_meas=noise.p_meas,
                        p_init=noise.p_init,
                    ),
                ).detector_error_model(decompose_errors=False),
            )

            pecos_errors = extract_errors(pecos_dem)
            stim_errors = extract_errors(stim_dem)
            assert set(pecos_errors) == set(stim_errors)
            for target in pecos_errors:
                rel_diff = abs(pecos_errors[target] - stim_errors[target]) / max(
                    pecos_errors[target],
                    stim_errors[target],
                    1e-12,
                )
                assert rel_diff < 0.005, (
                    f"{basis} traced-QIS DEM mismatch for {target}: "
                    f"PECOS={pecos_errors[target]:.8f}, Stim={stim_errors[target]:.8f}"
                )

    def test_traced_qis_native_topology_cache_is_shared_across_public_apis(self) -> None:
        """Public traced-QIS DEM and sampler helpers should reuse the shared topology cache."""
        from pecos.qec.surface import build_native_sampler
        from pecos.qec.surface.decode import (
            _cached_surface_native_dem_string,
            _cached_surface_native_topology,
            generate_circuit_level_dem_from_builder,
        )

        _require_selene_runtime()

        patch = SurfacePatch.create(distance=3)
        noise_a = NoiseModel(p1=0.001, p2=0.001, p_meas=0.001, p_init=0.001)
        noise_b = NoiseModel(p1=0.002, p2=0.002, p_meas=0.002, p_init=0.002)

        _cached_surface_native_topology.cache_clear()
        _cached_surface_native_dem_string.cache_clear()

        generate_circuit_level_dem_from_builder(
            patch,
            num_rounds=2,
            noise=noise_a,
            basis="Z",
            decompose_errors=True,
            circuit_source="traced_qis",
        )
        after_dem = _cached_surface_native_topology.cache_info()
        after_dem_str = _cached_surface_native_dem_string.cache_info()
        assert after_dem.misses == 1
        assert after_dem_str.misses == 1

        sampler = build_native_sampler(
            patch,
            num_rounds=2,
            noise=noise_a,
            basis="Z",
            circuit_source="traced_qis",
        )
        det_events, obs_flips = sampler.sample(2, seed=11)
        assert det_events.shape == (2, sampler.num_detectors)
        assert obs_flips.shape == (2, sampler.num_observables)

        after_sampler = _cached_surface_native_topology.cache_info()
        after_sampler_dem_str = _cached_surface_native_dem_string.cache_info()
        assert after_sampler.misses == after_dem.misses
        assert after_sampler.hits >= after_dem.hits + 1
        assert after_sampler_dem_str.misses == after_dem_str.misses
        assert after_sampler_dem_str.hits >= after_dem_str.hits + 1

        build_native_sampler(
            patch,
            num_rounds=2,
            noise=noise_b,
            basis="Z",
            circuit_source="traced_qis",
        )
        after_second_noise = _cached_surface_native_dem_string.cache_info()
        assert after_second_noise.misses == after_sampler_dem_str.misses + 1

    def test_generate_dem_from_tick_circuit_maximal_decomposition_prefers_singletons(self) -> None:
        """Maximal decomposition should no longer be a no-op."""
        patch = SurfacePatch.create(distance=3)
        tc = generate_tick_circuit_from_patch(patch, num_rounds=20, basis="X")
        params = {"p1": 0.0, "p2": 0.00235, "p_meas": 0.01972626855445279, "p_init": 0.0010045162906914633}

        decomposed_dem = generate_dem_from_tick_circuit(tc, **params, decompose_errors=True)
        maximal_dem = generate_dem_from_tick_circuit(
            tc,
            **params,
            decompose_errors=False,
            maximal_decomposition=True,
        )

        assert maximal_dem != decomposed_dem
        assert _count_singleton_error_parts(maximal_dem) > _count_singleton_error_parts(decomposed_dem)

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
    """Integration tests for noisy simulation.

    The Selene runtime and the ``selene_sim`` Python package are expected to be
    installed in every environment that runs this test tree. A missing runtime
    now raises rather than being silently skipped (see ``_require_selene_runtime``).
    """

    def test_run_noisy_memory_experiment_import(self) -> None:
        """Test that run_noisy_memory_experiment can be imported."""
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

    def test_noiseless_simulation(self) -> None:
        """Noiseless simulation should have zero logical error rate."""
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
