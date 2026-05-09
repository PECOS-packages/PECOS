# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Tests for analysis helpers with DEM-backed sampling backends."""

import pytest
from pecos.qec.analysis import empirical_correlation_table, fit_dem_from_simulation
from pecos.qec.surface import SurfacePatch
from pecos.qec.surface.decode import _build_surface_tick_circuit_for_native_model
from pecos_rslib_exp import depolarizing


@pytest.fixture
def d3_circuit_and_noise():
    patch = SurfacePatch.create(distance=3)
    tc = _build_surface_tick_circuit_for_native_model(patch, 6, "Z", circuit_source="abstract")
    noise = depolarizing().p1(0.005).p2(0.005).p_meas(0.005).p_prep(0.005)
    return tc, noise


class TestEmpiricalCorrelationTable:
    def test_meas_sampling_returns_nonempty(self, d3_circuit_and_noise):
        tc, noise = d3_circuit_and_noise
        table = empirical_correlation_table(
            tc,
            noise,
            shots=5000,
            max_order=1,
            backend="meas_sampling",
            seed=42,
        )
        assert len(table) > 0, "Should return at least one rate entry"

    def test_meas_sampling_label_shape(self, d3_circuit_and_noise):
        tc, noise = d3_circuit_and_noise
        table = empirical_correlation_table(
            tc,
            noise,
            shots=5000,
            max_order=1,
            backend="meas_sampling",
            seed=42,
        )
        # Each entry is (detector_indices_tuple, probability)
        for indices, prob in table:
            assert isinstance(indices, tuple)
            assert len(indices) >= 1
            assert isinstance(prob, float)
            assert 0.0 <= prob <= 1.0

    def test_meas_sampling_rates_close_to_stabilizer(self, d3_circuit_and_noise):
        tc, noise = d3_circuit_and_noise
        shots = 20000

        meas_table = empirical_correlation_table(
            tc,
            noise,
            shots=shots,
            max_order=1,
            backend="meas_sampling",
            seed=42,
        )
        stab_table = empirical_correlation_table(
            tc,
            noise,
            shots=shots,
            max_order=1,
            backend="stabilizer",
            seed=42,
        )

        # Both should have the same entries (same detectors)
        meas_dict = dict(meas_table)
        stab_dict = dict(stab_table)

        assert set(meas_dict.keys()) == set(stab_dict.keys()), "Same detector indices should appear in both"

        # Rates should be statistically close (within 20% relative for active detectors)
        close_count = 0
        active_count = 0
        for key, d in meas_dict.items():
            s = stab_dict[key]
            if s > 0.005:
                active_count += 1
                if abs(d - s) / s < 0.20:
                    close_count += 1

        assert active_count > 0, "Should have active detectors"
        assert close_count >= active_count * 0.8, f"Only {close_count}/{active_count} rates within 20% of stabilizer"


class TestFitDemFromSimulation:
    def test_meas_sampling_returns_dem_string(self, d3_circuit_and_noise):
        tc, noise = d3_circuit_and_noise
        dem_str = fit_dem_from_simulation(
            tc,
            noise,
            shots=10000,
            backend="meas_sampling",
            seed=42,
        )
        assert isinstance(dem_str, str)
        assert "error(" in dem_str, "DEM string should contain error(...) lines"

    def test_meas_sampling_has_multiple_mechanisms(self, d3_circuit_and_noise):
        tc, noise = d3_circuit_and_noise
        dem_str = fit_dem_from_simulation(
            tc,
            noise,
            shots=10000,
            backend="meas_sampling",
            seed=42,
        )
        error_lines = [line for line in dem_str.strip().split("\n") if line.strip().startswith("error(")]
        assert len(error_lines) > 10, f"Expected many mechanisms, got {len(error_lines)}"


class TestInvalidBackend:
    def test_empirical_correlation_table_rejects_unknown(self, d3_circuit_and_noise):
        tc, noise = d3_circuit_and_noise
        with pytest.raises(ValueError, match=r"Unknown backend.*'bogus'"):
            empirical_correlation_table(tc, noise, shots=10, backend="bogus")

    def test_fit_dem_from_simulation_rejects_unknown(self, d3_circuit_and_noise):
        tc, noise = d3_circuit_and_noise
        with pytest.raises(ValueError, match=r"Unknown backend.*'nope'"):
            fit_dem_from_simulation(tc, noise, shots=10, backend="nope")
