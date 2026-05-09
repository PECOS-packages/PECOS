# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Integration tests for meas_sampling() sim_neo backend.

Tests the d=3 surface code 57/48 regression and method dispatch.
"""

import json

import pytest
from pecos.qec.surface import SurfacePatch
from pecos.qec.surface.decode import _build_surface_tick_circuit_for_native_model
from pecos_rslib_exp import depolarizing, meas_sampling, sim_neo, stabilizer


@pytest.fixture
def d3_tc():
    patch = SurfacePatch.create(distance=3)
    return _build_surface_tick_circuit_for_native_model(patch, 6, "Z", circuit_source="abstract")


@pytest.fixture
def depol():
    return depolarizing().p1(0.0005).p2(0.005).p_meas(0.005).p_prep(0.005)


@pytest.fixture
def coherent():
    return depolarizing().p1(0.0005).p2(0.005).p_meas(0.005).p_prep(0.005).idle_rz(0.05)


class TestD3SurfaceCode57vs48:
    def test_raw_output_is_57_measurements(self, d3_tc, depol):
        r = sim_neo(d3_tc).quantum(meas_sampling()).noise(depol).shots(10).seed(42).run()
        assert len(r[0]) == 57

    def test_nondet_measurement_mean_half(self, d3_tc, depol):
        shots = 5000
        r = sim_neo(d3_tc).quantum(meas_sampling()).noise(depol).shots(shots).seed(42).run()
        mean_0 = sum(s[0] for s in r) / shots
        assert abs(mean_0 - 0.5) < 0.05, f"meas[0]={mean_0:.3f}"

    def test_det_measurement_mean_low(self, d3_tc, depol):
        shots = 5000
        r = sim_neo(d3_tc).quantum(meas_sampling()).noise(depol).shots(shots).seed(42).run()
        mean_4 = sum(s[4] for s in r) / shots
        assert mean_4 < 0.1, f"meas[4]={mean_4:.3f}"

    def test_z_type_detection_rates_match_stabilizer(self, d3_tc, depol):
        """Z-type detector rates match stabilizer (known-good subset)."""
        shots = 10000
        det_json = json.loads(d3_tc.get_meta("detectors"))
        num_meas = int(d3_tc.get_meta("num_measurements"))
        num_dets = len(det_json)

        def rates(results):
            r = [0.0] * num_dets
            for shot in results:
                for i, det in enumerate(det_json):
                    val = 0
                    for rec in det["records"]:
                        idx = num_meas + rec
                        if 0 <= idx < len(shot):
                            val ^= shot[idx]
                    if val:
                        r[i] += 1.0 / len(results)
            return r

        meas_r = sim_neo(d3_tc).quantum(meas_sampling()).noise(depol).shots(shots).seed(42).run()
        stab_r = sim_neo(d3_tc).quantum(stabilizer()).noise(depol).shots(shots).seed(42).run()

        meas_rates = rates(meas_r)
        stab_rates = rates(stab_r)

        # Z-type detectors (deterministic measurements) should match.
        close_count = sum(
            1 for d, s in zip(meas_rates, stab_rates, strict=False) if s > 0.001 and abs(d - s) / s < 0.15
        )
        total_active = sum(1 for s in stab_rates if s > 0.001)

        # At least half the detectors should match (Z-type ones)
        assert (
            close_count >= total_active // 2
        ), f"Only {close_count}/{total_active} detectors within 15% of stabilizer."

    def test_all_detection_rates_match_stabilizer(self, d3_tc, depol):
        """ALL detector rates should match stabilizer (target correctness)."""
        shots = 20000
        det_json = json.loads(d3_tc.get_meta("detectors"))
        num_meas = int(d3_tc.get_meta("num_measurements"))
        num_dets = len(det_json)

        def rates(results):
            r = [0.0] * num_dets
            for shot in results:
                for i, det in enumerate(det_json):
                    val = 0
                    for rec in det["records"]:
                        idx = num_meas + rec
                        if 0 <= idx < len(shot):
                            val ^= shot[idx]
                    if val:
                        r[i] += 1.0 / len(results)
            return r

        meas_r = sim_neo(d3_tc).quantum(meas_sampling()).noise(depol).shots(shots).seed(42).run()
        stab_r = sim_neo(d3_tc).quantum(stabilizer()).noise(depol).shots(shots).seed(42).run()

        meas_rates = rates(meas_r)
        stab_rates = rates(stab_r)

        max_diff = max(abs(d - s) / max(s, 1e-10) for d, s in zip(meas_rates, stab_rates, strict=False) if s > 0.001)
        assert max_diff < 0.15, f"Max relative det rate diff: {max_diff:.1%}"

    def test_observable_flip_rates_match_stabilizer(self, d3_tc):
        """Observable record extraction should match the stabilizer backend."""
        shots = 5000
        noise = depolarizing().p1(0.001).p2(0.01).p_meas(0.01).p_prep(0.01)
        obs_json = json.loads(d3_tc.get_meta("observables") or "[]")
        num_meas = int(d3_tc.get_meta("num_measurements"))
        assert obs_json, "surface-code memory circuit should define observables"

        def rates(results):
            r = [0.0] * len(obs_json)
            for shot in results:
                for i, obs in enumerate(obs_json):
                    val = 0
                    for rec in obs["records"]:
                        idx = num_meas + rec
                        if 0 <= idx < len(shot):
                            val ^= shot[idx]
                    if val:
                        r[i] += 1.0 / len(results)
            return r

        meas_r = sim_neo(d3_tc).quantum(meas_sampling()).noise(noise).shots(shots).seed(42).run()
        stab_r = sim_neo(d3_tc).quantum(stabilizer()).noise(noise).shots(shots).seed(43).run()

        meas_rates = rates(meas_r)
        stab_rates = rates(stab_r)
        for i, (meas_rate, stab_rate) in enumerate(zip(meas_rates, stab_rates, strict=False)):
            abs_diff = abs(meas_rate - stab_rate)
            rel_diff = abs_diff / max(stab_rate, 1e-12)
            assert (
                abs_diff < 0.03 or rel_diff < 0.5
            ), f"Observable L{i} rate mismatch: meas_sampling={meas_rate:.4f}, stabilizer={stab_rate:.4f}"


class TestMethodDispatch:
    def test_auto_no_idle_rz(self, d3_tc, depol):
        r = sim_neo(d3_tc).quantum(meas_sampling("auto")).noise(depol).shots(10).seed(42).run()
        assert len(r[0]) == 57

    def test_auto_with_idle_rz(self, d3_tc, coherent):
        r = sim_neo(d3_tc).quantum(meas_sampling("auto")).noise(coherent).shots(10).seed(42).run()
        assert len(r[0]) == 57

    def test_stochastic_rejects_idle_rz(self, d3_tc, coherent):
        with pytest.raises(Exception, match="idle_rz"):
            sim_neo(d3_tc).quantum(meas_sampling("stochastic")).noise(coherent).shots(10).seed(42).run()

    def test_coherent_no_idle_rz(self, d3_tc, depol):
        r = sim_neo(d3_tc).quantum(meas_sampling("coherent")).noise(depol).shots(10).seed(42).run()
        assert len(r[0]) == 57

    def test_coherent_with_idle_rz(self, d3_tc, coherent):
        r = sim_neo(d3_tc).quantum(meas_sampling("coherent")).noise(coherent).shots(10).seed(42).run()
        assert len(r[0]) == 57

    def test_invalid_method(self, d3_tc, depol):
        with pytest.raises(Exception, match="Unknown"):
            sim_neo(d3_tc).quantum(meas_sampling("bogus")).noise(depol).shots(10).seed(42).run()

    def test_no_noise_errors(self, d3_tc):
        with pytest.raises(Exception, match="noise"):
            sim_neo(d3_tc).quantum(meas_sampling()).shots(10).seed(42).run()
