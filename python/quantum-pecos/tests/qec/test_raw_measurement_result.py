# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Tests proving RawMeasurementResult protocol compatibility across backends.

All sim_neo backends now return RawMeasurementResult, a common Rust-backed
type that supports indexing, iteration, len(), and get(). This test verifies
the output contract is identical for stabilizer and meas_sampling.
"""

import pytest
from pecos.qec.surface import SurfacePatch
from pecos.qec.surface.decode import _build_surface_tick_circuit_for_native_model
from pecos_rslib_exp import depolarizing, meas_sampling, sim_neo, stabilizer


@pytest.fixture
def d3_results():
    """Run both backends on the same circuit and return their results."""
    patch = SurfacePatch.create(distance=3)
    tc = _build_surface_tick_circuit_for_native_model(patch, 6, "Z", circuit_source="abstract")
    depol = depolarizing().p1(0.005).p2(0.005).p_meas(0.005).p_prep(0.005)

    stab_r = sim_neo(tc).quantum(stabilizer()).noise(depol).shots(100).seed(42).run()
    meas_r = sim_neo(tc).quantum(meas_sampling()).noise(depol).shots(100).seed(42).run()
    return stab_r, meas_r


class TestCommonProtocol:
    """Both backends return objects with the same interface."""

    def test_len(self, d3_results):
        stab_r, meas_r = d3_results
        assert len(stab_r) == 100
        assert len(meas_r) == 100

    def test_indexing(self, d3_results):
        stab_r, meas_r = d3_results
        # r[shot] returns a sequence of u8 values
        s0 = stab_r[0]
        d0 = meas_r[0]
        assert len(s0) == len(d0) == 57  # d=3 surface code has 57 measurements

    def test_item_values(self, d3_results):
        stab_r, meas_r = d3_results
        # Individual values are 0 or 1
        for val in stab_r[0]:
            assert val in (0, 1)
        for val in meas_r[0]:
            assert val in (0, 1)

    def test_list_conversion(self, d3_results):
        stab_r, meas_r = d3_results
        s_row = list(stab_r[0])
        d_row = list(meas_r[0])
        assert all(isinstance(v, int) for v in s_row)
        assert all(isinstance(v, int) for v in d_row)
        assert len(s_row) == len(d_row) == 57

    def test_iteration(self, d3_results):
        stab_r, meas_r = d3_results
        stab_count = 0
        for row in stab_r:
            stab_count += 1
            assert len(row) == 57
        assert stab_count == 100

        dem_count = 0
        for row in meas_r:
            dem_count += 1
            assert len(row) == 57
        assert dem_count == 100

    def test_out_of_range_raises_index_error(self, d3_results):
        stab_r, meas_r = d3_results
        with pytest.raises(IndexError):
            stab_r[100]
        with pytest.raises(IndexError):
            meas_r[100]

    def test_num_shots_property(self, d3_results):
        stab_r, meas_r = d3_results
        assert stab_r.num_shots == 100
        assert meas_r.num_shots == 100

    def test_num_measurements_property(self, d3_results):
        stab_r, meas_r = d3_results
        assert stab_r.num_measurements == 57
        assert meas_r.num_measurements == 57

    def test_get_method(self, d3_results):
        stab_r, meas_r = d3_results
        # get(shot, meas) returns 0 or 1
        assert stab_r.get(0, 0) in (0, 1)
        assert meas_r.get(0, 0) in (0, 1)

    def test_get_out_of_range(self, d3_results):
        stab_r, meas_r = d3_results
        with pytest.raises(IndexError):
            stab_r.get(100, 0)
        with pytest.raises(IndexError):
            meas_r.get(0, 57)

    def test_get_shot_method(self, d3_results):
        stab_r, meas_r = d3_results
        s = stab_r.get_shot(0)
        d = meas_r.get_shot(0)
        assert len(s) == len(d) == 57

    def test_to_list(self, d3_results):
        stab_r, meas_r = d3_results
        sl = stab_r.to_list()
        dl = meas_r.to_list()
        assert len(sl) == len(dl) == 100
        assert len(sl[0]) == len(dl[0]) == 57

    def test_negative_index_raises_index_error(self, d3_results):
        """Negative indexing raises IndexError, never OverflowError."""
        stab_r, meas_r = d3_results
        with pytest.raises(IndexError):
            stab_r[-1]
        with pytest.raises(IndexError):
            meas_r[-1]

    def test_negative_get_raises_index_error(self, d3_results):
        stab_r, meas_r = d3_results
        # Negative shot
        with pytest.raises(IndexError):
            stab_r.get(-1, 0)
        with pytest.raises(IndexError):
            meas_r.get(-1, 0)
        # Negative measurement
        with pytest.raises(IndexError):
            stab_r.get(0, -1)
        with pytest.raises(IndexError):
            meas_r.get(0, -1)

    def test_get_shot_negative_raises_index_error(self, d3_results):
        stab_r, meas_r = d3_results
        with pytest.raises(IndexError):
            stab_r.get_shot(-1)
        with pytest.raises(IndexError):
            meas_r.get_shot(-1)

    def test_out_of_range_uses_len(self, d3_results):
        """result[len(result)] raises IndexError."""
        stab_r, meas_r = d3_results
        with pytest.raises(IndexError):
            stab_r[len(stab_r)]
        with pytest.raises(IndexError):
            meas_r[len(meas_r)]


class TestGenericConsumer:
    """A generic helper that works unchanged for any backend."""

    def compute_measurement_means(self, result):
        """Compute per-measurement mean across all shots — works for any backend."""
        shots = len(result)
        if shots == 0:
            return []
        n_meas = len(result[0])
        means = [0.0] * n_meas
        for row in result:
            for i, val in enumerate(row):
                means[i] += val
        return [m / shots for m in means]

    def test_generic_consumer_stabilizer(self):
        patch = SurfacePatch.create(distance=3)
        tc = _build_surface_tick_circuit_for_native_model(patch, 6, "Z", circuit_source="abstract")
        depol = depolarizing().p1(0.005).p2(0.005).p_meas(0.005).p_prep(0.005)
        result = sim_neo(tc).quantum(stabilizer()).noise(depol).shots(1000).seed(42).run()

        means = self.compute_measurement_means(result)
        assert len(means) == 57
        # Non-det measurements should be ~0.5, det should be ~0
        nondet = sum(1 for m in means if abs(m - 0.5) < 0.15)
        assert nondet > 0

    def test_generic_consumer_meas_sampling(self):
        patch = SurfacePatch.create(distance=3)
        tc = _build_surface_tick_circuit_for_native_model(patch, 6, "Z", circuit_source="abstract")
        depol = depolarizing().p1(0.005).p2(0.005).p_meas(0.005).p_prep(0.005)
        result = sim_neo(tc).quantum(meas_sampling()).noise(depol).shots(1000).seed(42).run()

        means = self.compute_measurement_means(result)
        assert len(means) == 57
        nondet = sum(1 for m in means if abs(m - 0.5) < 0.15)
        assert nondet > 0
