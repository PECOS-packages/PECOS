# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Tests for SampleBatch columnar storage and validation."""

import pytest
from pecos_rslib.qec import DemSampler, SampleBatch


class TestSampleBatchConstruction:
    def test_round_trip_get_syndrome(self):
        batch = SampleBatch([[1, 0], [0, 1]], [1, 0])
        assert list(batch.get_syndrome(0)) == [1, 0]
        assert list(batch.get_syndrome(1)) == [0, 1]

    def test_round_trip_get_observable_mask(self):
        batch = SampleBatch([[1, 0], [0, 1]], [1, 0])
        assert batch.get_observable_mask(0) == 1
        assert batch.get_observable_mask(1) == 0

    def test_num_shots(self):
        batch = SampleBatch([[0, 0], [1, 1], [0, 1]], [0, 0, 0])
        assert batch.num_shots == 3

    def test_ragged_rows_longer_rejected(self):
        with pytest.raises(ValueError, match=r"row 1.*length 3.*expected 2"):
            SampleBatch([[1, 0], [0, 1, 1]], [0, 0])

    def test_ragged_rows_shorter_rejected(self):
        with pytest.raises(ValueError, match=r"row 2.*length 1.*expected 2"):
            SampleBatch([[1, 0], [0, 1], [0]], [0, 0, 0])

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="must have same length"):
            SampleBatch([[1, 0]], [0, 0])

    def test_empty_batch(self):
        batch = SampleBatch([], [])
        assert batch.num_shots == 0


class TestGeneratedSampleBatch:
    @pytest.fixture
    def d3_setup(self):
        from pecos.qec.surface import SurfacePatch
        from pecos.qec.surface.decode import _build_surface_tick_circuit_for_native_model

        patch = SurfacePatch.create(distance=3)
        tc = _build_surface_tick_circuit_for_native_model(
            patch,
            6,
            "Z",
            circuit_source="abstract",
        )
        sampler = DemSampler.from_circuit(
            tc,
            p1=0.005,
            p2=0.005,
            p_meas=0.005,
            p_prep=0.005,
        )
        return sampler, tc

    def test_num_shots(self, d3_setup):
        sampler, _ = d3_setup
        batch = sampler.generate_samples(100, seed=42)
        assert batch.num_shots == 100

    def test_get_syndrome_shape(self, d3_setup):
        sampler, _ = d3_setup
        batch = sampler.generate_samples(10, seed=42)
        syn = batch.get_syndrome(0)
        assert len(syn) == sampler.num_detectors

    def test_get_observable_mask_type(self, d3_setup):
        sampler, _ = d3_setup
        batch = sampler.generate_samples(10, seed=42)
        mask = batch.get_observable_mask(0)
        assert isinstance(mask, int)

    def test_decode_count(self, d3_setup):
        import stim
        from pecos.qec.surface.circuit_builder import tick_circuit_to_stim

        sampler, tc = d3_setup
        noise = {"p1": 0.005, "p2": 0.005, "p_meas": 0.005, "p_prep": 0.005}
        stim_str = tick_circuit_to_stim(tc, **noise)
        dem_str = str(
            stim.Circuit(stim_str).detector_error_model(decompose_errors=True),
        )

        batch = sampler.generate_samples(1000, seed=42)
        errors = batch.decode_count(dem_str, "pymatching")
        assert isinstance(errors, int)
        assert 0 <= errors <= 1000
