# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Tests for SampleBatch columnar storage and validation."""

import pytest
from pecos.decoders import pymatching
from pecos_rslib.qec import DemSampler, ParsedDem, SampleBatch


class TestSampleBatchConstruction:
    def test_round_trip_get_syndrome(self):
        batch = SampleBatch([[1, 0], [0, 1]], [1, 0])
        assert list(batch.get_syndrome(0)) == [1, 0]
        assert list(batch.get_syndrome(1)) == [0, 1]

    def test_round_trip_get_observable_flips(self):
        batch = SampleBatch([[1, 0], [0, 1]], [1, 0])
        assert batch.get_observable_flips(0).mask == 1
        assert batch.get_observable_flips(1).mask == 0

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

    def test_observable_flips_preserves_explicit_width_for_empty_masks(self):
        batch = SampleBatch([[1, 0], [0, 1]], [0, 0], num_observables=3)
        assert batch.observable_flips() == [[False, False, False], [False, False, False]]

    def test_observable_flips_infers_width_from_highest_set_bit(self):
        batch = SampleBatch([[1, 0], [0, 1]], [0, 1 << 3])
        assert batch.observable_flips() == [
            [False, False, False, False],
            [False, False, False, True],
        ]

    def test_explicit_observable_width_rejects_out_of_range_mask_bit(self):
        with pytest.raises(ValueError, match=r"mask bit 3.*num_observables=3"):
            SampleBatch([[1, 0]], [1 << 3], num_observables=3)


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
        batch = sampler.sample_batch(100, seed=42)
        assert type(batch) is SampleBatch
        assert batch.num_shots == 100

    def test_bulk_accessors_match_per_shot_accessors(self, d3_setup):
        sampler, _ = d3_setup
        batch = sampler.sample_batch(73, seed=1234)

        detector_events = batch.detector_events()
        observable_flips = batch.observable_flips()
        assert len(detector_events) == batch.num_shots
        assert len(observable_flips) == batch.num_shots
        for shot in range(batch.num_shots):
            assert detector_events[shot] == [bool(value) for value in batch.get_syndrome(shot)]
            mask = batch.get_observable_flips(shot).mask
            assert observable_flips[shot] == [
                bool(mask & (1 << observable)) for observable in range(sampler.num_observables)
            ]

    def test_parsed_dem_sample_batch_returns_sample_batch(self):
        parsed = ParsedDem.from_string("error(0.25) D0 L0")
        assert type(parsed.sample_batch(5, seed=7)) is SampleBatch

    def test_parsed_dem_row_conversion_preserves_exact_column_positions(self):
        parsed = ParsedDem.from_string("error(1.0) D0 D2 L0")
        batch = parsed.sample_batch(4, seed=7)

        assert batch.detector_events() == [[True, False, True]] * 4
        assert batch.observable_flips() == [[True]] * 4

    def test_get_syndrome_shape(self, d3_setup):
        sampler, _ = d3_setup
        batch = sampler.sample_batch(10, seed=42)
        syn = batch.get_syndrome(0)
        assert len(syn) == sampler.num_detectors

    def test_get_observable_flips_mask_type(self, d3_setup):
        sampler, _ = d3_setup
        batch = sampler.sample_batch(10, seed=42)
        mask = batch.get_observable_flips(0).mask
        assert isinstance(mask, int)

    def test_decode(self, d3_setup):
        import stim
        from pecos.qec.surface.circuit_builder import tick_circuit_to_stim

        sampler, tc = d3_setup
        noise = {"p1": 0.005, "p2": 0.005, "p_meas": 0.005, "p_prep": 0.005}
        stim_str = tick_circuit_to_stim(tc, **noise)
        dem_str = str(
            stim.Circuit(stim_str).detector_error_model(decompose_errors=True),
        )

        batch = sampler.sample_batch(1000, seed=42)
        errors = batch.decode(dem_str, pymatching(correlated=True)).num_errors
        assert isinstance(errors, int)
        assert 0 <= errors <= 1000


def test_seeded_healthy_decoder_counts_and_stats_are_unchanged() -> None:
    dem = "error(0.1) D0 L0\nerror(0.2) D0\n"
    batch = DemSampler.from_dem_string(dem).sample_batch(257, seed=314159)

    sequential = batch.decode(
        dem,
        pymatching(correlated=True),
        workers=1,
        timing=True,
    )
    parallel = batch.decode(
        dem,
        pymatching(correlated=True),
        workers=3,
        timing=True,
    )
    native = batch.decode(dem, pymatching(correlated=False))

    assert sequential.execution_path == "sequential"
    assert parallel.execution_path == "parallel"
    assert native.execution_path == "native_batch"
    assert sequential.num_errors == parallel.num_errors == native.num_errors == 48
    assert sequential.num_shots == parallel.num_shots == native.num_shots == 257
    assert sequential.logical_error_rate == parallel.logical_error_rate == native.logical_error_rate == 48 / 257
    assert sequential.stats is not None
    assert parallel.stats is not None
    assert sequential.stats.num_timing_samples == parallel.stats.num_timing_samples == 257
