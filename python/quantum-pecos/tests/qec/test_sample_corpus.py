# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Python coverage for self-describing SampleBatch shot corpora."""

from __future__ import annotations

import pytest

pytest.importorskip("pecos_rslib")

from pecos_rslib.qec import DemSampler, SampleBatch


def test_generated_batch_round_trip_preserves_shots_and_provenance(tmp_path) -> None:
    dem = "error(0.125) D0 L0\n"
    metadata = '{ "decoder": "pymatching", "decoder_seed": 17 }'
    batch = DemSampler.from_dem_string(dem).generate_samples(130, seed=42)
    path = tmp_path / "round-trip.pecos"

    batch.save(path, dem=dem, metadata_json=metadata)
    loaded = SampleBatch.load(path)

    assert loaded.num_shots == batch.num_shots
    assert loaded.seed == batch.seed == 42
    assert loaded.dem == dem
    assert loaded.metadata_json == metadata
    assert loaded.format_version == 1
    assert [loaded.get_syndrome(i) for i in range(loaded.num_shots)] == [
        batch.get_syndrome(i) for i in range(batch.num_shots)
    ]
    assert [loaded.get_observable_mask_wide(i) for i in range(loaded.num_shots)] == [
        batch.get_observable_mask_wide(i) for i in range(batch.num_shots)
    ]


def test_wide_observable_above_bit_63_round_trips(tmp_path) -> None:
    dem = "error(0.125) D0 L64\n"
    batch = SampleBatch([[1], [0]], [1 << 64, 0])
    path = tmp_path / "wide.pecos"

    batch.save(path, dem=dem)
    loaded = SampleBatch.load(path)

    assert loaded.get_observable_mask_wide(0) == 1 << 64
    assert loaded.get_observable_mask_wide(1) == 0


def test_save_rejects_mismatched_dem_dimensions(tmp_path) -> None:
    batch = SampleBatch([[0]], [0])

    with pytest.raises(ValueError, match="DEM dimensions do not match SampleBatch"):
        batch.save(tmp_path / "wrong-dem.pecos", dem="error(0.1) D1\n")


def test_save_rejects_invalid_metadata_json(tmp_path) -> None:
    batch = SampleBatch([[0]], [0])

    with pytest.raises(ValueError, match="metadata_json is not valid JSON"):
        batch.save(
            tmp_path / "bad-metadata.pecos",
            dem="error(0.1) D0\n",
            metadata_json="{",
        )


def test_load_maps_malformed_files_to_value_error(tmp_path) -> None:
    path = tmp_path / "bad-magic.pecos"
    path.write_bytes(b"not a PECOS corpus")

    with pytest.raises(ValueError, match="bad shot-corpus magic"):
        SampleBatch.load(path)


def test_load_maps_filesystem_failures_to_io_error(tmp_path) -> None:
    with pytest.raises(OSError, match="No such file or directory"):
        SampleBatch.load(tmp_path / "missing.pecos")


def test_generate_samples_records_resolved_and_explicit_seeds() -> None:
    sampler = DemSampler.from_dem_string("error(0.125) D0 L0\n")

    resolved = sampler.generate_samples(130)
    explicit = sampler.generate_samples(1, seed=0xDEADBEEF)
    replayed = sampler.generate_samples(resolved.num_shots, seed=resolved.seed)

    assert isinstance(resolved.seed, int)
    assert 0 <= resolved.seed <= (1 << 64) - 1
    assert explicit.seed == 0xDEADBEEF
    assert [replayed.get_syndrome(i) for i in range(replayed.num_shots)] == [
        resolved.get_syndrome(i) for i in range(resolved.num_shots)
    ]
    assert [replayed.get_observable_mask_wide(i) for i in range(replayed.num_shots)] == [
        resolved.get_observable_mask_wide(i) for i in range(resolved.num_shots)
    ]


def test_compare_decoders_counts_survive_corpus_round_trip(tmp_path) -> None:
    dem = "error(0.1) D0 L0\nerror(0.1) D0\n"
    batch = DemSampler.from_dem_string(dem).generate_samples(257, seed=314159)
    before = batch.compare_decoders(dem, "pymatching", "pymatching")
    path = tmp_path / "comparison.pecos"

    batch.save(path, dem=dem)
    loaded = SampleBatch.load(path)
    after = loaded.compare_decoders(
        loaded.dem,
        "pymatching",
        "pymatching",
    )

    assert after.counts == before.counts
