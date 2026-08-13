# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Python coverage for self-describing SampleBatch shot corpora."""

from __future__ import annotations

import errno

import pytest

pytest.importorskip("pecos_rslib")

from pecos_rslib.qec import DemSampler, SampleBatch


def test_generated_batch_round_trip_preserves_shots_and_provenance(tmp_path) -> None:
    dem = "error(0.125) D0 L0\n"
    metadata = '{ "decoder": "pymatching", "decoder_seed": 17 }'
    batch = DemSampler.from_dem_string(dem).sample_batch(130, seed=42)
    path = tmp_path / "round-trip.pecos"

    batch.save(path, dem=dem, metadata_json=metadata)
    loaded = SampleBatch.load(path)

    assert loaded.num_shots == batch.num_shots
    assert loaded.seed == batch.seed == 42
    assert loaded.dem == dem
    assert loaded.metadata_json == metadata
    assert loaded.generator.startswith("pecos-rslib ")
    assert loaded.format_version == 1
    assert [loaded.get_syndrome(i) for i in range(loaded.num_shots)] == [
        batch.get_syndrome(i) for i in range(batch.num_shots)
    ]
    assert [loaded.get_observable_flips(i).mask for i in range(loaded.num_shots)] == [
        batch.get_observable_flips(i).mask for i in range(batch.num_shots)
    ]


def test_wide_observable_above_bit_63_round_trips(tmp_path) -> None:
    dem = "error(0.125) D0 L64\n"
    batch = SampleBatch([[1], [0]], [1 << 64, 0])
    path = tmp_path / "wide.pecos"

    batch.save(path, dem=dem)
    loaded = SampleBatch.load(path)

    assert loaded.get_observable_flips(0).mask == 1 << 64
    assert loaded.get_observable_flips(1).mask == 0


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


def test_load_rejects_a_tampered_header_or_payload(tmp_path) -> None:
    """Integrity is checked over the whole file before anything is parsed.

    A header edit that stays internally consistent -- bumping the declared
    num_shots -- was accepted by an earlier design that validated fields
    individually. The content digest is what makes that unreachable, so both
    a header edit and a payload bit-flip must be rejected.
    """
    dem = "error(0.125) D0 L0\n"
    path = tmp_path / "tampered.pecos"
    DemSampler.from_dem_string(dem).sample_batch(65, seed=7).save(path, dem=dem)

    pristine = path.read_bytes()
    header_edit = bytearray(pristine)
    field = header_edit.find(b'"num_shots"')
    assert field > 0, "corpus header must declare num_shots"
    count = header_edit.find(b"65", field)
    assert count > 0, "corpus header must declare the shot count"
    header_edit[count : count + 2] = b"66"
    path.write_bytes(bytes(header_edit))
    with pytest.raises(ValueError, match="content SHA-256 mismatch"):
        SampleBatch.load(path)

    payload_edit = bytearray(pristine)
    payload_edit[-3] ^= 0x01
    path.write_bytes(bytes(payload_edit))
    with pytest.raises(ValueError, match="content SHA-256 mismatch"):
        SampleBatch.load(path)


def test_load_maps_filesystem_failures_to_io_error(tmp_path) -> None:
    path = tmp_path / "missing.pecos"

    with pytest.raises(FileNotFoundError, match="No such file or directory") as exc_info:
        SampleBatch.load(path)

    assert exc_info.value.errno == errno.ENOENT
    assert exc_info.value.filename == str(path)


def test_resave_preserves_metadata_unless_explicitly_cleared(tmp_path) -> None:
    dem = "error(0.125) D0 L0\n"
    metadata = '{"source": "original"}'
    original_path = tmp_path / "original.pecos"
    preserved_path = tmp_path / "preserved.pecos"
    cleared_path = tmp_path / "cleared.pecos"
    batch = SampleBatch([[1]], [1])
    batch.save(original_path, dem=dem, metadata_json=metadata)
    loaded = SampleBatch.load(original_path)

    loaded.save(preserved_path, dem=dem)
    loaded.save(cleared_path, dem=dem, clear_metadata=True)

    assert SampleBatch.load(preserved_path).metadata_json == metadata
    assert SampleBatch.load(cleared_path).metadata_json is None


def test_loaded_batch_requires_its_embedded_dem_unless_opted_out(tmp_path) -> None:
    embedded_dem = "error(0.125) D0 L0\n"
    different_dem = "error(0.25) D0 L0\n"
    path = tmp_path / "dem-bound.pecos"
    batch = SampleBatch([[1]], [1])
    batch.save(path, dem=embedded_dem)
    loaded = SampleBatch.load(path)

    with pytest.raises(ValueError, match="differs from the DEM embedded"):
        loaded.compare_decoders(
            different_dem,
            "pymatching",
            "pymatching",
        )

    result = loaded.compare_decoders(
        different_dem,
        "pymatching",
        "pymatching",
        allow_dem_mismatch=True,
    )
    assert result.total_shots == 1


def test_sample_batch_records_resolved_and_explicit_seeds() -> None:
    sampler = DemSampler.from_dem_string("error(0.125) D0 L0\n")

    resolved = sampler.sample_batch(130)
    explicit = sampler.sample_batch(1, seed=0xDEADBEEF)
    replayed = sampler.sample_batch(resolved.num_shots, seed=resolved.seed)

    assert isinstance(resolved.seed, int)
    assert 0 <= resolved.seed <= (1 << 64) - 1
    assert explicit.seed == 0xDEADBEEF
    assert [replayed.get_syndrome(i) for i in range(replayed.num_shots)] == [
        resolved.get_syndrome(i) for i in range(resolved.num_shots)
    ]
    assert [replayed.get_observable_flips(i).mask for i in range(replayed.num_shots)] == [
        resolved.get_observable_flips(i).mask for i in range(resolved.num_shots)
    ]


def test_compare_decoders_counts_survive_corpus_round_trip(tmp_path) -> None:
    dem = "error(0.1) D0 L0\nerror(0.1) D0\n"
    batch = DemSampler.from_dem_string(dem).sample_batch(257, seed=314159)
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
