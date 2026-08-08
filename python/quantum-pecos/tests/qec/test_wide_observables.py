# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Wide (>64) observable support for ``LogicalSubgraphDecoder``.

Observable flips are packed into a mask; historically a ``u64`` capped decoders
at 64 observables. These tests exercise the wide ``ObsMask`` path: construction,
per-shot ``decode``/``decode_batch`` returning arbitrary-precision Python ints,
and ``decode_count`` comparing wide masks end-to-end.
"""

from __future__ import annotations

import pytest
from pecos_rslib.qec import LogicalSubgraphDecoder, ParsedDem, SampleBatch


def _wide_dem(n: int) -> tuple[str, list[list[int]]]:
    """A DEM with ``n`` observables, each on its own detector ``D_k``/``L_k``."""
    dem = "".join(f"detector({k},0,0) D{k}\n" for k in range(n)) + "".join(f"error(0.1) D{k} L{k}\n" for k in range(n))
    membership = [[k] for k in range(n)]
    return dem, membership


def test_decode_returns_big_int_above_64_observables() -> None:
    n = 65
    dem, membership = _wide_dem(n)
    dec = LogicalSubgraphDecoder.from_membership(dem, membership, "pecos_uf:fast")
    assert dec.num_observables() == n

    # Flipping detector 64 sets observable bit 64 -- not representable in a u64.
    syn = [0] * n
    syn[64] = 1
    result = dec.decode(syn)
    assert isinstance(result, int)
    assert (result >> 64) & 1 == 1
    assert result == 1 << 64

    # Two flips, one beyond bit 63.
    syn2 = [0] * n
    syn2[0] = 1
    syn2[64] = 1
    assert dec.decode(syn2) == (1 << 64) | 1


def test_decode_batch_above_64_observables() -> None:
    n = 65
    dem, membership = _wide_dem(n)
    dec = LogicalSubgraphDecoder.from_membership(dem, membership, "pecos_uf:fast")

    syndromes = [[0] * n, [0] * n]
    syndromes[0][64] = 1
    syndromes[1][0] = 1
    results = dec.decode_batch(syndromes)
    assert results[0] == 1 << 64
    assert results[1] == 1


def test_sample_batch_big_int_truth_masks() -> None:
    # The SampleBatch constructor accepts arbitrary-precision Python ints as
    # observable masks (bit 64 = observable 64), and decode_count compares the
    # wide truth mask against the wide prediction with no truncation.
    n = 65
    dem, membership = _wide_dem(n)
    dec = LogicalSubgraphDecoder.from_membership(dem, membership, "pecos_uf:fast")

    # Two shots, each flipping detector 64 (so the decoder predicts observable 64).
    syn = [0] * n
    syn[64] = 1
    detection_events = [syn, syn]

    # Truth that MATCHES the prediction (obs 64) -> zero logical errors.
    matching = SampleBatch(detection_events, [1 << 64, 1 << 64])
    assert dec.decode_count(matching) == 0

    # Truth that MISMATCHES (obs 0, not 64) -> both shots are errors.
    mismatching = SampleBatch(detection_events, [1, 1])
    assert dec.decode_count(mismatching) == 2


def test_decode_count_above_64_observables() -> None:
    # The wide compare path (predicted vs truth ObsMask) must run end-to-end on a
    # >64-observable batch without erroring or truncating.
    n = 65
    dem, membership = _wide_dem(n)
    dec = LogicalSubgraphDecoder.from_membership(dem, membership, "pecos_uf:fast")
    batch = ParsedDem.from_string(dem).to_dem_sampler().sample_batch(2000, seed=1)
    count = dec.decode_count(batch)
    assert 0 <= count <= 2000


def test_observable_flips_matches_wide_per_shot_masks() -> None:
    n = 70
    dem, _ = _wide_dem(n)
    batch = ParsedDem.from_string(dem).to_dem_sampler().sample_batch(73, seed=17)

    observable_flips = batch.observable_flips()
    assert all(len(row) == n for row in observable_flips)
    for shot, row in enumerate(observable_flips):
        mask = batch.get_observable_mask_wide(shot)
        assert row == [bool(mask & (1 << observable)) for observable in range(n)]


def test_u64_observable_getter_rejects_wide_batch() -> None:
    # get_observable_mask returns a u64 and cannot represent observable >= 64, so
    # it rejects a wide batch; get_observable_mask_wide returns the full Python
    # int. (The decode methods, by contrast, compare wide ObsMasks and do not
    # reject -- see below.)
    n = 65
    _dem, _ = _wide_dem(n)
    syn = [0] * n
    wide = SampleBatch([syn, syn], [1 << 64, 1 << 64])

    with pytest.raises(ValueError, match="64-observable"):
        wide.get_observable_mask(0)
    assert wide.get_observable_mask_wide(0) == 1 << 64


def test_sample_batch_decode_count_batch_handles_wide_dem() -> None:
    # decode_count_batch builds wide ObsMask predictions from PyMatching's batch
    # output, so a >64-observable DEM is decoded and compared with no truncation
    # or panic (no `1 << j` overflow).
    n = 70
    dem, _ = _wide_dem(n)
    syn = [0] * n
    syn[69] = 1  # detector 69 fires => boundary error flips observable 69
    batch = SampleBatch([syn, syn], [1 << 69, 1 << 69])  # truth: observable 69 set
    assert batch.decode_count_batch(dem) == 0


def test_generic_decode_handles_wide_dem() -> None:
    # With wide inner decoders (PyMatching now packs an ObsMask directly), the
    # generic decode_each / decode_count handle a >64-observable DEM end-to-end:
    # predictions are wide Python ints with no truncation or panic.
    n = 70
    dem, _ = _wide_dem(n)
    syn = [0] * n
    syn[69] = 1  # detector 69 => boundary error flips observable 69
    batch = SampleBatch([syn, syn], [1 << 69, 1 << 69])  # truth: observable 69 set

    preds = batch.decode_each(dem, "pymatching")
    assert preds == [1 << 69, 1 << 69]  # observable 69 predicted, not truncated
    assert batch.decode_count(dem, "pymatching") == 0  # predictions match truth
    assert batch.decode_count_parallel(dem, "pymatching") == 0


def test_decode_each_returns_python_ints() -> None:
    # decode_each returns Python ints (arbitrary precision) rather than u64, so
    # the value is not truncated; for the <=64 case it equals the historical u64.
    n = 5
    dem, _ = _wide_dem(n)
    batch = ParsedDem.from_string(dem).to_dem_sampler().sample_batch(8, seed=1)
    preds = batch.decode_each(dem, "pymatching")
    assert len(preds) == 8
    assert all(isinstance(p, int) for p in preds)


@pytest.mark.parametrize("decoder_type", ["fusion_blossom_serial", "pecos_uf", "k_mwpm"])
def test_matching_decoders_fail_loud_on_wide_dem(decoder_type: str) -> None:
    # The graphlike matching decoders pack observables into a u64 (1 << index), so
    # they support at most 64 observables. On a >64-observable DEM they must raise a
    # clean RuntimeError directing to a wide decoder -- NOT overflow-panic. (A panic
    # surfaces as a pyo3 PanicException, a BaseException that pytest.raises(RuntimeError)
    # would not catch, so this test fails if the guard regresses to a panic.)
    n = 70
    dem, _ = _wide_dem(n)
    syn = [0] * n
    syn[69] = 1
    batch = SampleBatch([syn, syn], [1 << 69, 1 << 69])
    with pytest.raises(RuntimeError, match="64"):
        batch.decode_count(dem, decoder_type)


def test_fusion_blossom_direct_constructors_reject_wide() -> None:
    # The FusionBlossom Python constructors (from_check_matrix, manual builder) sit
    # outside the decoder_type sweep, so they need their own coverage: an observable
    # index >= 64 must fail loud rather than overflow-panic when the u64 mask is packed.
    import numpy as np
    from pecos_rslib.decoders import FusionBlossomDecoder

    h = np.zeros((1, 65), dtype=np.uint8)
    h[0, 64] = 1  # observable (column) 64 -> would overflow `1 << 64`
    with pytest.raises(RuntimeError, match="64"):
        FusionBlossomDecoder.from_check_matrix(h)

    d = FusionBlossomDecoder(num_nodes=2, num_observables=65)
    with pytest.raises(RuntimeError, match="64"):
        d.add_edge(0, 1, [64], 1.0)
    with pytest.raises(RuntimeError, match="64"):
        d.add_boundary_edge(0, [64], 1.0)

    # The 64-observable boundary (indices 0..63) must still construct fine.
    ok = FusionBlossomDecoder(num_nodes=2, num_observables=64)
    ok.add_edge(0, 1, [63], 1.0)
