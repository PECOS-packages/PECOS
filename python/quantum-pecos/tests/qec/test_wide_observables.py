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
    batch = ParsedDem.from_string(dem).to_dem_sampler().generate_samples(2000, seed=1)
    count = dec.decode_count(batch)
    assert 0 <= count <= 2000


def test_narrow_sample_batch_apis_reject_wide_observables() -> None:
    # The legacy u64-based SampleBatch APIs cannot represent >64 observables and
    # must reject a wide batch up front rather than panicking or truncating.
    n = 65
    dem, _ = _wide_dem(n)
    syn = [0] * n
    syn[64] = 1
    wide = SampleBatch([syn, syn], [1 << 64, 1 << 64])

    with pytest.raises(ValueError, match="64-observable"):
        wide.get_observable_mask(0)
    with pytest.raises(ValueError, match="64-observable"):
        wide.decode_count(dem, "pecos_uf:fast")
    with pytest.raises(ValueError, match="64-observable"):
        wide.decode_stats(dem, "pecos_uf:fast")

    # The wide getter returns the full mask as a Python int with no truncation.
    assert wide.get_observable_mask_wide(0) == 1 << 64
