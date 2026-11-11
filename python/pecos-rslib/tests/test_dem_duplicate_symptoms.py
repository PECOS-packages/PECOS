# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Red/green and compatibility contracts for DEM duplicate-symptom merging."""

import pytest
from pecos_rslib.decoders import (
    BpLsdDecoder,
    BpOsdDecoder,
    DemAwareDecoder,
    MinSumBpBuilder,
    MinSumBpDecoder,
    RelayBpBuilder,
    RelayBpDecoder,
)


CONVENIENCE = {
    "bp_lsd": BpLsdDecoder,
    "bp_osd": BpOsdDecoder,
    "min_sum_bp": MinSumBpDecoder,
    "relay_bp": RelayBpDecoder,
}


@pytest.mark.parametrize("family", sorted(CONVENIENCE))
@pytest.mark.parametrize("convenience", [False, True])
def test_duplicate_dem_observables_preserve_original_mechanism_count(family, convenience):
    """Red/green: merged decoding preserves observables and the original mechanism count."""
    dem = "\n".join(
        [
            "error(0.1) D0 D1 L0",
            "error(0.2) D2 L65",
            "error(0.1) D0 D1 L0",
            "error(0.3) D2 L65",
        ]
    )
    if convenience:
        decoder = CONVENIENCE[family].from_dem(dem)
    else:
        decoder = DemAwareDecoder.from_dem(dem, family)
    assert decoder.num_mechanisms == 4
    assert "mechanisms=4" in repr(decoder)
    for syndrome, expected in [
        ([1, 1, 0], [1, 0]),
        ([0, 0, 1], [0, 1]),
        ([1, 1, 1], [1, 1]),
        ([0, 0, 0], [0, 0]),
    ]:
        result = decoder.decode_syndrome(syndrome)
        assert result.converged
        assert list(result.observable_flips) == [expected[0]] + [0] * 64 + [expected[1]]


@pytest.mark.parametrize("family", ["min_sum_bp", "relay_bp"])
@pytest.mark.parametrize("error_rate", [None, 0.15])
def test_unique_dem_matches_original_raw_matrix_path(family, error_rate):
    """Compatibility: compare observable predictions and convergence to the pre-merge path."""
    dem = "error(0.1) D0 L0\nerror(0.2) D0 D1 L1\nerror(0.3) D1 L0 L1"
    priors = [0.1, 0.2, 0.3] if error_rate is None else [error_rate] * 3
    if family == "min_sum_bp":
        original = MinSumBpBuilder([[1, 1, 0], [0, 1, 1]], priors).max_iter(100).build()
    else:
        original = RelayBpBuilder([[1, 1, 0], [0, 1, 1]], priors).build()
    decoder = DemAwareDecoder.from_dem(dem, family, error_rate=error_rate)
    assert decoder.num_mechanisms == 3
    for syndrome in [[0, 0], [1, 0], [0, 1], [1, 1], [1, 0]]:
        before = original.decode(syndrome)
        after = decoder.decode_syndrome(syndrome)
        correction = list(before.decoding)
        assert after.converged == before.converged
        assert after.iterations == before.iterations
        assert list(after.observable_flips) == [
            correction[0] ^ correction[2],
            correction[1] ^ correction[2],
        ]


@pytest.mark.parametrize("family", sorted(CONVENIENCE))
@pytest.mark.parametrize("error_rate", [float("nan"), -0.1, 1.5])
@pytest.mark.parametrize("convenience", [False, True])
def test_merged_families_reject_invalid_error_rate(family, error_rate, convenience):
    """Red/green: an override that is not a probability is rejected before merging."""
    dem = "error(0.1) D0 L0\nerror(0.1) D0 L0"
    with pytest.raises(ValueError, match="error_rate must be finite and in"):
        if convenience:
            CONVENIENCE[family].from_dem(dem, error_rate=error_rate)
        else:
            DemAwareDecoder.from_dem(dem, family, error_rate=error_rate)
