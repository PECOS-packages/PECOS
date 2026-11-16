# Copyright 2026 The PECOS Developers

"""Experimental specifications implement the internal decoder-provider protocol."""

import pecos_rslib_exp as exp
import pytest

DEM = "error(0.4) D0\nerror(0.4) D1\nerror(0.1) D0 D1 D2 L70"


@pytest.mark.parametrize("spec", [exp.frontier(), exp.bp_trellis()], ids=["frontier", "bp_trellis"])
def test_prepare_is_only_for_recovery_specs(spec):
    assert spec._pecos_decoder_api_version == 1
    with pytest.raises(TypeError, match="requires a recovery-configured frontier"):
        spec._pecos_prepare_decoder(DEM)


def test_direct_v1_build_of_recovery_spec_matches_the_v2_preparation():
    spec = exp.frontier(
        k=1,
        delta=0,
        score_alpha=0,
        column_order="time_order",
        delta_max=3,
        on_no_path="fallback",
        fallback="feasible_then_zero",
    )
    assert spec._pecos_decoder_api_version == 2
    prepared = spec._pecos_prepare_decoder(DEM)._pecos_build_decoder()
    direct = spec._pecos_build_decoder(DEM)
    for syndrome in [bytes([0, 0, 0]), bytes([0, 0, 1])]:
        assert direct._pecos_decode_outcome(syndrome) == prepared._pecos_decode_outcome(syndrome)
