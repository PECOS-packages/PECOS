# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Tests for the experimental native BP-trellis decoder bindings."""

from __future__ import annotations

import pytest

pecos_rslib_exp = pytest.importorskip("pecos_rslib_exp")

from pecos_rslib_exp import BpTrellisDecoder  # noqa: E402

SMALL_DEM = """\
error(0.1) D0 L0
error(0.2) D1
"""


def test_bptrellis_defaults_and_decode_shapes() -> None:
    decoder = BpTrellisDecoder.from_dem(SMALL_DEM)

    dense = decoder.decode_syndrome([1, 0])
    sparse = decoder.decode_from_defects([0])
    batch = decoder.decode_batch([[1, 0], [0, 1]])

    assert dense.observable_flips.mask == 1
    assert list(dense.observable_flips) == [True]
    assert sparse.observable_flips.mask == dense.observable_flips.mask
    assert batch[0].observable_flips.mask == dense.observable_flips.mask
    assert batch[1].observable_flips.mask == 0
    assert decoder.build_seconds >= 0.0
    assert not hasattr(decoder, "decode")


def test_bptrellis_ordering_variants_and_validation() -> None:
    for ordering in (
        "deadline",
        "backward_deadline",
        "time_order",
        [1, 0],
    ):
        decoder = BpTrellisDecoder.from_dem(SMALL_DEM, ordering=ordering)
        assert decoder.decode_syndrome([1, 0]).observable_flips.mask == 1

    with pytest.raises(ValueError, match="invalid ordering"):
        BpTrellisDecoder.from_dem(SMALL_DEM, ordering="not_an_order")


def test_bptrellis_escalation_ladder_kwarg_and_result_getter() -> None:
    dem = """\
error(0.4) D0
error(0.4) D1
error(0.1) D0 D1 D2 L0
"""
    bare_k16 = BpTrellisDecoder.from_dem(
        dem,
        k=16,
        bp_score_iterations=0,
        merge_indistinguishable=False,
        ordering="time_order",
        escalation_ks=None,
    ).decode_syndrome([0, 0, 1])
    escalated = BpTrellisDecoder.from_dem(
        dem,
        k=2,
        bp_score_iterations=0,
        merge_indistinguishable=False,
        ordering="time_order",
        escalation_ks=[16],
    ).decode_syndrome([0, 0, 1])

    assert bare_k16.escalation_rungs_used == 0
    assert escalated.observable_flips.mask == bare_k16.observable_flips.mask == 1
    assert escalated.escalation_rungs_used == 1
    assert escalated.transitions > bare_k16.transitions
