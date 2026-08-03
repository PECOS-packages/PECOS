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

"""Tests for the experimental native Frontier decoder bindings."""

from __future__ import annotations

import pytest

pecos_rslib_exp = pytest.importorskip("pecos_rslib_exp")

from pecos_rslib_exp import (  # noqa: E402
    FrontierCommitteeDecoder,
    FrontierDecoder,
)


SMALL_DEM = """\
error(0.1) D0 L0
error(0.2) D1
"""


def test_sparse_and_dense_decode_agree() -> None:
    decoder = FrontierDecoder.from_dem(SMALL_DEM)

    # Syndrome D0=1, D1=0 forces the first mechanism on and the second off,
    # so logical observable L0 flips and the expected mask is 1.
    dense = decoder.decode_syndrome([1, 0])
    sparse = decoder.decode([0])

    assert dense.observables_mask == 1
    assert sparse.observables_mask == dense.observables_mask
    assert sparse.log_evidence == dense.log_evidence
    assert sparse.logical_masses == dense.logical_masses
    assert dense.observable_bits(2) == [1, 0]


def test_wide_observable_mask_is_a_python_int() -> None:
    decoder = FrontierDecoder.from_dem("error(0.9) D0 L64\n")

    result = decoder.decode_syndrome([1])

    assert isinstance(result.observables_mask, int)
    assert result.observables_mask == 1 << 64
    assert result.observable_bits(65)[64] == 1
    assert result.logical_masses[0][0] == 1 << 64


def test_column_order_variants_and_validation() -> None:
    expected = 1
    for column_order in (
        "deadline_reorder",
        "time_order",
        "backward_deadline_reorder",
        [1, 0],
    ):
        decoder = FrontierDecoder.from_dem(SMALL_DEM, column_order=column_order)
        assert decoder.decode_syndrome([1, 0]).observables_mask == expected

    with pytest.raises(ValueError, match="invalid column_order"):
        FrontierDecoder.from_dem(SMALL_DEM, column_order="not_an_order")
    with pytest.raises(ValueError, match="column_order must be"):
        FrontierDecoder.from_dem(SMALL_DEM, column_order=object())
    with pytest.raises(RuntimeError, match="permutation"):
        FrontierDecoder.from_dem(SMALL_DEM, column_order=[0, 0])


def test_committee_easy_tie_selects_forward() -> None:
    committee = FrontierCommitteeDecoder.from_dem("error(0.1) D0 L0\n", column_order="time_order")

    result = committee.decode([0])

    assert result.observables_mask == 1
    assert result.direction == "forward"
    assert result.forward_log_evidence == result.log_evidence
    assert result.backward_log_evidence == result.log_evidence


def test_unexplainable_and_out_of_range_syndromes_raise_runtime_error() -> None:
    decoder = FrontierDecoder.from_dem("error(0.1) D0 D1\n")

    with pytest.raises(RuntimeError, match="unexplainable"):
        decoder.decode_syndrome([1, 0])
    with pytest.raises(RuntimeError, match="Invalid node index"):
        decoder.decode([2])


def test_decode_batch_matches_individual_decodes() -> None:
    shots = [[0, 0], [1, 0], [0, 1], [1, 1]]
    batch_decoder = FrontierDecoder.from_dem(SMALL_DEM)
    individual_decoder = FrontierDecoder.from_dem(SMALL_DEM)

    batch = batch_decoder.decode_batch(shots)
    individual = [individual_decoder.decode_syndrome(shot) for shot in shots]

    assert [result.observables_mask for result in batch] == [result.observables_mask for result in individual]
    assert [result.log_evidence for result in batch] == [result.log_evidence for result in individual]
    assert [result.logical_masses for result in batch] == [result.logical_masses for result in individual]
