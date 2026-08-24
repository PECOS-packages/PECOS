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

import math

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

N1_FACTORS = [
    [(0.60, [], []), (0.25, [0], [0]), (0.15, [0], [])],
    [(0.80, [], []), (0.20, [0], [])],
]


def test_sparse_and_dense_decode_agree() -> None:
    decoder = FrontierDecoder.from_dem(SMALL_DEM)

    # Syndrome D0=1, D1=0 forces the first mechanism on and the second off,
    # so logical observable L0 flips and the expected mask is 1.
    dense = decoder.decode_syndrome([1, 0])
    sparse = decoder.decode_from_defects([0])

    assert dense.observable_flips.mask == 1
    assert sparse.observable_flips.mask == dense.observable_flips.mask
    assert sparse.log_evidence == dense.log_evidence
    assert sparse.logical_masses == dense.logical_masses
    assert list(dense.observable_flips) == [True]
    assert dense.transitions == 4
    assert dense.dropped_states == 0
    assert dense.dropped_log_mass == float("-inf")
    assert dense.status == "exact"
    assert dense.bp_seconds == 0.0
    assert dense.escalation_rungs_used == 0
    assert decoder.build_seconds >= 0.0
    assert not hasattr(decoder, "decode")


def test_wide_observable_mask_is_a_python_int() -> None:
    decoder = FrontierDecoder.from_dem("error(0.9) D0 L64\n")

    result = decoder.decode_syndrome([1])

    assert isinstance(result.observable_flips.mask, int)
    assert result.observable_flips.mask == 1 << 64
    assert len(result.observable_flips) == 65
    assert result.observable_flips[64]
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
        assert decoder.decode_syndrome([1, 0]).observable_flips.mask == expected

    with pytest.raises(ValueError, match="invalid column_order"):
        FrontierDecoder.from_dem(SMALL_DEM, column_order="not_an_order")
    with pytest.raises(ValueError, match="column_order must be"):
        FrontierDecoder.from_dem(SMALL_DEM, column_order=object())
    with pytest.raises(RuntimeError, match="permutation"):
        FrontierDecoder.from_dem(SMALL_DEM, column_order=[0, 0])


def test_factor_model_decode_preserves_degeneracy_mass() -> None:
    decoder = FrontierDecoder.from_factors(
        N1_FACTORS,
        1,
        1,
        k=10**9,
        delta=1e6,
    )

    result = decoder.decode_syndrome([1])
    logical_masses = dict(result.logical_masses)

    assert result.observable_flips.mask == 0
    assert logical_masses[0] == pytest.approx(math.log(0.24), abs=1e-9)
    assert logical_masses[1] == pytest.approx(math.log(0.20), abs=1e-9)


def test_maxlog_metric_can_flip_the_prediction() -> None:
    dem = "error(0.20) D0 L0\nerror(0.15) D0\nerror(0.15) D0\n"
    default_result = FrontierDecoder.from_dem(dem).decode_syndrome([1])
    maxlog_result = FrontierDecoder.from_dem(
        dem,
        k=10**9,
        delta=1e6,
        metric_mode="frontier_lite",
    ).decode_syndrome([1])

    assert default_result.observable_flips.mask == 0
    assert maxlog_result.observable_flips.mask == 1
    assert maxlog_result.status == "exact"
    assert all(isinstance(log_mass, float) for _, log_mass in maxlog_result.logical_masses)


@pytest.mark.parametrize(
    "metric_mode",
    [
        "logsumexp_float",
        "float",
        "exact",
        "frontierLite",
        "frontier_lite",
        "frontier-lite",
        "frontierlite",
        "maxlog_int",
        "max_log_int",
        "viterbi_int",
    ],
)
def test_metric_mode_aliases_are_accepted_with_surrounding_whitespace(metric_mode: str) -> None:
    decoder = FrontierDecoder.from_dem(
        "error(0.1) D0\n",
        metric_mode=f"  {metric_mode}  ",
    )
    assert decoder.decode_syndrome([0]).status == "exact"


def test_metric_and_factor_model_error_paths() -> None:
    with pytest.raises(ValueError, match=r"logsumexp_float.*maxlog_int"):
        FrontierDecoder.from_dem(SMALL_DEM, metric_mode="not_a_metric")
    with pytest.raises(RuntimeError, match="integer max-log metric is not supported"):
        FrontierCommitteeDecoder.from_dem(SMALL_DEM, metric_mode="maxlog_int")
    with pytest.raises(RuntimeError, match="delta must be finite under maxlog_int"):
        FrontierDecoder.from_dem(
            SMALL_DEM,
            delta=float("inf"),
            metric_mode="maxlog_int",
        )
    with pytest.raises(RuntimeError, match="BP-guided pruning requires a binary model"):
        FrontierDecoder.from_factors(
            N1_FACTORS,
            1,
            1,
            bp_score_iterations=1,
        )
    with pytest.raises(ValueError, match="outcome probabilities must sum to 1"):
        FrontierDecoder.from_factors(
            [[(0.6, [], []), (0.3, [0], [])]],
            1,
            0,
        )


@pytest.mark.parametrize("column_order", ["deadline_reorder", "backward_deadline_reorder"])
def test_factor_model_named_column_orders_construct_and_decode(column_order: str) -> None:
    decoder = FrontierDecoder.from_factors(
        N1_FACTORS,
        1,
        1,
        column_order=column_order,
    )
    assert decoder.decode_syndrome([1]).status == "exact"


def test_binary_factor_model_delegates_identically_to_dem() -> None:
    factors = [
        [(0.9, [], []), (0.1, [0], [0])],
        [(0.8, [], []), (0.2, [1], [])],
    ]
    config = {
        "k": 10**9,
        "delta": 1e6,
        "score_alpha": 0.7,
        "bp_score_iterations": 2,
        "column_order": "time_order",
    }
    factor_decoder = FrontierDecoder.from_factors(factors, 2, 1, **config)
    dem_decoder = FrontierDecoder.from_dem(SMALL_DEM, **config)

    for syndrome in ([0, 0], [1, 0], [0, 1], [1, 1]):
        factor_result = factor_decoder.decode_syndrome(syndrome)
        dem_result = dem_decoder.decode_syndrome(syndrome)
        assert factor_result.observable_flips.mask == dem_result.observable_flips.mask
        assert factor_result.log_evidence == dem_result.log_evidence
        assert factor_result.runner_up_gap == dem_result.runner_up_gap
        assert factor_result.logical_masses == dem_result.logical_masses
        assert factor_result.status == dem_result.status


def test_committee_easy_tie_selects_forward() -> None:
    committee = FrontierCommitteeDecoder.from_dem("error(0.1) D0 L0\n", column_order="time_order")

    result = committee.decode_from_defects([0])

    assert result.observable_flips.mask == 1
    assert result.direction == "forward"
    assert result.forward_log_evidence == result.log_evidence
    assert result.backward_log_evidence == result.log_evidence
    assert result.status == "exact"
    assert result.transitions == 2
    assert committee.build_seconds >= 0.0
    assert not hasattr(committee, "decode")


def test_bp_score_iterations_and_telemetry_are_exposed() -> None:
    decoder = FrontierDecoder.from_dem(
        SMALL_DEM,
        k=1,
        delta=float("inf"),
        bp_score_iterations=3,
    )
    committee = FrontierCommitteeDecoder.from_dem(
        SMALL_DEM,
        k=1,
        delta=float("inf"),
        bp_score_iterations=3,
    )

    assert decoder.decode_syndrome([1, 0]).bp_seconds >= 0.0
    assert committee.decode_syndrome([1, 0]).bp_seconds >= 0.0


def test_pruning_telemetry_and_status_strings() -> None:
    width = FrontierDecoder.from_dem("error(0.25) L0\n", k=1, delta=float("inf"))
    width_result = width.decode_syndrome([])
    assert width_result.dropped_states == 1
    assert math.exp(width_result.dropped_log_mass) == pytest.approx(0.25)
    assert width_result.status == "pruned:k"

    delta = FrontierDecoder.from_dem("error(0.25) L0\n", k=2, delta=0.5)
    assert delta.decode_syndrome([]).status == "pruned:delta"

    both = FrontierDecoder.from_dem(
        "error(0.5) L0\nerror(0.1) L1\n",
        k=3,
        delta=0.5,
    )
    assert both.decode_syndrome([]).status == "pruned:k+delta"


def test_unexplainable_and_out_of_range_syndromes_raise_runtime_error() -> None:
    decoder = FrontierDecoder.from_dem("error(0.1) D0 D1\n")

    with pytest.raises(RuntimeError, match="unexplainable"):
        decoder.decode_syndrome([1, 0])
    with pytest.raises(RuntimeError, match="Invalid node index"):
        decoder.decode_from_defects([2])


def test_decode_batch_matches_individual_decodes() -> None:
    shots = [[0, 0], [1, 0], [0, 1], [1, 1]]
    batch_decoder = FrontierDecoder.from_dem(SMALL_DEM)
    individual_decoder = FrontierDecoder.from_dem(SMALL_DEM)

    batch = batch_decoder.decode_batch(shots)
    individual = [individual_decoder.decode_syndrome(shot) for shot in shots]

    assert [result.observable_flips.mask for result in batch] == [result.observable_flips.mask for result in individual]
    assert [result.log_evidence for result in batch] == [result.log_evidence for result in individual]
    assert [result.logical_masses for result in batch] == [result.logical_masses for result in individual]


DUPLICATE_DEM = """\
error(0.3) D0 L0
error(0.3) D0 L0
"""


def test_merge_indistinguishable_kwarg_defaults_off_and_merges_when_enabled() -> None:
    default_decoder = FrontierDecoder.from_dem(DUPLICATE_DEM, column_order="time_order")
    merged_decoder = FrontierDecoder.from_dem(
        DUPLICATE_DEM,
        column_order="time_order",
        merge_indistinguishable=True,
    )

    default_result = default_decoder.decode_from_defects([0])
    merged_result = merged_decoder.decode_from_defects([0])
    assert default_result.processed_columns == 2
    assert merged_result.processed_columns == 1
    # decode_from_defects([0]) passes sparse fired-detector indices: detector 0 fired. The only
    # explanation is the merged mechanism firing, whose XOR probability is
    # 0.3*0.7 + 0.7*0.3 = 0.42, so the total evidence is exactly 0.42.
    assert math.isclose(math.exp(merged_result.log_evidence), 0.42, abs_tol=1e-15)


def test_committee_merge_kwarg_defaults_off_and_merges_when_enabled() -> None:
    default_committee = FrontierCommitteeDecoder.from_dem(DUPLICATE_DEM, column_order="time_order")
    merged_committee = FrontierCommitteeDecoder.from_dem(
        DUPLICATE_DEM,
        column_order="time_order",
        merge_indistinguishable=True,
    )

    assert default_committee.decode_from_defects([0]).processed_columns == 2
    assert merged_committee.decode_from_defects([0]).processed_columns == 1
