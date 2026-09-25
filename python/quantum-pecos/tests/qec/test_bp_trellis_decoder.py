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

    assert not dense.no_path
    assert dense.bp_runs == 1
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


@pytest.mark.parametrize("observable", [0, 70])
def test_no_path_reports_across_direct_methods(observable: int) -> None:
    dem = (
        f"error(1) L{observable}\n"
        "error(0.2) D3 D4\n"
        "error(0.4) D0\nerror(0.4) D1\nerror(0.1) D0 D1 D2\n"
        "detector D5\n"
    )
    decoder = BpTrellisDecoder.from_dem(
        dem,
        k=1,
        delta=100.0,
        score_alpha=0.0,
        bp_score_iterations=0,
        merge_indistinguishable=False,
        ordering="time_order",
        escalation=[(1, 100.0)],
    )
    shots = [[0] * 6, [0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0]]
    sequential = decoder.decode_batch(shots, on_no_path="report")
    parallel = decoder.decode_batch(shots, workers=4, on_no_path="report")
    for index, (shot, expected, actual) in enumerate(zip(shots, sequential, parallel, strict=True)):
        dense = decoder.decode_syndrome(shot, on_no_path="report")
        sparse = decoder.decode_from_defects([i for i, bit in enumerate(shot) if bit], on_no_path="report")
        for result in (actual, dense, sparse):
            assert type(result) is type(expected)
            assert result.no_path == (index != 0)
            assert result.observable_flips.mask == 1 << observable
            assert len(result.observable_flips) == observable + 1
            assert result.bp_runs == 0
            assert result.bp_seconds == 0.0
            assert result.transitions == expected.transitions
            if index:
                assert isinstance(result, pecos_rslib_exp.BpTrellisNoPath)
                assert result.cause == ["residual", "infeasible", "exhausted"][index - 1]
                assert result.detector == (5 if index == 1 else None)
                assert result.rungs_tried == (1 if index == 3 else 0)
                assert result.cause in repr(result)
                if index == 1:
                    assert "detector=5" in repr(result)
            else:
                assert isinstance(result, pecos_rslib_exp.BpTrellisResult)
    for index, message in [
        (1, "detector 5 has a residual"),
        (2, "under the detector error model"),
        (3, "after 1 escalation rung"),
    ]:
        with pytest.raises(RuntimeError, match=message):
            decoder.decode_syndrome(shots[index])
        with pytest.raises(RuntimeError, match=message):
            decoder.decode_from_defects([i for i, bit in enumerate(shots[index]) if bit], on_no_path="raise")
    for workers in (1, 4):
        with pytest.raises(RuntimeError, match=r"shot 1:.*detector 5 has a residual"):
            decoder.decode_batch(shots, workers=workers, on_no_path="raise")
        for policy in ("raise", "report"):
            with pytest.raises(RuntimeError, match="shot 1:"):
                decoder.decode_batch([shots[0], []], workers=workers, on_no_path=policy)
            with pytest.raises(RuntimeError):
                decoder.decode_syndrome([], on_no_path=policy)
        with pytest.raises(RuntimeError, match="shot 2:"):
            decoder.decode_batch([shots[0], shots[1], []], workers=workers, on_no_path="report")
        with pytest.raises(ValueError, match="on_no_path"):
            decoder.decode_batch([], workers=workers, on_no_path="unknown")
    with pytest.raises(ValueError, match="on_no_path"):
        decoder.decode_syndrome([], on_no_path="unknown")
    with pytest.raises(ValueError, match="on_no_path"):
        decoder.decode_from_defects([], on_no_path="unknown")


@pytest.mark.parametrize("bp_iterations", [0, 5])
def test_no_path_exhausted_absolute_telemetry(bp_iterations: int) -> None:
    dem = "error(0.4) D0\nerror(0.4) D1\nerror(0.1) D0 D1 D2 L0\n"
    decoder = BpTrellisDecoder.from_dem(
        dem,
        k=1,
        delta=100.0,
        score_alpha=0.0,
        bp_score_iterations=bp_iterations,
        merge_indistinguishable=False,
        ordering="time_order",
        escalation=[(1, 100.0), (2, 100.0)],
    )
    reports = decoder.decode_batch([[0, 0, 1]] * 2, workers=1, on_no_path="report")
    assert len(reports) == 2
    for report in reports:
        assert isinstance(report, pecos_rslib_exp.BpTrellisNoPath)
        assert report.cause == "exhausted"
        assert report.rungs_tried == 2
        # Widths [1, 1, 2] evaluate [6, 6, 10] branches on this BP-off
        # fixture. With score_alpha=0, BP leaves those retained paths unchanged.
        assert report.transitions == 22
        assert report.bp_runs == int(bp_iterations > 0)
        if bp_iterations:
            assert report.bp_seconds > 0.0
        else:
            assert report.bp_seconds == 0.0
