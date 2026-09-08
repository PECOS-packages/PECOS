# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

"""Honest windowing-mode reporting for the logical-circuit decoder (Layer 0).

The windowed-budget path used to silently perform a single-window full decode
(the per-observable sub-DEMs were serialized without detector coordinates, so the
inner windowed decoder degenerated to one window) while the API advertised a
bounded-latency budget. This pins that the effective mode is now surfaced
explicitly: ``effective_windowing`` / ``actual_num_windows`` are introspectable,
``can_window`` distinguishes "real windowing is possible" from "real windowing is
enabled", and a ``strict`` request hard-errors instead of silently falling back.

See pecos-docs/design/windowed-logical-subgraph-proper-solution.md.
"""

from __future__ import annotations

import pytest
import stim
from pecos.qec.surface import LogicalCircuitBuilder, SurfacePatch
from pecos.qec.surface.logical_circuit import _validate_boundary_cardinality
from pecos_rslib.qec import LogicalAlgorithmDecoder, LogicalCircuitDecoder


def _memory_descriptor(d: int, rounds: int) -> dict:
    patch = SurfacePatch.create(d)
    b = LogicalCircuitBuilder()
    b.add_patch(patch, "A")
    b.add_memory("A", rounds, "Z")
    return b.build_algorithm_descriptor(p1=0.001, p2=0.001, p_meas=0.001)


def _h_boundary_descriptor() -> dict:
    patch = SurfacePatch.create(3)
    b = LogicalCircuitBuilder()
    b.add_patch(patch, "A")
    b.add_memory("A", 3, "Z")
    b.add_transversal_h("A")
    b.add_memory("A", 3, "X")
    desc = b.build_algorithm_descriptor(p1=0.001, p2=0.001, p_meas=0.001)
    assert desc["boundary_gates"][0][0]["type"] == "Hadamard"
    return desc


def _decision_descriptor() -> dict:
    patch = SurfacePatch.create(3)
    num_qubits = patch.geometry.num_qubits
    builder = LogicalCircuitBuilder()
    builder.add_patch(patch, "data", qubit_offset=0)
    builder.add_patch(patch, "ancilla", qubit_offset=num_qubits)
    builder.add_t_via_injection("data", "ancilla", rounds_before=2, rounds_after=2)
    return builder.build_algorithm_descriptor(p1=0.001, p2=0.001, p_meas=0.001)


def test_algorithm_descriptor_happy_path_keeps_segment_and_boundary_schema():
    desc = _h_boundary_descriptor()
    assert len(desc["segments"]) == 2
    assert len(desc["boundary_gates"]) == 1
    assert desc["boundary_gates"][0][0]["type"] == "Hadamard"
    assert desc["num_observables"] == stim.DetectorErrorModel(desc["full_dem"]).num_observables
    assert desc["num_frame_slots"] == 2


def test_algorithm_descriptor_rejects_trailing_logical_gates():
    builder = LogicalCircuitBuilder()
    builder.add_patch(SurfacePatch.create(3), "A")
    builder.add_memory("A", 2, "Z")
    builder.add_transversal_h("A")

    with pytest.raises(
        ValueError,
        match=r"Hadamard.*after final data measurement.*terminal-segment support.*#595",
    ):
        builder.build_algorithm_descriptor(p1=0.001, p2=0.001, p_meas=0.001)


def test_algorithm_descriptor_rejects_leading_logical_gates():
    builder = LogicalCircuitBuilder()
    builder.add_patch(SurfacePatch.create(3), "A")
    builder.add_transversal_h("A")
    builder.add_memory("A", 2, "Z")

    with pytest.raises(
        ValueError,
        match=r"leading logical gates before any syndrome round.*no representable boundary.*Hadamard",
    ) as exc_info:
        builder.build_algorithm_descriptor(p1=0.001, p2=0.001, p_meas=0.001)

    message = str(exc_info.value)
    assert "terminal-segment support" not in message
    assert "#595" not in message


def test_algorithm_descriptor_rejects_empty_segment_list():
    builder = LogicalCircuitBuilder()
    builder.add_patch(SurfacePatch.create(3), "A")

    with pytest.raises(ValueError, match=r"must contain at least one segment"):
        builder.build_algorithm_descriptor(p1=0.001, p2=0.001, p_meas=0.001)


def test_algorithm_descriptor_rejects_boundary_cardinality_mismatch():
    with pytest.raises(ValueError, match=r"0 boundary gate lists and 2 segments"):
        _validate_boundary_cardinality([object(), object()], [])


def test_unlimited_budget_reports_unlimited():
    dec = LogicalCircuitDecoder(_memory_descriptor(3, 9), budget="unlimited")
    assert dec.effective_windowing == "unlimited"
    assert dec.can_window is False
    assert dec.actual_num_windows == []
    assert dec.has_decision_points() is False
    assert dec.num_decision_points() == 0


def test_windowed_budget_is_explicit_full_fallback_not_silent():
    """The windowed budget must NOT silently claim bounded latency: it reports a
    full-decode fallback with one window per observable, while still signalling
    that genuine windowing is possible for this (deep enough) circuit."""
    dec = LogicalCircuitDecoder(_memory_descriptor(3, 9), budget="windowed")
    assert dec.effective_windowing == "full_fallback"
    assert len(dec.actual_num_windows) >= 1
    assert all(n == 1 for n in dec.actual_num_windows)
    # The circuit is deep enough that real windowing *could* happen (coords are
    # preserved in the plan); it is just not enabled until the anti-snake work.
    assert dec.can_window is True


def test_strict_windowed_budget_hard_errors():
    """With strict=True, an unmet bounded-latency budget is a hard error rather
    than a silent full-decode fallback."""
    desc = _memory_descriptor(3, 9)
    with pytest.raises(Exception, match="strict"):
        LogicalCircuitDecoder(desc, budget="windowed", strict=True)


def test_strict_accepts_shallow_circuit_using_real_distance():
    """`can_window`/`strict` must use the REAL physical code distance from the
    descriptor, not a fake distance derived from the patch count. A single d=5
    patch with only 2 rounds is one window at step=d=5, so strict=True must NOT
    reject and can_window must be False. (The prior code derived distance=1 from
    the 1-patch count and wrongly reported real windowing / rejected.)"""
    desc = _memory_descriptor(5, 2)
    assert desc["distance"] == 5
    dec = LogicalCircuitDecoder(desc, budget="windowed", strict=True)  # must not raise
    assert dec.can_window is False
    assert dec.effective_windowing == "full_fallback"


def test_windowed_full_fallback_still_decodes():
    """The full-fallback path is still a working decoder (accurate per-observable
    decode), not a stub."""
    desc = _memory_descriptor(3, 9)
    dec = LogicalCircuitDecoder(desc, budget="windowed")
    ndet = sum(1 for ln in desc["full_dem"].splitlines() if ln.strip().startswith("detector("))
    # Zero syndrome -> zero correction.
    assert dec.decode([0] * ndet) == 0


def test_logical_circuit_decoder_rejects_empty_segments():
    """Malformed algorithm descriptors should raise a Python error, not panic."""
    desc = _memory_descriptor(3, 3)
    desc["segments"] = []

    with pytest.raises(ValueError, match="must contain at least one segment"):
        LogicalCircuitDecoder(desc, budget="unlimited")


def test_logical_circuit_decoder_rejects_missing_boundary_gate_bit():
    """Boundary gate descriptors must fail loudly when required bit fields are absent."""
    desc = _h_boundary_descriptor()
    del desc["boundary_gates"][0][0]["x_obs_bit"]

    with pytest.raises(ValueError, match="missing required field 'x_obs_bit'"):
        LogicalCircuitDecoder(desc, budget="unlimited")


def test_logical_circuit_decoder_rejects_out_of_range_boundary_gate_bit():
    """Boundary gate bits must fit the descriptor's logical frame schema."""
    desc = _h_boundary_descriptor()
    desc["boundary_gates"][0][0]["x_obs_bit"] = desc["num_frame_slots"]

    with pytest.raises(ValueError, match=r"frame slot 2.*num_frame_slots is 2"):
        LogicalCircuitDecoder(desc, budget="unlimited")


def test_logical_circuit_decoder_rejects_boundary_cardinality_mismatch():
    desc = _memory_descriptor(3, 3)
    desc["boundary_gates"].append([])

    with pytest.raises(ValueError, match=r"1 boundary gate lists and 1 segments"):
        LogicalCircuitDecoder(desc, budget="unlimited")


@pytest.mark.parametrize("decoder_type", [LogicalAlgorithmDecoder, LogicalCircuitDecoder])
@pytest.mark.parametrize(
    ("num_frame_slots", "message"),
    [(0, "must be greater than zero"), (3, "must be even")],
)
def test_python_logical_decoders_validate_frame_schema_before_full_dem_build(
    decoder_type,
    num_frame_slots,
    message,
):
    desc = _memory_descriptor(3, 3)
    desc["num_frame_slots"] = num_frame_slots
    desc["full_dem"] = "not a detector error model"
    kwargs = {} if decoder_type is LogicalAlgorithmDecoder else {"budget": "unlimited"}

    with pytest.raises(ValueError, match=message):
        decoder_type(desc, **kwargs)


@pytest.mark.parametrize("decoder_type", [LogicalAlgorithmDecoder, LogicalCircuitDecoder])
def test_python_logical_decoders_reject_decision_points(decoder_type):
    desc = _decision_descriptor()
    desc["full_dem"] = "not a detector error model"
    kwargs = {} if decoder_type is LogicalAlgorithmDecoder else {"budget": "unlimited"}

    with pytest.raises(
        ValueError,
        match=r"descriptor contains feed-forward decision points.*issue #596",
    ):
        decoder_type(desc, **kwargs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
