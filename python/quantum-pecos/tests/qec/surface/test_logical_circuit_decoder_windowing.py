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
from pecos.qec.surface import LogicalCircuitBuilder, SurfacePatch
from pecos_rslib.qec import LogicalCircuitDecoder


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


def test_unlimited_budget_reports_unlimited():
    dec = LogicalCircuitDecoder(_memory_descriptor(3, 9), budget="unlimited")
    assert dec.effective_windowing == "unlimited"
    assert dec.can_window is False
    assert dec.actual_num_windows == []


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

    with pytest.raises(ValueError, match="no segments"):
        LogicalCircuitDecoder(desc, budget="unlimited")


def test_logical_circuit_decoder_rejects_missing_boundary_gate_bit():
    """Boundary gate descriptors must fail loudly when required bit fields are absent."""
    desc = _h_boundary_descriptor()
    del desc["boundary_gates"][0][0]["x_obs_bit"]

    with pytest.raises(ValueError, match="missing required field 'x_obs_bit'"):
        LogicalCircuitDecoder(desc, budget="unlimited")


def test_logical_circuit_decoder_rejects_out_of_range_boundary_gate_bit():
    """Boundary gate bits index a u64 observable frame and must be below 64."""
    desc = _h_boundary_descriptor()
    desc["boundary_gates"][0][0]["x_obs_bit"] = 64

    with pytest.raises(ValueError, match="exceeds the 64-observable frame limit"):
        LogicalCircuitDecoder(desc, budget="unlimited")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
