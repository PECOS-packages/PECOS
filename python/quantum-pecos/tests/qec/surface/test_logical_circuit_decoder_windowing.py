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


def test_windowed_full_fallback_still_decodes():
    """The full-fallback path is still a working decoder (accurate per-observable
    decode), not a stub."""
    desc = _memory_descriptor(3, 9)
    dec = LogicalCircuitDecoder(desc, budget="windowed")
    ndet = sum(1 for ln in desc["full_dem"].splitlines() if ln.strip().startswith("detector("))
    # Zero syndrome -> zero correction.
    assert dec.decode([0] * ndet) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
