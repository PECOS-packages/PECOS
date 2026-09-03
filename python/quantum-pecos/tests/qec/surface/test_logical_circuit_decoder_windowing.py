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
from pecos.qec.surface.patch import PatchOrientation
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


def test_algorithm_segments_keep_structured_boundary_correlations():
    """A local segment DEM may overlap the next detector round without moving
    the streaming boundary or projecting away the cross-round mechanism."""
    desc = _h_boundary_descriptor()
    full_detector_count = sum(line.startswith("detector(") for line in desc["full_dem"].splitlines())
    local_detector_counts = [
        sum(line.startswith("detector(") for line in segment["dem"].splitlines()) for segment in desc["segments"]
    ]

    assert sum(segment["num_detectors"] for segment in desc["segments"]) == full_detector_count
    assert local_detector_counts[0] > desc["segments"][0]["num_detectors"]

    first_dem = desc["segments"][0]["dem"]
    detector_times = {}
    for line in first_dem.splitlines():
        if line.startswith("detector("):
            coords, detector = line.split(") D")
            detector_times[int(detector)] = float(coords.rsplit(",", maxsplit=1)[-1])
    assert any(
        min(times) < 3 <= max(times)
        for line in first_dem.splitlines()
        if line.startswith("error(")
        and (times := [detector_times[int(token[1:])] for token in line.split()[1:] if token.startswith("D")])
    )


def test_memory_provider_reuses_bounded_templates_and_preserves_detector_order(monkeypatch):
    """The public memory path caches a bounded compile, not an algorithm DEM."""
    from pecos.qec.surface.logical_circuit import _cached_surface_memory_dem_templates

    _cached_surface_memory_dem_templates.cache_clear()
    patch = SurfacePatch.create(3)
    builder = LogicalCircuitBuilder()
    builder.add_patch(patch, "A")
    builder.add_memory("A", 7, "Z")

    oracle, _, _ = builder._build_structured_dem(  # noqa: SLF001
        p1=0.001,
        p2=0.001,
        p_meas=0.001,
        p_prep=0.0,
    )
    descriptor = builder.build_algorithm_descriptor(
        p1=0.001,
        p2=0.001,
        p_meas=0.001,
        p_prep=0.0,
    )
    assert descriptor["full_dem"] == oracle.to_string()
    after_first = _cached_surface_memory_dem_templates.cache_info()
    assert after_first.misses == 1
    assert after_first.currsize == 1

    equivalent = LogicalCircuitBuilder()
    equivalent.add_patch(
        SurfacePatch.create(3),
        "renamed",
        qubit_offset=29,
        coord_offset=(17.0, -8.0),
    )
    equivalent.add_memory("renamed", 11, "Z")
    equivalent.build_dem(p1=0.001, p2=0.001, p_meas=0.001, p_prep=0.0)
    after_second = _cached_surface_memory_dem_templates.cache_info()
    assert after_second.misses == after_first.misses
    assert after_second.hits == after_first.hits + 1

    equivalent.build_dem(p1=0.002, p2=0.001, p_meas=0.001, p_prep=0.0)
    assert _cached_surface_memory_dem_templates.cache_info().misses == after_second.misses + 1

    def reject_full_compile(*_args, **_kwargs):
        message = "a warm bounded-template request compiled the full circuit"
        raise AssertionError(message)

    monkeypatch.setattr(LogicalCircuitBuilder, "_build_structured_dem", reject_full_compile)
    warm = LogicalCircuitBuilder()
    warm.add_patch(SurfacePatch.create(3), "another")
    warm.add_memory("another", 13, "Z")
    warm.build_dem(p1=0.001, p2=0.001, p_meas=0.001, p_prep=0.0)


@pytest.mark.parametrize(
    ("dx", "dz", "basis", "rounds"),
    [
        (2, 2, "Z", 2),
        (3, 3, "X", 5),
        (2, 3, "Z", 4),
        (3, 2, "X", 6),
    ],
)
def test_cached_memory_provider_matches_full_compile_across_geometries(dx, dz, basis, rounds):
    patch = SurfacePatch.create(dx=dx, dz=dz)
    builder = LogicalCircuitBuilder()
    builder.add_patch(patch, "data", qubit_offset=7, coord_offset=(11.0, -4.0))
    builder.add_memory("data", rounds, basis)
    oracle, _, _ = builder._build_structured_dem(  # noqa: SLF001
        p1=0.002,
        p2=0.003,
        p_meas=0.004,
        p_prep=0.005,
    )
    assert builder.build_dem(p1=0.002, p2=0.003, p_meas=0.004, p_prep=0.005) == oracle.to_string()


def test_logical_h_provider_reuses_bounded_templates(monkeypatch):
    """The H provider caches its boundary families independently of depth."""
    from pecos.qec.surface.logical_circuit import (
        _cached_surface_h_dem_templates,
        _cached_surface_memory_dem_templates,
    )

    _cached_surface_h_dem_templates.cache_clear()
    _cached_surface_memory_dem_templates.cache_clear()
    patch = SurfacePatch.create(3)
    builder = LogicalCircuitBuilder()
    builder.add_patch(patch, "A")
    builder.add_memory("A", 3, "Z")
    builder.add_transversal_h("A")
    builder.add_memory("A", 3, "X")
    oracle, _, _ = builder._build_structured_dem(  # noqa: SLF001
        p1=0.001,
        p2=0.002,
        p_meas=0.003,
        p_prep=0.004,
    )
    descriptor = builder.build_algorithm_descriptor(
        p1=0.001,
        p2=0.002,
        p_meas=0.003,
        p_prep=0.004,
    )
    assert descriptor["full_dem"] == oracle.to_string()
    after_first = _cached_surface_h_dem_templates.cache_info()
    assert after_first.misses == 1
    assert after_first.currsize == 1
    assert _cached_surface_memory_dem_templates.cache_info().currsize == 0

    equivalent = LogicalCircuitBuilder()
    equivalent.add_patch(
        SurfacePatch.create(3),
        "renamed",
        qubit_offset=29,
        coord_offset=(17.0, -8.0),
    )
    equivalent.add_memory("renamed", 7, "Z")
    equivalent.add_transversal_h("renamed")
    equivalent.add_memory("renamed", 5, "X")
    equivalent.build_dem(p1=0.001, p2=0.002, p_meas=0.003, p_prep=0.004)
    after_second = _cached_surface_h_dem_templates.cache_info()
    assert after_second.misses == after_first.misses
    assert after_second.hits == after_first.hits + 1

    equivalent.build_dem(p1=0.002, p2=0.002, p_meas=0.003, p_prep=0.004)
    assert _cached_surface_h_dem_templates.cache_info().misses == after_second.misses + 1

    def reject_full_compile(*_args, **_kwargs):
        message = "a warm bounded H-template request compiled the full circuit"
        raise AssertionError(message)

    monkeypatch.setattr(LogicalCircuitBuilder, "_build_structured_dem", reject_full_compile)
    warm = LogicalCircuitBuilder()
    warm.add_patch(SurfacePatch.create(3), "another")
    warm.add_memory("another", 9, "Z")
    warm.add_transversal_h("another")
    warm.add_memory("another", 4, "X")
    warm.build_algorithm_descriptor(p1=0.001, p2=0.002, p_meas=0.003, p_prep=0.004)


@pytest.mark.parametrize(
    ("distance", "orientation", "initial_basis", "final_basis", "before_rounds", "after_rounds"),
    [
        (2, PatchOrientation.X_TOP_BOTTOM, "Z", "X", 2, 2),
        (3, PatchOrientation.Z_TOP_BOTTOM, "X", "Z", 5, 3),
        (3, PatchOrientation.X_TOP_BOTTOM, "Z", "Z", 3, 6),
        (4, PatchOrientation.X_TOP_BOTTOM, "X", "X", 6, 4),
    ],
)
def test_cached_logical_h_provider_matches_full_compile_across_families(
    distance,
    orientation,
    initial_basis,
    final_basis,
    before_rounds,
    after_rounds,
):
    patch = SurfacePatch.create(distance, orientation=orientation)
    builder = LogicalCircuitBuilder()
    builder.add_patch(patch, "data", qubit_offset=7, coord_offset=(11.0, -4.0))
    builder.add_memory("data", before_rounds, initial_basis)
    builder.add_transversal_h("data")
    builder.add_memory("data", after_rounds, final_basis)
    oracle, _, _ = builder._build_structured_dem(  # noqa: SLF001
        p1=0.002,
        p2=0.003,
        p_meas=0.004,
        p_prep=0.005,
    )
    assert builder.build_dem(p1=0.002, p2=0.003, p_meas=0.004, p_prep=0.005) == oracle.to_string()


def test_unsupported_logical_gate_and_shallow_h_retain_full_fallback():
    """Only complete bounded H families bypass full structured construction."""
    from pecos.qec.surface.logical_circuit import _cached_surface_h_dem_templates

    _cached_surface_h_dem_templates.cache_clear()
    sz_builder = LogicalCircuitBuilder()
    sz_builder.add_patch(SurfacePatch.create(3), "A")
    sz_builder.add_memory("A", 3, "Z")
    sz_builder.add_transversal_sz("A")
    sz_builder.add_memory("A", 3, "Z")
    sz_builder.build_dem()

    shallow_h_builder = LogicalCircuitBuilder()
    shallow_h_builder.add_patch(SurfacePatch.create(3), "A")
    shallow_h_builder.add_memory("A", 1, "Z")
    shallow_h_builder.add_transversal_h("A")
    shallow_h_builder.add_memory("A", 3, "X")
    shallow_h_builder.build_dem()
    assert _cached_surface_h_dem_templates.cache_info().currsize == 0


def test_explicit_algorithm_buffer_cannot_truncate_a_boundary_correlation():
    patch = SurfacePatch.create(3)
    builder = LogicalCircuitBuilder()
    builder.add_patch(patch, "A")
    builder.add_memory("A", 3, "Z")
    builder.add_transversal_h("A")
    builder.add_memory("A", 3, "X")

    with pytest.raises(ValueError, match="requires at least 1 look-ahead rounds"):
        builder.build_algorithm_descriptor(buffer=0)


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
