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
from pecos.qec.surface.logical_circuit import (
    _TWO_PATCH_IDENTITY,
    _append_two_patch_gate_transform,
    _canonical_two_patch_suffix,
    _validate_boundary_cardinality,
)
from pecos.qec.surface.patch import PatchOrientation
from pecos_rslib.qec import LogicalAlgorithmDecoder, LogicalCircuitDecoder, transform_two_patch_pauli


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
    assert all(segment["num_commit_detectors"] == segment["num_detectors"] for segment in desc["segments"])
    assert [segment["num_window_detectors"] for segment in desc["segments"]] == local_detector_counts

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


def test_single_round_memory_provider_reuses_bounded_templates(monkeypatch):
    """A one-round experiment is its own constant-depth boundary family."""
    from pecos.qec.surface.logical_circuit import (
        _cached_surface_memory_dem_templates,
        _cached_surface_singleton_memory_dem_templates,
    )

    _cached_surface_memory_dem_templates.cache_clear()
    _cached_surface_singleton_memory_dem_templates.cache_clear()
    builder = LogicalCircuitBuilder()
    builder.add_patch(SurfacePatch.create(3), "A", coord_offset=(-7.0, 5.0))
    builder.add_memory("A", 1, "X")
    oracle, _, _ = builder._build_structured_dem(  # noqa: SLF001
        p1=0.001,
        p2=0.002,
        p_meas=0.003,
        p_prep=0.004,
    )
    assert builder.build_dem(p1=0.001, p2=0.002, p_meas=0.003, p_prep=0.004) == oracle.to_string()
    after_first = _cached_surface_singleton_memory_dem_templates.cache_info()
    assert after_first.misses == 1
    assert after_first.currsize == 1
    assert _cached_surface_memory_dem_templates.cache_info().currsize == 0

    equivalent = LogicalCircuitBuilder()
    equivalent.add_patch(
        SurfacePatch.create(3),
        "renamed",
        qubit_offset=47,
        coord_offset=(23.0, -11.0),
    )
    equivalent.add_memory("renamed", 1, "X")
    equivalent.build_algorithm_descriptor(p1=0.001, p2=0.002, p_meas=0.003, p_prep=0.004)
    after_second = _cached_surface_singleton_memory_dem_templates.cache_info()
    assert after_second.misses == after_first.misses
    assert after_second.hits == after_first.hits + 1

    def reject_full_compile(*_args, **_kwargs):
        message = "a warm one-round template request compiled the full circuit"
        raise AssertionError(message)

    monkeypatch.setattr(LogicalCircuitBuilder, "_build_structured_dem", reject_full_compile)
    warm = LogicalCircuitBuilder()
    warm.add_patch(SurfacePatch.create(3), "warm", qubit_offset=17, coord_offset=(2.0, 19.0))
    warm.add_memory("warm", 1, "X")
    warm.build_dem(p1=0.001, p2=0.002, p_meas=0.003, p_prep=0.004)


@pytest.mark.parametrize(
    ("rounds", "shapes", "bases"),
    [
        (1, [(2, 2), (4, 3)], ["X", "Z"]),
        (3, [(3, 3), (2, 3)], ["Z", "X"]),
        (7, [(2, 2), (3, 2), (3, 3)], ["X", "Z", "X"]),
    ],
)
def test_multi_patch_memory_provider_matches_full_compile(rounds, shapes, bases):
    """Independent patch streams retain full ordering and placement."""
    builder = LogicalCircuitBuilder()
    labels = []
    for patch_index, (dx, dz) in enumerate(shapes):
        label = f"patch_{patch_index}"
        labels.append(label)
        builder.add_patch(
            SurfacePatch.create(dx=dx, dz=dz),
            label,
            qubit_offset=100 * patch_index + 7,
            coord_offset=(31.0 * patch_index - 9.0, 13.0 * patch_index + 2.0),
        )
    builder.add_memory(labels, rounds, dict(zip(labels, bases, strict=True)))
    oracle, _, _ = builder._build_structured_dem(  # noqa: SLF001
        p1=0.001,
        p2=0.002,
        p_meas=0.003,
        p_prep=0.004,
    )
    assert builder.build_dem(p1=0.001, p2=0.002, p_meas=0.003, p_prep=0.004) == oracle.to_string()


def test_multi_patch_memory_provider_reuses_physical_family(monkeypatch):
    """Labels, requested depth, qubit IDs, and placement are instance state."""
    from pecos.qec.surface.logical_circuit import _cached_surface_multi_memory_dem_templates

    _cached_surface_multi_memory_dem_templates.cache_clear()

    def build(rounds, labels, offsets, coords):
        builder = LogicalCircuitBuilder()
        shapes = [(3, 3), (2, 3)]
        bases = ["Z", "X"]
        for label, (dx, dz), offset, coord in zip(labels, shapes, offsets, coords, strict=True):
            builder.add_patch(
                SurfacePatch.create(dx=dx, dz=dz),
                label,
                qubit_offset=offset,
                coord_offset=coord,
            )
        builder.add_memory(labels, rounds, dict(zip(labels, bases, strict=True)))
        return builder

    first = build(3, ["A", "B"], [0, 50], [(-7.0, 5.0), (29.0, -3.0)])
    first.build_dem(p1=0.001, p2=0.002, p_meas=0.003, p_prep=0.004)
    after_first = _cached_surface_multi_memory_dem_templates.cache_info()
    assert after_first.misses == 1
    assert after_first.currsize == 1

    second = build(11, ["left", "right"], [19, 119], [(41.0, 23.0), (-17.0, 12.0)])
    second.build_algorithm_descriptor(p1=0.001, p2=0.002, p_meas=0.003, p_prep=0.004)
    after_second = _cached_surface_multi_memory_dem_templates.cache_info()
    assert after_second.misses == after_first.misses
    assert after_second.hits == after_first.hits + 1

    def reject_full_compile(*_args, **_kwargs):
        message = "a warm multi-patch memory request compiled the full circuit"
        raise AssertionError(message)

    monkeypatch.setattr(LogicalCircuitBuilder, "_build_structured_dem", reject_full_compile)
    warm = build(5, ["warm_0", "warm_1"], [31, 231], [(3.0, -19.0), (71.0, 8.0)])
    warm.build_dem(p1=0.001, p2=0.002, p_meas=0.003, p_prep=0.004)


def test_default_patch_placement_handles_heterogeneous_sizes_and_memory_order():
    """Automatic coordinates cannot merge streams from differently sized patches."""
    builder = LogicalCircuitBuilder()
    builder.add_patch(SurfacePatch.create(5), "A", qubit_offset=0)
    builder.add_patch(SurfacePatch.create(2), "B", qubit_offset=100)
    builder.add_memory(["B", "A"], 3, "Z")

    assert builder._patches["A"].coord_offset == (0.0, 0.0)  # noqa: SLF001
    assert builder._patches["B"].coord_offset == (12.0, 0.0)  # noqa: SLF001
    descriptor = builder.build_algorithm_descriptor(p1=0.001, p2=0.002, p_meas=0.003, p_prep=0.004)
    assert descriptor["segments"][0]["num_detectors"] == 80


def test_surface_template_provider_caches_are_bounded():
    """Noise sweeps cannot retain every compiled physical fixture forever."""
    from pecos.qec.surface.logical_circuit import (
        _cached_surface_cx_dem_templates,
        _cached_surface_h_dem_templates,
        _cached_surface_memory_dem_templates,
        _cached_surface_mixed_dem_templates,
        _cached_surface_multi_memory_dem_templates,
        _cached_surface_singleton_memory_dem_templates,
    )

    providers = (
        _cached_surface_memory_dem_templates,
        _cached_surface_singleton_memory_dem_templates,
        _cached_surface_multi_memory_dem_templates,
        _cached_surface_h_dem_templates,
        _cached_surface_cx_dem_templates,
        _cached_surface_mixed_dem_templates,
    )
    assert {provider.cache_info().maxsize for provider in providers} == {16}


@pytest.mark.parametrize(
    ("dx", "dz", "basis", "rounds"),
    [
        (3, 3, "Z", 1),
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


@pytest.mark.parametrize(
    ("orientation", "rounds", "bases"),
    [
        (PatchOrientation.X_TOP_BOTTOM, [3, 4, 5], ["Z", "X", "Z"]),
        (PatchOrientation.Z_TOP_BOTTOM, [2, 3, 4, 2], ["X", "Z", "X", "Z"]),
        (PatchOrientation.X_TOP_BOTTOM, [2, 2, 3, 2, 4], ["Z", "X", "Z", "X", "Z"]),
    ],
)
def test_repeated_logical_h_provider_matches_full_compile(orientation, rounds, bases):
    """Two through four H gates cover all physical/frame-parity families."""
    builder = LogicalCircuitBuilder()
    builder.add_patch(
        SurfacePatch.create(3, orientation=orientation),
        "data",
        qubit_offset=17,
        coord_offset=(-11.0, 6.0),
    )
    for segment_index, (segment_rounds, basis) in enumerate(zip(rounds, bases, strict=True)):
        if segment_index:
            builder.add_transversal_h("data")
        builder.add_memory("data", segment_rounds, basis)

    oracle, _, _ = builder._build_structured_dem(  # noqa: SLF001
        p1=0.002,
        p2=0.003,
        p_meas=0.004,
        p_prep=0.005,
    )
    assert builder.build_dem(p1=0.002, p2=0.003, p_meas=0.004, p_prep=0.005) == oracle.to_string()
    descriptor = builder.build_algorithm_descriptor(
        p1=0.002,
        p2=0.003,
        p_meas=0.004,
        p_prep=0.005,
    )
    assert descriptor["full_dem"] == oracle.to_string()
    assert len(descriptor["segments"]) == len(rounds)
    assert len(descriptor["boundary_gates"]) == len(rounds) - 1


def test_repeated_logical_h_provider_reuses_constant_boundary_families(monkeypatch):
    """Repeated-H cache cardinality and compilation are independent of depth."""
    from pecos.qec.surface.logical_circuit import _cached_surface_h_dem_templates

    _cached_surface_h_dem_templates.cache_clear()

    def build(rounds, *, label, qubit_offset, coord_offset):
        builder = LogicalCircuitBuilder()
        builder.add_patch(
            SurfacePatch.create(3),
            label,
            qubit_offset=qubit_offset,
            coord_offset=coord_offset,
        )
        for segment_index, segment_rounds in enumerate(rounds):
            if segment_index:
                builder.add_transversal_h(label)
            builder.add_memory(label, segment_rounds, "Z" if segment_index % 2 == 0 else "X")
        return builder

    first = build([3, 4, 5], label="A", qubit_offset=0, coord_offset=(0.0, 0.0))
    first.build_dem(p1=0.001, p2=0.002, p_meas=0.003, p_prep=0.004)
    after_first = _cached_surface_h_dem_templates.cache_info()
    assert after_first.misses == 2
    assert after_first.currsize == 2

    second = build([7, 2, 9], label="renamed", qubit_offset=29, coord_offset=(17.0, -8.0))
    second.build_dem(p1=0.001, p2=0.002, p_meas=0.003, p_prep=0.004)
    after_second = _cached_surface_h_dem_templates.cache_info()
    assert after_second.misses == after_first.misses
    assert after_second.hits == after_first.hits + 2

    def reject_full_compile(*_args, **_kwargs):
        message = "a warm repeated-H template request compiled the full circuit"
        raise AssertionError(message)

    monkeypatch.setattr(LogicalCircuitBuilder, "_build_structured_dem", reject_full_compile)
    warm = build([11, 6, 3], label="warm", qubit_offset=41, coord_offset=(-2.0, 13.0))
    warm.build_algorithm_descriptor(p1=0.001, p2=0.002, p_meas=0.003, p_prep=0.004)


def test_logical_cx_provider_reuses_bounded_templates_and_routes_patch_coordinates(monkeypatch):
    """CX templates ignore depth and independently translate both patches."""
    from pecos.qec.surface.logical_circuit import _cached_surface_cx_dem_templates

    _cached_surface_cx_dem_templates.cache_clear()
    builder = LogicalCircuitBuilder()
    builder.add_patch(SurfacePatch.create(3), "control", qubit_offset=11, coord_offset=(-13.0, 7.0))
    builder.add_patch(SurfacePatch.create(3), "target", qubit_offset=51, coord_offset=(27.0, -9.0))
    builder.add_memory(["control", "target"], 3, {"control": "Z", "target": "X"})
    builder.add_transversal_cx("control", "target")
    builder.add_memory(["control", "target"], 3, {"control": "X", "target": "Z"})
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
    assert descriptor["boundary_gates"][0][0]["type"] == "Cnot"
    after_first = _cached_surface_cx_dem_templates.cache_info()
    assert after_first.misses == 1
    assert after_first.currsize == 1

    equivalent = LogicalCircuitBuilder()
    equivalent.add_patch(SurfacePatch.create(3), "C", qubit_offset=5, coord_offset=(101.0, 53.0))
    equivalent.add_patch(SurfacePatch.create(3), "T", qubit_offset=85, coord_offset=(-41.0, 12.0))
    equivalent.add_memory(["C", "T"], 7, {"C": "Z", "T": "X"})
    equivalent.add_transversal_cx("C", "T")
    equivalent.add_memory(["C", "T"], 5, {"C": "X", "T": "Z"})
    equivalent.build_dem(p1=0.001, p2=0.002, p_meas=0.003, p_prep=0.004)
    after_second = _cached_surface_cx_dem_templates.cache_info()
    assert after_second.misses == after_first.misses
    assert after_second.hits == after_first.hits + 1

    equivalent.build_dem(p1=0.001, p2=0.003, p_meas=0.003, p_prep=0.004)
    assert _cached_surface_cx_dem_templates.cache_info().misses == after_second.misses + 1

    def reject_full_compile(*_args, **_kwargs):
        message = "a warm bounded CX-template request compiled the full circuit"
        raise AssertionError(message)

    monkeypatch.setattr(LogicalCircuitBuilder, "_build_structured_dem", reject_full_compile)
    warm = LogicalCircuitBuilder()
    warm.add_patch(SurfacePatch.create(3), "left", qubit_offset=17, coord_offset=(9.0, -2.0))
    warm.add_patch(SurfacePatch.create(3), "right", qubit_offset=117, coord_offset=(39.0, 22.0))
    warm.add_memory(["left", "right"], 9, {"left": "Z", "right": "X"})
    warm.add_transversal_cx("left", "right")
    warm.add_memory(["left", "right"], 4, {"left": "X", "right": "Z"})
    warm.build_algorithm_descriptor(p1=0.001, p2=0.002, p_meas=0.003, p_prep=0.004)


@pytest.mark.parametrize(
    (
        "distance",
        "control_orientation",
        "target_orientation",
        "initial_bases",
        "final_bases",
        "before_rounds",
        "after_rounds",
    ),
    [
        (
            2,
            PatchOrientation.X_TOP_BOTTOM,
            PatchOrientation.X_TOP_BOTTOM,
            ("Z", "X"),
            ("X", "Z"),
            2,
            2,
        ),
        (
            3,
            PatchOrientation.X_TOP_BOTTOM,
            PatchOrientation.Z_TOP_BOTTOM,
            ("X", "Z"),
            ("Z", "X"),
            5,
            3,
        ),
        (
            4,
            PatchOrientation.Z_TOP_BOTTOM,
            PatchOrientation.Z_TOP_BOTTOM,
            ("Z", "Z"),
            ("X", "X"),
            3,
            6,
        ),
    ],
)
def test_cached_logical_cx_provider_matches_full_compile_across_families(
    distance,
    control_orientation,
    target_orientation,
    initial_bases,
    final_bases,
    before_rounds,
    after_rounds,
):
    builder = LogicalCircuitBuilder()
    builder.add_patch(
        SurfacePatch.create(distance, orientation=control_orientation),
        "control",
        qubit_offset=7,
        coord_offset=(11.0, -4.0),
    )
    builder.add_patch(
        SurfacePatch.create(distance, orientation=target_orientation),
        "target",
        qubit_offset=107,
        coord_offset=(-23.0, 31.0),
    )
    initial = dict(zip(("control", "target"), initial_bases, strict=True))
    final = dict(zip(("control", "target"), final_bases, strict=True))
    builder.add_memory(["control", "target"], before_rounds, initial)
    builder.add_transversal_cx("control", "target")
    builder.add_memory(["control", "target"], after_rounds, final)
    oracle, _, _ = builder._build_structured_dem(  # noqa: SLF001
        p1=0.002,
        p2=0.003,
        p_meas=0.004,
        p_prep=0.005,
    )
    assert builder.build_dem(p1=0.002, p2=0.003, p_meas=0.004, p_prep=0.005) == oracle.to_string()


@pytest.mark.parametrize(
    ("final_basis", "rounds"),
    [
        ("Z", [3, 4, 5]),
        ("X", [2, 3, 2, 4]),
    ],
)
def test_repeated_logical_cx_provider_matches_full_compile(final_basis, rounds):
    """Future CNOT parity is applied by one-to-many output routing."""
    builder = LogicalCircuitBuilder()
    builder.add_patch(SurfacePatch.create(3), "control", qubit_offset=7, coord_offset=(-9.0, 4.0))
    builder.add_patch(SurfacePatch.create(3), "target", qubit_offset=107, coord_offset=(31.0, -6.0))
    for segment_index, segment_rounds in enumerate(rounds):
        if segment_index:
            builder.add_transversal_cx("control", "target")
        builder.add_memory(
            ["control", "target"],
            segment_rounds,
            {"control": final_basis, "target": final_basis},
        )

    oracle, _, _ = builder._build_structured_dem(  # noqa: SLF001
        p1=0.002,
        p2=0.003,
        p_meas=0.004,
        p_prep=0.005,
    )
    assert builder.build_dem(p1=0.002, p2=0.003, p_meas=0.004, p_prep=0.005) == oracle.to_string()
    descriptor = builder.build_algorithm_descriptor(
        p1=0.002,
        p2=0.003,
        p_meas=0.004,
        p_prep=0.005,
    )
    assert descriptor["full_dem"] == oracle.to_string()
    assert len(descriptor["segments"]) == len(rounds)
    assert len(descriptor["boundary_gates"]) == len(rounds) - 1


def test_repeated_logical_cx_provider_reuses_one_physical_family(monkeypatch):
    """Repeated identical CX gates reuse one bounded physical compile."""
    from pecos.qec.surface.logical_circuit import _cached_surface_cx_dem_templates

    _cached_surface_cx_dem_templates.cache_clear()

    def build(rounds, *, labels, offsets, coords):
        control, target = labels
        builder = LogicalCircuitBuilder()
        builder.add_patch(SurfacePatch.create(3), control, qubit_offset=offsets[0], coord_offset=coords[0])
        builder.add_patch(SurfacePatch.create(3), target, qubit_offset=offsets[1], coord_offset=coords[1])
        for segment_index, segment_rounds in enumerate(rounds):
            if segment_index:
                builder.add_transversal_cx(control, target)
            builder.add_memory([control, target], segment_rounds, {control: "Z", target: "Z"})
        return builder

    first = build([3, 4, 5], labels=("C", "T"), offsets=(0, 50), coords=((0.0, 0.0), (20.0, 0.0)))
    first.build_dem(p1=0.001, p2=0.002, p_meas=0.003, p_prep=0.004)
    after_first = _cached_surface_cx_dem_templates.cache_info()
    assert after_first.misses == 1
    assert after_first.currsize == 1

    second = build(
        [7, 2, 9],
        labels=("left", "right"),
        offsets=(13, 113),
        coords=((-17.0, 8.0), (42.0, -3.0)),
    )
    second.build_dem(p1=0.001, p2=0.002, p_meas=0.003, p_prep=0.004)
    after_second = _cached_surface_cx_dem_templates.cache_info()
    assert after_second.misses == after_first.misses
    assert after_second.hits == after_first.hits + 1

    def reject_full_compile(*_args, **_kwargs):
        message = "a warm repeated-CX template request compiled the full circuit"
        raise AssertionError(message)

    monkeypatch.setattr(LogicalCircuitBuilder, "_build_structured_dem", reject_full_compile)
    warm = build(
        [11, 6, 3],
        labels=("warm_control", "warm_target"),
        offsets=(23, 223),
        coords=((-2.0, 13.0), (55.0, 21.0)),
    )
    warm.build_algorithm_descriptor(p1=0.001, p2=0.002, p_meas=0.003, p_prep=0.004)


@pytest.mark.parametrize(
    ("gate_names", "rounds", "initial_bases", "final_bases"),
    [
        (("h0", "h1", "cx"), (3, 4, 3, 5), ("Z", "X"), ("X", "X")),
        (("cx", "h0", "h1", "cx"), (2, 3, 2, 4, 3), ("Z", "Z"), ("Z", "Z")),
    ],
)
def test_mixed_h_cx_provider_matches_full_compile(gate_names, rounds, initial_bases, final_bases):
    """Finite logical-state keys compose H and CX families exactly."""
    builder = LogicalCircuitBuilder()
    builder.add_patch(SurfacePatch.create(3), "control", qubit_offset=7, coord_offset=(-5.0, 2.0))
    builder.add_patch(SurfacePatch.create(3), "target", qubit_offset=107, coord_offset=(40.0, -3.0))
    labels = ["control", "target"]
    builder.add_memory(labels, rounds[0], dict(zip(labels, initial_bases, strict=True)))
    for gate_name, segment_rounds in zip(gate_names, rounds[1:], strict=True):
        if gate_name == "h0":
            builder.add_transversal_h("control")
        elif gate_name == "h1":
            builder.add_transversal_h("target")
        else:
            builder.add_transversal_cx("control", "target")
        builder.add_memory(labels, segment_rounds, dict(zip(labels, final_bases, strict=True)))

    oracle, _, _ = builder._build_structured_dem(  # noqa: SLF001
        p1=0.002,
        p2=0.003,
        p_meas=0.004,
        p_prep=0.005,
    )
    assert builder.build_dem(p1=0.002, p2=0.003, p_meas=0.004, p_prep=0.005) == oracle.to_string()
    descriptor = builder.build_algorithm_descriptor(
        p1=0.002,
        p2=0.003,
        p_meas=0.004,
        p_prep=0.005,
    )
    assert descriptor["full_dem"] == oracle.to_string()
    assert len(descriptor["segments"]) == len(rounds)
    assert [gate[0]["type"] for gate in descriptor["boundary_gates"]] == [
        "Hadamard" if gate_name.startswith("h") else "Cnot" for gate_name in gate_names
    ]


def test_mixed_h_cx_future_action_normalizes_to_a_bounded_valid_word():
    """Every valid short schedule maps to the same canonical finite state."""
    frontier = [((), (False, False), _TWO_PATCH_IDENTITY)]
    for _ in range(6):
        next_frontier = []
        for word, swapped, transform in frontier:
            canonical = _canonical_two_patch_suffix((False, False), transform, swapped)
            replay_swapped = (False, False)
            replay_transform = _TWO_PATCH_IDENTITY
            for gate in canonical:
                if gate == "cx":
                    assert replay_swapped[0] == replay_swapped[1]
                elif gate == "h0":
                    replay_swapped = (not replay_swapped[0], replay_swapped[1])
                else:
                    replay_swapped = (replay_swapped[0], not replay_swapped[1])
                replay_transform = _append_two_patch_gate_transform(replay_transform, gate)
            assert replay_swapped == swapped
            assert replay_transform == transform

            for gate in ("h0", "h1", "cx"):
                if gate == "cx" and swapped[0] != swapped[1]:
                    continue
                child_swapped = swapped
                if gate == "h0":
                    child_swapped = (not swapped[0], swapped[1])
                elif gate == "h1":
                    child_swapped = (swapped[0], not swapped[1])
                child_transform = _append_two_patch_gate_transform(transform, gate)
                next_frontier.append(((*word, gate), child_swapped, child_transform))
        frontier = next_frontier


@pytest.mark.parametrize(
    ("gate", "basis_images"),
    [
        ("h0", (0b0010, 0b0001, 0b0100, 0b1000)),
        ("h1", (0b0001, 0b0010, 0b1000, 0b0100)),
        ("cx", (0b0101, 0b0010, 0b0100, 0b1010)),
    ],
)
def test_shared_two_patch_clifford_transform_matches_basis_images(gate, basis_images):
    assert tuple(transform_two_patch_pauli(pauli, gate) for pauli in _TWO_PATCH_IDENTITY) == basis_images


def test_shared_two_patch_clifford_transform_rejects_invalid_input():
    with pytest.raises(ValueError, match="four bits"):
        transform_two_patch_pauli(16, "cx")
    with pytest.raises(ValueError, match="unknown two-patch Clifford"):
        transform_two_patch_pauli(0, "cz")


def test_mixed_h_cx_with_history_sensitive_outputs_uses_full_fallback(monkeypatch):
    """Even CX parity must not erase the frontend's observable reliability state."""
    import pecos.qec.surface.logical_circuit as logical_circuit

    builder = LogicalCircuitBuilder()
    builder.add_patch(SurfacePatch.create(1), "A", qubit_offset=0)
    builder.add_patch(SurfacePatch.create(1), "B", qubit_offset=3)
    both = ["A", "B"]
    builder.add_memory(both, 2, {"A": "Z", "B": "Z"})
    builder.add_transversal_h("A")
    builder.add_memory(both, 2, {"A": "X", "B": "Z"})
    builder.add_transversal_h("B")
    builder.add_memory(both, 2, {"A": "X", "B": "X"})
    builder.add_transversal_cx("A", "B")
    builder.add_memory(both, 2, {"A": "X", "B": "Z"})
    builder.add_transversal_cx("A", "B")
    builder.add_memory(both, 2, {"A": "X", "B": "Z"})
    oracle, _, _ = builder._build_structured_dem(  # noqa: SLF001
        p1=0.001,
        p2=0.002,
        p_meas=0.003,
        p_prep=0.004,
    )
    assert builder._assembled_dem_output_ids() == []  # noqa: SLF001

    def reject_mixed_cache(*_args, **_kwargs):
        message = "history-sensitive output schema reached the mixed template cache"
        raise AssertionError(message)

    monkeypatch.setattr(logical_circuit, "_cached_surface_mixed_dem_templates", reject_mixed_cache)
    assert builder.build_dem(p1=0.001, p2=0.002, p_meas=0.003, p_prep=0.004) == oracle.to_string()
    assert "L0" not in oracle.to_string()
    assert "L1" not in oracle.to_string()


def test_cached_provider_checks_the_assembled_circuit_output_schema(monkeypatch):
    """A fixture cannot authorize its own observable declarations."""
    builder = LogicalCircuitBuilder()
    builder.add_patch(SurfacePatch.create(3), "data")
    builder.add_memory("data", 2, "Z")

    monkeypatch.setattr(builder, "_assembled_dem_output_ids", list)
    with pytest.raises(ValueError, match=r"assembled circuit expects \{\}"):
        builder.build_dem(p1=0.001, p2=0.002, p_meas=0.003, p_prep=0.004)


def test_mixed_h_cx_provider_reuses_normalized_boundary_families(monkeypatch):
    """Equivalent depths and placement reuse the finite mixed-state cache."""
    from pecos.qec.surface.logical_circuit import _cached_surface_mixed_dem_templates

    _cached_surface_mixed_dem_templates.cache_clear()

    def build(rounds, *, labels, offsets, coords):
        control, target = labels
        builder = LogicalCircuitBuilder()
        builder.add_patch(SurfacePatch.create(3), control, qubit_offset=offsets[0], coord_offset=coords[0])
        builder.add_patch(SurfacePatch.create(3), target, qubit_offset=offsets[1], coord_offset=coords[1])
        builder.add_memory([control, target], rounds[0], {control: "Z", target: "X"})
        builder.add_transversal_h(control)
        builder.add_memory([control, target], rounds[1], {control: "X", target: "Z"})
        builder.add_transversal_h(target)
        builder.add_memory([control, target], rounds[2], {control: "X", target: "X"})
        builder.add_transversal_cx(control, target)
        builder.add_memory([control, target], rounds[3], {control: "X", target: "X"})
        return builder

    first = build(
        (3, 4, 3, 5),
        labels=("C", "T"),
        offsets=(0, 50),
        coords=((0.0, 0.0), (20.0, 0.0)),
    )
    first.build_dem(p1=0.001, p2=0.002, p_meas=0.003, p_prep=0.004)
    after_first = _cached_surface_mixed_dem_templates.cache_info()
    assert after_first.misses == 3
    assert after_first.currsize == 3

    second = build(
        (7, 2, 9, 6),
        labels=("left", "right"),
        offsets=(13, 113),
        coords=((-17.0, 8.0), (42.0, -3.0)),
    )
    second.build_dem(p1=0.001, p2=0.002, p_meas=0.003, p_prep=0.004)
    after_second = _cached_surface_mixed_dem_templates.cache_info()
    assert after_second.misses == after_first.misses
    assert after_second.hits == after_first.hits + 3

    def reject_full_compile(*_args, **_kwargs):
        message = "a warm mixed H/CX template request compiled the full circuit"
        raise AssertionError(message)

    monkeypatch.setattr(LogicalCircuitBuilder, "_build_structured_dem", reject_full_compile)
    warm = build(
        (11, 6, 3, 8),
        labels=("warm_control", "warm_target"),
        offsets=(23, 223),
        coords=((-2.0, 13.0), (55.0, 21.0)),
    )
    warm.build_algorithm_descriptor(p1=0.001, p2=0.002, p_meas=0.003, p_prep=0.004)


def test_unsupported_logical_gate_and_shallow_boundaries_retain_full_fallback():
    """Only complete bounded H and CX families bypass full construction."""
    from pecos.qec.surface.logical_circuit import (
        _cached_surface_cx_dem_templates,
        _cached_surface_h_dem_templates,
    )

    _cached_surface_h_dem_templates.cache_clear()
    _cached_surface_cx_dem_templates.cache_clear()
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

    shallow_cx_builder = LogicalCircuitBuilder()
    shallow_cx_builder.add_patch(SurfacePatch.create(3), "C", qubit_offset=0)
    shallow_cx_builder.add_patch(SurfacePatch.create(3), "T", qubit_offset=50)
    shallow_cx_builder.add_memory(["C", "T"], 3, "Z")
    shallow_cx_builder.add_transversal_cx("C", "T")
    shallow_cx_builder.add_memory(["C", "T"], 1, "Z")
    shallow_cx_builder.build_dem()

    mixed_repeated_cx = LogicalCircuitBuilder()
    mixed_repeated_cx.add_patch(SurfacePatch.create(3), "C", qubit_offset=0)
    mixed_repeated_cx.add_patch(SurfacePatch.create(3), "T", qubit_offset=50)
    mixed_repeated_cx.add_memory(["C", "T"], 3, {"C": "Z", "T": "X"})
    mixed_repeated_cx.add_transversal_cx("C", "T")
    mixed_repeated_cx.add_memory(["C", "T"], 3, {"C": "X", "T": "Z"})
    mixed_repeated_cx.add_transversal_cx("C", "T")
    mixed_repeated_cx.add_memory(["C", "T"], 3, {"C": "Z", "T": "X"})
    mixed_repeated_cx.build_dem()

    mismatched_cx_builder = LogicalCircuitBuilder()
    mismatched_cx_builder.add_patch(SurfacePatch.create(dx=1, dz=4), "C", qubit_offset=0)
    mismatched_cx_builder.add_patch(SurfacePatch.create(dx=2, dz=2), "T", qubit_offset=50)
    mismatched_cx_builder.add_memory(["C", "T"], 3, "Z")
    mismatched_cx_builder.add_transversal_cx("C", "T")
    mismatched_cx_builder.add_memory(["C", "T"], 3, "Z")
    mismatched_cx_builder.build_dem()
    assert _cached_surface_h_dem_templates.cache_info().currsize == 0
    assert _cached_surface_cx_dem_templates.cache_info().currsize == 0


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
