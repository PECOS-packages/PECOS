# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Tests for the public surface metadata helpers.

These tests cover the public descriptor API exposed for surface-code memory
experiments, independent of the DEM decomposition internals.
"""

import json
from typing import TYPE_CHECKING

import pytest
from pecos.qec.surface import (
    SurfacePatch,
    classify_stabilizer_boundary,
    describe_surface_memory_experiment,
    generate_tick_circuit_from_patch,
    get_detector_descriptors_from_tick_circuit,
    get_measurement_order_from_tick_circuit,
    get_observable_descriptors_from_tick_circuit,
    get_stab_schedule,
    get_stabilizer_region,
    get_stabilizer_schedule_entries,
    get_stabilizer_schedule_metadata,
    get_stabilizer_touch_label,
)

if TYPE_CHECKING:
    from pecos.qec.surface import SurfacePatchDescriptor


def test_surface_schedule_helpers_expose_region_and_touch_labels() -> None:
    """Surface metadata helpers should expose stable boundary and touch labels."""
    patch = SurfacePatch.create(distance=3)

    x_top = patch.x_stabilizers[0]
    x_bulk = patch.x_stabilizers[1]
    z_left = patch.z_stabilizers[0]

    assert classify_stabilizer_boundary("X", x_top.data_qubits, patch.distance) == "top"
    assert classify_stabilizer_boundary("Z", z_left.data_qubits, patch.distance) == "left"

    assert get_stabilizer_region(x_top, patch) == "top+left"
    assert get_stabilizer_region(x_bulk, patch) == "top+right"
    assert get_stabilizer_region(z_left, patch) == "bottom+left"

    assert get_stabilizer_touch_label(x_top, patch, 0) == "left"
    assert get_stabilizer_touch_label(x_top, patch, 1) == "right"
    assert get_stabilizer_touch_label(z_left, patch, 3) == "top"
    assert get_stabilizer_touch_label(z_left, patch, 6) == "bottom"
    assert get_stabilizer_touch_label(x_bulk, patch, 1) == "TL"
    assert get_stabilizer_touch_label(x_bulk, patch, 2) == "TR"
    assert get_stabilizer_touch_label(x_bulk, patch, 4) == "BL"
    assert get_stabilizer_touch_label(x_bulk, patch, 5) == "BR"

    assert get_stab_schedule("X", x_top.data_qubits, x_top.is_boundary, patch.dx, patch.dz) == [(2, 1), (3, 0)]
    assert get_stab_schedule("Z", z_left.data_qubits, z_left.is_boundary, patch.dx, patch.dz) == [(0, 3), (1, 6)]

    assert get_stabilizer_schedule_entries(x_top, patch) == [
        {"round_0based": 2, "data_qubit": 1, "touch_label": "right"},
        {"round_0based": 3, "data_qubit": 0, "touch_label": "left"},
    ]
    assert get_stabilizer_schedule_entries(z_left, patch) == [
        {"round_0based": 0, "data_qubit": 3, "touch_label": "top"},
        {"round_0based": 1, "data_qubit": 6, "touch_label": "bottom"},
    ]

    x_top_meta = get_stabilizer_schedule_metadata(x_top, patch)
    assert x_top_meta["stabilizer_kind"] == "X"
    assert x_top_meta["stabilizer_index"] == 0
    assert x_top_meta["stabilizer_is_boundary"] is True
    assert x_top_meta["stabilizer_region"] == "top+left"
    assert x_top_meta["schedule_rounds"] == [2, 3]
    assert x_top_meta["schedule_start_round"] == 2
    assert x_top_meta["schedule_end_round"] == 3
    assert x_top_meta["schedule_entries"] == get_stabilizer_schedule_entries(x_top, patch)


def test_surface_patch_exposes_stabilizer_descriptors() -> None:
    """Surface patches should publish detailed stabilizer descriptors."""
    patch = SurfacePatch.create(distance=3)

    x0 = patch.get_stabilizer_descriptor("X", 0)
    assert x0["stabilizer_kind"] == "X"
    assert x0["stabilizer_index"] == 0
    assert x0["stabilizer_region"] == "top+left"
    assert x0["data_qubits"] == [0, 1]
    assert x0["data_qubit_positions"] == [[0, 0], [0, 1]]
    assert x0["weight"] == 2
    assert x0["schedule_rounds"] == [2, 3]
    assert x0["schedule_entries"] == [
        {"round_0based": 2, "data_qubit": 1, "touch_label": "right"},
        {"round_0based": 3, "data_qubit": 0, "touch_label": "left"},
    ]

    x_descriptors = list(patch.iter_stabilizer_descriptors("X"))
    z_descriptors = list(patch.iter_stabilizer_descriptors("Z"))
    all_descriptors = list(patch.iter_stabilizer_descriptors())

    assert len(x_descriptors) == len(patch.x_stabilizers)
    assert len(z_descriptors) == len(patch.z_stabilizers)
    assert len(all_descriptors) == len(patch.x_stabilizers) + len(patch.z_stabilizers)
    assert all(row["stabilizer_kind"] == "X" for row in x_descriptors)
    assert all(row["stabilizer_kind"] == "Z" for row in z_descriptors)


def test_surface_patch_exposes_patch_descriptor() -> None:
    """Surface patches should expose a compact patch descriptor."""
    patch = SurfacePatch.create(distance=3)

    descriptor: SurfacePatchDescriptor = patch.get_patch_descriptor()
    assert descriptor == {
        "distance": 3,
        "dx": 3,
        "dz": 3,
        "rotated": True,
        "orientation": "X_TOP_BOTTOM",
        "num_data": 9,
        "num_ancilla": 8,
        "num_qubits": 17,
    }


def test_surface_patch_exposes_logical_descriptors() -> None:
    """Surface patches should expose public logical support descriptors."""
    patch = SurfacePatch.create(distance=3)

    logical_x = patch.get_logical_descriptor("X")
    logical_z = patch.get_logical_descriptor("Z")
    logicals = list(patch.iter_logical_descriptors())

    assert logical_x["logical_type"] == "X"
    assert logical_x["data_qubits"] == list(patch.geometry.logical_x.data_qubits)
    assert logical_x["data_qubit_positions"] == [[0, 0], [1, 0], [2, 0]]
    assert logical_x["weight"] == len(patch.geometry.logical_x.data_qubits)
    assert logical_x["support_axis"] == "vertical"

    assert logical_z["logical_type"] == "Z"
    assert logical_z["data_qubits"] == list(patch.geometry.logical_z.data_qubits)
    assert logical_z["data_qubit_positions"] == [[0, 0], [0, 1], [0, 2]]
    assert logical_z["weight"] == len(patch.geometry.logical_z.data_qubits)
    assert logical_z["support_axis"] == "horizontal"

    assert logicals == [logical_x, logical_z]


def test_tick_circuit_exposes_detector_descriptors() -> None:
    """Tick circuits should publish detector descriptors consistent with cached metadata."""
    patch = SurfacePatch.create(distance=3)
    tc = generate_tick_circuit_from_patch(patch, num_rounds=2, basis="X")

    descriptors = get_detector_descriptors_from_tick_circuit(tc, patch)
    cached = json.loads(tc.get_meta("detector_descriptors") or "[]")

    assert descriptors == cached
    assert len(descriptors) == int(tc.get_meta("num_detectors") or "0")

    first_x = next(
        row
        for row in descriptors
        if row["stabilizer_kind"] == "X" and row["stabilizer_index"] == 0 and row["round"] == 0
    )
    assert first_x["coords"] == [0, 0, 0]
    assert first_x["stabilizer_region"] == "top+left"
    assert first_x["stabilizer_is_boundary"] is True
    assert first_x["data_qubits"] == [0, 1]
    assert first_x["data_qubit_positions"] == [[0, 0], [0, 1]]
    assert first_x["schedule_rounds"] == [2, 3]
    assert first_x["is_final_round"] is False

    final_x = next(
        row
        for row in descriptors
        if row["stabilizer_kind"] == "X" and row["stabilizer_index"] == 0 and row["is_final_round"]
    )
    assert final_x["round"] == 2
    assert final_x["coords"] == [0, 0, 2]


def test_tick_circuit_exposes_observable_descriptors() -> None:
    """Tick circuits should publish observable descriptors derived from logical metadata."""
    patch = SurfacePatch.create(distance=3)
    tc = generate_tick_circuit_from_patch(patch, num_rounds=2, basis="X")

    descriptors = get_observable_descriptors_from_tick_circuit(tc, patch)
    cached = json.loads(tc.get_meta("observable_descriptors") or "[]")
    logical_x = patch.get_logical_descriptor("X")

    assert descriptors == cached
    assert len(descriptors) == 1

    row = descriptors[0]
    assert row["observable_id"] == 0
    assert row["basis"] == "X"
    assert row["records"] == cached[0]["records"]
    for key in logical_x:
        assert row[key] == logical_x[key]


def test_tick_circuit_exposes_measurement_order() -> None:
    """Tick circuits should expose measurement order matching their MZ gates."""
    patch = SurfacePatch.create(distance=3)
    tc = generate_tick_circuit_from_patch(patch, num_rounds=2, basis="X")

    observed = get_measurement_order_from_tick_circuit(tc)

    expected: list[int] = []
    for tick_index in range(tc.num_ticks()):
        tick = tc.get_tick(tick_index)
        if tick is None:
            continue
        for gate in tick.gate_batches():
            if "MZ" not in str(gate.gate_type):
                continue
            for qubit in gate.qubits:
                if hasattr(qubit, "index"):
                    expected.append(qubit.index())
                else:
                    expected.append(int(qubit))

    assert observed == expected
    assert len(observed) == int(tc.get_meta("num_measurements") or "0")


def test_tick_circuit_respects_ancilla_budget_in_measurement_order() -> None:
    """Measurement ordering should reflect ancilla reuse when a budget is imposed."""
    patch = SurfacePatch.create(distance=3)
    full_tc = generate_tick_circuit_from_patch(patch, num_rounds=1, basis="Z")
    batched_tc = generate_tick_circuit_from_patch(
        patch,
        num_rounds=1,
        basis="Z",
        ancilla_budget=2,
    )

    full_order = get_measurement_order_from_tick_circuit(full_tc)
    batched_order = get_measurement_order_from_tick_circuit(batched_tc)
    num_ancilla = patch.geometry.num_ancilla

    full_ancilla_measures = full_order[:num_ancilla]
    batched_ancilla_measures = batched_order[:num_ancilla]

    assert len(set(full_ancilla_measures)) == num_ancilla
    assert len(set(batched_ancilla_measures)) == 2
    assert max(batched_ancilla_measures) == patch.num_data + 1
    assert batched_tc.get_meta("ancilla_budget") == "2"
    assert full_tc.get_meta("num_detectors") == batched_tc.get_meta("num_detectors")
    assert full_tc.get_meta("num_measurements") == batched_tc.get_meta("num_measurements")


@pytest.mark.parametrize(
    ("patch_kwargs", "basis"),
    [
        ({"distance": 3, "rotated": False}, "Z"),
        ({"distance": 5, "rotated": False}, "X"),
        ({"dx": 3, "dz": 5}, "X"),
        ({"dx": 5, "dz": 3}, "Z"),
    ],
)
def test_surface_metadata_helpers_support_nonrotated_and_asymmetric_patches(
    patch_kwargs: dict[str, object],
    basis: str,
) -> None:
    """Surface metadata helpers should also work on non-rotated and asymmetric patches."""
    patch = SurfacePatch.create(**patch_kwargs)
    summary = describe_surface_memory_experiment(patch, num_rounds=1, basis=basis)

    assert summary["patch"]["rotated"] == patch.rotated
    assert summary["patch"]["dx"] == patch.dx
    assert summary["patch"]["dz"] == patch.dz
    assert summary["basis"] == basis
    assert summary["num_rounds"] == 1
    assert summary["x_stabilizers"]
    assert summary["z_stabilizers"]
    assert summary["logicals"]
    assert summary["detectors"]
    assert summary["observables"]


def test_describe_surface_memory_experiment_returns_descriptor_bundle() -> None:
    """Experiment summaries should bundle patch, stabilizer, detector, and observable metadata."""
    patch = SurfacePatch.create(distance=3)
    tc = generate_tick_circuit_from_patch(patch, num_rounds=2, basis="X")
    summary = describe_surface_memory_experiment(patch, num_rounds=2, basis="X")

    assert summary["patch"] == {
        "distance": 3,
        "dx": 3,
        "dz": 3,
        "rotated": True,
        "orientation": "X_TOP_BOTTOM",
        "num_data": 9,
        "num_ancilla": 8,
        "num_qubits": 17,
    }
    assert summary["basis"] == "X"
    assert summary["num_rounds"] == 2
    assert summary["ancilla_budget"] is None
    assert len(summary["x_stabilizers"]) == len(patch.x_stabilizers)
    assert len(summary["z_stabilizers"]) == len(patch.z_stabilizers)
    assert summary["stabilizers"] == summary["x_stabilizers"] + summary["z_stabilizers"]
    assert summary["logicals"] == list(patch.iter_logical_descriptors())
    assert summary["detectors"] == get_detector_descriptors_from_tick_circuit(tc, patch)
    assert summary["observables"] == get_observable_descriptors_from_tick_circuit(tc, patch)
