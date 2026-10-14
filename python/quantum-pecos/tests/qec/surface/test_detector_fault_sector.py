# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Fault-sector metadata follows family keys, including H and fold boundaries."""

import json

import pytest
from pecos.qec.dem_spec import surface_memory_dem_spec
from pecos.qec.surface import SurfacePatch
from pecos.qec.surface.decode import _surface_native_topology, _surface_patch_cache_key
from pecos.qec.surface.teleportation_decision import t_teleportation_decision_model

from qec.surface.test_logical_builder_gadgets import make_builder


@pytest.mark.parametrize("source", ["abstract", "traced_qis"])
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_surface_fault_sectors_and_native_forward(source, basis):
    patch = SurfacePatch.create(distance=3)
    topology = _surface_native_topology(_surface_patch_cache_key(patch), 2, basis, None, source, False)
    detectors = json.loads(topology.detectors_json)
    assert detectors
    assert {d["fault_sector"] for d in detectors} == {"X", "Z"}
    assert all(d["fault_sector"] == ("X" if d["coords"][1] == 0 else "Z") for d in detectors)


@pytest.mark.parametrize(
    "shape",
    ["d3_mem_X", "d3_mem_Z", "d3_h_z_to_x", "d3_h_x_to_z", "d3_h_zero_final", "d3_cx_zz", "d3_cx_xx"],
)
def test_logical_fault_sectors_follow_current_family(shape):
    circuit = make_builder(shape).to_tick_circuit()
    keys = json.loads(circuit.get_meta("measurement_keys"))["stabilizer"]
    by_record = {ordinal: (family, segment, rnd) for _, family, _, segment, rnd, ordinal in keys}
    detectors = json.loads(circuit.get_meta("detectors"))
    for detector in detectors:
        # The latest syndrome record is the current family, including round 0 after H.
        current = max(record for record in detector["meas_ids"] if record in by_record)
        expected = by_record[current][0]
        if shape == "d3_h_zero_final" and any(record not in by_record for record in detector["meas_ids"]):
            # With no syndrome round after H, the current X family is read on data.
            expected = "X"
        assert detector["fault_sector"] == expected


@pytest.mark.parametrize("shape", ["d3_fold_s_mid", "d3_fold_s_first", "d3_fold_s_last", "d3_fold_pair_x", "d3_h_fold"])
def test_fold_involved_fault_sectors_unknown(shape):
    builder = make_builder(shape)
    circuit = builder.to_tick_circuit()
    keys = json.loads(circuit.get_meta("measurement_keys"))["stabilizer"]
    fold_segments = set()
    segment = 0
    for op in builder.operations:
        if op.is_segment:
            if op.fold:
                fold_segments.update((patch, segment) for patch in op.patches)
            segment += 1
    fold_records = {ordinal for patch, _, _, seg, _, ordinal in keys if (patch, seg) in fold_segments}
    assert fold_records
    detectors = json.loads(circuit.get_meta("detectors"))
    for detector in detectors:
        if fold_records.intersection(detector["meas_ids"]):
            assert detector["fault_sector"] is None
        else:
            assert detector["fault_sector"] in {"X", "Z"}
    assert any(d["fault_sector"] is None for d in detectors)


def test_surface_spec_fault_sector_survival():
    detectors, _ = surface_memory_dem_spec(3, 2, "Z")
    assert {d.metadata["fault_sector"] for d in detectors} == {"X", "Z"}


def test_teleportation_filter_fault_sector_survival():
    model = t_teleportation_decision_model(SurfacePatch.create(distance=3), rounds_before=1)
    detectors = json.loads(model.circuit.get_meta("detectors"))
    assert detectors
    assert all(d["fault_sector"] in {"X", "Z"} for d in detectors)


def test_surface_descriptor_fault_sector_survival():
    from pecos.qec.surface.circuit_builder import (
        generate_tick_circuit_from_patch,
        get_detector_descriptors_from_tick_circuit,
    )

    patch = SurfacePatch.create(distance=3)
    circuit = generate_tick_circuit_from_patch(patch, 2, "Z")
    expected = [d["fault_sector"] for d in circuit.circuit_definitions()["detectors"]]
    descriptors = get_detector_descriptors_from_tick_circuit(circuit, patch)
    assert [d["fault_sector"] for d in descriptors] == expected
    assert get_detector_descriptors_from_tick_circuit(circuit, patch) == descriptors


@pytest.mark.parametrize("dagger", [False, True])
@pytest.mark.parametrize("final_rounds", [0, 2])
def test_physical_sz_boundary_fault_sectors_unknown(dagger, final_rounds):
    from pecos.qec.surface import LogicalCircuitBuilder

    builder = LogicalCircuitBuilder()
    builder.add_patch(SurfacePatch.create(distance=3), "A")
    builder.add_memory("A", 2, "Z")
    if dagger:
        builder.add_transversal_szdg("A")
    else:
        builder.add_transversal_sz("A")
    builder.add_memory("A", final_rounds, "Z")
    circuit = builder.to_tick_circuit()
    keys = json.loads(circuit.get_meta("measurement_keys"))
    affected = {ordinal for _, _, _, segment, rnd, ordinal in keys["stabilizer"] if segment == 1 and rnd == 0}
    if final_rounds == 0:
        affected.update(ordinal for _, _, ordinal in keys["data"])
    detectors = json.loads(circuit.get_meta("detectors"))
    for detector in detectors:
        if affected.intersection(detector["meas_ids"]):
            assert detector["fault_sector"] is None
        else:
            assert detector["fault_sector"] in {"X", "Z"}
    assert any(d["fault_sector"] is None for d in detectors)


def test_physical_sz_phase_uncertainty_follows_transversal_cx():
    from pecos.qec.surface import LogicalCircuitBuilder

    patch = SurfacePatch.create(distance=3)
    builder = LogicalCircuitBuilder()
    builder.add_patch(patch, "A")
    builder.add_patch(
        SurfacePatch.create(distance=3),
        "B",
        qubit_offset=patch.geometry.num_qubits,
        coord_offset=(8.0, 0.0),
    )
    builder.add_memory(["A", "B"], 2, {"A": "Z", "B": "Z"})
    builder.add_transversal_sz("A")
    builder.add_transversal_cx("A", "B")
    builder.add_memory(["A", "B"], 2, {"A": "Z", "B": "Z"})
    circuit = builder.to_tick_circuit()
    keys = json.loads(circuit.get_meta("measurement_keys"))
    last_segment = max(entry[3] for entry in keys["stabilizer"] if entry[0] == "B")
    affected_b = {
        entry[5] for entry in keys["stabilizer"] if entry[0] == "B" and entry[3] == last_segment and entry[4] == 0
    }
    detectors = json.loads(circuit.get_meta("detectors"))
    touching_b = [d for d in detectors if affected_b.intersection(d["meas_ids"])]
    assert touching_b
    assert all(d["fault_sector"] is None for d in touching_b)


@pytest.mark.parametrize("policy", ["global_axis_cycle_f", "checkerboard_xzzx", "checkerboard_zxxz"])
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_clifford_deformed_surface_keeps_undeformed_family_sectors(policy, basis):
    from pecos.qec.surface.circuit_builder import generate_tick_circuit_from_patch

    def sectors(clifford_frame_policy):
        circuit = generate_tick_circuit_from_patch(
            SurfacePatch.create(distance=3),
            num_rounds=2,
            basis=basis,
            interaction_basis="szz",
            clifford_frame_policy=clifford_frame_policy,
        )
        return [d["fault_sector"] for d in json.loads(circuit.get_meta("detectors"))]

    deformed = sectors(policy)
    assert deformed == sectors(None)
    assert set(deformed) == {"X", "Z"}
