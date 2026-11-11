# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Measurement provenance exported by surface generators and their consumers."""

import hashlib
import json

import pytest
from pecos.qec.circuit_steps import CircuitStep, OpType
from pecos.qec.dem_spec import Detector, _resolve_dem_specs, _resolved_schema_from_validated_json, rec
from pecos.qec.surface import LogicalCircuitBuilder, SurfacePatch
from pecos.qec.surface.circuit_builder import (
    TickCircuitRenderer,
    build_surface_code_circuit,
    generate_tick_circuit_from_patch,
)
from pecos.qec.surface.decode import _copy_surface_tick_circuit_metadata
from pecos.qec.surface.gadgets import default_allocation
from pecos.qec.surface.teleportation_decision import t_teleportation_decision_model
from pecos_rslib import TickCircuit

from qec.surface.test_logical_builder_gadgets import make_builder


@pytest.mark.parametrize("generator", ["surface", "logical"])
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_memory_classes_cover_emission(generator, basis):
    circuit = (
        generate_tick_circuit_from_patch(SurfacePatch.create(distance=3), 2, basis)
        if generator == "surface"
        else make_builder(f"d3_mem_{basis}").to_tick_circuit()
    )
    raw = json.loads(circuit.get_meta("detectors"))
    classes = json.loads(circuit.get_meta("measurement_classes"))
    definitions = circuit.circuit_definitions()
    assert raw
    assert all("fault_sector" not in detector for detector in raw)
    assert len(classes) == definitions["num_measurements"]
    assert {detector["fault_sector"] for detector in definitions["detectors"]} == {"X", "Z"}
    if generator == "surface":
        assert {entry["block"] for entry in classes} == {"memory"}
        assert {entry["family"] for entry in classes if entry["round"] == -1} == {"X" if basis == "Z" else "Z"}
        assert all(entry["family"] == basis and entry["round"] == 2 for entry in classes[-9:])


def test_surface_unrecognized_measurement_is_unclassed():
    patch = SurfacePatch.create(distance=3)
    steps, allocation = build_surface_code_circuit(patch, 2, "Z")
    steps.append(CircuitStep(OpType.MEASURE, [0], "injection"))
    circuit = TickCircuitRenderer(add_typed_annotations=False).render(steps, allocation, patch, 2, "Z")
    classes = json.loads(circuit.get_meta("measurement_classes"))
    assert classes[-1] is None
    assert len(classes) == circuit.circuit_definitions()["num_measurements"]


def test_surface_without_detectors_has_no_measurement_classes():
    patch = SurfacePatch.create(distance=3)
    circuit = TickCircuitRenderer(add_detectors=False).render(
        [CircuitStep(OpType.ALLOC, [0]), CircuitStep(OpType.MEASURE, [0], "injection")],
        default_allocation(patch),
        patch,
        0,
        "Z",
    )
    assert circuit.get_meta("measurement_classes") is None
    assert circuit.circuit_definitions()["num_measurements"] == 1


def test_logical_classes_keep_register_family_and_global_epoch_after_h():
    builder = LogicalCircuitBuilder()
    builder.add_patch(SurfacePatch.create(distance=3), "A")
    builder.add_memory("A", 2, "Z")
    builder.add_transversal_h("A")
    builder.add_memory("A", 2, "X")
    circuit = builder.to_tick_circuit()
    classes = json.loads(circuit.get_meta("measurement_classes"))
    keys = json.loads(circuit.get_meta("measurement_keys"))
    for label, physical_family, _, segment, rnd, ordinal in keys["stabilizer"]:
        base = {"X": "Z", "Z": "X"}[physical_family] if segment else physical_family
        assert classes[ordinal] == {"block": label, "family": base, "round": 2 * segment + rnd}
    for label, _, ordinal in keys["data"]:
        assert classes[ordinal] == {"block": label, "family": "Z", "round": 4}
    epochs = [entry["round"] for entry in classes]
    assert epochs == sorted(epochs)


@pytest.mark.parametrize("label", ["", None, 1, True, [], {}])
def test_empty_patch_label_is_rejected(label):
    builder = LogicalCircuitBuilder()
    with pytest.raises(ValueError, match="Patch label must be a non-empty string"):
        builder.add_patch(SurfacePatch.create(distance=3), label)


@pytest.mark.parametrize("shape", ["d3_fold_s_mid", "d3_fold_s_first", "d3_fold_s_last", "d3_h_fold"])
def test_fold_measurements_are_untyped(shape):
    builder = make_builder(shape)
    circuit = builder.to_tick_circuit()
    classes = json.loads(circuit.get_meta("measurement_classes"))
    segments = [op for op in builder.operations if op.is_segment]
    keys = json.loads(circuit.get_meta("measurement_keys"))
    for label, _, _, segment, _, ordinal in keys["stabilizer"]:
        assert classes[ordinal]["block"] == label
        assert (classes[ordinal]["family"] is None) == bool(segments[segment].fold)


@pytest.mark.parametrize("dagger", [False, True])
@pytest.mark.parametrize("rounds", [0, 2])
def test_phase_uncertainty_classes_follow_cx_and_terminal_readout(dagger, rounds):
    builder = LogicalCircuitBuilder()
    patch = SurfacePatch.create(distance=3)
    builder.add_patch(patch, "A")
    builder.add_patch(patch, "B", qubit_offset=patch.geometry.num_qubits)
    builder.add_memory(["A", "B"], 2, "Z")
    if dagger:
        builder.add_transversal_szdg("A")
    else:
        builder.add_transversal_sz("A")
    builder.add_transversal_cx("A", "B")
    builder.add_memory(["A", "B"], rounds, "Z")
    circuit = builder.to_tick_circuit()
    classes = json.loads(circuit.get_meta("measurement_classes"))
    keys = json.loads(circuit.get_meta("measurement_keys"))
    for label, _, _, segment, rnd, ordinal in keys["stabilizer"]:
        assert classes[ordinal]["block"] == label
        assert (classes[ordinal]["family"] is None) == (segment == 1 and rnd == 0)
    for label, _, ordinal in keys["data"]:
        assert classes[ordinal] == {"block": label, "family": None if rounds == 0 else "Z", "round": 2 + rounds}


@pytest.mark.parametrize("remap", [None, {0: 1, 1: 0}])
def test_surface_copy_remaps_classes(remap):
    source, target = TickCircuit(), TickCircuit()
    source.tick().mz([0, 1])
    target.tick().mz([1, 0])
    classes = [{"block": "memory", "family": "X", "round": -1}, None]
    source.set_meta("measurement_classes", json.dumps(classes))
    source.set_meta("detectors", '[{"id":0,"records":[0]}]')
    _copy_surface_tick_circuit_metadata(source, target, measurement_index_remap=remap)
    assert json.loads(target.get_meta("measurement_classes")) == (classes if remap is None else classes[::-1])
    assert target.circuit_definitions()["detectors"][0]["fault_sector"] == "X"


def test_surface_copy_classes_use_emission_positions_not_meas_ids():
    source, target = TickCircuit(), TickCircuit()
    source.tick().mz_with_ids([0, 1], [17, 9])
    target.tick().mz_with_ids([1, 0], [200, 100])
    classes = [{"block": "memory", "family": "X", "round": 0}, None]
    source.set_meta("measurement_classes", json.dumps(classes))
    source.set_meta("detectors", '[{"id":0,"meas_ids":[17]}]')
    _copy_surface_tick_circuit_metadata(source, target, measurement_index_remap={0: 100, 1: 200})
    assert json.loads(target.get_meta("measurement_classes")) == classes[::-1]
    assert target.circuit_definitions()["detectors"][0]["fault_sector"] == "X"


def test_teleportation_trace_classes_preserve_decision_labels():
    model = t_teleportation_decision_model(SurfacePatch.create(distance=3), rounds_before=1)
    traced = model.traced_circuit
    classes = json.loads(traced.get_meta("measurement_classes"))
    assert len(classes) == traced.circuit_definitions()["num_measurements"]
    assert None in classes  # Injection and final data measurements are outside the decision map.
    assert {entry["block"] for entry in classes if entry is not None} == {"D", "A"}
    expected = [(entry["id"], entry["fault_sector"]) for entry in model.circuit.circuit_definitions()["detectors"]]
    assert [(entry["id"], entry["fault_sector"]) for entry in traced.circuit_definitions()["detectors"]] == expected
    assert model.build_dem().detector_fault_sectors() == expected


@pytest.mark.parametrize("remap", [{0: 0}, {0: 0, 1: 0}, {0: 0, 1: 2}, {0: 0, 1: 1, 2: 2}])
def test_surface_copy_rejects_incomplete_class_remap(remap):
    source, target = TickCircuit(), TickCircuit()
    source.tick().mz([0, 1])
    target.tick().mz([0, 1])
    source.set_meta("measurement_classes", "[null,null]")
    with pytest.raises(ValueError, match=r"remap|bindings"):
        _copy_surface_tick_circuit_metadata(source, target, measurement_index_remap=remap)


@pytest.mark.parametrize("route", ["specs", "validated_json"])
def test_schema_fingerprint_includes_classes_only_when_present(route):
    circuit = TickCircuit()
    circuit.tick().mz([0])
    circuit.set_meta("detectors", '[{"id":0,"records":[0]}]')
    circuit.set_meta("observables", "[]")

    def fingerprint():
        if route == "specs":
            schema = _resolve_dem_specs([Detector(rec[-1])], [], circuit=circuit, result_traces=[])
        else:
            schema = _resolved_schema_from_validated_json(
                circuit.get_meta("detectors"),
                "[]",
                circuit=circuit,
                result_traces=[],
            )
        return schema.schema_fingerprint

    entry = {"id": 0, "meas_ids": [0]} if route == "specs" else {"id": 0, "records": [0]}
    payload = {
        "detectors": [entry],
        "observables": [],
        "runtime_measurement_order": [0],
        "named_result_measurements": [],
    }
    assert (
        fingerprint() == hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    )
    circuit.set_meta("measurement_classes", '[{"block":"memory","family":"X","round":0}]')
    first = fingerprint()
    circuit.set_meta("measurement_classes", '[{"block":"memory","family":"Z","round":0}]')
    assert fingerprint() != first
