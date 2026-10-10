# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Consumers resolve ids and records through the circuit's shared reader."""

import json
from pathlib import Path

import pytest
from pecos._traced_circuit import measurement_ids_in_execution_order
from pecos.qec.dem_spec import _measurement_ids_in_runtime_order
from pecos.qec.surface import (
    SurfacePatch,
    extract_detection_events_and_observables,
    generate_tick_circuit_from_patch,
    get_detector_descriptors_from_tick_circuit,
    get_measurement_order_from_tick_circuit,
    get_observable_descriptors_from_tick_circuit,
    tick_circuit_to_stim,
)
from pecos.qec.surface.circuit_builder import generate_dem_from_tick_circuit_via_pauli_frame
from pecos.testing import simulate_tick_circuit
from pecos_rslib.quantum import TickCircuit


@pytest.fixture
def scrambled():
    circuit = TickCircuit()
    circuit.tick().mz_with_ids([0, 1], [17, 9])
    circuit.set_meta("num_measurements", "2")
    return circuit


@pytest.mark.parametrize("references", [{"meas_ids": [9]}, {"records": [1]}, {"records": [1], "meas_ids": [9]}])
def test_extraction_uses_positions(scrambled, references):
    scrambled.set_meta("detectors", json.dumps([{"id": 0, **references}]))
    assert extract_detection_events_and_observables(scrambled, [[0, 1]]) == ([[0]], [[]])


def test_extraction_annotation_only(scrambled):
    scrambled.detector([(0, 0, 1)])
    scrambled.observable([(0, 0, 1)])
    assert extract_detection_events_and_observables(scrambled, [[0, 1]]) == ([[0]], [[0]])


def test_extraction_sparse_ids_and_multiplicity(scrambled):
    scrambled.set_meta("detectors", '[{"id":8,"records":[1]},{"id":3,"records":[0]},{"id":2,"records":[1,1]}]')
    scrambled.set_meta("observables", '[{"id":7,"records":[1]},{"id":4,"records":[0]}]')
    assert extract_detection_events_and_observables(scrambled, [[1, 1]]) == ([[3, 8]], [[4, 7]])
    with pytest.raises(ValueError, match=r"row has length 1.*num_measurements=2"):
        extract_detection_events_and_observables(scrambled, [[1]])


def test_stim_resolves_scrambled_id(scrambled):
    scrambled.set_meta("detectors", '[{"id":0,"meas_ids":[17],"coords":[0,0,0]}]')
    assert tick_circuit_to_stim(scrambled) == "M 0 1\nDETECTOR(0.0, 0.0, 0.0) rec[-2]"


def test_stim_detector_id_order(scrambled):
    scrambled.set_meta("detectors", '[{"id":1,"records":[1]},{"id":0,"records":[0]}]')
    assert tick_circuit_to_stim(scrambled) == "M 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]"


def test_stim_annotation_only_observable(scrambled):
    scrambled.observable([(0, 0, 1)])
    assert tick_circuit_to_stim(scrambled) == "M 0 1\nOBSERVABLE_INCLUDE(0) rec[-1]"


def test_measurement_walkers_include_mx_and_mpz():
    circuit = TickCircuit()
    circuit.tick().mz_with_ids([4], [17])
    circuit.tick().mx([2])
    circuit.tick().add_gate("MPZ", [6, 1])
    circuit.assign_missing_meas_ids()
    assert measurement_ids_in_execution_order(circuit) == [17, 18, 19, 20]
    assert _measurement_ids_in_runtime_order(circuit) == [17, 18, 19, 20]
    assert get_measurement_order_from_tick_circuit(circuit) == [4, 2, 6, 1]


def test_runtime_walker_rejects_duplicates_and_unstamped_records():
    circuit = TickCircuit()
    circuit.tick().mz_with_ids([0], [7])
    circuit.tick().mz_with_ids([1], [7])
    with pytest.raises(ValueError, match="duplicate MeasId"):
        _measurement_ids_in_runtime_order(circuit)
    unstamped = TickCircuit()
    unstamped.tick().add_gate("MX", [0])
    with pytest.raises(ValueError, match="emission position 0 carries no MeasId"):
        measurement_ids_in_execution_order(unstamped)


def test_simulation_absolute_record():
    circuit = TickCircuit()
    circuit.tick().pz([0])
    circuit.tick().x([0])
    circuit.tick().mz_with_ids([0], [17])
    circuit.set_meta("detectors", '[{"id":8,"records":[0]}]')
    circuit.set_meta("observables", '[{"id":3,"records":[0]}]')
    assert simulate_tick_circuit(circuit) == ([1], 1, {3: 1})


def test_pauli_frame_generator_warns_at_caller(scrambled):
    scrambled.set_meta("detectors", '[{"id":0,"records":[-1],"coords":[0,0,0]}]')
    scrambled.set_meta("num_measurements", "2")
    with pytest.warns(DeprecationWarning, match="will be removed.*DetectorErrorModel.from_circuit") as warnings:
        generate_dem_from_tick_circuit_via_pauli_frame(scrambled)
    assert warnings[0].filename == __file__


def test_annotation_only_observable_descriptor(scrambled):
    scrambled.observable([(0, 0, 1)], label="ignored by descriptors")
    patch = SurfacePatch.create(distance=3)
    assert get_observable_descriptors_from_tick_circuit(scrambled, patch) == [
        {
            "id": 0,
            "observable_id": 0,
            "basis": "Z",
            "records": [-1],
            "logical_type": "Z",
            "data_qubits": [0, 1, 2],
            "data_qubit_positions": [[0, 0], [0, 1], [0, 2]],
            "weight": 3,
            "support_axis": "horizontal",
        },
    ]


def test_detector_descriptor_requires_coords(scrambled):
    scrambled.detector([(0, 0, 1)])
    with pytest.raises(ValueError, match="Detector 0 requires coordinates"):
        get_detector_descriptors_from_tick_circuit(scrambled, SurfacePatch.create(distance=3))


@pytest.mark.parametrize(("basis", "budget"), [("Z", None), ("X", None), ("Z", 2)])
def test_surface_descriptors_match_pre_reader_baseline(basis, budget):
    expected = json.loads((Path(__file__).parent / "data" / "reader_descriptor_baseline.json").read_text())
    patch = SurfacePatch.create(distance=3)
    circuit = generate_tick_circuit_from_patch(patch, num_rounds=1, basis=basis, ancilla_budget=budget)
    assert circuit.get_meta("detector_descriptors") is None
    assert circuit.get_meta("observable_descriptors") is None
    baseline = expected[f"{basis}-{budget}"]
    assert get_detector_descriptors_from_tick_circuit(circuit, patch) == baseline["detectors"]
    assert get_observable_descriptors_from_tick_circuit(circuit, patch) == baseline["observables"]
    assert circuit.get_meta("detector_descriptors") == json.dumps(baseline["detectors"])
    assert circuit.get_meta("observable_descriptors") == json.dumps(baseline["observables"])
    assert "DETECTOR" in tick_circuit_to_stim(circuit)


@pytest.mark.parametrize("gate", ["MeasureFree", "MX", "MPZ"])
def test_simulation_checks_reader_count_before_replay(gate):
    circuit = TickCircuit()
    circuit.tick().add_gate(gate, [0])
    circuit.set_meta("num_measurements", "0")
    with pytest.raises(ValueError, match="num_measurements=0 disagrees with the 1 measurement"):
        simulate_tick_circuit(circuit)


@pytest.mark.parametrize("consumer", ["extract", "stim", "simulate", "detectors", "observables"])
def test_consumers_reject_disagreeing_sources(scrambled, consumer):
    scrambled.detector([(0, 0, 1)])
    scrambled.set_meta("detectors", '[{"id":0,"records":[0],"coords":[0,0,0]}]')
    patch = SurfacePatch.create(distance=3)
    function, args = {
        "extract": (extract_detection_events_and_observables, (scrambled, [[0, 1]])),
        "stim": (tick_circuit_to_stim, (scrambled,)),
        "simulate": (simulate_tick_circuit, (scrambled,)),
        "detectors": (get_detector_descriptors_from_tick_circuit, (scrambled, patch)),
        "observables": (get_observable_descriptors_from_tick_circuit, (scrambled, patch)),
    }[consumer]
    with pytest.raises(ValueError, match="differ from annotation positions"):
        function(*args)


def test_descriptor_cache_is_returned_without_rereading(scrambled):
    scrambled.set_meta("detector_descriptors", '[{"cached":"detector"}]')
    scrambled.set_meta("observable_descriptors", '[{"cached":"observable"}]')
    scrambled.set_meta("detectors", 123)
    patch = SurfacePatch.create(distance=3)
    assert get_detector_descriptors_from_tick_circuit(scrambled, patch) == [{"cached": "detector"}]
    assert get_observable_descriptors_from_tick_circuit(scrambled, patch) == [{"cached": "observable"}]
