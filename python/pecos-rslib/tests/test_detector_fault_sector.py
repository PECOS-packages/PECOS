# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Detector fault sectors survive the public circuit and DEM boundaries."""

import json

import pytest
from pecos_rslib.qec import DetectorErrorModel
from pecos_rslib.quantum import TickCircuit


def circuit_with(entry):
    circuit = TickCircuit()
    circuit.tick().pz([0])
    circuit.tick().mz([0])
    circuit.set_meta("detectors", json.dumps([{"id": 0, "records": [-1], **entry}]))
    return circuit


@pytest.mark.parametrize(
    "entry",
    [
        {},
        {"fault_sector": None},
        {"fault_sector": "X"},
        {"fault_sector": "Z"},
        {"fault_sector": "XX"},
        {"fault_sector": "sector-3"},
        {"fault_sector": "x"},
        {"fault_sector": "Y"},
        {"fault_sector": " "},
    ],
)
def test_fault_sector_reader_and_dem(entry):
    circuit = circuit_with(entry)
    circuit.detector([(1, 0, 0)])
    assert circuit.circuit_definitions()["detectors"][0]["fault_sector"] == entry.get("fault_sector")
    dem = DetectorErrorModel.from_circuit(circuit)
    assert dem.detector_fault_sectors() == [(0, entry.get("fault_sector"))]
    plain = DetectorErrorModel.from_circuit(circuit_with({}))
    for name in dir(dem):
        if name == "to_string" or name.startswith("to_string_") or name == "to_pecos_metadata_json":
            args = ("KeepDirect",) if name.endswith("_policy") else ()
            assert getattr(dem, name)(*args) == getattr(plain, name)(*args), name


@pytest.mark.parametrize("value", ["", 1, [], {}, True])
def test_fault_sector_rejected_with_id(value):
    circuit = circuit_with({"id": 37, "fault_sector": value})
    with pytest.raises(ValueError, match=r"detector 37.*fault_sector"):
        circuit.circuit_definitions()
    with pytest.raises(ValueError, match=r"detector 37.*fault_sector"):
        DetectorErrorModel.from_circuit(circuit)


@pytest.mark.parametrize("value", [None, "X", "Z", "XX", "sector-3", "x", "Y", " "])
def test_add_detector_fault_sector(value):
    circuit = circuit_with({})
    circuit.set_meta("detectors", "[]")
    circuit.add_detector([-1], detector_id=4, fault_sector=value)
    assert circuit.circuit_definitions()["detectors"][0]["fault_sector"] == value


@pytest.mark.parametrize("value", ["", 1, [], {}, True, object()])
@pytest.mark.parametrize(("detector_id", "expected_id"), [(37, 37), (None, 1)])
def test_add_detector_rejects_fault_sector(value, detector_id, expected_id):
    circuit = circuit_with({})
    with pytest.raises(ValueError, match=rf"detector {expected_id}.*fault_sector"):
        circuit.add_detector([-1], detector_id=detector_id, fault_sector=value)
    assert len(circuit.circuit_definitions()["detectors"]) == 1


@pytest.mark.parametrize("sector", ["X", "sector-3", "x", " "])
def test_fault_sector_conversion_and_noise_survival(sector):
    circuit = circuit_with({"fault_sector": sector})
    converted = circuit.to_dag_circuit().to_tick_circuit()
    assert converted.circuit_definitions()["detectors"][0]["fault_sector"] == sector
    noisy = circuit.with_noise(p_prep=0.01)
    assert noisy.circuit_definitions()["detectors"][0]["fault_sector"] == sector


def test_observable_fault_sector_remains_ignored():
    circuit = circuit_with({})
    circuit.set_meta("observables", '[{"id":0,"records":[-1],"fault_sector":[]}]')
    assert "fault_sector" not in circuit.circuit_definitions()["observables"][0]
