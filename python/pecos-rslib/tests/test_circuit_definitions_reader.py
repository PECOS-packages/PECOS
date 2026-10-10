# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""The Python boundary preserves shared-reader emission and definition semantics."""

import json

import pytest
from pecos_rslib.quantum import DagCircuit, TickCircuit


@pytest.fixture
def scrambled():
    circuit = TickCircuit()
    circuit.tick().mz_with_ids([0, 1], [17, 9])
    return circuit


def test_scrambled_emission(scrambled):
    assert scrambled.measurement_emission() == [(0, 17), (1, 9)]


def test_all_record_gates_and_unstamped_emission():
    circuit = TickCircuit()
    tick = circuit.tick()
    tick.mz([4, 2])
    tick.mx([7])
    tick.mz([9])
    tick = circuit.tick()
    tick.add_gate("MPZ", [3, 1])
    tick.mz_free([5])
    # Compatible MZ calls merge into the first batch, before the MX batch.
    assert circuit.measurement_emission() == [
        (4, 0),
        (2, 1),
        (9, 3),
        (7, 2),
        (3, None),
        (1, None),
        (5, 4),
    ]
    circuit.assign_missing_meas_ids()
    assert circuit.measurement_emission() == [
        (4, 0),
        (2, 1),
        (9, 3),
        (7, 2),
        (3, 5),
        (1, 6),
        (5, 4),
    ]

    unstamped = TickCircuit()
    unstamped.tick().add_gate("MZ", [0, 1])
    assert unstamped.measurement_emission() == [(0, None), (1, None)]


@pytest.mark.parametrize(
    ("references", "positions"),
    [
        ({"meas_ids": [9]}, [1]),
        ({"records": [-2]}, [0]),
        ({"records": [0]}, [0]),
        ({"records": [], "meas_ids": [9]}, [1]),
        ({"records": [-1], "meas_ids": [9]}, [1]),
        ({"records": [1, 1], "meas_ids": [9, 9]}, [1, 1]),
    ],
)
def test_resolved_definition_shape(scrambled, references, positions):
    scrambled.set_meta("detectors", json.dumps([{"id": 8, **references}]))
    scrambled.set_meta("observables", json.dumps([{"id": 3, **references, "label": "readout"}]))
    assert scrambled.circuit_definitions() == {
        "num_measurements": 2,
        "detectors": [{"id": 8, "measurements": positions, "coords": None, "label": None}],
        "observables": [{"id": 3, "measurements": positions, "label": "readout", "pauli": None}],
    }


def test_annotation_only_and_agreeing_metadata(scrambled):
    scrambled.detector([(0, 0, 1)], label="syndrome")
    scrambled.observable([(0, 0, 0)], label="logical")
    expected = {
        "num_measurements": 2,
        "detectors": [{"id": 0, "measurements": [1], "coords": None, "label": "syndrome"}],
        "observables": [{"id": 0, "measurements": [0], "label": "logical", "pauli": "+Z"}],
    }
    assert scrambled.circuit_definitions() == expected
    scrambled.set_meta("detectors", '[{"id":0,"records":[-1],"coords":[1,2,3]}]')
    scrambled.set_meta("observables", '[{"id":0,"meas_ids":[17]}]')
    expected["detectors"][0]["coords"] = [1.0, 2.0, 3.0]
    assert scrambled.circuit_definitions() == expected


@pytest.mark.parametrize("kind", ["detectors", "observables"])
def test_disagreeing_sources_raise_display_text(scrambled, kind):
    scrambled.detector([(0, 0, 1)])
    scrambled.observable([(0, 0, 1)])
    scrambled.set_meta(kind, '[{"id":0,"records":[0]}]')
    with pytest.raises(
        ValueError,
        match=r"id 0: metadata positions \[0\] differ from annotation positions \[1\]",
    ):
        scrambled.circuit_definitions()


@pytest.mark.parametrize("key", ["detectors", "observables", "num_measurements"])
def test_non_string_metadata_rejected(scrambled, key):
    scrambled.set_meta(key, 2)
    with pytest.raises(ValueError, match=f'attribute "{key}" must be a string'):
        scrambled.circuit_definitions()


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("detectors", '[{"id":0,"records":[0],"meas_ids":[9]}]', "records resolve to"),
        ("detectors", '[{"id":0,"records":[2]}]', "out of range"),
        ("observables", '[{"id":0,"meas_ids":[2]}]', "not present"),
        ("num_measurements", "18", "disagrees with the 2 measurement"),
    ],
)
def test_invalid_definitions_rejected(scrambled, key, value, message):
    scrambled.set_meta(key, value)
    with pytest.raises(ValueError, match=message):
        scrambled.circuit_definitions()


@pytest.mark.parametrize("circuit_type", [TickCircuit, DagCircuit])
@pytest.mark.parametrize("label", [None, "readout"])
def test_annotation_definition_ids(circuit_type, label):
    circuit = circuit_type()
    measurements = circuit.tick().mz([0, 1]) if circuit_type is TickCircuit else circuit.mz([0, 1])
    first, second = measurements

    ids = [
        circuit.observable([second], label=label),
        circuit.detector([first], label=label),
        circuit.detector([first, second], label=label),
        circuit.observable([first], label=label),
    ]
    assert ids == [0, 0, 1, 1]

    tick_circuit = circuit if circuit_type is TickCircuit else circuit.to_tick_circuit()
    definitions = tick_circuit.circuit_definitions()
    assert [(d["id"], d["measurements"]) for d in definitions["detectors"]] == [(0, [0]), (1, [0, 1])]
    assert [(o["id"], o["measurements"]) for o in definitions["observables"]] == [(0, [1]), (1, [0])]
    assert [a["label"] for a in circuit.annotations()] == [label] * 4
