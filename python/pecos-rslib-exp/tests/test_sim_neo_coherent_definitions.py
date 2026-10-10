"""Coherent raw-row synthesis consumes the shared circuit definition reader."""

import json

import pecos_rslib_exp as exp
import pytest
from pecos.quantum import TickCircuit


def _probe():
    circuit = TickCircuit()
    circuit.tick().pz([0, 1])
    circuit.tick().h([1])
    first, second = circuit.tick().mz_with_ids([0, 1], [17, 9])
    return circuit, first, second


def _sample(circuit):
    return [
        list(row)
        for row in (
            exp.sim_neo(circuit)
            .quantum(exp.meas_sampling())
            .noise(exp.depolarizing().idle_rz(0.125))
            .sampling(exp.monte_carlo(128))
            .seed(1046)
            .run()
        )
    ]


@pytest.mark.parametrize("reference", [{"records": [-2]}, {"meas_ids": [17]}, {"records": [0]}])
def test_coherent_detector_references_constrain_raw_rows(reference):
    circuit, _, _ = _probe()
    # Z rotations leave q0 in |0>; q1 is unconstrained. Honouring D0=m0
    # therefore gives m0=0 in every row, while ignoring D0 makes m0 a coin.
    circuit.set_meta("detectors", json.dumps([{"id": 7, **reference}]))
    rows = _sample(circuit)
    assert {row[0] for row in rows} == {0}
    assert {row[1] for row in rows} == {0, 1}

    equivalent, _, _ = _probe()
    equivalent.set_meta("detectors", '[{"id":7,"records":[-2]}]')
    assert rows == _sample(equivalent)


@pytest.mark.parametrize("empty_metadata", [None, "", "[]"])
def test_coherent_annotation_only_detector_constrains_raw_rows(empty_metadata):
    circuit, first, _ = _probe()
    circuit.detector([first])
    if empty_metadata is not None:
        circuit.set_meta("detectors", empty_metadata)
    assert {row[0] for row in _sample(circuit)} == {0}


def test_coherent_unconstrained_measurement_is_a_coin():
    circuit, _, _ = _probe()
    assert {row[0] for row in _sample(circuit)} == {0, 1}


@pytest.mark.parametrize("attribute", ["detectors", "observables"])
@pytest.mark.parametrize(("value", "message"), [("{", "metadata"), (42, "must be a string")])
def test_coherent_invalid_definition_metadata_raises(attribute, value, message):
    circuit, _, _ = _probe()
    circuit.set_meta(attribute, value)
    with pytest.raises(ValueError, match=message):
        _sample(circuit)


@pytest.mark.parametrize("attribute", ["detectors", "observables"])
def test_coherent_metadata_and_annotations_must_agree(attribute):
    circuit, first, _ = _probe()
    annotate = circuit.detector if attribute == "detectors" else circuit.observable
    annotate([first])
    circuit.set_meta(attribute, '[{"id":0,"records":[-1]}]')
    with pytest.raises(ValueError, match=r"metadata positions.*differ from annotation positions"):
        _sample(circuit)


@pytest.mark.parametrize("count", ["2", None, "3", "invalid"])
def test_coherent_measurement_count_is_inferred_and_validated(count):
    circuit, first, _ = _probe()
    circuit.detector([first])
    if count is not None:
        circuit.set_meta("num_measurements", count)
    if count in ("2", None):
        assert {row[0] for row in _sample(circuit)} == {0}
    else:
        with pytest.raises(ValueError, match="num_measurements"):
            _sample(circuit)
