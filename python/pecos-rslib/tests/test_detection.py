"""Detection names, explicit IDs, readout validation, and Stim rejection."""

import json

import pytest
from pecos_rslib import GateRegistry, TickCircuit, general_noise, sim
from pecos_rslib.programs import Qasm

FLAVORS = [
    ("DetectLeaked", "detect_leaked_with_ids", "DL"),
    ("DetectLost", "detect_lost_with_ids", "DLo"),
    ("DetectLeakedLost", "detect_leaked_lost_with_ids", "DLL"),
]


def program():
    return Qasm.from_string('OPENQASM 2.0; include "qelib1.inc"; qreg q[1]; creg c[1]; measure q[0] -> c[0];')


@pytest.mark.parametrize(("name", "method", "symbol"), FLAVORS)
def test_detection_helpers_and_case_sensitive_registry(name, method, symbol):
    circuit = TickCircuit()
    handle = circuit.tick()
    assert getattr(handle, method)([2, 0], [9, 3]) is handle
    gate = circuit.get_tick(0).gate_batches()[0]
    assert gate.gate_type.name == name
    assert gate.gate_type.is_detection()
    assert gate.qubits == [2, 0]
    assert gate.meas_ids == [9, 3]
    assert f"{symbol}[m9]" in circuit.render("ascii", show_measurement_ids=True)
    circuit.tick().add_gate(name, [1])
    assert circuit.get_tick(1).gate_batches()[0].meas_ids == []
    registry = GateRegistry()
    registry.define("DetectStep", 1).step(name, [0]).register_into(registry)
    assert registry.decompose("DetectStep", [2], []) == [(name, [2], [], {})]
    with pytest.raises(ValueError, match="Unknown gate type"):
        registry.define("Lowercase", 1).step(name.lower(), [0])


@pytest.mark.parametrize(("name", "method", "symbol"), FLAVORS)
def test_detection_fromstr_readout_and_noiseless_names(name, method, symbol):
    noise = general_noise().with_readout(name.lower(), {"present": {"0": 1.0}}).with_noiseless_gate(name.lower())
    assert sim(program()).noise(noise).run(1).to_dict()["c"] == [0]


@pytest.mark.parametrize(("name", "method", "symbol"), FLAVORS)
@pytest.mark.parametrize("json_config", [False, True])
@pytest.mark.parametrize("row", ["present", "leaked", "lost", "0", "1"])
def test_detection_python_readout_rows(name, method, symbol, json_config, row):
    rows = {row: {"0": 1.0}}
    if row in {"0", "1"}:

        def build_invalid_readout():
            noise = (
                general_noise().with_config_json(json.dumps({"measurement": {"readout": {name: rows}}}))
                if json_config
                else general_noise().with_readout(name, rows)
            )
            return sim(program()).noise(noise).run(1)

        with pytest.raises(ValueError, match="use row present, leaked, or lost"):
            build_invalid_readout()
        return
    noise = (
        general_noise().with_config_json(json.dumps({"measurement": {"readout": {name: rows}}}))
        if json_config
        else general_noise().with_readout(name, rows)
    )
    if row == "lost":
        with pytest.raises(ValueError, match="lost row needs qubit loss"):
            sim(program()).noise(noise).run(1)
    else:
        assert sim(program()).noise(noise).run(1).to_dict()["c"] == [0]


@pytest.mark.parametrize(("name", "method", "symbol"), FLAVORS)
def test_detection_python_readout_columns(name, method, symbol):
    with pytest.raises(ValueError, match="alphabet"):
        general_noise().with_readout(name, {"present": {"1": 1.0}})


@pytest.mark.parametrize(("name", "method", "symbol"), FLAVORS)
def test_stim_rejects_detection_before_noop_skip(name, method, symbol):
    from pecos.qec.surface.circuit_builder import tick_circuit_to_stim

    circuit = TickCircuit()
    circuit.tick().add_gate(name, [0])
    with pytest.raises(ValueError, match=f"{name} has no Stim form; remove it"):
        tick_circuit_to_stim(circuit)


@pytest.mark.parametrize(("name", "method", "symbol"), FLAVORS)
def test_detection_lost_row_with_enabled_loss(name, method, symbol):
    noise = general_noise().with_loss({"enabled": True}).with_readout(name, {"lost": {"0": 1.0}})
    assert sim(program()).noise(noise).run(1).to_dict()["c"] == [0]


@pytest.mark.parametrize("name", ["MZ", "MeasureZLeaked", "MeasureZLost", "MeasureZLeakedLost"])
def test_z_flavors_reject_present_row(name):
    with pytest.raises(ValueError, match="use row 0, 1, leaked, or lost"):
        general_noise().with_readout(name, {"present": {"0": 1.0}})
