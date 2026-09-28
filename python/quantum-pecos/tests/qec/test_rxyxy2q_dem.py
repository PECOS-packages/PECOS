# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Scheduled RXYXY2Q gates retain their two-qubit DEM noise rates."""

import math

import pytest
from pecos.qec import DetectorErrorModel
from pecos_rslib.qec import DemSampler
from pecos_rslib.quantum import TickCircuit


def _circuit(name: str, angles: list[float]) -> TickCircuit:
    circuit = TickCircuit()
    circuit.tick().pz([0, 1])
    circuit.tick().add_gate(name, [0, 1], angles)
    circuit.tick().add_gate(name, [0, 1], angles)
    circuit.tick().mz([0])
    circuit.tick().mz([1])
    circuit.set_meta("detectors", '[{"id":0,"records":[-2]},{"id":1,"records":[-1]}]')
    circuit.set_meta("num_measurements", "2")
    return circuit


@pytest.mark.parametrize("builder", [DetectorErrorModel, DemSampler])
def test_rxyxy2q_dem_scheduled_noise_rate(builder: type) -> None:
    actual = _circuit("RXYXY2Q", [math.pi / 2, math.pi / 2])
    reference = _circuit("RYY", [math.pi / 2])
    noise = {"p1": 0.0, "p2": 0.0, "p_meas": 0.0, "p_prep": 0.0}
    result = builder.from_circuit(actual, p2_gate_rates={"RXYXY2Q": 0.03}, **noise)
    expected = builder.from_circuit(reference, p2_gate_rates={"RYY": 0.03}, **noise)
    if builder is DemSampler:
        result = result.to_detector_error_model()
        expected = expected.to_detector_error_model()
    assert result.to_string() == expected.to_string()
    assert "error(" in result.to_string()


@pytest.mark.parametrize("builder", [DetectorErrorModel, DemSampler])
def test_rxyxy2q_dem_rejects_non_clifford(builder: type) -> None:
    circuit = _circuit("RXYXY2Q", [math.pi / 2, 0.123])
    with pytest.raises(ValueError, match="RXYXY2Q"):
        builder.from_circuit(circuit, p2=0.03)
