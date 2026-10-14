# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""XX/YY interference through the shared per-detector formula (#1209)."""

import json
import math

import pecos_rslib_exp as exp
import pytest
from pecos.quantum import TickCircuit


def _xx_yy_circuit(h_xx, h_yy):
    circuit = TickCircuit()
    circuit.tick().pz([0, 1])
    # exp(-i h_xx XX): Clifford skeleton is the identity.
    circuit.tick().h([0, 1])
    circuit.tick().cx([(0, 1)])
    circuit.tick().rz(2 * h_xx, [1])
    circuit.tick().cx([(0, 1)])
    circuit.tick().h([0, 1])
    # Change the same gadget's X axes to Y using S X S^dagger = Y.
    circuit.tick().szdg([0, 1])
    circuit.tick().h([0, 1])
    circuit.tick().cx([(0, 1)])
    circuit.tick().rz(2 * h_yy, [1])
    circuit.tick().cx([(0, 1)])
    circuit.tick().h([0, 1])
    circuit.tick().sz([0, 1])
    circuit.tick().mz([0, 1])
    circuit.set_meta("detectors", '[{"id":0,"records":[-2]},{"id":1,"records":[-1]}]')
    circuit.set_meta("num_measurements", "2")
    return circuit


@pytest.mark.parametrize("h_yy", [0.05, -0.05])
@pytest.mark.parametrize("formula", ["taylor", "sin_squared", "exact_commuting", "exact_subset"])
def test_xx_yy_per_detector_uses_phase_aware_crate_formula(h_yy, formula):
    circuit = _xx_yy_circuit(0.1, h_yy)
    # XX YY = -ZZ on |00>. Exact coherent angle is h_xx - h_yy.
    expected = {
        "taylor": (0.1 - h_yy) ** 2,
        "sin_squared": math.sin(0.1 - h_yy) ** 2,
        "exact_subset": math.sin(0.1 - h_yy) ** 2,
        # This variant intentionally ignores pair interference.
        "exact_commuting": (1 - math.cos(0.2) * math.cos(2 * h_yy)) / 2,
    }[formula]
    rates = dict(exp.eeg_per_detector(circuit, h_formula=formula))
    assert rates == pytest.approx({0: expected, 1: expected}, abs=1e-12)
    events = exp.perturbative_dem_events(circuit, h_formula=formula)
    assert len(events) == 1
    assert events[0][0] == pytest.approx(expected, abs=1e-12)
    assert events[0][1] == [0, 1]


def _over_limit_circuit(*, split_events=False):
    circuit = TickCircuit()
    qubits = list(range(26))
    circuit.tick().pz(qubits)
    circuit.tick().h(qubits)
    circuit.tick().rz(0.002, qubits)
    circuit.tick().h(qubits)
    circuit.tick().mz(qubits)
    # The Clifford skeleton is identity. Each RZ becomes a distinct H_Xj,
    # and all 26 labels flip the same global measurement parity.
    detectors = [{"id": 0, "records": list(range(-26, 0))}]
    if split_events:
        detectors.append({"id": 1, "records": [-26]})
    circuit.set_meta("detectors", json.dumps(detectors))
    circuit.set_meta("num_measurements", "26")
    return circuit


@pytest.mark.parametrize(
    "function",
    [exp.perturbative_dem, exp.perturbative_dem_events, exp.eeg_per_detector],
)
def test_exact_subset_limit_is_value_error(function):
    with pytest.raises(ValueError, match=r"ExactSubset limit is 25 H labels per event; got 26 for event"):
        function(_over_limit_circuit(), h_formula="exact_subset")


def test_exact_subset_limit_validates_single_detector_pooling():
    # Full events have sizes 1 and 25, but pooling for detector 0 has size 26.
    with pytest.raises(ValueError, match=r"ExactSubset limit is 25 H labels per event; got 26 for event"):
        exp.eeg_per_detector(_over_limit_circuit(split_events=True), h_formula="exact_subset")
