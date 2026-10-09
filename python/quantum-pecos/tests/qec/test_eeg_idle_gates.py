# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Identity and idle gates pass through the EEG bindings as valid gates."""

import pytest
from pecos_rslib.quantum import TickCircuit
from pecos_rslib_exp import exact_detection_rates


@pytest.mark.parametrize("kind", ["idle", "i"])
def test_identity_gate_between_hadamards_keeps_categorical_rate(kind: str) -> None:
    # The EEG expansion validates every gate, so an Idle must keep its duration
    # and an I must stay an I. Neither acts on the state, so two Hadamard
    # locations with depolarizing p give (1 - (1 - 4p/3)^2) / 2.
    p = 0.1
    circuit = TickCircuit()
    circuit.tick().pz([0])
    circuit.tick().h([0])
    if kind == "idle":
        circuit.tick().idle(5, [0])
    else:
        circuit.tick().i([0])
    circuit.tick().h([0])
    circuit.tick().mz([0])
    circuit.set_meta("detectors", '[{"id":0,"records":[-1]}]')
    circuit.set_meta("num_measurements", "1")

    rates = dict(exact_detection_rates(circuit, p1=p, prune=0.0))
    expected = (1.0 - (1.0 - 4.0 * p / 3.0) ** 2) / 2.0
    assert rates[0] == pytest.approx(expected, abs=1e-12)
