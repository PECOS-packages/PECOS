# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""DemBuilder and EEG bind every reference form to the same measurement (#1067).

One gate measures qubits 0 and 1 with stamped ids 17 and 9, so emission order
and id order disagree. Only qubit 0 sees gate noise. Detector D0 reads qubit 0
and D1 reads qubit 1, so in both components every error must fire D0 and none
may fire D1, whichever form defines the detectors.
"""

import json
import re

import pytest
from pecos.qec import DetectorErrorModel
from pecos_rslib.quantum import TickCircuit
from pecos_rslib_exp import noise_characterization

# Emission order is [(qubit 0, id 17), (qubit 1, id 9)]: rec[-2] and id 17 name
# qubit 0; rec[-1] and id 9 name qubit 1.
_METADATA_FORMS = {
    "records": ({"records": [-2]}, {"records": [-1]}),
    "meas_ids": ({"meas_ids": [17]}, {"meas_ids": [9]}),
    "both": ({"records": [-2], "meas_ids": [17]}, {"records": [-1], "meas_ids": [9]}),
}


def _circuit(form):
    circuit = TickCircuit()
    circuit.tick().pz([0, 1])
    circuit.tick().h([0])
    circuit.tick().h([0])
    qubit0, qubit1 = circuit.tick().mz_with_ids([0, 1], [17, 9])
    circuit.set_meta("num_measurements", "2")
    if form == "annotations":
        circuit.detector([qubit0])
        circuit.detector([qubit1])
    else:
        d0, d1 = _METADATA_FORMS[form]
        circuit.set_meta("detectors", json.dumps([{"id": 0, **d0}, {"id": 1, **d1}]))
    return circuit


def _fired_detectors(dem):
    fired = set()
    for line in str(dem).splitlines():
        match = re.fullmatch(r"error\([^)]+\) (.*)", line.strip())
        if match:
            fired.add(frozenset(target for target in match[1].split() if target.startswith("D")))
    return fired


@pytest.mark.parametrize("form", [*_METADATA_FORMS, "annotations"])
def test_dem_builder_and_eeg_bind_the_same_measurement(form):
    dem_builder = _fired_detectors(
        DetectorErrorModel.from_circuit(_circuit(form), p1=0.01, p2=0.0, p_meas=0.0, p_prep=0.0),
    )
    _, eeg_dem, _ = noise_characterization(_circuit(form), p1=0.01, prune=0.0)
    eeg = _fired_detectors(eeg_dem)
    assert dem_builder == {frozenset({"D0"})}
    assert eeg == {frozenset({"D0"})}
