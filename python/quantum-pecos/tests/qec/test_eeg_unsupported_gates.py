# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Unsupported Cliffords fail at the EEG Python boundary."""

import pytest
from pecos_rslib.quantum import TickCircuit
from pecos_rslib_exp import exact_detection_rates


@pytest.mark.parametrize("kind", ["f", "fdg"])
def test_eeg_rejects_unsupported_clifford(kind: str) -> None:
    circuit = TickCircuit()
    circuit.tick().pz([0])
    getattr(circuit.tick(), kind)([0])
    circuit.tick().mz([0])
    circuit.set_meta("detectors", '[{"id":0,"records":[-1]}]')
    circuit.set_meta("num_measurements", "1")

    with pytest.raises(ValueError, match="unsupported gate type"):
        exact_detection_rates(circuit, p1=0.01, prune=0.0)
