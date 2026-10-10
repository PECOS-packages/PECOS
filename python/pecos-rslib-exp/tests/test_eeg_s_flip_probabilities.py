# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""S injection rates encode physical flip probabilities (issue #1174)."""

import math

import pecos_rslib_exp as exp
import pytest
from pecos.quantum import TickCircuit


def _circuit():
    circuit = TickCircuit()
    circuit.tick().pz([0])
    circuit.tick().mz([0])
    circuit.set_meta("detectors", '[{"id":0,"records":[-1]}]')
    circuit.set_meta("num_measurements", "1")
    return circuit


def test_exact_detection_rates_frozen_before_1174():
    # Captured from the built extension before any #1174 edits: [(0, 0.25)].
    assert dict(exp.exact_detection_rates(_circuit(), p_meas=0.25, idle_rz=0.125, prune=0.0)) == {0: 0.25}


def test_forward_measurement_flip_probability():
    events = exp.perturbative_dem_events(_circuit(), p_meas=0.25)
    assert len(events) == 1
    probability, detectors, observables = events[0]
    assert probability == pytest.approx(0.25, abs=1e-12)
    assert detectors == [0]
    assert observables == []


@pytest.mark.parametrize(
    "function",
    [
        exp.perturbative_dem,
        exp.perturbative_dem_events,
        exp.eeg_summary,
        exp.eeg_event_diagnostics,
        exp.eeg_per_detector,
    ],
)
@pytest.mark.parametrize("probability", [0.5, 0.75, 1.0])
def test_forward_bindings_reject_unrepresentable_s_probabilities(function, probability):
    with pytest.raises(ValueError, match=r"S injection at gate .*forward EEG requires 0 <= p < 0.5"):
        function(_circuit(), p_meas=probability)


def test_sim_neo_measurement_sampling_1174():
    shots = 40_000
    rows = (
        exp.sim_neo(_circuit())
        .quantum(exp.meas_sampling())
        .noise(exp.depolarizing().p_meas(0.25).idle_rz(0.125))
        .sampling(exp.monte_carlo(shots))
        .seed(1174)
        .run()
    )
    firing_rate = sum(row[0] for row in rows) / shots
    assert abs(firing_rate - 0.25) < 6 * math.sqrt(0.25 * 0.75 / shots)
