# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Exact EEG rates use categorical depolarizing channels at each location."""

import json

import pytest
from pecos_rslib.quantum import TickCircuit
from pecos_rslib_exp import (
    exact_correlation_table,
    exact_detection_rates,
    exact_pairwise_rates,
    noise_characterization,
)


def _two_hadamards() -> TickCircuit:
    circuit = TickCircuit()
    circuit.tick().pz([0])
    circuit.tick().h([0])
    circuit.tick().h([0])
    circuit.tick().mz([0])
    circuit.set_meta("detectors", '[{"id":0,"records":[-1]}]')
    circuit.set_meta("num_measurements", "1")
    return circuit


@pytest.mark.parametrize("p", [0.0, 0.01, 0.1, 0.3, 0.75, 1.0])
def test_two_hadamards_have_categorical_detection_rate(p: float) -> None:
    rates = dict(exact_detection_rates(_two_hadamards(), p1=p, prune=0.0))
    expected = (1.0 - (1.0 - 4.0 * p / 3.0) ** 2) / 2.0
    assert rates[0] == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("p", [0.1, 0.9375, 1.0])
def test_two_qubit_marginals_and_correlations_are_categorical(p: float) -> None:
    circuit = TickCircuit()
    circuit.tick().pz([0, 1])
    circuit.tick().cx([(0, 1)])
    circuit.tick().mz([0])
    circuit.tick().mz([1])
    circuit.set_meta("detectors", '[{"id":0,"records":[-2]},{"id":1,"records":[-1]}]')
    circuit.set_meta("num_measurements", "2")

    rates = dict(exact_detection_rates(circuit, p2=p, prune=0.0))
    assert rates == pytest.approx({0: 8.0 * p / 15.0, 1: 8.0 * p / 15.0}, abs=1e-12)
    pairwise = dict(exact_pairwise_rates(circuit, p2=p, prune=0.0))
    assert pairwise[(0, 1)] == pytest.approx(4.0 * p / 15.0, abs=1e-12)
    correlations = {
        tuple(nodes): probability for nodes, probability in exact_correlation_table(circuit, p2=p, prune=0.0)
    }
    assert correlations[("D0", "D1")] == pytest.approx(4.0 * p / 15.0, abs=1e-12)


@pytest.mark.parametrize("compress", [False, True])
def test_compression_keeps_original_categorical_correlation_targets(*, compress: bool) -> None:
    data, _raw, _decomposed = noise_characterization(
        _two_hadamards(),
        p1=0.75,
        max_order=1,
        prune=0.0,
        compress=compress,
    )
    correlations = json.loads(data)["correlations"]
    assert len(correlations) == 1
    assert correlations[0]["nodes"] == ["D0"]
    assert correlations[0]["probability"] == pytest.approx(0.5, abs=1e-12)
