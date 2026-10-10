# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Exact independent depolarizing flips and their domain (issue #1202)."""

import math

import pecos_rslib_exp as exp
import pytest
from pecos.quantum import TickCircuit


def _circuit():
    circuit = TickCircuit()
    circuit.tick().pz([0])
    circuit.tick().h([0])
    circuit.tick().h([0])
    circuit.tick().mz([0])
    circuit.set_meta("detectors", '[{"id":0,"records":[-1]}]')
    circuit.set_meta("num_measurements", "1")
    return circuit


@pytest.mark.parametrize(
    "function",
    [
        exp.perturbative_dem,
        exp.perturbative_dem_events,
        exp.eeg_summary,
        exp.eeg_event_diagnostics,
        exp.eeg_per_detector,
        exp.coherent_dem_exact,
        exp.coherent_dem_decomposed,
        exp.compress_noise,
        exp.noise_characterization,
    ],
)
@pytest.mark.parametrize("parameter,limit,fraction", [("p1", 0.75, "3/4"), ("p2", 0.9375, "15/16")])
def test_generator_bindings_reject_out_of_domain_before_reading_circuit(function, parameter, limit, fraction):
    # An unusable circuit proves validation happens before circuit work.
    with pytest.raises(ValueError, match=f"generator view requires {parameter} <= {fraction}"):
        function(None, **{parameter: math.nextafter(limit, math.inf)})


@pytest.mark.parametrize(
    "function",
    [
        exp.exact_detection_rates,
        exp.exact_pairwise_rates,
        exp.exact_correlation_table,
        exp.correlation_matching_dem,
    ],
)
def test_channel_only_bindings_accept_negative_eigenvalues(function):
    result = function(_circuit(), p1=0.9, p2=1.0)
    if function is exp.exact_detection_rates:
        # Two H locations attenuate Z by lambda^2, lambda = 1 - 4*p1/3.
        expected = (1 - (1 - 4 * 0.9 / 3) ** 2) / 2
        assert dict(result)[0] == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("method", ["auto", "coherent", "coherent_approx", "coherent_exact"])
@pytest.mark.parametrize("parameter,limit,fraction", [("p1", 0.75, "3/4"), ("p2", 0.9375, "15/16")])
def test_coherent_measurement_sampling_validates_generator_domain(method, parameter, limit, fraction):
    noise = exp.depolarizing().idle_rz(0.125)
    noise = getattr(noise, parameter)(math.nextafter(limit, math.inf))
    simulation = (
        exp.sim_neo(_circuit())
        .quantum(exp.meas_sampling(method))
        .noise(noise)
        .sampling(exp.monte_carlo(1))
    )
    with pytest.raises(ValueError, match=f"generator view requires {parameter} <= {fraction}"):
        simulation.run()


@pytest.mark.parametrize("method", ["auto", "stochastic", "coherent", "coherent_approx", "coherent_exact"])
def test_measurement_sampling_without_coherent_noise_keeps_categorical_domain(method):
    result = (
        exp.sim_neo(_circuit())
        .quantum(exp.meas_sampling(method))
        .noise(exp.depolarizing().p1(0.9).p2(1.0))
        .sampling(exp.monte_carlo(1))
        .seed(1202)
        .run()
    )
    assert len(result) == 1
