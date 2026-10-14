"""Coherent raw rows preserve surface-code detector marginals and correlations."""

import json

import numpy as np
import pecos_rslib_exp as exp
import pytest
from pecos.qec.surface import SurfacePatch
from pecos.qec.surface.decode import _build_surface_tick_circuit_for_native_model

SHOTS = 40_000


def _detector_statistics(circuit, method):
    noise = exp.depolarizing().p1(1e-3).p2(1e-3).p_meas(1e-3).p_prep(1e-3)
    if method != "stochastic":
        noise = noise.idle_rz(1e-4)
    rows = np.asarray(
        [
            list(row)
            for row in exp.sim_neo(circuit)
            .quantum(exp.meas_sampling(method))
            .noise(noise)
            .sampling(exp.monte_carlo(SHOTS))
            .seed(7)
            .run()
        ],
        dtype=np.int64,
    )
    definitions = json.loads(circuit.get_meta("detectors"))
    events = np.column_stack(
        [rows[:, [rows.shape[1] + rec for rec in definition["records"]]].sum(axis=1) % 2 for definition in definitions],
    )
    # E[D_i D_j] is the pairwise detector-correlation estimate: each joint
    # firing is itself a Bernoulli observation, with a binomial standard error.
    joint = (events.T @ events) / SHOTS
    return events.mean(axis=0), joint[np.triu_indices(len(definitions), k=1)]


@pytest.fixture(scope="module")
def surface_reference():
    circuit = _build_surface_tick_circuit_for_native_model(
        SurfacePatch.create(distance=3),
        6,
        "Z",
        circuit_source="abstract",
    )
    definitions = json.loads(circuit.get_meta("detectors"))
    assert [len(definition["records"]) for definition in definitions[-4:]] == [3, 5, 5, 3]
    return circuit, _detector_statistics(circuit, "stochastic")


@pytest.mark.parametrize("method", ["coherent", "coherent_approx", "coherent_exact"])
def test_coherent_surface_detector_statistics(surface_reference, method, record_property):
    circuit, stochastic = surface_reference
    coherent = _detector_statistics(circuit, method)
    record_property("method", method)
    for statistic, actual, reference in zip(("rates", "joint_rates"), coherent, stochastic, strict=True):
        record_property(f"coherent_{statistic}", json.dumps(actual.tolist()))
        record_property(f"stochastic_{statistic}", json.dumps(reference.tolist()))
    for statistic, actual, reference in zip(("rates", "joint_rates"), coherent, stochastic, strict=True):
        # Six standard errors of the difference cover all 52 marginals and
        # 1326 pairs with a wide multiple-comparison margin. Six counts also
        # cover the discrete, very rare-event regime. At 40k shots this still
        # rejects suppressing the ~1% physical noise, as well as random rows.
        standard_error = np.sqrt((actual * (1 - actual) + reference * (1 - reference)) / SHOTS)
        tolerance = 6 * standard_error + 6 / SHOTS
        assert np.all(np.abs(actual - reference) <= tolerance), (
            method,
            statistic,
            np.max(np.abs(actual - reference) - tolerance),
            actual,
            reference,
        )
