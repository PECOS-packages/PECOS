"""Tests for the public general-noise configuration API."""

import json

import pytest
from guppylang import guppy
from guppylang.std.angles import pi
from guppylang.std.builtins import result
from guppylang.std.quantum import measure, qubit, rz
from pecos_selene_general_noise import (
    GateNoise,
    GeneralNoiseParameters,
    GeneralNoisePlugin,
)
from selene_sim import Stim
from selene_sim.build import build


def config(parameters: GeneralNoiseParameters) -> dict[str, object]:
    return json.loads(GeneralNoisePlugin(parameters=parameters).get_init_args()[0])


def test_default_is_explicitly_noiseless() -> None:
    payload = config(GeneralNoiseParameters())
    assert payload["preparation"]["probability"] is None
    assert payload["single_qubit"]["probability"] is None
    assert payload["two_qubit"]["probability"] is None


def test_uniform_convenience_model() -> None:
    payload = config(GeneralNoiseParameters.uniform(0.001))
    assert payload["preparation"]["probability"] == 0.001
    assert payload["measurement"]["p0_to_1"] == 0.001
    assert payload["measurement"]["p1_to_0"] == 0.001
    assert payload["single_qubit"]["probability"] == 0.001
    assert payload["two_qubit"]["probability"] == 0.001


def test_exposes_current_pecos_capabilities() -> None:
    parameters = (
        GeneralNoiseParameters()
        .with_average_p1(1e-4)
        .with_p2(2e-3)
        .with_p2_angle_params(1.0, 0.0, 1.0, 0.0)
        .with_idle_after_2q(5e-6)
        .with_p_idle_linear(0.01, {"Z": 1.0})
        .with_p_idle_sin_squared(0.02, {"Z": 1.0})
        .with_p_idle_coherent(0.03, {"RZ": 1.0})
        .with_local_crosstalk_groups((0, 1), (2, 3))
    )
    payload = config(parameters)
    assert payload["single_qubit"]["average_infidelity"] == 1e-4
    assert payload["idle"]["coherent_rate"] == 0.03
    assert payload["measurement"]["local_groups"] == [[0, 1], [2, 3]]


def test_rejects_ambiguous_infidelity() -> None:
    with pytest.raises(ValueError, match="not both"):
        GeneralNoiseParameters(single_qubit=GateNoise(probability=0.1, average_infidelity=0.1))


def test_rejects_bad_distribution() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        GeneralNoiseParameters(single_qubit=GateNoise(pauli_model={"X": 0.2, "Z": 0.2}))


def test_accepts_crosstalk_transition_model_per_input_state() -> None:
    model = {
        "0->0": 0.8,
        "0->1": 0.2,
        "1->0": 0.1,
        "1->1": 0.9,
    }
    parameters = GeneralNoiseParameters().with_p_meas_crosstalk_model(model)
    assert config(parameters)["measurement"]["crosstalk_model"] == model


def test_fluent_setters_are_immutable_and_last_infidelity_wins() -> None:
    original = GeneralNoiseParameters()
    updated = original.with_average_p1(0.01).with_p1(0.02)
    assert original.single_qubit.probability is None
    assert updated.single_qubit.probability == 0.02
    assert updated.single_qubit.average_infidelity is None


def test_matches_all_configurable_pecos_builder_setters() -> None:
    expected = {
        "with_average_p1",
        "with_average_p2",
        "with_average_p_prep_crosstalk",
        "with_emission_scale",
        "with_idle_after_2q",
        "with_idle_scale",
        "with_leakage_scale",
        "with_meas_scale",
        "with_noiseless_gate",
        "with_p1",
        "with_p1_emission_model",
        "with_p1_emission_ratio",
        "with_p1_pauli_model",
        "with_p1_scale",
        "with_p1_seepage_prob",
        "with_p2",
        "with_p2_angle_params",
        "with_p2_angle_power",
        "with_p2_emission_model",
        "with_p2_emission_ratio",
        "with_p2_pauli_model",
        "with_p2_scale",
        "with_p2_seepage_prob",
        "with_p_idle_coherent",
        "with_p_idle_linear",
        "with_p_idle_sin_squared",
        "with_p_meas",
        "with_p_meas_0",
        "with_p_meas_1",
        "with_p_meas_crosstalk",
        "with_p_meas_crosstalk_global",
        "with_p_meas_crosstalk_local",
        "with_p_meas_crosstalk_model",
        "with_p_meas_crosstalk_scale",
        "with_p_prep",
        "with_p_prep_crosstalk",
        "with_p_prep_crosstalk_scale",
        "with_prep_leak_ratio",
        "with_prep_scale",
        "with_scale",
        "with_seepage_prob",
    }
    assert expected <= set(dir(GeneralNoiseParameters))


def test_plugin_executes_in_selene() -> None:
    """A deterministic preparation fault traverses both plugin boundaries."""

    @guppy
    def main() -> None:
        q = qubit()
        outcome = measure(q)
        result("outcome", outcome)

    runner = build(main.compile())
    noise = GeneralNoisePlugin(
        parameters=GeneralNoiseParameters().with_p_prep(1.0),
        random_seed=7,
    )
    results = dict(runner.run(Stim(random_seed=11), n_qubits=1, error_model=noise))
    assert results["outcome"] == 1


def test_default_parameters_are_noiseless_in_selene() -> None:
    """Default parameters preserve a deterministic noiseless outcome."""

    @guppy
    def main() -> None:
        q = qubit()
        outcome = measure(q)
        result("outcome", outcome)

    runner = build(main.compile())
    noise = GeneralNoisePlugin(parameters=GeneralNoiseParameters(), random_seed=7)
    results = dict(runner.run(Stim(random_seed=11), n_qubits=1, error_model=noise))
    assert results["outcome"] == 0


def test_plugin_executes_with_pecos_statevec() -> None:
    """The error model composes with PECOS's in-tree Selene simulator."""
    statevec_module = pytest.importorskip("pecos_selene_statevec")

    @guppy
    def main() -> None:
        q = qubit()
        outcome = measure(q)
        result("outcome", outcome)

    runner = build(main.compile())
    noise = GeneralNoisePlugin(
        parameters=GeneralNoiseParameters().with_p_prep(1.0),
        random_seed=7,
    )
    simulator = statevec_module.StateVecPlugin(random_seed=11)
    results = dict(runner.run(simulator, n_qubits=1, error_model=noise))
    assert results["outcome"] == 1


def test_asymmetric_measurement_noise_executes_in_selene() -> None:
    """A configured 0-to-1 readout fault is deterministic."""

    @guppy
    def main() -> None:
        q = qubit()
        outcome = measure(q)
        result("outcome", outcome)

    runner = build(main.compile())
    noise = GeneralNoisePlugin(
        parameters=GeneralNoiseParameters().with_p_meas_0(1.0),
        random_seed=7,
    )
    results = dict(runner.run(Stim(random_seed=11), n_qubits=1, error_model=noise))
    assert results["outcome"] == 1


def test_custom_single_qubit_pauli_channel_executes_in_selene() -> None:
    """A deterministic Pauli model is honored after a single-qubit gate."""

    @guppy
    def main() -> None:
        q = qubit()
        rz(q, pi / 2)
        outcome = measure(q)
        result("outcome", outcome)

    runner = build(main.compile())
    parameters = GeneralNoiseParameters().with_p1(1.0).with_p1_pauli_model({"X": 1.0})
    noise = GeneralNoisePlugin(parameters=parameters, random_seed=7)
    results = dict(runner.run(Stim(random_seed=11), n_qubits=1, error_model=noise))
    assert results["outcome"] == 1
