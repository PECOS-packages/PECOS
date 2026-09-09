"""Tests for the public general-noise configuration API."""

import inspect
import json
import re
from collections.abc import Callable
from pathlib import Path

import pytest
from guppylang import guppy
from guppylang.std.angles import pi
from guppylang.std.builtins import result
from guppylang.std.quantum import measure, qubit, rz
from pecos_selene_general_noise import (
    GateNoise,
    GeneralNoiseParameters,
    GeneralNoisePlugin,
    IdleNoise,
    MeasurementNoise,
    NoiseScaling,
    PreparationNoise,
    TwoQubitGateNoise,
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


@pytest.mark.parametrize(
    "parameters",
    [
        pytest.param(
            lambda: GeneralNoiseParameters(preparation=PreparationNoise(probability=-0.01)),
            id="negative-probability",
        ),
        pytest.param(
            lambda: GeneralNoiseParameters(measurement=MeasurementNoise(p0_to_1=1.01)),
            id="probability-above-one",
        ),
        pytest.param(
            lambda: GeneralNoiseParameters(single_qubit=GateNoise(probability=float("nan"))),
            id="nan-probability",
        ),
        pytest.param(
            lambda: GeneralNoiseParameters(two_qubit=TwoQubitGateNoise(angle_power=0.0)),
            id="non-positive-angle-power",
        ),
        pytest.param(
            lambda: GeneralNoiseParameters(idle=IdleNoise(linear_rate=float("inf"))),
            id="infinite-idle-rate",
        ),
        pytest.param(
            lambda: GeneralNoiseParameters(scaling=NoiseScaling(overall=-1.0)),
            id="negative-scale",
        ),
    ],
)
def test_rejects_non_physical_numeric_boundaries(parameters: Callable[[], GeneralNoiseParameters]) -> None:
    """Every numeric family rejects non-finite or out-of-domain values."""
    with pytest.raises(ValueError, match="must"):
        parameters()


@pytest.mark.parametrize(
    "parameters",
    [
        pytest.param(
            lambda: GeneralNoiseParameters(single_qubit=GateNoise(pauli_model={"A": 1.0})),
            id="invalid-one-qubit-key",
        ),
        pytest.param(
            lambda: GeneralNoiseParameters(single_qubit=GateNoise(pauli_model={})),
            id="empty-distribution",
        ),
        pytest.param(
            lambda: GeneralNoiseParameters(two_qubit=TwoQubitGateNoise(emission_model={"III": 1.0})),
            id="invalid-two-qubit-key",
        ),
        pytest.param(
            lambda: GeneralNoiseParameters(idle=IdleNoise(coherent_model={"X": 1.0})),
            id="invalid-coherent-axis",
        ),
        pytest.param(
            lambda: GeneralNoiseParameters(idle=IdleNoise(linear_model={"X": 1.0})),
            id="idle-model-without-rate",
        ),
        pytest.param(
            lambda: GeneralNoiseParameters(measurement=MeasurementNoise(crosstalk_model={"0->0": 1.0, "1->0": 0.5})),
            id="incomplete-transition-model",
        ),
        pytest.param(
            lambda: GeneralNoiseParameters(measurement=MeasurementNoise(local_groups=((0, -1),))),
            id="negative-local-qubit",
        ),
    ],
)
def test_rejects_malformed_channel_shapes(parameters: Callable[[], GeneralNoiseParameters]) -> None:
    """Channel keys, transition rows, and abstract topology are validated early."""
    with pytest.raises(ValueError, match=r"unsupported|cannot|requires|must sum"):
        parameters()


def test_rejects_average_infidelity_outside_supported_conversions() -> None:
    """Average-infidelity setters enforce the dimensional conversion domains."""
    with pytest.raises(ValueError, match="2/3"):
        GeneralNoiseParameters().with_average_p1(2.0 / 3.0 + 1e-9)
    with pytest.raises(ValueError, match=r"0\.8"):
        GeneralNoiseParameters().with_average_p2(0.8 + 1e-9)
    with pytest.raises(ValueError, match="5/18"):
        GeneralNoiseParameters().with_average_p_prep_crosstalk(5.0 / 18.0 + 1e-9)


def test_rejects_invalid_plugin_seed() -> None:
    """Selene-owned randomness rejects a negative seed before native loading."""
    with pytest.raises(ValueError, match="non-negative"):
        GeneralNoisePlugin(random_seed=-1)


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


def test_channel_models_are_deeply_immutable() -> None:
    """Input dictionaries cannot mutate a validated reusable configuration."""
    model = {"X": 1.0}
    parameters = GeneralNoiseParameters().with_p1_pauli_model(model)
    model.clear()
    model["BAD"] = 2.0
    assert parameters.single_qubit.pauli_model == {"X": 1.0}
    with pytest.raises(TypeError):
        parameters.single_qubit.pauli_model["X"] = 0.5  # type: ignore[index]


@pytest.mark.parametrize(
    "parameters",
    [
        pytest.param(lambda: GeneralNoiseParameters().with_p1(0.8).with_scale(2.0), id="global-p1-scale"),
        pytest.param(
            lambda: GeneralNoiseParameters().with_p_meas_0(0.6).with_meas_scale(2.0),
            id="measurement-scale",
        ),
        pytest.param(
            lambda: GeneralNoiseParameters().with_p2(0.8).with_p2_angle_params(1.0, 1.0, 0.0, 1.0),
            id="angle-scaling",
        ),
    ],
)
def test_rejects_out_of_range_effective_probabilities(parameters: Callable[[], GeneralNoiseParameters]) -> None:
    """Cross-field scaling may not escape PECOS's probability domain."""
    with pytest.raises(ValueError, match=r"effective .* probability"):
        GeneralNoisePlugin(parameters=parameters())


def test_compensating_scales_are_order_independent() -> None:
    """Fluent construction may pass through an invalid intermediate scale."""
    parameters = GeneralNoiseParameters().with_p1(0.8).with_scale(2.0).with_p1_scale(0.5)
    GeneralNoisePlugin(parameters=parameters)


def test_matches_all_configurable_pecos_builder_setters() -> None:
    workspace = Path(__file__).parents[4]
    builder_source = (workspace / "crates/pecos-engines/src/noise/general/builder.rs").read_text()
    rust_setters = set(re.findall(r"pub fn (with_[a-zA-Z0-9_]+)", builder_source)) - {"with_seed"}
    python_setters = {
        name
        for name, member in inspect.getmembers(GeneralNoiseParameters, predicate=inspect.isfunction)
        if name.startswith("with_")
    }
    assert rust_setters == python_setters - {"with_local_crosstalk_groups"}


def test_plugin_executes_in_selene() -> None:
    """A deterministic preparation fault traverses both plugin boundaries."""

    @guppy
    def main() -> None:
        q = qubit()
        outcome = measure(q).read()
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
        outcome = measure(q).read()
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
        outcome = measure(q).read()
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
        outcome = measure(q).read()
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
        outcome = measure(q).read()
        result("outcome", outcome)

    runner = build(main.compile())
    parameters = GeneralNoiseParameters().with_p1(1.0).with_p1_pauli_model({"X": 1.0})
    noise = GeneralNoisePlugin(parameters=parameters, random_seed=7)
    results = dict(runner.run(Stim(random_seed=11), n_qubits=1, error_model=noise))
    assert results["outcome"] == 1
