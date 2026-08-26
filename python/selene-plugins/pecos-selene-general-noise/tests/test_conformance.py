"""Device-neutral behavioral conformance tests for PECOS general noise."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest
from general_noise_conformance import ConformanceExperiment, ExpectedDistribution
from guppylang import guppy
from guppylang.std.angles import pi
from guppylang.std.builtins import result
from guppylang.std.quantum import cx, measure, qubit, rz, x
from pecos_selene_general_noise import GeneralNoiseParameters
from selene_sim import Stim
from selene_sim.build import build

if TYPE_CHECKING:
    from selene_sim.instance import SeleneInstance


@guppy
def prepared_zero() -> None:
    """Prepare and measure one computational-basis qubit."""
    q = qubit()
    result("outcome", measure(q).read())


@guppy
def prepared_one() -> None:
    """Prepare one, then measure it to exercise asymmetric readout."""
    q = qubit()
    x(q)
    result("outcome", measure(q).read())


@guppy
def one_qubit_gate() -> None:
    """Apply a diagonal 1Q gate so an injected X is directly observable."""
    q = qubit()
    rz(q, pi / 2)
    result("outcome", measure(q).read())


@guppy
def emission_replacement_probe() -> None:
    """Apply an X whose removal is distinguishable from an emitted Z."""
    q = qubit()
    x(q)
    result("outcome", measure(q).read())


@guppy
def two_qubit_gate() -> None:
    """Apply a standard entangling operation and measure both qubits."""
    q0 = qubit()
    q1 = qubit()
    cx(q0, q1)
    result("q0", measure(q0).read())
    result("q1", measure(q1).read())


@guppy
def measurement_crosstalk_probe() -> None:
    """Prepare both qubits before measuring a source and its crosstalk victim."""
    source = qubit()
    victim = qubit()
    cx(source, victim)
    trigger = measure(source).read()
    if trigger:
        x(victim)
    result("source", trigger)
    result("victim", measure(victim).read())


ZERO = ExpectedDistribution({(0,): 1.0})
ONE = ExpectedDistribution({(1,): 1.0})
ZERO_ZERO = ExpectedDistribution({(0, 0): 1.0})


@pytest.mark.parametrize(
    ("parameters", "expected"),
    [
        pytest.param(GeneralNoiseParameters().with_p_prep(0.3), {(0,): 0.7, (1,): 0.3}, id="preparation"),
        pytest.param(GeneralNoiseParameters().with_p_meas_0(0.3), {(0,): 0.7, (1,): 0.3}, id="readout-0"),
    ],
)
def test_single_qubit_bernoulli_channels(
    parameters: GeneralNoiseParameters,
    expected: dict[tuple[int, ...], float],
) -> None:
    """Preparation and readout rates agree with their Bernoulli specifications."""
    experiment = ConformanceExperiment(
        runner=build(prepared_zero.compile()),
        n_qubits=1,
        result_tags=("outcome",),
        parameters=parameters,
        expected=ExpectedDistribution(expected),
        comparison=ZERO,
        shots=512,
    )
    experiment.assert_conforms(Stim(random_seed=23), n_processes=2)


def test_one_to_zero_readout_channel() -> None:
    """The 1-to-0 readout parameter is independently observable."""
    experiment = ConformanceExperiment(
        runner=build(prepared_one.compile()),
        n_qubits=1,
        result_tags=("outcome",),
        parameters=GeneralNoiseParameters().with_p_meas_1(0.3),
        expected=ExpectedDistribution({(0,): 0.3, (1,): 0.7}),
        comparison=ONE,
        shots=512,
    )
    experiment.assert_conforms(Stim(random_seed=23), n_processes=2)


def test_one_to_zero_readout_is_applied_after_leakage() -> None:
    """A leaked Boolean measurement is forced to one before readout noise."""
    parameters = GeneralNoiseParameters().with_p_prep(1.0).with_prep_leak_ratio(1.0).with_p_meas_1(1.0)
    experiment = ConformanceExperiment(
        runner=build(prepared_zero.compile()),
        n_qubits=1,
        result_tags=("outcome",),
        parameters=parameters,
        expected=ZERO,
        comparison=ONE,
        shots=32,
        seed=29,
    )
    experiment.assert_conforms(Stim(random_seed=31))


@pytest.mark.parametrize(
    ("runner", "expected"),
    [
        pytest.param(build(prepared_zero.compile()), {(0,): 0.7, (1,): 0.3}, id="zero"),
        pytest.param(build(prepared_one.compile()), {(0,): 0.3, (1,): 0.7}, id="one"),
    ],
)
def test_symmetric_measurement_alias(
    runner: SeleneInstance,
    expected: dict[tuple[int, ...], float],
) -> None:
    """The symmetric measurement convenience method configures both transitions."""
    experiment = ConformanceExperiment(
        runner=runner,
        n_qubits=1,
        result_tags=("outcome",),
        parameters=GeneralNoiseParameters().with_p_meas(0.3),
        expected=ExpectedDistribution(expected),
        comparison=ZERO if expected[(0,)] < 0.5 else ONE,
        shots=512,
    )
    experiment.assert_conforms(Stim(random_seed=23), n_processes=2)


@pytest.mark.parametrize(
    "parameters",
    [
        pytest.param(GeneralNoiseParameters().with_p1(0.3), id="process-infidelity"),
        pytest.param(GeneralNoiseParameters().with_average_p1(0.2), id="average-infidelity"),
    ],
)
def test_single_qubit_error_probability(parameters: GeneralNoiseParameters) -> None:
    """Process and average infidelity spellings resolve to the documented fault rate."""
    parameters = parameters.with_p1_pauli_model({"X": 1.0})
    experiment = ConformanceExperiment(
        runner=build(one_qubit_gate.compile()),
        n_qubits=1,
        result_tags=("outcome",),
        parameters=parameters,
        expected=ExpectedDistribution({(0,): 0.7, (1,): 0.3}),
        comparison=ZERO,
        shots=512,
    )
    experiment.assert_conforms(Stim(random_seed=23), n_processes=2)


@pytest.mark.parametrize(
    "parameters",
    [
        pytest.param(
            GeneralNoiseParameters().with_p1(1.0).with_p1_pauli_model({"X": 1.0}),
            id="pauli",
        ),
        pytest.param(
            GeneralNoiseParameters().with_p1(1.0).with_p1_emission_ratio(1.0).with_p1_emission_model({"X": 1.0}),
            id="emission",
        ),
    ],
)
def test_one_qubit_fault_families(parameters: GeneralNoiseParameters) -> None:
    """Both Pauli and non-leaking emission paths reach the simulator correctly."""
    experiment = ConformanceExperiment(
        runner=build(one_qubit_gate.compile()),
        n_qubits=1,
        result_tags=("outcome",),
        parameters=parameters,
        expected=ONE,
        comparison=ZERO,
        shots=64,
    )
    experiment.assert_conforms(Stim(random_seed=23))


@pytest.mark.parametrize(
    "parameters",
    [
        pytest.param(
            GeneralNoiseParameters().with_p_idle_linear(1.0, {"X": 1.0}).with_idle_after_2q(1.0),
            id="linear",
        ),
        pytest.param(
            GeneralNoiseParameters().with_p_idle_sin_squared(math.pi / 2.0, {"X": 1.0}).with_idle_after_2q(1.0),
            id="sine-squared",
        ),
        pytest.param(
            GeneralNoiseParameters().with_p_idle_coherent(math.pi, {"RX": 1.0}).with_idle_after_2q(1.0),
            id="coherent",
        ),
    ],
)
def test_post_two_qubit_idle_families(parameters: GeneralNoiseParameters) -> None:
    """Every current PECOS idle family traverses the Selene bridge."""
    experiment = ConformanceExperiment(
        runner=build(two_qubit_gate.compile()),
        n_qubits=2,
        result_tags=("q0", "q1"),
        parameters=parameters,
        expected=ExpectedDistribution({(1, 1): 1.0}),
        comparison=ZERO_ZERO,
        shots=64,
    )
    experiment.assert_conforms(Stim(random_seed=23))


def test_seeded_experiment_is_reproducible() -> None:
    """Identical Selene component seeds reproduce the complete shot sequence."""
    experiment = ConformanceExperiment(
        runner=build(prepared_zero.compile()),
        n_qubits=1,
        result_tags=("outcome",),
        parameters=GeneralNoiseParameters().with_p_prep(0.3),
        expected=ExpectedDistribution({(0,): 0.7, (1,): 0.3}),
        comparison=ZERO,
        shots=128,
    )
    first = experiment.sample(Stim(random_seed=23))
    second = experiment.sample(Stim(random_seed=23))
    assert first.outcomes == second.outcomes


@pytest.mark.parametrize(
    ("parameters", "expected"),
    [
        pytest.param(
            GeneralNoiseParameters().with_p2(1.0).with_p2_pauli_model({"XI": 1.0}),
            ExpectedDistribution({(1, 0): 1.0}),
            id="pauli",
        ),
        pytest.param(
            GeneralNoiseParameters().with_average_p2(0.8).with_p2_pauli_model({"XI": 1.0}),
            ExpectedDistribution({(1, 0): 1.0}),
            id="average-infidelity",
        ),
        pytest.param(
            GeneralNoiseParameters().with_p2(1.0).with_p2_emission_ratio(1.0).with_p2_emission_model({"XI": 1.0}),
            ExpectedDistribution({(1, 0): 0.5, (1, 1): 0.5}),
            id="emission",
        ),
        pytest.param(
            GeneralNoiseParameters().with_p2(1.0).with_p2_emission_ratio(1.0).with_p2_emission_model({"IL": 1.0}),
            ExpectedDistribution({(0, 1): 1.0}),
            id="emission-leak-second",
        ),
        pytest.param(
            GeneralNoiseParameters().with_p2(1.0).with_p2_emission_ratio(1.0).with_p2_emission_model({"LI": 1.0}),
            ExpectedDistribution({(1, 0): 0.5, (1, 1): 0.5}),
            id="emission-leak-first",
        ),
        pytest.param(
            GeneralNoiseParameters().with_p2(1.0).with_p2_emission_ratio(1.0).with_p2_emission_model({"LL": 1.0}),
            ExpectedDistribution({(1, 1): 1.0}),
            id="emission-leak-both",
        ),
    ],
)
def test_two_qubit_fault_families(
    parameters: GeneralNoiseParameters,
    expected: ExpectedDistribution,
) -> None:
    """Two-qubit Pauli, emission, and leakage channels preserve target ordering."""
    experiment = ConformanceExperiment(
        runner=build(two_qubit_gate.compile()),
        n_qubits=2,
        result_tags=("q0", "q1"),
        parameters=parameters,
        expected=expected,
        comparison=ZERO_ZERO,
        shots=512,
    )
    experiment.assert_conforms(Stim(random_seed=23), n_processes=2)


def test_angle_scaling_can_suppress_two_qubit_noise() -> None:
    """Angle coefficients affect the RZZ fault rate used by a compiled entangler."""
    parameters = (
        GeneralNoiseParameters().with_p2(1.0).with_p2_pauli_model({"XI": 1.0}).with_p2_angle_params(0.0, 0.0, 0.0, 0.0)
    )
    experiment = ConformanceExperiment(
        runner=build(two_qubit_gate.compile()),
        n_qubits=2,
        result_tags=("q0", "q1"),
        parameters=parameters,
        expected=ZERO_ZERO,
        comparison=ExpectedDistribution({(1, 0): 1.0}),
        shots=64,
    )
    experiment.assert_conforms(Stim(random_seed=23))


def test_angle_power_controls_two_qubit_fault_probability() -> None:
    """The angle exponent produces the analytic half-rate for a compiled pi/2 RZZ."""
    parameters = (
        GeneralNoiseParameters()
        .with_p2(1.0)
        .with_p2_pauli_model({"XI": 1.0})
        .with_p2_angle_params(1.0, 0.0, 1.0, 0.0)
        .with_p2_angle_power(1.0)
    )
    experiment = ConformanceExperiment(
        runner=build(two_qubit_gate.compile()),
        n_qubits=2,
        result_tags=("q0", "q1"),
        parameters=parameters,
        expected=ExpectedDistribution({(0, 0): 0.5, (1, 0): 0.5}),
        comparison=ZERO_ZERO,
        shots=512,
    )
    experiment.assert_conforms(Stim(random_seed=23), n_processes=2)


@pytest.mark.parametrize(
    "parameters",
    [
        pytest.param(
            GeneralNoiseParameters().with_p1(1.0).with_p1_pauli_model({"X": 1.0}).with_scale(0.0),
            id="global-scale",
        ),
        pytest.param(
            GeneralNoiseParameters().with_p1(1.0).with_p1_pauli_model({"X": 1.0}).with_p1_scale(0.0),
            id="one-qubit-scale",
        ),
        pytest.param(
            GeneralNoiseParameters().with_p1(1.0).with_p1_pauli_model({"X": 1.0}).with_noiseless_gate("RZ"),
            id="noiseless-gate",
        ),
    ],
)
def test_noise_suppression_controls(parameters: GeneralNoiseParameters) -> None:
    """Global, family, and gate-level controls can each suppress an otherwise certain fault."""
    experiment = ConformanceExperiment(
        runner=build(one_qubit_gate.compile()),
        n_qubits=1,
        result_tags=("outcome",),
        parameters=parameters,
        expected=ZERO,
        comparison=ONE,
        shots=64,
    )
    experiment.assert_conforms(Stim(random_seed=23))


@pytest.mark.parametrize(
    "parameters",
    [
        pytest.param(GeneralNoiseParameters().with_p_prep(1.0).with_prep_scale(0.0), id="preparation-scale"),
        pytest.param(GeneralNoiseParameters().with_p_meas_0(1.0).with_meas_scale(0.0), id="measurement-scale"),
    ],
)
def test_spam_scale_controls(parameters: GeneralNoiseParameters) -> None:
    """Preparation and measurement scales suppress their respective channels."""
    experiment = ConformanceExperiment(
        runner=build(prepared_zero.compile()),
        n_qubits=1,
        result_tags=("outcome",),
        parameters=parameters,
        expected=ZERO,
        comparison=ONE,
        shots=64,
    )
    experiment.assert_conforms(Stim(random_seed=23))


def test_emission_scale_changes_fault_family() -> None:
    """Zero emission scale selects the configured Pauli path instead of emission."""
    parameters = (
        GeneralNoiseParameters()
        .with_p1(1.0)
        .with_p1_pauli_model({"Z": 1.0})
        .with_p1_emission_ratio(1.0)
        .with_p1_emission_model({"X": 1.0})
        .with_emission_scale(0.0)
    )
    experiment = ConformanceExperiment(
        runner=build(one_qubit_gate.compile()),
        n_qubits=1,
        result_tags=("outcome",),
        parameters=parameters,
        expected=ZERO,
        comparison=ONE,
        shots=64,
    )
    experiment.assert_conforms(Stim(random_seed=23))


def test_one_qubit_emission_replaces_original_gate() -> None:
    """The emission branch substitutes its fault instead of following the ideal gate."""
    parameters = GeneralNoiseParameters().with_p1(1.0).with_p1_emission_ratio(1.0).with_p1_emission_model({"Z": 1.0})
    experiment = ConformanceExperiment(
        runner=build(emission_replacement_probe.compile()),
        n_qubits=1,
        result_tags=("outcome",),
        parameters=parameters,
        expected=ZERO,
        comparison=ONE,
        shots=64,
    )
    experiment.assert_conforms(Stim(random_seed=23))


@pytest.mark.parametrize(
    "parameters",
    [
        pytest.param(
            GeneralNoiseParameters().with_p2(1.0).with_p2_pauli_model({"XI": 1.0}).with_p2_scale(0.0),
            id="two-qubit-scale",
        ),
        pytest.param(
            GeneralNoiseParameters().with_p_idle_linear(1.0, {"X": 1.0}).with_idle_after_2q(1.0).with_idle_scale(0.0),
            id="linear-idle-scale",
        ),
        pytest.param(
            GeneralNoiseParameters()
            .with_p_idle_sin_squared(math.pi / 2.0, {"X": 1.0})
            .with_idle_after_2q(1.0)
            .with_idle_scale(0.0),
            id="sine-squared-idle-scale",
        ),
        pytest.param(
            GeneralNoiseParameters()
            .with_p_idle_coherent(math.pi, {"RX": 1.0})
            .with_idle_after_2q(1.0)
            .with_idle_scale(0.0),
            id="coherent-idle-scale",
        ),
    ],
)
def test_two_qubit_and_idle_scale_controls(parameters: GeneralNoiseParameters) -> None:
    """Two-qubit and idle family scales suppress their observable faults."""
    experiment = ConformanceExperiment(
        runner=build(two_qubit_gate.compile()),
        n_qubits=2,
        result_tags=("q0", "q1"),
        parameters=parameters,
        expected=ZERO_ZERO,
        comparison=ExpectedDistribution({(1, 0): 1.0}),
        shots=64,
    )
    experiment.assert_conforms(Stim(random_seed=23))


def test_measurement_crosstalk_scale_can_disable_channel() -> None:
    """The measurement-crosstalk scale is independently observable."""
    parameters = (
        GeneralNoiseParameters()
        .with_p_meas_crosstalk_global(1.0)
        .with_p_meas_crosstalk_model({"0->1": 1.0, "1->0": 1.0})
        .with_p_meas_crosstalk_scale(0.0)
    )
    experiment = ConformanceExperiment(
        runner=build(measurement_crosstalk_probe.compile()),
        n_qubits=2,
        result_tags=("source", "victim"),
        parameters=parameters,
        expected=ZERO_ZERO,
        comparison=ExpectedDistribution({(0, 1): 1.0}),
        shots=64,
    )
    experiment.assert_conforms(Stim(random_seed=23))


def test_preparation_leakage_skips_gates() -> None:
    """A leaked qubit ignores a following X and has the documented normal-readout value."""
    parameters = GeneralNoiseParameters().with_p_prep(1.0).with_prep_leak_ratio(1.0)
    experiment = ConformanceExperiment(
        runner=build(prepared_one.compile()),
        n_qubits=1,
        result_tags=("outcome",),
        parameters=parameters,
        expected=ONE,
        comparison=ZERO,
        shots=64,
    )
    experiment.assert_conforms(Stim(random_seed=23))


@pytest.mark.parametrize(
    "parameters",
    [
        pytest.param(
            GeneralNoiseParameters()
            .with_p_prep(1.0)
            .with_prep_leak_ratio(1.0)
            .with_p1(1.0)
            .with_p1_emission_ratio(1.0)
            .with_p1_seepage_prob(1.0),
            id="one-qubit-specific",
        ),
        pytest.param(
            GeneralNoiseParameters()
            .with_p_prep(1.0)
            .with_prep_leak_ratio(1.0)
            .with_p1(1.0)
            .with_p1_emission_ratio(1.0)
            .with_seepage_prob(1.0),
            id="shared-alias",
        ),
    ],
)
def test_seepage_releases_a_leaked_qubit(parameters: GeneralNoiseParameters) -> None:
    """Certain seepage resets a leaked qubit to the documented random bit."""
    experiment = ConformanceExperiment(
        runner=build(prepared_one.compile()),
        n_qubits=1,
        result_tags=("outcome",),
        parameters=parameters,
        expected=ExpectedDistribution({(0,): 0.5, (1,): 0.5}),
        comparison=ONE,
        shots=512,
    )
    experiment.assert_conforms(Stim(random_seed=23), n_processes=2)


def test_two_qubit_seepage_releases_leaked_pair() -> None:
    """Two-qubit seepage independently returns both leaked operands to random bits."""
    parameters = (
        GeneralNoiseParameters()
        .with_p_prep(1.0)
        .with_prep_leak_ratio(1.0)
        .with_p2(1.0)
        .with_p2_emission_ratio(1.0)
        .with_p2_seepage_prob(1.0)
    )
    experiment = ConformanceExperiment(
        runner=build(two_qubit_gate.compile()),
        n_qubits=2,
        result_tags=("q0", "q1"),
        parameters=parameters,
        expected=ExpectedDistribution({(0, 0): 0.25, (0, 1): 0.25, (1, 0): 0.25, (1, 1): 0.25}),
        comparison=ExpectedDistribution({(1, 1): 1.0}),
        shots=1024,
    )
    experiment.assert_conforms(Stim(random_seed=23), n_processes=2)


def test_leakage_scale_converts_leakage_to_depolarization() -> None:
    """Zero leakage scale converts an otherwise certain leak to a random bit."""
    parameters = GeneralNoiseParameters().with_p_prep(1.0).with_prep_leak_ratio(1.0).with_leakage_scale(0.0)
    experiment = ConformanceExperiment(
        runner=build(prepared_zero.compile()),
        n_qubits=1,
        result_tags=("outcome",),
        parameters=parameters,
        expected=ExpectedDistribution({(0,): 0.5, (1,): 0.5}),
        comparison=ONE,
        shots=512,
    )
    experiment.assert_conforms(Stim(random_seed=23), n_processes=2)


@pytest.mark.parametrize(
    "parameters",
    [
        pytest.param(
            GeneralNoiseParameters()
            .with_p_meas_crosstalk_global(1.0)
            .with_p_meas_crosstalk_model({"0->1": 1.0, "1->0": 1.0}),
            id="global",
        ),
        pytest.param(
            GeneralNoiseParameters()
            .with_p_meas_crosstalk_local(1.0)
            .with_p_meas_crosstalk_model({"0->1": 1.0, "1->0": 1.0})
            .with_local_crosstalk_groups((0, 1)),
            id="local-group",
        ),
    ],
)
def test_measurement_crosstalk_is_device_neutral(parameters: GeneralNoiseParameters) -> None:
    """Global and user-defined local crosstalk project an unmeasured victim."""
    experiment = ConformanceExperiment(
        runner=build(measurement_crosstalk_probe.compile()),
        n_qubits=2,
        result_tags=("source", "victim"),
        parameters=parameters,
        expected=ExpectedDistribution({(0, 1): 1.0}),
        comparison=ZERO_ZERO,
        shots=64,
    )
    experiment.assert_conforms(Stim(random_seed=23))


def test_measurement_crosstalk_alias_enables_global_channel() -> None:
    """The shared crosstalk convenience method includes the global channel."""
    parameters = (
        GeneralNoiseParameters().with_p_meas_crosstalk(1.0).with_p_meas_crosstalk_model({"0->1": 1.0, "1->0": 1.0})
    )
    experiment = ConformanceExperiment(
        runner=build(measurement_crosstalk_probe.compile()),
        n_qubits=2,
        result_tags=("source", "victim"),
        parameters=parameters,
        expected=ExpectedDistribution({(0, 1): 1.0}),
        comparison=ZERO_ZERO,
        shots=64,
    )
    experiment.assert_conforms(Stim(random_seed=23))


def test_conformance_case_runs_with_pecos_statevec() -> None:
    """The framework and native error model are independent of simulator implementation."""
    statevec_module = pytest.importorskip("pecos_selene_statevec")
    parameters = GeneralNoiseParameters().with_p1(1.0).with_p1_pauli_model({"X": 1.0})
    experiment = ConformanceExperiment(
        runner=build(one_qubit_gate.compile()),
        n_qubits=1,
        result_tags=("outcome",),
        parameters=parameters,
        expected=ONE,
        comparison=ZERO,
        shots=64,
    )
    experiment.assert_conforms(statevec_module.StateVecPlugin(random_seed=23))


def test_sensitivity_guard_rejects_a_vacuous_reference_case() -> None:
    """A circuit cannot claim coverage when its noise and comparison distributions coincide."""
    with pytest.raises(AssertionError, match="not sensitive enough"):
        ZERO.assert_sensitive_to(ZERO, shots=512)


@pytest.mark.parametrize(
    ("probabilities", "error"),
    [
        pytest.param({}, "cannot be empty", id="empty"),
        pytest.param({(0,): 0.4, (1,): 0.4}, "sum to one", id="not-normalized"),
        pytest.param({(0,): -0.1, (1,): 1.1}, "between zero and one", id="out-of-range"),
    ],
)
def test_reference_distribution_validation(
    probabilities: dict[tuple[int, ...], float],
    error: str,
) -> None:
    """Malformed analytic references fail before any expensive simulation runs."""
    with pytest.raises(ValueError, match=error):
        ExpectedDistribution(probabilities)
