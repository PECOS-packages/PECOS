"""Broader conformance tests driven by an independent exact reference model."""

from __future__ import annotations

import pytest
from basis_state_reference import BasisNoise, BasisStateReference
from general_noise_conformance import ConformanceExperiment, ExpectedDistribution
from guppylang import guppy
from guppylang.std.angles import pi
from guppylang.std.builtins import result
from guppylang.std.quantum import cz, h, measure, qubit, reset, rz, x
from pecos_selene_general_noise import GeneralNoiseParameters
from selene_sim import Stim
from selene_sim.build import build


@guppy
def mixed_basis_sequence() -> None:
    """Exercise three independently noisy gates while remaining in the Z basis."""
    q = qubit()
    x(q)
    rz(q, pi / 2)
    x(q)
    result("outcome", measure(q).read())


@guppy
def parallel_basis_sequences() -> None:
    """Run distinct basis-preserving sequences on two live qubits."""
    q0 = qubit()
    q1 = qubit()
    x(q0)
    x(q1)
    rz(q0, pi / 2)
    rz(q1, pi / 2)
    x(q0)
    result("q0", measure(q0).read())
    result("q1", measure(q1).read())


@guppy
def preparation_crosstalk_probe() -> None:
    """Reset one qubit after a harmless dependency on a live superposition."""
    victim = qubit()
    source = qubit()
    h(victim)
    cz(victim, source)
    reset(source)
    h(victim)
    result("victim", measure(victim).read())
    result("source", measure(source).read())


@pytest.mark.parametrize(
    ("error_seed", "simulator_seed"),
    [
        pytest.param(17, 31, id="primary"),
        pytest.param(101, 103, id="repeat-1", marks=pytest.mark.slow),
        pytest.param(211, 223, id="repeat-2", marks=pytest.mark.slow),
    ],
)
def test_combined_basis_channels_match_independent_reference(error_seed: int, simulator_seed: int) -> None:
    """Combined SPAM, Pauli, emission, leakage, and seepage match an exact oracle."""
    reference_noise = BasisNoise(
        preparation_probability=0.12,
        preparation_leakage_ratio=0.25,
        p1=0.2,
        p1_pauli_model=(("X", 0.5), ("Z", 0.5)),
        p1_emission_ratio=0.3,
        p1_emission_model=(("X", 0.5), ("L", 0.5)),
        p1_seepage_probability=0.4,
        measurement_0_to_1=0.1,
        measurement_1_to_0=0.2,
    )
    expected = (
        BasisStateReference(1, reference_noise)
        .reset(0)
        .one_qubit_gate(0, flips_basis=True)
        .one_qubit_gate(0, flips_basis=False)
        .one_qubit_gate(0, flips_basis=True)
        .measurement_distribution((0,))
    )
    parameters = (
        GeneralNoiseParameters()
        .with_p_prep(0.12)
        .with_prep_leak_ratio(0.25)
        .with_p1(0.2)
        .with_p1_pauli_model({"X": 0.5, "Z": 0.5})
        .with_p1_emission_ratio(0.3)
        .with_p1_emission_model({"X": 0.5, "L": 0.5})
        .with_p1_seepage_prob(0.4)
        .with_p_meas_0(0.1)
        .with_p_meas_1(0.2)
    )
    experiment = ConformanceExperiment(
        runner=build(mixed_basis_sequence.compile()),
        n_qubits=1,
        result_tags=("outcome",),
        parameters=parameters,
        expected=expected,
        comparison=ExpectedDistribution({(0,): 1.0}),
        shots=2048,
        seed=error_seed,
    )
    experiment.assert_conforms(Stim(random_seed=simulator_seed), n_processes=2)


def test_parallel_sequences_match_joint_reference_distribution() -> None:
    """The oracle checks joint behavior and batched one-qubit gate delivery."""
    reference_noise = BasisNoise(
        preparation_probability=0.1,
        p1=0.25,
        p1_pauli_model=(("X", 0.6), ("Z", 0.4)),
        measurement_0_to_1=0.05,
        measurement_1_to_0=0.15,
    )
    expected = (
        BasisStateReference(2, reference_noise)
        .reset(0)
        .reset(1)
        .one_qubit_gate(0, flips_basis=True)
        .one_qubit_gate(1, flips_basis=True)
        .one_qubit_gate(0, flips_basis=False)
        .one_qubit_gate(1, flips_basis=False)
        .one_qubit_gate(0, flips_basis=True)
        .measurement_distribution((0, 1))
    )
    parameters = (
        GeneralNoiseParameters()
        .with_p_prep(0.1)
        .with_p1(0.25)
        .with_p1_pauli_model({"X": 0.6, "Z": 0.4})
        .with_p_meas_0(0.05)
        .with_p_meas_1(0.15)
    )
    experiment = ConformanceExperiment(
        runner=build(parallel_basis_sequences.compile()),
        n_qubits=2,
        result_tags=("q0", "q1"),
        parameters=parameters,
        expected=expected,
        comparison=ExpectedDistribution({(0, 1): 1.0}),
        shots=2048,
        seed=307,
    )
    experiment.assert_conforms(Stim(random_seed=311), n_processes=2)


@pytest.mark.parametrize(
    "parameters",
    [
        pytest.param(GeneralNoiseParameters().with_p_prep_crosstalk(1.0), id="process-probability"),
        pytest.param(
            GeneralNoiseParameters().with_average_p_prep_crosstalk(5.0 / 18.0),
            id="average-probability",
        ),
    ],
)
@pytest.mark.noise_channel("preparation-crosstalk", oracle="analytic")
def test_preparation_crosstalk_is_device_neutral(parameters: GeneralNoiseParameters) -> None:
    """Preparation crosstalk follows public PECOS semantics without a device layout."""
    experiment = ConformanceExperiment(
        runner=build(preparation_crosstalk_probe.compile()),
        n_qubits=2,
        result_tags=("victim", "source"),
        parameters=parameters,
        expected=ExpectedDistribution({(0, 0): 0.5, (1, 0): 0.5}),
        comparison=ExpectedDistribution({(0, 0): 1.0}),
        shots=1024,
    )
    experiment.assert_conforms(Stim(random_seed=37), n_processes=2)


@pytest.mark.noise_channel("preparation-crosstalk", oracle="analytic")
def test_preparation_crosstalk_scale_can_disable_channel() -> None:
    """The preparation-crosstalk scale is behaviorally observable."""
    parameters = GeneralNoiseParameters().with_p_prep_crosstalk(1.0).with_p_prep_crosstalk_scale(0.0)
    experiment = ConformanceExperiment(
        runner=build(preparation_crosstalk_probe.compile()),
        n_qubits=2,
        result_tags=("victim", "source"),
        parameters=parameters,
        expected=ExpectedDistribution({(0, 0): 1.0}),
        comparison=ExpectedDistribution({(0, 0): 0.5, (1, 0): 0.5}),
        shots=512,
    )
    experiment.assert_conforms(Stim(random_seed=37))
