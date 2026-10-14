"""Coherent conformance tests driven by an independent qutrit oracle."""

from __future__ import annotations

import math
from itertools import product
from typing import TYPE_CHECKING

import numpy as np
import pytest
from general_noise_conformance import ConformanceExperiment
from guppylang import guppy
from guppylang.std.angles import pi
from guppylang.std.builtins import result
from guppylang.std.quantum import cx, h, measure, qubit
from guppylang.std.quantum import rx as quantum_rx
from guppylang.std.quantum import ry as quantum_ry
from guppylang.std.quantum import rz as quantum_rz
from pecos_selene_general_noise import GeneralNoiseParameters
from qutrit_reference import QutritNoise, QutritReference, controlled_x, hadamard, rx, ry, rz
from selene_sim import Stim
from selene_sim.build import build

if TYPE_CHECKING:
    from qutrit_reference import Matrix
    from selene_sim.instance import SeleneInstance


@guppy
def x_axis_echo() -> None:
    """Traverse a superposition and return deterministically to zero."""
    q = qubit()
    quantum_rx(q, pi / 2)
    quantum_rz(q, pi)
    quantum_rx(q, pi / 2)
    result("outcome", measure(q).read())


@guppy
def y_axis_echo() -> None:
    """Traverse a superposition and return deterministically to one."""
    q = qubit()
    quantum_ry(q, pi / 2)
    quantum_rz(q, pi)
    quantum_ry(q, -pi / 2)
    result("outcome", measure(q).read())


@guppy
def six_layer_clifford_zero() -> None:
    """A mixed-axis Clifford sequence with deterministic ideal zero readout."""
    q = qubit()
    quantum_ry(q, pi / 2)
    quantum_rx(q, -pi / 2)
    quantum_rz(q, pi / 2)
    quantum_ry(q, -pi / 2)
    quantum_rx(q, pi / 2)
    quantum_rz(q, pi / 2)
    result("outcome", measure(q).read())


@guppy
def six_layer_clifford_one() -> None:
    """A second mixed-axis Clifford sequence with deterministic ideal one readout."""
    q = qubit()
    quantum_ry(q, -pi / 2)
    quantum_ry(q, pi)
    quantum_rz(q, pi / 2)
    quantum_rx(q, -pi / 2)
    quantum_rz(q, pi)
    quantum_rz(q, pi / 2)
    result("outcome", measure(q).read())


@guppy
def twelve_layer_clifford_zero() -> None:
    """A deeper standard-operation sequence with deterministic ideal readout."""
    q = qubit()
    quantum_ry(q, pi / 2)
    quantum_rx(q, -pi / 2)
    quantum_rz(q, pi / 2)
    quantum_ry(q, -pi / 2)
    quantum_rx(q, pi / 2)
    quantum_rz(q, pi / 2)
    quantum_ry(q, pi / 2)
    quantum_rx(q, -pi / 2)
    quantum_rz(q, pi / 2)
    quantum_ry(q, -pi / 2)
    quantum_rx(q, pi / 2)
    quantum_rz(q, pi / 2)
    result("outcome", measure(q).read())


@guppy
def parallel_coherent_sequences() -> None:
    """Run three different coherent standard-gate sequences in parallel."""
    q0 = qubit()
    q1 = qubit()
    q2 = qubit()
    quantum_rx(q0, pi / 2)
    quantum_ry(q1, pi / 2)
    quantum_ry(q2, pi / 2)
    quantum_rz(q0, pi)
    quantum_rz(q1, pi)
    quantum_rx(q2, -pi / 2)
    quantum_rx(q0, pi / 2)
    quantum_ry(q1, -pi / 2)
    quantum_rz(q2, pi / 2)
    quantum_ry(q2, -pi / 2)
    quantum_rx(q2, pi / 2)
    quantum_rz(q2, pi / 2)
    result("q0", measure(q0).read())
    result("q1", measure(q1).read())
    result("q2", measure(q2).read())


@guppy
def bell_state() -> None:
    """Prepare a Bell state with standard device-neutral gates."""
    q0 = qubit()
    q1 = qubit()
    h(q0)
    cx(q0, q1)
    result("q0", measure(q0).read())
    result("q1", measure(q1).read())


@guppy
def anti_correlated_bell_state() -> None:
    """Prepare an orthogonal Bell state with anti-correlated readout."""
    q0 = qubit()
    q1 = qubit()
    h(q0)
    cx(q0, q1)
    quantum_rx(q1, pi)
    result("q0", measure(q0).read())
    result("q1", measure(q1).read())


@guppy
def three_qubit_entangling_chain() -> None:
    """Entangle a three-qubit chain using only standard operations."""
    q0 = qubit()
    q1 = qubit()
    q2 = qubit()
    h(q0)
    cx(q0, q1)
    cx(q1, q2)
    result("q0", measure(q0).read())
    result("q1", measure(q1).read())
    result("q2", measure(q2).read())


SIX_LAYER_ZERO = (
    ry(math.pi / 2),
    rx(-math.pi / 2),
    rz(math.pi / 2),
    ry(-math.pi / 2),
    rx(math.pi / 2),
    rz(math.pi / 2),
)

COHERENT_CASES: dict[str, tuple[SeleneInstance, tuple[Matrix, ...]]] = {
    "x-axis-echo": (
        build(x_axis_echo.compile()),
        (rx(math.pi / 2), rz(math.pi), rx(math.pi / 2)),
    ),
    "y-axis-echo": (
        build(y_axis_echo.compile()),
        (ry(math.pi / 2), rz(math.pi), ry(-math.pi / 2)),
    ),
    "six-layer-zero": (
        build(six_layer_clifford_zero.compile()),
        SIX_LAYER_ZERO,
    ),
    "six-layer-one": (
        build(six_layer_clifford_one.compile()),
        (
            ry(-math.pi / 2),
            ry(math.pi),
            rz(math.pi / 2),
            rx(-math.pi / 2),
            rz(math.pi),
            rz(math.pi / 2),
        ),
    ),
    "twelve-layer-zero": (
        build(twelve_layer_clifford_zero.compile()),
        SIX_LAYER_ZERO + SIX_LAYER_ZERO,
    ),
}

REFERENCE_NOISE = QutritNoise(
    preparation_probability=0.08,
    preparation_leakage_ratio=0.3,
    p1=0.18,
    p1_pauli_model=(("X", 0.2), ("Y", 0.3), ("Z", 0.5)),
    p1_emission_ratio=0.25,
    p1_emission_model=(("X", 0.4), ("Y", 0.2), ("L", 0.4)),
    p1_seepage_probability=0.35,
    measurement_0_to_1=0.06,
    measurement_1_to_0=0.11,
)

PLUGIN_PARAMETERS = (
    GeneralNoiseParameters()
    .with_p_prep(0.08)
    .with_prep_leak_ratio(0.3)
    .with_p1(0.18)
    .with_p1_pauli_model({"X": 0.2, "Y": 0.3, "Z": 0.5})
    .with_p1_emission_ratio(0.25)
    .with_p1_emission_model({"X": 0.4, "Y": 0.2, "L": 0.4})
    .with_p1_seepage_prob(0.35)
    .with_p_meas_0(0.06)
    .with_p_meas_1(0.11)
)

UNIFORM_P2_MODEL = {first + second: 1.0 / 15.0 for first, second in product("IXYZ", repeat=2) if first + second != "II"}
TWO_QUBIT_NOISE = QutritNoise(p2=0.3)
TWO_QUBIT_PARAMETERS = GeneralNoiseParameters().with_p2(0.3).with_p2_pauli_model(UNIFORM_P2_MODEL)
THREE_QUBIT_NOISE = QutritNoise(p2=0.2)
THREE_QUBIT_PARAMETERS = GeneralNoiseParameters().with_p2(0.2).with_p2_pauli_model(UNIFORM_P2_MODEL)


def _reference(gates: tuple[Matrix, ...], noise: QutritNoise) -> QutritReference:
    reference = QutritReference(1, noise).reset(0)
    for gate in gates:
        reference.one_qubit_gate(0, gate)
    return reference


def _parallel_reference(noise: QutritNoise) -> QutritReference:
    reference = QutritReference(3, noise).reset(0).reset(1).reset(2)
    operations = (
        (0, rx(math.pi / 2)),
        (1, ry(math.pi / 2)),
        (2, ry(math.pi / 2)),
        (0, rz(math.pi)),
        (1, rz(math.pi)),
        (2, rx(-math.pi / 2)),
        (0, rx(math.pi / 2)),
        (1, ry(-math.pi / 2)),
        (2, rz(math.pi / 2)),
        (2, ry(-math.pi / 2)),
        (2, rx(math.pi / 2)),
        (2, rz(math.pi / 2)),
    )
    for qubit_id, gate in operations:
        reference.one_qubit_gate(qubit_id, gate)
    return reference


def _bell_reference(noise: QutritNoise, *, anti_correlated: bool) -> QutritReference:
    reference = QutritReference(2, noise).reset(0).reset(1)
    reference.one_qubit_gate(0, hadamard()).two_qubit_gate((0, 1), controlled_x())
    if anti_correlated:
        reference.one_qubit_gate(1, rx(math.pi))
    return reference


def _three_qubit_reference(noise: QutritNoise) -> QutritReference:
    reference = QutritReference(3, noise).reset(0).reset(1).reset(2)
    return (
        reference.one_qubit_gate(0, hadamard())
        .two_qubit_gate(
            (0, 1),
            controlled_x(),
        )
        .two_qubit_gate((1, 2), controlled_x())
    )


@pytest.mark.parametrize("case_name", COHERENT_CASES)
@pytest.mark.parametrize(
    ("error_seed", "simulator_seed"),
    [
        pytest.param(401, 409, id="primary"),
        pytest.param(419, 421, id="repeat-1", marks=pytest.mark.slow),
        pytest.param(431, 433, id="repeat-2", marks=pytest.mark.slow),
    ],
)
def test_coherent_clifford_matrix_matches_qutrit_reference(
    case_name: str,
    error_seed: int,
    simulator_seed: int,
) -> None:
    """Coherent Clifford-style circuits agree for several independent seeds."""
    runner, gates = COHERENT_CASES[case_name]
    expected = _reference(gates, REFERENCE_NOISE).measurement_distribution((0,))
    comparison = _reference(gates, QutritNoise()).measurement_distribution((0,))
    experiment = ConformanceExperiment(
        runner=runner,
        n_qubits=1,
        result_tags=("outcome",),
        parameters=PLUGIN_PARAMETERS,
        expected=expected,
        comparison=comparison,
        shots=2048,
        seed=error_seed,
    )
    experiment.assert_conforms(Stim(random_seed=simulator_seed), n_processes=2)


@pytest.mark.parametrize(
    ("error_seed", "simulator_seed"),
    [
        pytest.param(443, 449, id="primary"),
        pytest.param(457, 461, id="repeat-1", marks=pytest.mark.slow),
        pytest.param(463, 467, id="repeat-2", marks=pytest.mark.slow),
    ],
)
def test_parallel_coherent_sequences_match_joint_qutrit_reference(
    error_seed: int,
    simulator_seed: int,
) -> None:
    """Three-qubit coherent sequences match the full joint qutrit distribution."""
    experiment = ConformanceExperiment(
        runner=build(parallel_coherent_sequences.compile()),
        n_qubits=3,
        result_tags=("q0", "q1", "q2"),
        parameters=PLUGIN_PARAMETERS,
        expected=_parallel_reference(REFERENCE_NOISE).measurement_distribution((0, 1, 2)),
        comparison=_parallel_reference(QutritNoise()).measurement_distribution((0, 1, 2)),
        shots=4096,
        seed=error_seed,
    )
    experiment.assert_conforms(Stim(random_seed=simulator_seed), n_processes=2)


@pytest.mark.parametrize(
    ("runner", "anti_correlated"),
    [
        pytest.param(build(bell_state.compile()), False, id="bell-state"),
        pytest.param(build(anti_correlated_bell_state.compile()), True, id="anti-correlated-bell"),
    ],
)
@pytest.mark.parametrize(
    ("error_seed", "simulator_seed"),
    [
        pytest.param(479, 487, id="primary"),
        pytest.param(491, 499, id="repeat-1", marks=pytest.mark.slow),
        pytest.param(503, 509, id="repeat-2", marks=pytest.mark.slow),
    ],
)
def test_entangling_cliffords_match_two_qutrit_reference(
    runner: SeleneInstance,
    anti_correlated: bool,
    error_seed: int,
    simulator_seed: int,
) -> None:
    """Entangling Clifford circuits match an independent two-qutrit oracle."""
    experiment = ConformanceExperiment(
        runner=runner,
        n_qubits=2,
        result_tags=("q0", "q1"),
        parameters=TWO_QUBIT_PARAMETERS,
        expected=_bell_reference(TWO_QUBIT_NOISE, anti_correlated=anti_correlated).measurement_distribution((0, 1)),
        comparison=_bell_reference(QutritNoise(), anti_correlated=anti_correlated).measurement_distribution((0, 1)),
        shots=4096,
        seed=error_seed,
    )
    experiment.assert_conforms(Stim(random_seed=simulator_seed), n_processes=2)


@pytest.mark.slow
def test_three_qubit_entangling_chain_matches_qutrit_reference() -> None:
    """A multi-edge entangling network agrees with the independent oracle."""
    experiment = ConformanceExperiment(
        runner=build(three_qubit_entangling_chain.compile()),
        n_qubits=3,
        result_tags=("q0", "q1", "q2"),
        parameters=THREE_QUBIT_PARAMETERS,
        expected=_three_qubit_reference(THREE_QUBIT_NOISE).measurement_distribution((0, 1, 2)),
        comparison=_three_qubit_reference(QutritNoise()).measurement_distribution((0, 1, 2)),
        shots=4096,
        seed=521,
    )
    experiment.assert_conforms(Stim(random_seed=523), n_processes=2)


def test_qutrit_reference_preserves_a_physical_density_operator() -> None:
    """The independent channel remains normalized, Hermitian, and positive."""
    reference = _reference(COHERENT_CASES["six-layer-zero"][1], REFERENCE_NOISE)
    assert np.isclose(np.trace(reference.rho), 1.0)
    assert np.allclose(reference.rho, reference.rho.conj().T)
    assert np.linalg.eigvalsh(reference.rho).min() >= -1e-12


def test_qutrit_reference_clips_only_floating_point_probability_residue() -> None:
    """Cross-platform BLAS residue at zero and one is not treated as physics."""
    reference = QutritReference(1)
    reference.rho[0, 0] = 1.0 + 5e-16
    reference.rho[1, 1] = -5e-16
    assert reference.measurement_distribution((0,)).probabilities == {(0,): 1.0, (1,): 0.0}


def test_qutrit_reference_rejects_materially_nonphysical_probabilities() -> None:
    """The oracle must not hide real loss of density-matrix positivity."""
    reference = QutritReference(1)
    reference.rho[0, 0] = 1.01
    reference.rho[1, 1] = -0.01
    with pytest.raises(ValueError, match="outside the physical probability range"):
        reference.measurement_distribution((0,))
