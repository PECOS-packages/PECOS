"""Deterministic tests for the Rust-backed qudit reference simulators."""

from __future__ import annotations

import math

import pytest

from pecos_rslib.simulators import (
    QuditDensityMatrix,
    QuditStateVec,
    QutritDensityMatrix,
    QutritStateVec,
    basis_swap,
    embedded_qubit_unitary,
    qutrit_leakage_channel,
    qutrit_seepage_channel,
)

TEST_SEED = 0x5EED_588


def test_qutrit_wrappers_fix_the_local_dimension() -> None:
    state = QutritStateVec(2, seed=TEST_SEED)
    density = QutritDensityMatrix(2, seed=TEST_SEED)

    assert state.local_dimension == 3
    assert density.local_dimension == 3
    assert state.dimension == 9
    assert density.dimension == 9
    assert state.state == [1 + 0j, *([0j] * 8)]


def test_qudit_basis_swap_supports_more_than_three_levels() -> None:
    state = QuditStateVec(1, 4, seed=TEST_SEED)
    density = QuditDensityMatrix(1, 4, seed=TEST_SEED)
    swap = basis_swap(4, 0, 3)
    state.apply_operator([0], swap)
    density.apply_operator([0], swap)

    assert state.outcome_probabilities(0) == [0.0, 0.0, 0.0, 1.0]
    assert density.outcome_probabilities(0) == [0.0, 0.0, 0.0, 1.0]
    assert state.measure(0) == 3


def test_seeded_trajectory_is_reproducible() -> None:
    inv_sqrt_two = 1 / math.sqrt(2)
    hadamard = [inv_sqrt_two, inv_sqrt_two, inv_sqrt_two, -inv_sqrt_two]
    channel = qutrit_leakage_channel(0.3)

    first = QutritStateVec(1, seed=TEST_SEED)
    second = QutritStateVec(1, seed=TEST_SEED)
    for simulator in (first, second):
        simulator.apply_embedded_qubit_unitary(0, hadamard)

    first_sample = first.apply_kraus([0], channel)
    second_sample = second.apply_kraus([0], channel)

    assert first_sample.operator_index == second_sample.operator_index
    assert first_sample.probability == second_sample.probability
    assert first.state == second.state
    assert first.measure(0) == second.measure(0)


def test_exact_leakage_channel_has_the_expected_distribution() -> None:
    inv_sqrt_two = 1 / math.sqrt(2)
    hadamard = embedded_qubit_unitary(
        3,
        [inv_sqrt_two, inv_sqrt_two, inv_sqrt_two, -inv_sqrt_two],
    )
    density = QutritDensityMatrix(1, seed=TEST_SEED)
    density.apply_operator([0], hadamard)
    density.apply_kraus([0], qutrit_leakage_channel(0.2))

    assert density.outcome_probabilities(0) == pytest.approx([0.4, 0.4, 0.2])
    assert density.trace() == pytest.approx(1 + 0j)
    assert density.diagnostics().is_physical(1e-10)


def test_state_import_and_reduced_density_matrix_use_python_sequences() -> None:
    state = QutritStateVec.from_state(
        1,
        [1 / math.sqrt(2), 1j / math.sqrt(2), 0j],
        seed=TEST_SEED,
    )
    reduced = state.reduced_density_matrix([0])

    assert reduced == pytest.approx(
        [0.5, -0.5j, 0j, 0.5j, 0.5, 0j, 0j, 0j, 0j]
    )


def test_measurements_instruments_reset_and_seepage_are_exposed() -> None:
    projectors = []
    for level in range(3):
        operator = [0j] * 9
        operator[level * 3 + level] = 1 + 0j
        projectors.append([operator])

    trajectory = QutritStateVec(1, seed=TEST_SEED)
    assert trajectory.instrument_probabilities([0], projectors) == [1.0, 0.0, 0.0]
    sample = trajectory.measure_instrument([0], projectors)
    assert sample.outcome == 0
    assert sample.operator_index == 0
    assert sample.outcome_probability == 1.0
    assert sample.branch_probability == 1.0

    exact = QutritDensityMatrix(1, seed=TEST_SEED)
    exact_sample = exact.measure_instrument([0], projectors)
    assert exact_sample.outcome == 0
    assert exact_sample.probability == 1.0

    exact.prepare_basis(0, 2)
    exact.apply_kraus([0], qutrit_seepage_channel(0.4, 0.25))
    assert exact.outcome_probabilities(0) == pytest.approx([0.1, 0.3, 0.6])
    partition = exact.measure_partition([0], [[0, 1], [2]])
    assert partition.outcome in (0, 1)
    exact.reset(0)
    assert exact.measure_computational(0) is False


def test_invalid_inputs_become_python_exceptions() -> None:
    with pytest.raises(ValueError, match="local dimension"):
        QuditStateVec(1, 1, seed=TEST_SEED)

    state = QutritStateVec(1, seed=TEST_SEED)
    with pytest.raises(IndexError, match="out of range"):
        state.measure(1)
    with pytest.raises(ValueError, match="not unitary"):
        state.apply_embedded_qubit_unitary(0, [1, 1, 0, 1])
