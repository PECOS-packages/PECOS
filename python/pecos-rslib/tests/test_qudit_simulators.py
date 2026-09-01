"""Deterministic tests for the Rust-backed qudit reference simulators."""

from __future__ import annotations

import math

import pytest

from pecos_rslib.simulators import (
    QuditDensityMatrix,
    QuditError,
    QuditIndexError,
    QuditMemoryError,
    QuditStateVec,
    QuditValueError,
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

    assert reduced == pytest.approx([0.5, -0.5j, 0j, 0.5j, 0.5, 0j, 0j, 0j, 0j])


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
    exact.reset_site(0)
    assert exact.measure_computational(0) is False


def test_invalid_inputs_become_python_exceptions() -> None:
    with pytest.raises(ValueError, match="local dimension"):
        QuditStateVec(1, 1, seed=TEST_SEED)

    state = QutritStateVec(1, seed=TEST_SEED)
    with pytest.raises(IndexError, match="out of range"):
        state.measure(1)
    with pytest.raises(ValueError, match="not unitary"):
        state.apply_embedded_qubit_unitary(0, [1, 1, 0, 1])


# A non-symmetric, complex two-site unitary: a cyclic shift of the nine joint
# basis states with per-column phases. Non-symmetric so a transposed operator is
# detectable, complex so a dropped or conjugated imaginary part is detectable.
_TWO_SITE_PHASES = [1, 1j, -1, -1j, 1, 1j, -1, -1j, 1]
TWO_SITE_UNITARY = [0j] * 81
for _column, _phase in enumerate(_TWO_SITE_PHASES):
    TWO_SITE_UNITARY[((_column + 1) % 9) * 9 + _column] = complex(_phase)

# Entangled, non-product input state.
TWO_SITE_STATE = [0j, 0.5 + 0j, 0j, 0.5 + 0j, 0.5 + 0j, 0.5 + 0j, 0j, 0j, 0j]

# Expected values derived independently of this module: the operator is applied
# by dense matrix multiplication, with targets [1, 0] expressed as S @ U @ S for
# the digit-swap permutation S. Site 0 is the least-significant radix digit.
EXPECTED_STATE_01 = [0j, 0j, 0.5j, 0j, -0.5j, 0.5 + 0j, 0.5j, 0j, 0j]
EXPECTED_STATE_10 = [0j, 0j, 0j, 0j, -0.5j, 0j, 0.5j, 0.5 + 0j, -0.5j]
EXPECTED_JOINT_01 = [0.0, 0.0, 0.25, 0.0, 0.25, 0.25, 0.25, 0.0, 0.0]
EXPECTED_JOINT_10 = [0.0, 0.0, 0.25, 0.0, 0.25, 0.0, 0.25, 0.25, 0.0]
EXPECTED_RDM_SITE0 = [0.25 + 0j, 0j, 0j, 0j, 0.25 + 0j, -0.25j, 0j, 0.25j, 0.5 + 0j]
EXPECTED_RDM_SITE1 = [0.25 + 0j, 0.25j, 0j, -0.25j, 0.5 + 0j, 0j, 0j, 0j, 0.25 + 0j]


def test_two_site_target_order_is_not_symmetric() -> None:
    """`[0, 1]` and `[1, 0]` must be different operations, in both backends."""
    forward = QutritStateVec.from_state(2, TWO_SITE_STATE, seed=TEST_SEED)
    forward.apply_operator([0, 1], TWO_SITE_UNITARY)
    reversed_ = QutritStateVec.from_state(2, TWO_SITE_STATE, seed=TEST_SEED)
    reversed_.apply_operator([1, 0], TWO_SITE_UNITARY)

    assert forward.state == pytest.approx(EXPECTED_STATE_01)
    assert reversed_.state == pytest.approx(EXPECTED_STATE_10)
    assert forward.state != pytest.approx(reversed_.state)

    # The density backend is an independent code path and must agree.
    density = QutritDensityMatrix.from_density_matrix(
        2,
        [a * b.conjugate() for a in TWO_SITE_STATE for b in TWO_SITE_STATE],
        seed=TEST_SEED,
    )
    density.apply_operator([0, 1], TWO_SITE_UNITARY)
    assert density.density_matrix == pytest.approx(
        [a * b.conjugate() for a in EXPECTED_STATE_01 for b in EXPECTED_STATE_01]
    )


def test_joint_outcome_and_reduced_matrix_orderings_are_literal() -> None:
    """`targets[0]` is the least-significant digit of outcomes and reduced rows."""
    state = QutritStateVec.from_state(2, TWO_SITE_STATE, seed=TEST_SEED)
    state.apply_operator([0, 1], TWO_SITE_UNITARY)

    assert state.joint_outcome_probabilities([0, 1]) == pytest.approx(EXPECTED_JOINT_01)
    assert state.joint_outcome_probabilities([1, 0]) == pytest.approx(EXPECTED_JOINT_10)

    # Single-site reduced matrices differ between the sites, so a target mix-up
    # or a transposed/conjugated partial trace is caught.
    assert state.reduced_density_matrix([0]) == pytest.approx(EXPECTED_RDM_SITE0)
    assert state.reduced_density_matrix([1]) == pytest.approx(EXPECTED_RDM_SITE1)

    density = QutritDensityMatrix.from_density_matrix(
        2,
        [a * b.conjugate() for a in EXPECTED_STATE_01 for b in EXPECTED_STATE_01],
        seed=TEST_SEED,
    )
    assert density.joint_outcome_probabilities([0, 1]) == pytest.approx(
        EXPECTED_JOINT_01
    )
    assert density.joint_outcome_probabilities([1, 0]) == pytest.approx(
        EXPECTED_JOINT_10
    )
    assert density.reduced_density_matrix([0]) == pytest.approx(EXPECTED_RDM_SITE0)
    assert density.reduced_density_matrix([1]) == pytest.approx(EXPECTED_RDM_SITE1)


def test_seepage_channel_keyword_matches_the_rust_core() -> None:
    assert qutrit_seepage_channel(probability=0.4, zero_fraction=0.25) == (
        qutrit_seepage_channel(0.4, 0.25)
    )


def test_errors_are_distinguishable_without_matching_message_text() -> None:
    """`kind` identifies the condition; builtin bases stay backward compatible."""
    state = QutritStateVec(1, seed=TEST_SEED)
    state.prepare_basis(0, 2)

    with pytest.raises(QuditError) as leaked:
        state.measure_computational(0)
    assert leaked.value.kind == "LeakagePopulation"
    # The historical builtin base still catches it.
    assert isinstance(leaked.value, ValueError)
    assert isinstance(leaked.value, QuditValueError)

    with pytest.raises(QuditIndexError) as out_of_range:
        state.measure(1)
    assert out_of_range.value.kind == "TargetOutOfRange"
    assert isinstance(out_of_range.value, IndexError)

    with pytest.raises(QuditMemoryError) as too_big:
        QuditStateVec(50, 2, seed=TEST_SEED)
    assert too_big.value.kind == "AllocationFailed"
    assert isinstance(too_big.value, MemoryError)

    # Distinct conditions that previously shared a bare ValueError.
    kinds = {}
    for label, call in (
        ("empty", lambda: state.measure_joint([])),
        ("duplicate", lambda: state.apply_operator([0, 0], [0j] * 81)),
        ("nonunitary", lambda: state.apply_embedded_qubit_unitary(0, [1, 1, 0, 1])),
    ):
        with pytest.raises(QuditError) as caught:
            call()
        kinds[label] = caught.value.kind
    assert kinds == {
        "empty": "EmptyTargets",
        "duplicate": "DuplicateTarget",
        "nonunitary": "NonUnitary",
    }
