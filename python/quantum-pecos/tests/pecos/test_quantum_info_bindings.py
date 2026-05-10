from __future__ import annotations

from pecos.quantum_info import (
    ChiMatrix,
    ChoiMatrix,
    PauliChannel,
    Ptm,
    Stinespring,
    SuperOp,
    average_gate_fidelity,
    entropy,
    gate_error,
    hellinger_distance,
    hellinger_fidelity,
    logarithmic_negativity,
    negativity,
    pauli_channel_diamond_distance,
    pauli_channel_diamond_norm,
    process_fidelity,
    purity,
    random_density_matrix,
    random_quantum_channel,
    schmidt_decomposition,
    shannon_entropy,
    state_fidelity,
    state_fidelity_with_density_matrix,
)


def assert_close(actual: float, expected: float, tol: float = 1e-12) -> None:
    assert abs(actual - expected) < tol


def assert_matrix_close(actual: list[list[complex]], expected: list[list[complex]]) -> None:
    assert len(actual) == len(expected)
    for actual_row, expected_row in zip(actual, expected, strict=True):
        assert len(actual_row) == len(expected_row)
        for actual_value, expected_value in zip(actual_row, expected_row, strict=True):
            assert abs(actual_value - expected_value) < 1e-12


def test_pauli_channel_exposes_probabilities_and_ptm() -> None:
    channel = PauliChannel.one_qubit(0.1, 0.2, 0.0)

    assert channel.num_qubits() == 1
    assert_close(channel.total_error_rate(), 0.3)
    assert channel.probabilities() == {"I": 0.7, "X": 0.1, "Y": 0.2}

    ptm = channel.to_ptm()
    assert ptm.num_qubits() == 1
    assert_close(ptm.entry(0, 0), 1.0)

    other = PauliChannel.one_qubit(0.0, 0.2, 0.3)
    assert_close(pauli_channel_diamond_norm(channel, other), 0.6)
    assert_close(pauli_channel_diamond_distance(channel, other), 0.3)


def test_choi_and_kraus_wrappers_round_trip_identity_channel() -> None:
    identity = Ptm.identity(1)
    choi = identity.to_choi()

    assert isinstance(choi, ChoiMatrix)
    assert choi.is_completely_positive()
    assert choi.is_trace_preserving()
    assert choi.is_cptp()
    assert choi.is_unital()
    assert_matrix_close(
        choi.partial_trace_output(),
        [[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]],
    )

    kraus = choi.to_kraus()
    assert kraus.num_qubits() == 1
    assert kraus.is_trace_preserving()
    assert_close(process_fidelity(kraus.to_ptm(), identity), 1.0)
    assert_close(average_gate_fidelity(kraus.to_ptm(), identity), 1.0)
    assert_close(gate_error(kraus.to_ptm(), identity), 0.0)

    superop = kraus.to_superop()
    assert isinstance(superop, SuperOp)
    assert_close(process_fidelity(superop.to_ptm(), identity), 1.0)

    chi = kraus.to_chi()
    assert isinstance(chi, ChiMatrix)
    assert_close(process_fidelity(chi.to_ptm(), identity), 1.0)

    stinespring = kraus.to_stinespring()
    assert isinstance(stinespring, Stinespring)
    assert stinespring.environment_dim() == 1
    assert_close(process_fidelity(stinespring.to_kraus().to_ptm(), identity), 1.0)


def test_state_measure_wrappers() -> None:
    zero = [1.0 + 0.0j, 0.0 + 0.0j]
    plus = [2.0**-0.5 + 0.0j, 2.0**-0.5 + 0.0j]
    zero_density = [[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]]
    bell = [2.0**-0.5 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 2.0**-0.5 + 0.0j]
    bell_density = [
        [0.5 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.5 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        [0.5 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.5 + 0.0j],
    ]

    assert_close(state_fidelity(zero, zero), 1.0)
    assert_close(state_fidelity(zero, plus), 0.5)
    assert_close(state_fidelity_with_density_matrix(zero_density, zero), 1.0)
    assert_close(purity(zero_density), 1.0)
    assert_close(entropy(zero_density), 0.0)
    assert_close(shannon_entropy([0.5, 0.5], 2.0), 1.0)
    assert_close(negativity(bell_density, [2, 2], 1), 0.5)
    assert_close(logarithmic_negativity(bell_density, [2, 2], 1), 1.0)
    assert_close(hellinger_distance([1.0, 0.0], [0.0, 1.0]), 1.0)
    assert_close(hellinger_fidelity([0.25, 0.75], [0.25, 0.75]), 1.0)
    schmidt = schmidt_decomposition(bell, [2, 2], [0])
    assert len(schmidt) == 2
    assert_close(schmidt[0][0], 2.0**-0.5)
    assert_close(schmidt[1][0], 2.0**-0.5)


def test_random_generators_are_seed_reproducible_and_valid() -> None:
    rho = random_density_matrix(1, 123)
    same_rho = random_density_matrix(1, 123)
    different_rho = random_density_matrix(1, 124)

    assert rho == same_rho
    assert rho != different_rho
    assert_close((rho[0][0] + rho[1][1]).real, 1.0)

    channel = random_quantum_channel(1, 2, 123)
    same_channel = random_quantum_channel(1, 2, 123)
    assert channel.operators() == same_channel.operators()
    assert channel.is_trace_preserving()
