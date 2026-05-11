from __future__ import annotations

import pytest
from pecos.quantum import X, Z
from pecos.quantum_info import (
    ChiMatrix,
    ChoiMatrix,
    PauliChannel,
    ProcessTomographyDesign,
    Ptm,
    Stinespring,
    SuperOp,
    average_gate_fidelity,
    entropy,
    gate_error,
    hellinger_distance,
    hellinger_fidelity,
    logarithmic_negativity,
    matrix_unit_basis,
    negativity,
    partial_trace_qubits,
    partial_trace_subsystems,
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
from pecos_rslib import PauliString


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


def test_pauli_channel_accepts_pauli_string_probability_keys() -> None:
    channel = PauliChannel.from_probabilities(
        2,
        {
            PauliString.I(): 0.97,
            X(0): 0.01,
            Z(1): 0.02,
        },
    )

    assert channel.probabilities() == {"II": 0.97, "IX": 0.01, "ZI": 0.02}
    assert_close(channel.total_error_rate(), 0.03)

    from_sequence = PauliChannel.from_probabilities(
        2,
        [
            (PauliString.I(), 0.97),
            (X(0), 0.01),
            (Z(1), 0.02),
        ],
    )
    assert from_sequence.probabilities() == channel.probabilities()


def test_pauli_channel_rejects_ambiguous_pauli_string_keys() -> None:
    with pytest.raises(ValueError, match="unphased"):
        PauliChannel.from_probabilities(1, {-X(0): 1.0})

    with pytest.raises(ValueError, match="outside num_qubits"):
        PauliChannel.from_probabilities(1, {Z(1): 1.0})

    with pytest.raises(ValueError, match="duplicate"):
        PauliChannel.from_probabilities(1, [("X", 0.5), (X(0), 0.5)])


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


def test_superop_compose_and_tensor_wrappers() -> None:
    identity = Ptm.identity(1).to_superop()
    x_channel = PauliChannel.one_qubit(1.0, 0.0, 0.0).to_ptm().to_superop()

    composed = x_channel.compose(x_channel)
    assert isinstance(composed, SuperOp)
    assert composed.num_qubits() == 1
    assert_matrix_close(composed.matrix(), identity.matrix())

    tensor = identity.tensor(identity)
    assert tensor.num_qubits() == 2
    assert_matrix_close(
        tensor.matrix(),
        [[1.0 + 0.0j if row == col else 0.0 + 0.0j for col in range(16)] for row in range(16)],
    )

    scalar_identity = Ptm.identity(0).to_superop()
    assert scalar_identity.num_qubits() == 0
    assert_matrix_close(scalar_identity.matrix(), [[1.0 + 0.0j]])

    with pytest.raises(ValueError, match="channel qubit count mismatch"):
        identity.compose(tensor)


def test_zero_qubit_channel_wrappers_round_trip_scalar_identity() -> None:
    identity = Ptm.identity(0)

    assert identity.num_qubits() == 0
    assert identity.matrix() == [[1.0]]

    choi = identity.to_choi()
    assert choi.num_qubits() == 0
    assert_matrix_close(choi.matrix(), [[1.0 + 0.0j]])

    kraus = identity.to_kraus()
    assert kraus.num_qubits() == 0
    assert kraus.operators() == [[[1.0 + 0.0j]]]

    superop = identity.to_superop()
    assert superop.num_qubits() == 0
    assert_matrix_close(superop.matrix(), [[1.0 + 0.0j]])

    chi = identity.to_chi()
    assert chi.num_qubits() == 0
    assert_matrix_close(chi.matrix(), [[1.0 + 0.0j]])

    stinespring = kraus.to_stinespring()
    assert stinespring.num_qubits() == 0
    assert stinespring.environment_dim() == 1
    assert_matrix_close(stinespring.isometry(), [[1.0 + 0.0j]])


def test_process_tomography_design_reconstructs_identity_channel() -> None:
    design = ProcessTomographyDesign.matrix_unit(1)

    assert design.num_qubits() == 1
    assert design.dim() == 2
    assert design.num_inputs() == 4
    assert design.input_metadata_all() == [(0, 0, 0), (1, 1, 0), (2, 0, 1), (3, 1, 1)]
    assert design.input_index(1, 0) == 1
    assert_matrix_close(
        design.input_operator(2),
        [[0.0 + 0.0j, 1.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]],
    )
    assert design.input_operators() == matrix_unit_basis(1)

    choi = Ptm.identity(1).to_choi()
    outputs = design.simulate_outputs(choi)
    reconstructed = design.reconstruct_choi(outputs)
    assert_matrix_close(reconstructed.matrix(), choi.matrix())
    assert reconstructed.is_cptp()
    assert reconstructed.is_unital()


def test_process_tomography_design_reconstructs_random_two_qubit_channel() -> None:
    channel = random_quantum_channel(2, 2, 321)
    choi = channel.to_choi()
    design = ProcessTomographyDesign.matrix_unit(2)

    assert design.dim() == 4
    assert design.num_inputs() == 16

    outputs = design.simulate_outputs(choi)
    reconstructed = design.reconstruct_choi(outputs)

    assert_matrix_close(reconstructed.matrix(), choi.matrix())
    assert reconstructed.is_cptp()


def test_choi_from_matrix_unit_outputs_static_constructor() -> None:
    outputs = matrix_unit_basis(1)
    reconstructed = ChoiMatrix.from_matrix_unit_outputs(1, outputs)
    assert_matrix_close(reconstructed.matrix(), Ptm.identity(1).to_choi().matrix())


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
    expected_reduced = [[0.5 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.5 + 0.0j]]
    assert_matrix_close(partial_trace_qubits(bell_density, 2, [1]), expected_reduced)
    assert_matrix_close(partial_trace_subsystems(bell_density, [2, 2], [1]), expected_reduced)
    assert_close(hellinger_distance([1.0, 0.0], [0.0, 1.0]), 1.0)
    assert_close(hellinger_fidelity([0.25, 0.75], [0.25, 0.75]), 1.0)
    schmidt = schmidt_decomposition(bell, [2, 2], [0])
    assert len(schmidt) == 2
    assert_close(schmidt[0][0], 2.0**-0.5)
    assert_close(schmidt[1][0], 2.0**-0.5)


def test_quantum_info_wrappers_raise_value_errors_for_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="vector length mismatch"):
        state_fidelity([1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j])

    with pytest.raises(ValueError, match="state vector squared norm"):
        state_fidelity([1.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j])

    with pytest.raises(ValueError, match="matrix must be square"):
        purity([[1.0 + 0.0j, 0.0 + 0.0j]])

    with pytest.raises(ValueError, match="probability distribution must sum"):
        shannon_entropy([0.25, 0.25], 2.0)

    rho = [[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]]
    with pytest.raises(ValueError, match="duplicate subsystem"):
        partial_trace_subsystems(rho, [2], [0, 0])

    with pytest.raises(ValueError, match="outside"):
        partial_trace_subsystems(rho, [2], [1])

    with pytest.raises(ValueError, match="invalid subsystem dimensions"):
        schmidt_decomposition([1.0 + 0.0j, 0.0 + 0.0j], [2, 2], [0])

    with pytest.raises(ValueError, match="invalid matrix shape"):
        SuperOp(1, [[1.0 + 0.0j]])

    with pytest.raises(ValueError, match="not an isometry"):
        Stinespring(1, [[2.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 2.0 + 0.0j]])

    with pytest.raises(ValueError, match="qubit count mismatch"):
        process_fidelity(Ptm.identity(1), Ptm.identity(2))

    with pytest.raises(ValueError, match="qubit count mismatch"):
        average_gate_fidelity(Ptm.identity(1), Ptm.identity(2))

    with pytest.raises(ValueError, match="qubit count mismatch"):
        gate_error(Ptm.identity(1), Ptm.identity(2))


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

    two_qubit = random_quantum_channel(2, 2, 125)
    assert two_qubit.num_qubits() == 2
    assert two_qubit.is_trace_preserving()
    assert two_qubit.to_superop().num_qubits() == 2
    assert len(two_qubit.to_superop().matrix()) == 16
