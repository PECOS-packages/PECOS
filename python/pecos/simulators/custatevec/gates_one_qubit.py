# TODO: Include license information?

import cupy as cp

from cuquantum import custatevec as cusv
from cuquantum import cudaDataType


def _apply_one_qubit_matrix(state, qubit, matrix: cp.ndarray) -> None:
    """
    Apply the matrix to the state.

    Args:
        state: An instance of CuStateVec
        qubit: The index of the qubit where the gate is applied
        matrix: The matrix to be applied
    """
    cusv.apply_matrix(
        handle=state.libhandle,
        sv=state.vector.data.ptr,
        sv_data_type=state.cuda_type,
        n_index_bits=state.num_qubits,
        matrix=matrix.data.ptr,
        matrix_data_type=state.cuda_type,
        layout=cusv.MatrixLayout.ROW,
        adjoint=0,  # Don't use the adjoint
        targets=[qubit],
        n_targets=1,
        controls=[],
        control_bit_values=[],  # No value of control bit assigned
        n_controls=0,
        compute_type=state.compute_type,
        workspace=0,  # Let cuQuantum use the mempool we configured
        workspace_size=0,  # Let cuQuantum use the mempool we configured
    )
    state.stream.synchronize()


def I(state, qubit) -> None:
    """
    Identity gate.

    Args:
        state: An instance of CuStateVec
        qubit: The index of the qubit where the gate is applied
    """
    pass


def X(state, qubit) -> None:
    """
    Pauli X gate.

    Args:
        state: An instance of CuStateVec
        qubit: The index of the qubit where the gate is applied
    """
    matrix = cp.asarray(
        [
            0, 1,
            1, 0,
        ],
        dtype=state.cp_type,
    )
    _apply_one_qubit_matrix(state, qubit, matrix)


def Y(state, qubit) -> None:
    """
    Pauli Y gate.

    Args:
        state: An instance of CuStateVec
        qubit: The index of the qubit where the gate is applied
    """
    matrix = cp.asarray(
        [
            0, -1j,
            1j, 0,
        ],
        dtype=state.cp_type,
    )
    _apply_one_qubit_matrix(state, qubit, matrix)


def Z(state, qubit) -> None:
    """
    Pauli Z gate.

    Args:
        state: An instance of CuStateVec
        qubit: The index of the qubit where the gate is applied
    """
    matrix = cp.asarray(
        [
            1, 0,
            0, -1,
        ],
        dtype=state.cp_type,
    )
    _apply_one_qubit_matrix(state, qubit, matrix)


def H(state, qubit) -> None:
    """
    Apply Hadamard gate.

    Args:
        state: An instance of CuStateVec
        qubit: The index of the qubit where the gate is applied
    """
    matrix = 1/cp.sqrt(2) * cp.asarray(
        [
            1, 1,
            1, -1,
        ],
        dtype=state.cp_type,
    )
    _apply_one_qubit_matrix(state, qubit, matrix)