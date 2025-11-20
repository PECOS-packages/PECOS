# Copyright 2023 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Two-qubit gate operations for cuStateVec simulator.

This module provides GPU-accelerated two-qubit quantum gate operations for the NVIDIA cuStateVec simulator,
including CNOT gates, controlled gates, and other entangling operations using CUDA acceleration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp

if TYPE_CHECKING:
    from pecos.simulators.custatevec.state import CuStateVec
    from pecos.typing import SimulatorGateParams
from cuquantum.bindings import custatevec as cusv

import pecos as pc
from pecos.simulators.custatevec.gates_one_qubit import H


def _apply_controlled_matrix(
    state: CuStateVec,
    control: int,
    target: int,
    matrix: cp.ndarray,
) -> None:
    """Apply the matrix to the state. This should be faster for controlled gates.

    Args:
        state: An instance of CuStateVec
        control: The index of the qubit that acts as the control
        target: The index of the qubit that acts as the target
        matrix: The matrix to be applied
    """
    if control >= state.num_qubits or control < 0:
        msg = f"Qubit {control} out of range."
        raise ValueError(msg)
    if target >= state.num_qubits or target < 0:
        msg = f"Qubit {target} out of range."
        raise ValueError(msg)
    # CuStateVec uses smaller qubit index as least significant
    control = state.num_qubits - 1 - control
    target = state.num_qubits - 1 - target

    cusv.apply_matrix(
        handle=state.libhandle,
        sv=state.cupy_vector.data.ptr,
        sv_data_type=state.cuda_type,
        n_index_bits=state.num_qubits,
        matrix=matrix.data.ptr,
        matrix_data_type=state.cuda_type,
        layout=cusv.MatrixLayout.ROW,
        adjoint=0,  # Don't use the adjoint
        targets=[target],
        n_targets=1,
        controls=[control],
        control_bit_values=[],  # No value of control bit assigned
        n_controls=1,
        compute_type=state.compute_type,
        extra_workspace=0,  # Let cuQuantum use the mempool we configured
        extra_workspace_size_in_bytes=0,  # Let cuQuantum use the mempool we configured
    )
    state.stream.synchronize()


def CX(
    state: CuStateVec,
    qubits: tuple[int, int],
    **_params: SimulatorGateParams,
) -> None:
    """Apply controlled X gate.

    Args:
        state: An instance of CuStateVec
        qubits: A tuple with the index of the qubits where the gate is applied
            The one at `qubits[0]` is the control qubit.
    """
    matrix = cp.asarray(
        [
            0,
            1,
            1,
            0,
        ],
        dtype=state.cp_type,
    )
    _apply_controlled_matrix(state, qubits[0], qubits[1], matrix)


def CY(
    state: CuStateVec,
    qubits: tuple[int, int],
    **_params: SimulatorGateParams,
) -> None:
    """Apply controlled Y gate.

    Args:
        state: An instance of CuStateVec
        qubits: A tuple with the index of the qubits where the gate is applied
            The one at `qubits[0]` is the control qubit.
    """
    matrix = cp.asarray(
        [
            0,
            -1j,
            1j,
            0,
        ],
        dtype=state.cp_type,
    )
    _apply_controlled_matrix(state, qubits[0], qubits[1], matrix)


def CZ(
    state: CuStateVec,
    qubits: tuple[int, int],
    **_params: SimulatorGateParams,
) -> None:
    """Apply controlled Z gate.

    Args:
        state: An instance of CuStateVec
        qubits: A tuple with the index of the qubits where the gate is applied
            The one at `qubits[0]` is the control qubit.
    """
    matrix = cp.asarray(
        [
            1,
            0,
            0,
            -1,
        ],
        dtype=state.cp_type,
    )
    _apply_controlled_matrix(state, qubits[0], qubits[1], matrix)


def _apply_two_qubit_matrix(
    state: CuStateVec,
    qubits: tuple[int, int],
    matrix: cp.ndarray,
) -> None:
    """Apply the matrix to the state.

    Args:
        state: An instance of CuStateVec
        qubits: A tuple of two qubit indices where the gate is applied
        matrix: The matrix to be applied
    """
    if qubits[0] >= state.num_qubits or qubits[0] < 0:
        msg = f"Qubit {qubits[0]} out of range."
        raise ValueError(msg)
    if qubits[1] >= state.num_qubits or qubits[1] < 0:
        msg = f"Qubit {qubits[1]} out of range."
        raise ValueError(msg)
    # CuStateVec uses smaller qubit index as least significant
    q0 = state.num_qubits - 1 - qubits[0]
    q1 = state.num_qubits - 1 - qubits[1]

    cusv.apply_matrix(
        handle=state.libhandle,
        sv=state.cupy_vector.data.ptr,
        sv_data_type=state.cuda_type,
        n_index_bits=state.num_qubits,
        matrix=matrix.data.ptr,
        matrix_data_type=state.cuda_type,
        layout=cusv.MatrixLayout.ROW,
        adjoint=0,  # Don't use the adjoint
        targets=[q0, q1],
        n_targets=2,
        controls=[],
        control_bit_values=[],  # No value of control bit assigned
        n_controls=0,
        compute_type=state.compute_type,
        extra_workspace=0,  # Let cuQuantum use the mempool we configured
        extra_workspace_size_in_bytes=0,  # Let cuQuantum use the mempool we configured
    )
    state.stream.synchronize()


def RXX(
    state: CuStateVec,
    qubits: tuple[int, int],
    angles: tuple[float],
    **_params: SimulatorGateParams,
) -> None:
    """Apply a rotation about XX.

    Args:
        state: An instance of CuStateVec
        qubits: A tuple with the index of the qubits where the gate is applied
        angles: A tuple containing a single angle in radians
    """
    if len(angles) != 1:
        msg = "Gate must be given 1 angle parameter."
        raise ValueError(msg)
    theta = angles[0]

    matrix = cp.asarray(
        [
            pc.cos(theta / 2),
            0,
            0,
            -1j * pc.sin(theta / 2),
            0,
            pc.cos(theta / 2),
            -1j * pc.sin(theta / 2),
            0,
            0,
            -1j * pc.sin(theta / 2),
            pc.cos(theta / 2),
            0,
            -1j * pc.sin(theta / 2),
            0,
            0,
            pc.cos(theta / 2),
        ],
        dtype=state.cp_type,
    )
    _apply_two_qubit_matrix(state, qubits, matrix)


def RYY(
    state: CuStateVec,
    qubits: tuple[int, int],
    angles: tuple[float],
    **_params: SimulatorGateParams,
) -> None:
    """Apply a rotation about YY.

    Args:
        state: An instance of CuStateVec
        qubits: A tuple with the index of the qubits where the gate is applied
        angles: A tuple containing a single angle in radians
    """
    if len(angles) != 1:
        msg = "Gate must be given 1 angle parameter."
        raise ValueError(msg)
    theta = angles[0]

    matrix = cp.asarray(
        [
            pc.cos(theta / 2),
            0,
            0,
            1j * pc.sin(theta / 2),
            0,
            pc.cos(theta / 2),
            -1j * pc.sin(theta / 2),
            0,
            0,
            -1j * pc.sin(theta / 2),
            pc.cos(theta / 2),
            0,
            1j * pc.sin(theta / 2),
            0,
            0,
            pc.cos(theta / 2),
        ],
        dtype=state.cp_type,
    )
    _apply_two_qubit_matrix(state, qubits, matrix)


def RZZ(
    state: CuStateVec,
    qubits: tuple[int, int],
    angles: tuple[float],
    **_params: SimulatorGateParams,
) -> None:
    """Apply a rotation about ZZ.

    Args:
        state: An instance of CuStateVec
        qubits: A tuple with the index of the qubits where the gate is applied
        angles: A tuple containing a single angle in radians
    """
    if len(angles) != 1:
        msg = "Gate must be given 1 angle parameter."
        raise ValueError(msg)
    theta = angles[0]

    matrix = cp.asarray(
        [
            cp.exp(-1j * theta / 2),
            0,
            0,
            0,
            0,
            cp.exp(1j * theta / 2),
            0,
            0,
            0,
            0,
            cp.exp(1j * theta / 2),
            0,
            0,
            0,
            0,
            cp.exp(-1j * theta / 2),
        ],
        dtype=state.cp_type,
    )
    _apply_two_qubit_matrix(state, qubits, matrix)


def R2XXYYZZ(
    state: CuStateVec,
    qubits: tuple[int, int],
    angles: tuple[float, float, float],
    **_params: SimulatorGateParams,
) -> None:
    """Apply RXX*RYY*RZZ.

    Args:
        state: An instance of CuStateVec
        qubits: A tuple with the index of the qubits where the gate is applied
        angles: A tuple containing three angles in radians, for XX, YY and ZZ, in that order
    """
    if len(angles) != 3:
        msg = "Gate must be given 3 angle parameters."
        raise ValueError(msg)

    RXX(state, qubits, (angles[0],))
    RYY(state, qubits, (angles[1],))
    RZZ(state, qubits, (angles[2],))


def SXX(
    state: CuStateVec,
    qubits: tuple[int, int],
    **_params: SimulatorGateParams,
) -> None:
    """Apply a square root of XX gate.

    Args:
        state: An instance of CuStateVec
        qubits: A tuple with the index of the qubits where the gate is applied
    """
    RXX(state, qubits, angles=(pc.f64.frac_pi_2,))


def SXXdg(
    state: CuStateVec,
    qubits: tuple[int, int],
    **_params: SimulatorGateParams,
) -> None:
    """Apply adjoint of a square root of XX gate.

    Args:
        state: An instance of CuStateVec
        qubits: A tuple with the index of the qubits where the gate is applied
    """
    RXX(state, qubits, angles=(-pc.f64.frac_pi_2,))


def SYY(
    state: CuStateVec,
    qubits: tuple[int, int],
    **_params: SimulatorGateParams,
) -> None:
    """Apply a square root of YY gate.

    Args:
        state: An instance of CuStateVec
        qubits: A tuple with the index of the qubits where the gate is applied
    """
    RYY(state, qubits, angles=(pc.f64.frac_pi_2,))


def SYYdg(
    state: CuStateVec,
    qubits: tuple[int, int],
    **_params: SimulatorGateParams,
) -> None:
    """Apply adjoint of a square root of YY gate.

    Args:
        state: An instance of CuStateVec
        qubits: A tuple with the index of the qubits where the gate is applied
    """
    RYY(state, qubits, angles=(-pc.f64.frac_pi_2,))


def SZZ(
    state: CuStateVec,
    qubits: tuple[int, int],
    **_params: SimulatorGateParams,
) -> None:
    """Apply a square root of ZZ gate.

    Args:
        state: An instance of CuStateVec
        qubits: A tuple with the index of the qubits where the gate is applied
    """
    RZZ(state, qubits, angles=(pc.f64.frac_pi_2,))


def SZZdg(
    state: CuStateVec,
    qubits: tuple[int, int],
    **_params: SimulatorGateParams,
) -> None:
    """Apply adjoint of a square root of ZZ gate.

    Args:
        state: An instance of CuStateVec
        qubits: A tuple with the index of the qubits where the gate is applied
    """
    RZZ(state, qubits, angles=(-pc.f64.frac_pi_2,))


def SWAP(
    state: CuStateVec,
    qubits: tuple[int, int],
    **_params: SimulatorGateParams,
) -> None:
    """Apply a SWAP gate.

    Args:
        state: An instance of CuStateVec
        qubits: A tuple with the index of the qubits where the gate is applied
    """
    if qubits[0] >= state.num_qubits or qubits[0] < 0:
        msg = f"Qubit {qubits[0]} out of range."
        raise ValueError(msg)
    if qubits[1] >= state.num_qubits or qubits[1] < 0:
        msg = f"Qubit {qubits[1]} out of range."
        raise ValueError(msg)
    # CuStateVec uses smaller qubit index as least significant
    q0 = state.num_qubits - 1 - qubits[0]
    q1 = state.num_qubits - 1 - qubits[1]

    # Possibly faster since it may just be an internal qubit relabelling or sv reshape
    cusv.apply_generalized_permutation_matrix(
        handle=state.libhandle,
        sv=state.cupy_vector.data.ptr,
        sv_data_type=state.cuda_type,
        n_index_bits=state.num_qubits,
        permutation=[
            0,
            2,
            1,
            3,
        ],  # Leave |00> and |11> where they are, swap the other two
        diagonals=0,  # Don't apply a diagonal gate
        diagonals_data_type=state.cuda_type,
        adjoint=0,  # Don't use the adjoint
        targets=[q0, q1],
        n_targets=2,
        controls=[],
        control_bit_values=[],  # No value of control bit assigned
        n_controls=0,
        extra_workspace=0,  # Let cuQuantum use the mempool we configured
        extra_workspace_size_in_bytes=0,  # Let cuQuantum use the mempool we configured
    )
    state.stream.synchronize()


def G(
    state: CuStateVec,
    qubits: tuple[int, int],
    **_params: SimulatorGateParams,
) -> None:
    """'G': (('I', 'H'), 'CNOT', ('H', 'H'), 'CNOT', ('I', 'H'))."""
    H(state, qubits[1])
    CX(state, qubits)
    H(state, qubits[0])
    H(state, qubits[1])
    CX(state, qubits)
    H(state, qubits[1])
