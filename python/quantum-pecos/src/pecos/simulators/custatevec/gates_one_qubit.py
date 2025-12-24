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

"""Single-qubit gate operations for cuStateVec simulator.

This module provides GPU-accelerated single-qubit quantum gate operations for the NVIDIA cuStateVec simulator,
including Pauli gates, rotation gates, and other fundamental single-qubit operations using CUDA acceleration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp

if TYPE_CHECKING:
    from pecos.simulators.custatevec.state import CuStateVec
    from pecos.typing import SimulatorGateParams
from cuquantum.bindings import custatevec as cusv

import pecos as pc


def _apply_one_qubit_matrix(state: CuStateVec, qubit: int, matrix: cp.ndarray) -> None:
    """Apply the matrix to the state.

    Args:
        state: An instance of CuStateVec
        qubit: The index of the qubit where the gate is applied
        matrix: The matrix to be applied
    """
    if qubit >= state.num_qubits or qubit < 0:
        msg = f"Qubit {qubit} out of range."
        raise ValueError(msg)
    # CuStateVec uses smaller qubit index as least significant
    target = state.num_qubits - 1 - qubit

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
        controls=[],
        control_bit_values=[],  # No value of control bit assigned
        n_controls=0,
        compute_type=state.compute_type,
        extra_workspace=0,  # Let cuQuantum use the mempool we configured
        extra_workspace_size_in_bytes=0,  # Let cuQuantum use the mempool we configured
    )
    state.stream.synchronize()


def identity(state: CuStateVec, qubit: int, **_params: SimulatorGateParams) -> None:
    """Identity gate.

    Args:
        state: An instance of CuStateVec
        qubit: The index of the qubit where the gate is applied
    """


def X(state: CuStateVec, qubit: int, **_params: SimulatorGateParams) -> None:
    """Pauli X gate.

    Args:
        state: An instance of CuStateVec
        qubit: The index of the qubit where the gate is applied
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
    _apply_one_qubit_matrix(state, qubit, matrix)


def Y(state: CuStateVec, qubit: int, **_params: SimulatorGateParams) -> None:
    """Pauli Y gate.

    Args:
        state: An instance of CuStateVec
        qubit: The index of the qubit where the gate is applied
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
    _apply_one_qubit_matrix(state, qubit, matrix)


def Z(state: CuStateVec, qubit: int, **_params: SimulatorGateParams) -> None:
    """Pauli Z gate.

    Args:
        state: An instance of CuStateVec
        qubit: The index of the qubit where the gate is applied
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
    _apply_one_qubit_matrix(state, qubit, matrix)


def RX(
    state: CuStateVec,
    qubit: int,
    angles: tuple[float],
    **_params: SimulatorGateParams,
) -> None:
    """Apply an RX gate.

    Args:
        state: An instance of CuStateVec
        qubit: The index of the qubit where the gate is applied
        angles: A tuple containing a single angle in radians
    """
    if len(angles) != 1:
        msg = "Gate must be given 1 angle parameter."
        raise ValueError(msg)
    theta = angles[0]

    matrix = cp.asarray(
        [
            pc.cos(theta / 2),
            -1j * pc.sin(theta / 2),
            -1j * pc.sin(theta / 2),
            pc.cos(theta / 2),
        ],
        dtype=state.cp_type,
    )
    _apply_one_qubit_matrix(state, qubit, matrix)


def RY(
    state: CuStateVec,
    qubit: int,
    angles: tuple[float],
    **_params: SimulatorGateParams,
) -> None:
    """Apply an RY gate.

    Args:
        state: An instance of CuStateVec
        qubit: The index of the qubit where the gate is applied
        angles: A tuple containing a single angle in radians
    """
    if len(angles) != 1:
        msg = "Gate must be given 1 angle parameter."
        raise ValueError(msg)
    theta = angles[0]

    matrix = cp.asarray(
        [
            pc.cos(theta / 2),
            -pc.sin(theta / 2),
            pc.sin(theta / 2),
            pc.cos(theta / 2),
        ],
        dtype=state.cp_type,
    )
    _apply_one_qubit_matrix(state, qubit, matrix)


def RZ(
    state: CuStateVec,
    qubit: int,
    angles: tuple[float],
    **_params: SimulatorGateParams,
) -> None:
    """Apply an RZ gate.

    Args:
        state: An instance of CuStateVec
        qubit: The index of the qubit where the gate is applied
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
            cp.exp(1j * theta / 2),
        ],
        dtype=state.cp_type,
    )
    _apply_one_qubit_matrix(state, qubit, matrix)


def R1XY(
    state: CuStateVec,
    qubit: int,
    angles: tuple[float, float],
    **_params: SimulatorGateParams,
) -> None:
    """Apply an R1XY gate.

    Args:
        state: An instance of CuStateVec
        qubit: The index of the qubit where the gate is applied
        angles: A tuple containing two angles in radians
    """
    if len(angles) != 2:
        msg = "Gate must be given 2 angle parameters."
        raise ValueError(msg)
    theta = angles[0]
    phi = angles[1]

    # Gate is equal to RZ(phi-pi/2)*RY(theta)*RZ(-phi+pi/2)
    RZ(state, qubit, angles=(-phi + pc.f64.frac_pi_2,))
    RY(state, qubit, angles=(theta,))
    RZ(state, qubit, angles=(phi - pc.f64.frac_pi_2,))


def SX(state: CuStateVec, qubit: int, **_params: SimulatorGateParams) -> None:
    """Apply a square-root of X.

    Args:
        state: An instance of CuStateVec
        qubit: The index of the qubit where the gate is applied
    """
    RX(state, qubit, angles=(pc.f64.frac_pi_2,))


def SXdg(state: CuStateVec, qubit: int, **_params: SimulatorGateParams) -> None:
    """Apply adjoint of the square-root of X.

    Args:
        state: An instance of CuStateVec
        qubit: The index of the qubit where the gate is applied
    """
    RX(state, qubit, angles=(-pc.f64.frac_pi_2,))


def SY(state: CuStateVec, qubit: int, **_params: SimulatorGateParams) -> None:
    """Apply a square-root of Y.

    Args:
        state: An instance of CuStateVec
        qubit: The index of the qubit where the gate is applied
    """
    RY(state, qubit, angles=(pc.f64.frac_pi_2,))


def SYdg(state: CuStateVec, qubit: int, **_params: SimulatorGateParams) -> None:
    """Apply adjoint of the square-root of Y.

    Args:
        state: An instance of CuStateVec
        qubit: The index of the qubit where the gate is applied
    """
    RY(state, qubit, angles=(-pc.f64.frac_pi_2,))


def SZ(state: CuStateVec, qubit: int, **_params: SimulatorGateParams) -> None:
    """Apply a square-root of Z.

    Args:
        state: An instance of CuStateVec
        qubit: The index of the qubit where the gate is applied
    """
    RZ(state, qubit, angles=(pc.f64.frac_pi_2,))


def SZdg(state: CuStateVec, qubit: int, **_params: SimulatorGateParams) -> None:
    """Apply adjoint of the square-root of Z.

    Args:
        state: An instance of CuStateVec
        qubit: The index of the qubit where the gate is applied
    """
    RZ(state, qubit, angles=(-pc.f64.frac_pi_2,))


def H(state: CuStateVec, qubit: int, **_params: SimulatorGateParams) -> None:
    """Apply Hadamard gate.

    Args:
        state: An instance of CuStateVec
        qubit: The index of the qubit where the gate is applied
    """
    matrix = (
        1
        / cp.sqrt(2)
        * cp.asarray(
            [
                1,
                1,
                1,
                -1,
            ],
            dtype=state.cp_type,
        )
    )
    _apply_one_qubit_matrix(state, qubit, matrix)


def F(state: CuStateVec, qubit: int, **_params: SimulatorGateParams) -> None:
    """Apply face rotation of an octahedron #1 (X->Y->Z->X).

    Args:
        state: An instance of CuStateVec
        qubit: The index of the qubit where the gate is applied
    """
    RX(state, qubit, angles=(pc.f64.frac_pi_2,))
    RZ(state, qubit, angles=(pc.f64.frac_pi_2,))


def Fdg(state: CuStateVec, qubit: int, **_params: SimulatorGateParams) -> None:
    """Apply adjoint of face rotation of an octahedron #1 (X<-Y<-Z<-X).

    Args:
        state: An instance of CuStateVec
        qubit: The index of the qubit where the gate is applied
    """
    RZ(state, qubit, angles=(-pc.f64.frac_pi_2,))
    RX(state, qubit, angles=(-pc.f64.frac_pi_2,))


def T(state: CuStateVec, qubit: int, **_params: SimulatorGateParams) -> None:
    """Apply a T gate.

    Args:
        state: An instance of CuStateVec
        qubit: The index of the qubit where the gate is applied
    """
    RZ(state, qubit, angles=(pc.f64.frac_pi_4,))


def Tdg(state: CuStateVec, qubit: int, **_params: SimulatorGateParams) -> None:
    """Apply adjoint of a T gate.

    Args:
        state: An instance of CuStateVec
        qubit: The index of the qubit where the gate is applied
    """
    RZ(state, qubit, angles=(-pc.f64.frac_pi_4,))


def H2(state: CuStateVec, qubit: int, **_params: SimulatorGateParams) -> None:
    """'H2': ('S', 'S', 'H', 'S', 'S')."""
    Z(state, qubit)
    H(state, qubit)
    Z(state, qubit)


def H3(state: CuStateVec, qubit: int, **_params: SimulatorGateParams) -> None:
    """'H3': ('H', 'S', 'S', 'H', 'S',)."""
    X(state, qubit)
    SZ(state, qubit)


def H4(state: CuStateVec, qubit: int, **_params: SimulatorGateParams) -> None:
    """'H4': ('H', 'S', 'S', 'H', 'S', 'S', 'S',)."""
    X(state, qubit)
    SZdg(state, qubit)


def H5(state: CuStateVec, qubit: int, **_params: SimulatorGateParams) -> None:
    """'H5': ('S', 'S', 'S', 'H', 'S')."""
    SZdg(state, qubit)
    H(state, qubit)
    SZ(state, qubit)


def H6(state: CuStateVec, qubit: int, **_params: SimulatorGateParams) -> None:
    """'H6': ('S', 'H', 'S', 'S', 'S',)."""
    SZ(state, qubit)
    H(state, qubit)
    SZdg(state, qubit)


def F2(state: CuStateVec, qubit: int, **_params: SimulatorGateParams) -> None:
    """'F2': ('S', 'S', 'H', 'S')."""
    Z(state, qubit)
    H(state, qubit)
    SZ(state, qubit)


def F2d(state: CuStateVec, qubit: int, **_params: SimulatorGateParams) -> None:
    """'F2d': ('S', 'S', 'S', 'H', 'S', 'S')."""
    SZdg(state, qubit)
    H(state, qubit)
    Z(state, qubit)


def F3(state: CuStateVec, qubit: int, **_params: SimulatorGateParams) -> None:
    """'F3': ('S', 'H', 'S', 'S')."""
    SZ(state, qubit)
    H(state, qubit)
    Z(state, qubit)


def F3d(state: CuStateVec, qubit: int, **_params: SimulatorGateParams) -> None:
    """'F3d': ('S', 'S', 'H', 'S', 'S', 'S')."""
    Z(state, qubit)
    H(state, qubit)
    SZdg(state, qubit)


def F4(state: CuStateVec, qubit: int, **_params: SimulatorGateParams) -> None:
    """'F4': ('H', 'S', 'S', 'S')."""
    H(state, qubit)
    SZdg(state, qubit)


def F4d(state: CuStateVec, qubit: int, **_params: SimulatorGateParams) -> None:
    """'F4d': ('S', 'H')."""
    SZ(state, qubit)
    H(state, qubit)
