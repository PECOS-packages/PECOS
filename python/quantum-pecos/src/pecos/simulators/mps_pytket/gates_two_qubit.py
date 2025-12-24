# Copyright 2024 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Two-qubit gate operations for MPS PyTket simulator.

This module provides two-qubit quantum gate operations for the Matrix Product State PyTket simulator, including
CNOT gates, controlled gates, and other entangling operations with MPS bond dimension management.
"""

from __future__ import annotations

import cmath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pecos.simulators.mps_pytket.state import MPS
    from pecos.typing import SimulatorGateParams

import cupy as cp
from pytket import Qubit

import pecos as pc
from pecos.simulators.mps_pytket.gates_one_qubit import H


def _apply_two_qubit_matrix(
    state: MPS,
    qubits: tuple[int, int],
    matrix: cp.ndarray,
) -> None:
    """Apply the matrix to the state.

    Args:
        state: An instance of MPS
        qubits: The index of the qubits where the gate is applied
        matrix: The matrix to be applied
    """
    if qubits[0] >= state.num_qubits or qubits[0] < 0:
        msg = f"Qubit {qubits[0]} out of range."
        raise ValueError(msg)
    if qubits[1] >= state.num_qubits or qubits[1] < 0:
        msg = f"Qubit {qubits[1]} out of range."
        raise ValueError(msg)

    state.mps.apply_unitary(matrix, [Qubit(q) for q in qubits])


def CX(state: MPS, qubits: tuple[int, int], **_params: SimulatorGateParams) -> None:
    """Apply controlled X gate.

    Args:
        state: An instance of MPS
        qubits: A tuple with the index of the qubits where the gate is applied
            The one at `qubits[0]` is the control qubit.
    """
    matrix = cp.asarray(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=state.dtype,
    )
    _apply_two_qubit_matrix(state, qubits, matrix)


def CY(state: MPS, qubits: tuple[int, int], **_params: SimulatorGateParams) -> None:
    """Apply controlled Y gate.

    Args:
        state: An instance of MPS
        qubits: A tuple with the index of the qubits where the gate is applied
            The one at `qubits[0]` is the control qubit.
    """
    matrix = cp.asarray(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, -1j],
            [0, 0, 1j, 0],
        ],
        dtype=state.dtype,
    )
    _apply_two_qubit_matrix(state, qubits, matrix)


def CZ(state: MPS, qubits: tuple[int, int], **_params: SimulatorGateParams) -> None:
    """Apply controlled Z gate.

    Args:
        state: An instance of MPS
        qubits: A tuple with the index of the qubits where the gate is applied
            The one at `qubits[0]` is the control qubit.
    """
    matrix = cp.asarray(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1],
        ],
        dtype=state.dtype,
    )
    _apply_two_qubit_matrix(state, qubits, matrix)


def RXX(
    state: MPS,
    qubits: tuple[int, int],
    angles: tuple[float],
    **_params: SimulatorGateParams,
) -> None:
    """Apply a rotation about XX.

    Args:
        state: An instance of MPS
        qubits: A tuple with the index of the qubits where the gate is applied
        angles: A tuple containing a single angle in radians
    """
    if len(angles) != 1:
        msg = "Gate must be given 1 angle parameter."
        raise ValueError(msg)
    theta = angles[0]

    matrix = cp.asarray(
        [
            [pc.cos(theta / 2), 0, 0, -1j * pc.sin(theta / 2)],
            [0, pc.cos(theta / 2), -1j * pc.sin(theta / 2), 0],
            [0, -1j * pc.sin(theta / 2), pc.cos(theta / 2), 0],
            [-1j * pc.sin(theta / 2), 0, 0, pc.cos(theta / 2)],
        ],
        dtype=state.dtype,
    )
    _apply_two_qubit_matrix(state, qubits, matrix)


def RYY(
    state: MPS,
    qubits: tuple[int, int],
    angles: tuple[float],
    **_params: SimulatorGateParams,
) -> None:
    """Apply a rotation about YY.

    Args:
        state: An instance of MPS
        qubits: A tuple with the index of the qubits where the gate is applied
        angles: A tuple containing a single angle in radians
    """
    if len(angles) != 1:
        msg = "Gate must be given 1 angle parameter."
        raise ValueError(msg)
    theta = angles[0]

    matrix = cp.asarray(
        [
            [pc.cos(theta / 2), 0, 0, 1j * pc.sin(theta / 2)],
            [0, pc.cos(theta / 2), -1j * pc.sin(theta / 2), 0],
            [0, -1j * pc.sin(theta / 2), pc.cos(theta / 2), 0],
            [1j * pc.sin(theta / 2), 0, 0, pc.cos(theta / 2)],
        ],
        dtype=state.dtype,
    )
    _apply_two_qubit_matrix(state, qubits, matrix)


def RZZ(
    state: MPS,
    qubits: tuple[int, int],
    angles: tuple[float],
    **_params: SimulatorGateParams,
) -> None:
    """Apply a rotation about ZZ.

    Args:
        state: An instance of MPS
        qubits: A tuple with the index of the qubits where the gate is applied
        angles: A tuple containing a single angle in radians
    """
    if len(angles) != 1:
        msg = "Gate must be given 1 angle parameter."
        raise ValueError(msg)
    theta = angles[0]

    matrix = cp.asarray(
        [
            [cmath.exp(-1j * theta / 2), 0, 0, 0],
            [0, cmath.exp(1j * theta / 2), 0, 0],
            [0, 0, cmath.exp(1j * theta / 2), 0],
            [0, 0, 0, cmath.exp(-1j * theta / 2)],
        ],
        dtype=state.dtype,
    )
    _apply_two_qubit_matrix(state, qubits, matrix)


def R2XXYYZZ(
    state: MPS,
    qubits: tuple[int, int],
    angles: tuple[float, float, float],
    **_params: SimulatorGateParams,
) -> None:
    """Apply RXX*RYY*RZZ.

    Args:
        state: An instance of MPS
        qubits: A tuple with the index of the qubits where the gate is applied
        angles: A tuple containing three angles in radians, for XX, YY and ZZ, in that order
    """
    if len(angles) != 3:
        msg = "Gate must be given 3 angle parameters."
        raise ValueError(msg)

    RXX(state, qubits, (angles[0],))
    RYY(state, qubits, (angles[1],))
    RZZ(state, qubits, (angles[2],))


def SXX(state: MPS, qubits: tuple[int, int], **_params: SimulatorGateParams) -> None:
    """Apply a square root of XX gate.

    Args:
        state: An instance of MPS
        qubits: A tuple with the index of the qubits where the gate is applied
    """
    RXX(state, qubits, angles=(pc.f64.frac_pi_2,))


def SXXdg(state: MPS, qubits: tuple[int, int], **_params: SimulatorGateParams) -> None:
    """Apply adjoint of a square root of XX gate.

    Args:
        state: An instance of MPS
        qubits: A tuple with the index of the qubits where the gate is applied
    """
    RXX(state, qubits, angles=(-pc.f64.frac_pi_2,))


def SYY(state: MPS, qubits: tuple[int, int], **_params: SimulatorGateParams) -> None:
    """Apply a square root of YY gate.

    Args:
        state: An instance of MPS
        qubits: A tuple with the index of the qubits where the gate is applied
    """
    RYY(state, qubits, angles=(pc.f64.frac_pi_2,))


def SYYdg(state: MPS, qubits: tuple[int, int], **_params: SimulatorGateParams) -> None:
    """Apply adjoint of a square root of YY gate.

    Args:
        state: An instance of MPS
        qubits: A tuple with the index of the qubits where the gate is applied
    """
    RYY(state, qubits, angles=(-pc.f64.frac_pi_2,))


def SZZ(state: MPS, qubits: tuple[int, int], **_params: SimulatorGateParams) -> None:
    """Apply a square root of ZZ gate.

    Args:
        state: An instance of MPS
        qubits: A tuple with the index of the qubits where the gate is applied
    """
    RZZ(state, qubits, angles=(pc.f64.frac_pi_2,))


def SZZdg(state: MPS, qubits: tuple[int, int], **_params: SimulatorGateParams) -> None:
    """Apply adjoint of a square root of ZZ gate.

    Args:
        state: An instance of MPS
        qubits: A tuple with the index of the qubits where the gate is applied
    """
    RZZ(state, qubits, angles=(-pc.f64.frac_pi_2,))


def SWAP(state: MPS, qubits: tuple[int, int], **_params: SimulatorGateParams) -> None:
    """Apply a SWAP gate.

    Args:
        state: An instance of MPS
        qubits: A tuple with the index of the qubits where the gate is applied
    """
    matrix = cp.asarray(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=state.dtype,
    )
    _apply_two_qubit_matrix(state, qubits, matrix)


def G(state: MPS, qubits: tuple[int, int], **_params: SimulatorGateParams) -> None:
    """'G': (('I', 'H'), 'CNOT', ('H', 'H'), 'CNOT', ('I', 'H'))."""
    H(state, qubits[1])
    CX(state, qubits)
    H(state, qubits[0])
    H(state, qubits[1])
    CX(state, qubits)
    H(state, qubits[1])
