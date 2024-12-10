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

import cmath
from typing import Any

import numpy as np
import pyquest.unitaries as qgate

from pecos.simulators.quest.gates_one_qubit import H


def CX(state, qubits: tuple[int, int], **params: Any) -> None:
    """
    Apply controlled X gate.

    Args:
        state: An instance of QuEST
        qubits: A tuple with the index of the qubits where the gate is applied
            The one at `qubits[0]` is the control qubit.
    """
    # QuEST uses qubit index 0 as the least significant bit
    control = state.num_qubits - qubits[0] - 1
    target = state.num_qubits - qubits[1] - 1
    state.quest_state.apply_operator(qgate.X(target, controls=[control]))


def CY(state, qubits: tuple[int, int], **params: Any) -> None:
    """
    Apply controlled Y gate.

    Args:
        state: An instance of QuEST
        qubits: A tuple with the index of the qubits where the gate is applied
            The one at `qubits[0]` is the control qubit.
    """
    # QuEST uses qubit index 0 as the least significant bit
    control = state.num_qubits - qubits[0] - 1
    target = state.num_qubits - qubits[1] - 1
    state.quest_state.apply_operator(qgate.Y(target, controls=[control]))


def CZ(state, qubits: tuple[int, int], **params: Any) -> None:
    """
    Apply controlled Z gate.

    Args:
        state: An instance of QuEST
        qubits: A tuple with the index of the qubits where the gate is applied
            The one at `qubits[0]` is the control qubit.
    """
    # QuEST uses qubit index 0 as the least significant bit
    control = state.num_qubits - qubits[0] - 1
    target = state.num_qubits - qubits[1] - 1
    state.quest_state.apply_operator(qgate.Z(target, controls=[control]))


def RXX(state, qubits: tuple[int, int], angles: tuple[float], **params: Any) -> None:
    """
    Apply a rotation about XX.

    Args:
        state: An instance of QuEST
        qubits: A tuple with the index of the qubits where the gate is applied
        angles: A tuple containing a single angle in radians
    """
    if len(angles) != 1:
        msg = "Gate must be given 1 angle parameter."
        raise ValueError(msg)
    theta = angles[0]

    # QuEST uses qubit index 0 as the least significant bit
    idxs = [state.num_qubits - q - 1 for q in qubits]

    matrix = np.array(
        [
            [cmath.cos(theta / 2), 0, 0, -1j * cmath.sin(theta / 2)],
            [0, cmath.cos(theta / 2), -1j * cmath.sin(theta / 2), 0],
            [0, -1j * cmath.sin(theta / 2), cmath.cos(theta / 2), 0],
            [-1j * cmath.sin(theta / 2), 0, 0, cmath.cos(theta / 2)],
        ],
    )
    state.quest_state.apply_operator(qgate.U(targets=idxs, matrix=matrix))


def RYY(state, qubits: tuple[int, int], angles: tuple[float], **params: Any) -> None:
    """
    Apply a rotation about YY.

    Args:
        state: An instance of QuEST
        qubits: A tuple with the index of the qubits where the gate is applied
        angles: A tuple containing a single angle in radians
    """
    if len(angles) != 1:
        msg = "Gate must be given 1 angle parameter."
        raise ValueError(msg)
    theta = angles[0]

    # QuEST uses qubit index 0 as the least significant bit
    idxs = [state.num_qubits - q - 1 for q in qubits]

    matrix = np.array(
        [
            [cmath.cos(theta / 2), 0, 0, 1j * cmath.sin(theta / 2)],
            [0, cmath.cos(theta / 2), -1j * cmath.sin(theta / 2), 0],
            [0, -1j * cmath.sin(theta / 2), cmath.cos(theta / 2), 0],
            [1j * cmath.sin(theta / 2), 0, 0, cmath.cos(theta / 2)],
        ],
    )
    state.quest_state.apply_operator(qgate.U(targets=idxs, matrix=matrix))


def RZZ(state, qubits: tuple[int, int], angles: tuple[float], **params: Any) -> None:
    """
    Apply a rotation about ZZ.

    Args:
        state: An instance of QuEST
        qubits: A tuple with the index of the qubits where the gate is applied
        angles: A tuple containing a single angle in radians
    """
    if len(angles) != 1:
        msg = "Gate must be given 1 angle parameter."
        raise ValueError(msg)
    theta = angles[0]

    # QuEST uses qubit index 0 as the least significant bit
    idxs = [state.num_qubits - q - 1 for q in qubits]

    matrix = np.array(
        [
            [cmath.exp(-1j * theta / 2), 0, 0, 0],
            [0, cmath.exp(1j * theta / 2), 0, 0],
            [0, 0, cmath.exp(1j * theta / 2), 0],
            [0, 0, 0, cmath.exp(-1j * theta / 2)],
        ],
    )
    state.quest_state.apply_operator(qgate.U(targets=idxs, matrix=matrix))


def R2XXYYZZ(
    state,
    qubits: tuple[int, int],
    angles: tuple[float, float, float],
    **params: Any,
) -> None:
    """
    Apply RXX*RYY*RZZ.

    Args:
        state: An instance of QuEST
        qubits: A tuple with the index of the qubits where the gate is applied
        angles: A tuple containing three angles in radians, for XX, YY and ZZ, in that order
    """
    if len(angles) != 3:
        msg = "Gate must be given 3 angle parameters."
        raise ValueError(msg)

    RXX(state, qubits, (angles[0],))
    RYY(state, qubits, (angles[1],))
    RZZ(state, qubits, (angles[2],))


def SXX(state, qubits: tuple[int, int], **params: Any) -> None:
    """
    Apply a square root of XX gate.

    Args:
        state: An instance of QuEST
        qubits: A tuple with the index of the qubits where the gate is applied
    """
    RXX(state, qubits, angles=(cmath.pi / 2,))


def SXXdg(state, qubits: tuple[int, int], **params: Any) -> None:
    """
    Apply adjoint of a square root of XX gate.

    Args:
        state: An instance of QuEST
        qubits: A tuple with the index of the qubits where the gate is applied
    """
    RXX(state, qubits, angles=(-cmath.pi / 2,))


def SYY(state, qubits: tuple[int, int], **params: Any) -> None:
    """
    Apply a square root of YY gate.

    Args:
        state: An instance of QuEST
        qubits: A tuple with the index of the qubits where the gate is applied
    """
    RYY(state, qubits, angles=(cmath.pi / 2,))


def SYYdg(state, qubits: tuple[int, int], **params: Any) -> None:
    """
    Apply adjoint of a square root of YY gate.

    Args:
        state: An instance of QuEST
        qubits: A tuple with the index of the qubits where the gate is applied
    """
    RYY(state, qubits, angles=(-cmath.pi / 2,))


def SZZ(state, qubits: tuple[int, int], **params: Any) -> None:
    """
    Apply a square root of ZZ gate.

    Args:
        state: An instance of QuEST
        qubits: A tuple with the index of the qubits where the gate is applied
    """
    RZZ(state, qubits, angles=(cmath.pi / 2,))


def SZZdg(state, qubits: tuple[int, int], **params: Any) -> None:
    """
    Apply adjoint of a square root of ZZ gate.

    Args:
        state: An instance of QuEST
        qubits: A tuple with the index of the qubits where the gate is applied
    """
    RZZ(state, qubits, angles=(-cmath.pi / 2,))


def SWAP(state, qubits: tuple[int, int], **params: Any) -> None:
    """
    Apply a SWAP gate.

    Args:
        state: An instance of QuEST
        qubits: A tuple with the index of the qubits where the gate is applied
    """
    # QuEST uses qubit index 0 as the least significant bit
    idxs = [state.num_qubits - q - 1 for q in qubits]

    state.quest_state.apply_operator(qgate.Swap(targets=idxs))


def G(state, qubits: tuple[int, int], **params: Any) -> None:
    """'G': (('I', 'H'), 'CNOT', ('H', 'H'), 'CNOT', ('I', 'H'))"""
    H(state, qubits[1])
    CX(state, qubits)
    H(state, qubits[0])
    H(state, qubits[1])
    CX(state, qubits)
    H(state, qubits[1])
