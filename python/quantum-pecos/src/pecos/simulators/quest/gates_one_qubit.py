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

from typing import Any

import numpy as np
import pyquest.unitaries as qgate


def identity(state, qubit: int, **params: Any) -> None:
    """
    Identity gate.

    Args:
        state: An instance of QuEST
        qubit: The index of the qubit where the gate is applied
    """


def X(state, qubit: int, **params: Any) -> None:
    """
    Pauli X gate.

    Args:
        state: An instance of QuEST
        qubit: The index of the qubit where the gate is applied
    """
    # QuEST uses qubit index 0 as the least significant bit
    idx = state.num_qubits - qubit - 1
    state.quest_state.apply_operator(qgate.X(idx))


def Y(state, qubit: int, **params: Any) -> None:
    """
    Pauli Y gate.

    Args:
        state: An instance of QuEST
        qubit: The index of the qubit where the gate is applied
    """
    # QuEST uses qubit index 0 as the least significant bit
    idx = state.num_qubits - qubit - 1
    state.quest_state.apply_operator(qgate.Y(idx))


def Z(state, qubit: int, **params: Any) -> None:
    """
    Pauli Z gate.

    Args:
        state: An instance of QuEST
        qubit: The index of the qubit where the gate is applied
    """
    # QuEST uses qubit index 0 as the least significant bit
    idx = state.num_qubits - qubit - 1
    state.quest_state.apply_operator(qgate.Z(idx))


def RX(state, qubit: int, angles: tuple[float], **params: Any) -> None:
    """
    Apply an RX gate.

    Args:
        state: An instance of QuEST
        qubit: The index of the qubit where the gate is applied
        angles: A tuple containing a single angle in radians
    """
    if len(angles) != 1:
        msg = "Gate must be given 1 angle parameter."
        raise ValueError(msg)
    theta = angles[0]

    # QuEST uses qubit index 0 as the least significant bit
    idx = state.num_qubits - qubit - 1
    state.quest_state.apply_operator(qgate.Rx(idx, theta))


def RY(state, qubit: int, angles: tuple[float], **params: Any) -> None:
    """
    Apply an RY gate.

    Args:
        state: An instance of QuEST
        qubit: The index of the qubit where the gate is applied
        angles: A tuple containing a single angle in radians
    """
    if len(angles) != 1:
        msg = "Gate must be given 1 angle parameter."
        raise ValueError(msg)
    theta = angles[0]

    # QuEST uses qubit index 0 as the least significant bit
    idx = state.num_qubits - qubit - 1
    state.quest_state.apply_operator(qgate.Ry(idx, theta))


def RZ(state, qubit: int, angles: tuple[float], **params: Any) -> None:
    """
    Apply an RZ gate.

    Args:
        state: An instance of QuEST
        qubit: The index of the qubit where the gate is applied
        angles: A tuple containing a single angle in radians
    """
    if len(angles) != 1:
        msg = "Gate must be given 1 angle parameter."
        raise ValueError(msg)
    theta = angles[0]

    # QuEST uses qubit index 0 as the least significant bit
    idx = state.num_qubits - qubit - 1
    state.quest_state.apply_operator(qgate.Rz(idx, theta))


def R1XY(state, qubit: int, angles: tuple[float, float], **params: Any) -> None:
    """
    Apply an R1XY gate.

    Args:
        state: An instance of QuEST
        qubit: The index of the qubit where the gate is applied
        angles: A tuple containing two angles in radians
    """
    if len(angles) != 2:
        msg = "Gate must be given 2 angle parameters."
        raise ValueError(msg)
    theta = angles[0]
    phi = angles[1]

    # Gate is equal to RZ(phi-pi/2)*RY(theta)*RZ(-phi+pi/2)
    RZ(state, qubit, angles=(-phi + np.pi / 2,))
    RY(state, qubit, angles=(theta,))
    RZ(state, qubit, angles=(phi - np.pi / 2,))


def SX(state, qubit: int, **params: Any) -> None:
    """
    Apply a square-root of X.

    Args:
        state: An instance of QuEST
        qubit: The index of the qubit where the gate is applied
    """
    RX(state, qubit, angles=(np.pi / 2,))


def SXdg(state, qubit: int, **params: Any) -> None:
    """
    Apply adjoint of the square-root of X.

    Args:
        state: An instance of QuEST
        qubit: The index of the qubit where the gate is applied
    """
    RX(state, qubit, angles=(-np.pi / 2,))


def SY(state, qubit: int, **params: Any) -> None:
    """
    Apply a square-root of Y.

    Args:
        state: An instance of QuEST
        qubit: The index of the qubit where the gate is applied
    """
    RY(state, qubit, angles=(np.pi / 2,))


def SYdg(state, qubit: int, **params: Any) -> None:
    """
    Apply adjoint of the square-root of Y.

    Args:
        state: An instance of QuEST
        qubit: The index of the qubit where the gate is applied
    """
    RY(state, qubit, angles=(-np.pi / 2,))


def SZ(state, qubit: int, **params: Any) -> None:
    """
    Apply a square-root of Z.

    Args:
        state: An instance of QuEST
        qubit: The index of the qubit where the gate is applied
    """
    RZ(state, qubit, angles=(np.pi / 2,))


def SZdg(state, qubit: int, **params: Any) -> None:
    """
    Apply adjoint of the square-root of Z.

    Args:
        state: An instance of QuEST
        qubit: The index of the qubit where the gate is applied
    """
    RZ(state, qubit, angles=(-np.pi / 2,))


def H(state, qubit: int, **params: Any) -> None:
    """
    Apply Hadamard gate.

    Args:
        state: An instance of QuEST
        qubit: The index of the qubit where the gate is applied
    """
    # QuEST uses qubit index 0 as the least significant bit
    idx = state.num_qubits - qubit - 1
    state.quest_state.apply_operator(qgate.H(idx))


def F(state, qubit: int, **params: Any) -> None:
    """
    Apply face rotation of an octahedron #1 (X->Y->Z->X).

    Args:
        state: An instance of QuEST
        qubit: The index of the qubit where the gate is applied
    """
    RX(state, qubit, angles=(np.pi / 2,))
    RZ(state, qubit, angles=(np.pi / 2,))


def Fdg(state, qubit: int, **params: Any) -> None:
    """
    Apply adjoint of face rotation of an octahedron #1 (X<-Y<-Z<-X).

    Args:
        state: An instance of QuEST
        qubit: The index of the qubit where the gate is applied
    """
    RZ(state, qubit, angles=(-np.pi / 2,))
    RX(state, qubit, angles=(-np.pi / 2,))


def T(state, qubit: int, **params: Any) -> None:
    """
    Apply a T gate.

    Args:
        state: An instance of QuEST
        qubit: The index of the qubit where the gate is applied
    """
    RZ(state, qubit, angles=(np.pi / 4,))


def Tdg(state, qubit: int, **params: Any) -> None:
    """
    Apply adjoint of a T gate.

    Args:
        state: An instance of QuEST
        qubit: The index of the qubit where the gate is applied
    """
    RZ(state, qubit, angles=(-np.pi / 4,))
