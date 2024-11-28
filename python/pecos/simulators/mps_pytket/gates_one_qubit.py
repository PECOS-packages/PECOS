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
import math
from typing import Any

import cupy as cp
from pytket import Qubit


def _apply_one_qubit_matrix(state, qubit: int, matrix: cp.ndarray) -> None:
    """
    Apply the matrix to the state.

    Args:
        state: An instance of MPS
        qubit: The index of the qubit where the gate is applied
        matrix: The matrix to be applied
    """
    if qubit >= state.num_qubits or qubit < 0:
        msg = f"Qubit {qubit} out of range."
        raise ValueError(msg)

    state.mps.apply_unitary(matrix, [Qubit(qubit)])


def identity(state, qubit: int, **params: Any) -> None:
    """
    Identity gate.

    Args:
        state: An instance of MPS
        qubit: The index of the qubit where the gate is applied
    """


def X(state, qubit: int, **params: Any) -> None:
    """
    Pauli X gate.

    Args:
        state: An instance of MPS
        qubit: The index of the qubit where the gate is applied
    """
    matrix = cp.asarray(
        [
            [0, 1],
            [1, 0],
        ],
        dtype=state.dtype,
    )
    _apply_one_qubit_matrix(state, qubit, matrix)


def Y(state, qubit: int, **params: Any) -> None:
    """
    Pauli Y gate.

    Args:
        state: An instance of MPS
        qubit: The index of the qubit where the gate is applied
    """
    matrix = cp.asarray(
        [
            [0, -1j],
            [1j, 0],
        ],
        dtype=state.dtype,
    )
    _apply_one_qubit_matrix(state, qubit, matrix)


def Z(state, qubit: int, **params: Any) -> None:
    """
    Pauli Z gate.

    Args:
        state: An instance of MPS
        qubit: The index of the qubit where the gate is applied
    """
    matrix = cp.asarray(
        [
            [1, 0],
            [0, -1],
        ],
        dtype=state.dtype,
    )
    _apply_one_qubit_matrix(state, qubit, matrix)


def RX(state, qubit: int, angles: tuple[float], **params: Any) -> None:
    """
    Apply an RX gate.

    Args:
        state: An instance of MPS
        qubit: The index of the qubit where the gate is applied
        angles: A tuple containing a single angle in radians
    """
    if len(angles) != 1:
        msg = "Gate must be given 1 angle parameter."
        raise ValueError(msg)
    theta = angles[0]

    matrix = cp.asarray(
        [
            [math.cos(theta / 2), -1j * math.sin(theta / 2)],
            [-1j * math.sin(theta / 2), math.cos(theta / 2)],
        ],
        dtype=state.dtype,
    )
    _apply_one_qubit_matrix(state, qubit, matrix)


def RY(state, qubit: int, angles: tuple[float], **params: Any) -> None:
    """
    Apply an RY gate.

    Args:
        state: An instance of MPS
        qubit: The index of the qubit where the gate is applied
        angles: A tuple containing a single angle in radians
    """
    if len(angles) != 1:
        msg = "Gate must be given 1 angle parameter."
        raise ValueError(msg)
    theta = angles[0]

    matrix = cp.asarray(
        [
            [math.cos(theta / 2), -math.sin(theta / 2)],
            [math.sin(theta / 2), math.cos(theta / 2)],
        ],
        dtype=state.dtype,
    )
    _apply_one_qubit_matrix(state, qubit, matrix)


def RZ(state, qubit: int, angles: tuple[float], **params: Any) -> None:
    """
    Apply an RZ gate.

    Args:
        state: An instance of MPS
        qubit: The index of the qubit where the gate is applied
        angles: A tuple containing a single angle in radians
    """
    if len(angles) != 1:
        msg = "Gate must be given 1 angle parameter."
        raise ValueError(msg)
    theta = angles[0]

    matrix = cp.asarray(
        [
            [cmath.exp(-1j * theta / 2), 0],
            [0, cmath.exp(1j * theta / 2)],
        ],
        dtype=state.dtype,
    )
    _apply_one_qubit_matrix(state, qubit, matrix)


def R1XY(state, qubit: int, angles: tuple[float, float], **params: Any) -> None:
    """
    Apply an R1XY gate.

    Args:
        state: An instance of MPS
        qubit: The index of the qubit where the gate is applied
        angles: A tuple containing two angles in radians
    """
    if len(angles) != 2:
        msg = "Gate must be given 2 angle parameters."
        raise ValueError(msg)
    theta = angles[0]
    phi = angles[1]

    # Gate is equal to RZ(phi-pi/2)*RY(theta)*RZ(-phi+pi/2)
    RZ(state, qubit, angles=(-phi + math.pi / 2,))
    RY(state, qubit, angles=(theta,))
    RZ(state, qubit, angles=(phi - math.pi / 2,))


def SX(state, qubit: int, **params: Any) -> None:
    """
    Apply a square-root of X.

    Args:
        state: An instance of MPS
        qubit: The index of the qubit where the gate is applied
    """
    RX(state, qubit, angles=(math.pi / 2,))


def SXdg(state, qubit: int, **params: Any) -> None:
    """
    Apply adjoint of the square-root of X.

    Args:
        state: An instance of MPS
        qubit: The index of the qubit where the gate is applied
    """
    RX(state, qubit, angles=(-math.pi / 2,))


def SY(state, qubit: int, **params: Any) -> None:
    """
    Apply a square-root of Y.

    Args:
        state: An instance of MPS
        qubit: The index of the qubit where the gate is applied
    """
    RY(state, qubit, angles=(math.pi / 2,))


def SYdg(state, qubit: int, **params: Any) -> None:
    """
    Apply adjoint of the square-root of Y.

    Args:
        state: An instance of MPS
        qubit: The index of the qubit where the gate is applied
    """
    RY(state, qubit, angles=(-math.pi / 2,))


def SZ(state, qubit: int, **params: Any) -> None:
    """
    Apply a square-root of Z.

    Args:
        state: An instance of MPS
        qubit: The index of the qubit where the gate is applied
    """
    RZ(state, qubit, angles=(math.pi / 2,))


def SZdg(state, qubit: int, **params: Any) -> None:
    """
    Apply adjoint of the square-root of Z.

    Args:
        state: An instance of MPS
        qubit: The index of the qubit where the gate is applied
    """
    RZ(state, qubit, angles=(-math.pi / 2,))


def H(state, qubit: int, **params: Any) -> None:
    """
    Apply Hadamard gate.

    Args:
        state: An instance of MPS
        qubit: The index of the qubit where the gate is applied
    """
    matrix = (
        1
        / cp.sqrt(2)
        * cp.asarray(
            [
                [1, 1],
                [1, -1],
            ],
            dtype=state.dtype,
        )
    )
    _apply_one_qubit_matrix(state, qubit, matrix)


def F(state, qubit: int, **params: Any) -> None:
    """
    Apply face rotation of an octahedron #1 (X->Y->Z->X).

    Args:
        state: An instance of MPS
        qubit: The index of the qubit where the gate is applied
    """
    RX(state, qubit, angles=(math.pi / 2,))
    RZ(state, qubit, angles=(math.pi / 2,))


def Fdg(state, qubit: int, **params: Any) -> None:
    """
    Apply adjoint of face rotation of an octahedron #1 (X<-Y<-Z<-X).

    Args:
        state: An instance of MPS
        qubit: The index of the qubit where the gate is applied
    """
    RZ(state, qubit, angles=(-math.pi / 2,))
    RX(state, qubit, angles=(-math.pi / 2,))


def T(state, qubit: int, **params: Any) -> None:
    """
    Apply a T gate.

    Args:
        state: An instance of MPS
        qubit: The index of the qubit where the gate is applied
    """
    RZ(state, qubit, angles=(math.pi / 4,))


def Tdg(state, qubit: int, **params: Any) -> None:
    """
    Apply adjoint of a T gate.

    Args:
        state: An instance of MPS
        qubit: The index of the qubit where the gate is applied
    """
    RZ(state, qubit, angles=(-math.pi / 4,))


# The definition of the extra Clifford gates added below come from
# circuit_converters/std2chs.py
def H2(state, qubit: int, **params: Any) -> None:
    """'H2': ('S', 'S', 'H', 'S', 'S')"""
    Z(state, qubit)
    H(state, qubit)
    Z(state, qubit)


def H3(state, qubit: int, **params: Any) -> None:
    """'H3': ('H', 'S', 'S', 'H', 'S',)"""
    X(state, qubit)
    SZ(state, qubit)


def H4(state, qubit: int, **params: Any) -> None:
    """'H4': ('H', 'S', 'S', 'H', 'S', 'S', 'S',)"""
    X(state, qubit)
    SZdg(state, qubit)


def H5(state, qubit: int, **params: Any) -> None:
    """'H5': ('S', 'S', 'S', 'H', 'S')"""
    SZdg(state, qubit)
    H(state, qubit)
    SZ(state, qubit)


def H6(state, qubit: int, **params: Any) -> None:
    """'H6': ('S', 'H', 'S', 'S', 'S',)"""
    SZ(state, qubit)
    H(state, qubit)
    SZdg(state, qubit)


def F2(state, qubit: int, **params: Any) -> None:
    """'F2': ('S', 'S', 'H', 'S')"""
    Z(state, qubit)
    H(state, qubit)
    SZ(state, qubit)


def F2d(state, qubit: int, **params: Any) -> None:
    """'F2d': ('S', 'S', 'S', 'H', 'S', 'S')"""
    SZdg(state, qubit)
    H(state, qubit)
    Z(state, qubit)


def F3(state, qubit: int, **params: Any) -> None:
    """'F3': ('S', 'H', 'S', 'S')"""
    SZ(state, qubit)
    H(state, qubit)
    Z(state, qubit)


def F3d(state, qubit: int, **params: Any) -> None:
    """'F3d': ('S', 'S', 'H', 'S', 'S', 'S')"""
    Z(state, qubit)
    H(state, qubit)
    SZdg(state, qubit)


def F4(state, qubit: int, **params: Any) -> None:
    """'F4': ('H', 'S', 'S', 'S')"""
    H(state, qubit)
    SZdg(state, qubit)


def F4d(state, qubit: int, **params: Any) -> None:
    """'F4d': ('S', 'H')"""
    SZ(state, qubit)
    H(state, qubit)
