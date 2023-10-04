# Copyright 2019 The PECOS Developers
# Copyright 2018 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract
# DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from __future__ import annotations

from typing import Any

import numpy as np
from projectq import ops

from pecos.simulators.projectq.helper import MakeFunc


def Identity(state, qubit: int, **params: Any) -> None:
    """Identity does nothing.

    X -> X

    Z -> Z

    Y -> Y

    Args:
    ----
        state (PauliFaultProp):  The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """


def X(state, qubit: int, **params: Any) -> None:
    """Pauli X.

    X -> X

    Z -> -Z

    Y -> -Y

    Args:
    ----
        state (PauliFaultProp):  The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """
    ops.X | state.qids[qubit]


def Y(state, qubit: int, **params: Any) -> None:
    """X -> -X.

    Z -> -Z

    Y -> Y

    Args:
    ----
        state (PauliFaultProp):  The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """
    ops.Y | state.qids[qubit]


def Z(state, qubit: int, **params: Any) -> None:
    """X -> -X.

    Z -> Z

    Y -> -Y

    Args:
    ----
        state (PauliFaultProp):  The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """
    ops.Z | state.qids[qubit]


RX = MakeFunc(ops.Rx, angle=True).func  # Rotation about X (takes angle arg)
RY = MakeFunc(ops.Ry, angle=True).func  # Rotation about Y (takes angle arg)
RZ = MakeFunc(ops.Rz, angle=True).func  # Rotation about Z (takes angle arg)


def R1XY(state, qubit: int, angles: tuple[float, float], **params: Any) -> None:
    """R1XY(theta, phi) = U1q(theta, phi) = RZ(phi-pi/2)*RY(theta)*RZ(-phi+pi/2).

    Args:
    ----
        state:
        qubit:
        angles:
        **params:

    Returns:
    -------

    """
    theta = angles[0]
    phi = angles[1]

    RZ(state, qubit, angle=-phi + np.pi / 2)
    RY(state, qubit, angle=theta)
    RZ(state, qubit, angle=phi - np.pi / 2)


def SX(state, qubit: int, **params: Any) -> None:
    """Square-root of X gate class."""
    RX(state, qubit, angle=np.pi / 2)


def SXdg(state, qubit: int, **params: Any) -> None:
    """Adjoint of the square-root of X gate class."""
    RX(state, qubit, angle=-np.pi / 2)


def SY(state, qubit: int, **params: Any) -> None:
    """Square-root of Y gate class."""
    RY(state, qubit, angle=np.pi / 2)


def SYdg(state, qubit: int, **params: Any) -> None:
    """Adjoint of the square-root of Y gate class."""
    RY(state, qubit, angle=-np.pi / 2)


def SZ(state, qubit: int, **params: Any) -> None:
    """Square-root of Z gate class."""
    ops.S | state.qids[qubit]


def SZdg(state, qubit: int, **params: Any) -> None:
    """Adjoint of the square-root of Z gate class."""
    ops.Sdag | state.qids[qubit]


def H(state, qubit: int, **params: Any) -> None:
    """Square root of Z.

    X -> Z

    Z -> X

    Y -> -Y

    Args:
    ----
        state (PauliFaultProp):  The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """
    ops.H | state.qids[qubit]


def H2(state, qubit: int, **params: Any) -> None:
    # @property
    # def matrix(self):

    ops.Ry(np.pi / 2) | state.qids[qubit]
    ops.Z | state.qids[qubit]


def H3(state, qubit: int, **params: Any) -> None:
    # @property
    # def matrix(self):

    ops.S | state.qids[qubit]
    ops.Y | state.qids[qubit]


def H4(state, qubit: int, **params: Any) -> None:
    # @property
    # def matrix(self):

    ops.S | state.qids[qubit]
    ops.X | state.qids[qubit]


def H5(state, qubit: int, **params: Any) -> None:
    # @property
    # def matrix(self):

    ops.Rx(np.pi / 2) | state.qids[qubit]
    ops.Z | state.qids[qubit]


def H6(state, qubit: int, **params: Any) -> None:
    # @property
    # def matrix(self):

    ops.Rx(np.pi / 2) | state.qids[qubit]
    ops.Y | state.qids[qubit]


def F(state, qubit: int, **params: Any) -> None:
    """Face rotations of an octahedron #1."""
    # @property
    # def matrix(self):

    ops.Rx(np.pi / 2) | state.qids[qubit]
    ops.Rz(np.pi / 2) | state.qids[qubit]


def Fdg(state, qubit: int, **params: Any) -> None:
    """Adjoint of face rotations of an octahedron #1."""
    ops.Rz(-np.pi / 2) | state.qids[qubit]
    ops.Rx(-np.pi / 2) | state.qids[qubit]


def F2(state, qubit: int, **params: Any) -> None:
    """Face rotations of an octahedron #2."""
    # @property
    # def matrix(self):

    ops.Rz(np.pi / 2) | state.qids[qubit]
    ops.Rx(-np.pi / 2) | state.qids[qubit]


def F2dg(state, qubit: int, **params: Any) -> None:
    """Adjoint of face rotations of an octahedron #2."""
    ops.Rx(np.pi / 2) | state.qids[qubit]
    ops.Rz(-np.pi / 2) | state.qids[qubit]


def F3(state, qubit: int, **params: Any) -> None:
    """Face rotations of an octahedron #3."""
    # @property
    # def matrix(self):

    ops.Rx(-np.pi / 2) | state.qids[qubit]
    ops.Rz(np.pi / 2) | state.qids[qubit]


def F3dg(state, qubit: int, **params: Any) -> None:
    """Adjoint of face rotations of an octahedron #3."""
    ops.Rz(-np.pi / 2) | state.qids[qubit]
    ops.Rx(np.pi / 2) | state.qids[qubit]


def F4(state, qubit: int, **params: Any) -> None:
    """Face rotations of an octahedron #4."""
    # @property
    # def matrix(self):

    ops.Rz(np.pi / 2) | state.qids[qubit]
    ops.Rx(np.pi / 2) | state.qids[qubit]


def F4dg(state, qubit: int, **params: Any) -> None:
    """Adjoint of face rotations of an octahedron #4."""
    ops.Rx(-np.pi / 2) | state.qids[qubit]
    ops.Rz(-np.pi / 2) | state.qids[qubit]
