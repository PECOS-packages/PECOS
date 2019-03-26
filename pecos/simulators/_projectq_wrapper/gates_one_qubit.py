#  =========================================================================  #
#   Copyright 2018 National Technology & Engineering Solutions of Sandia,
#   LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS,
#   the U.S. Government retains certain rights in this software.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#  =========================================================================  #

import cmath
import numpy as np
import projectq.ops as ops
from projectq.ops import BasicGate, get_inverse
from .helper import MakeFunc


def I(state,
      qubit: int) -> None:
    """
    Identity does nothing.

    X -> X

    Z -> Z

    Y -> Y

    Args:
        state (PauliFaultProp):  The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """
    pass


def X(state,
      qubit: int) -> None:
    """
    Pauli X

    X -> X

    Z -> -Z

    Y -> -Y

    Args:
        state (PauliFaultProp):  The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """

    ops.X | state.qids[qubit]


def Y(state,
      qubit: int) -> None:
    """
    X -> -X

    Z -> -Z

    Y -> Y

    Args:
        state (PauliFaultProp):  The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """

    ops.Y | state.qids[qubit]


def Z(state,
      qubit: int) -> None:
    """
    X -> -X

    Z -> Z

    Y -> -Y

    Args:
        state (PauliFaultProp):  The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """

    ops.Z | state.qids[qubit]


def H(state,
      qubit: int) -> None:
    """
    Square root of Z.

    X -> Z

    Z -> X

    Y -> -Y

    Args:
        state (PauliFaultProp):  The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """

    ops.H | state.qids[qubit]


class QGate(BasicGate):
    """Square-root of X gate class"""
    @property
    def matrix(self):
        return 0.5 * np.matrix([[1+1j, 1-1j], [1-1j, 1+1j]])

    def __str__(self):
        return "Q"


Q = MakeFunc(QGate()).func
Qd = MakeFunc(get_inverse(QGate())).func

class RGate(BasicGate):
    """Square-root of Y gate class"""
    @property
    def matrix(self):
        return 0.5 * np.matrix([[1+1j, -1-1j], [1+1j, 1+1j]])

    def __str__(self):
        return "R"

R = MakeFunc(RGate()).func
Rd = MakeFunc(get_inverse(RGate())).func


S = MakeFunc(ops.S).func
Sd = MakeFunc(ops.Sdag).func


class H2Gate(BasicGate):
    """Hadmard-like gate #2"""
    @property
    def matrix(self):
        return 0.5 * np.matrix([[1+1j, -1-1j], [-1-1j, -1-1j]])

    def __str__(self):
        return "H2"


H2 = MakeFunc(H2Gate()).func


class H3Gate(BasicGate):
    """Hadmard-like gate #3"""
    @property
    def matrix(self):
        return np.matrix([[0, 1], [1j, 0]])

    def __str__(self):
        return "H3"


H3 = MakeFunc(H3Gate()).func


class H4Gate(BasicGate):
    """Hadmard-like gate #4"""
    @property
    def matrix(self):
        return np.matrix([[0, 1j], [1, 0]])

    def __str__(self):
        return "H4"


H4 = MakeFunc(H4Gate()).func


class H5Gate(BasicGate):
    """Hadmard-like gate #5"""
    @property
    def matrix(self):
        return 0.5 * np.matrix([[1+1j, 1-1j], [-1+1j, -1-1j]])

    def __str__(self):
        return "H5"


H5 = MakeFunc(H5Gate()).func


class H6Gate(BasicGate):
    """Hadmard-like gate #6"""
    @property
    def matrix(self):
        return 0.5 * np.matrix([[-1-1j, 1-1j], [-1+1j, 1+1j]])

    def __str__(self):
        return "H6"


H6 = MakeFunc(H6Gate()).func


class F1Gate(BasicGate):
    """Face rotations of an octahedron #1"""
    @property
    def matrix(self):
        return 0.5 * np.matrix([[1+1j, 1-1j], [1+1j, -1+1j]])

    def __str__(self):
        return "F1"


F1 = MakeFunc(F1Gate()).func
F1d = MakeFunc(get_inverse(F1Gate())).func


class F2Gate(BasicGate):
    """Face rotations of an octahedron #2"""
    @property
    def matrix(self):
        return 0.5 * np.matrix([[1-1j, -1+1j], [1+1j, 1+1j]])

    def __str__(self):
        return "F2"


F2 = MakeFunc(F2Gate()).func
F2d = MakeFunc(get_inverse(F2Gate())).func


class F3Gate(BasicGate):
    """Face rotations of an octahedron #3"""
    @property
    def matrix(self):
        return 0.5 * np.matrix([[1-1j, 1+1j], [-1+1j, 1+1j]])

    def __str__(self):
        return "F3"


F3 = MakeFunc(F3Gate()).func
F3d = MakeFunc(get_inverse(F3Gate())).func


class F4Gate(BasicGate):
    """Face rotations of an octahedron #3"""
    @property
    def matrix(self):
        return 0.5 * np.matrix([[1+1j, 1+1j], [1-1j, -1+1j]])

    def __str__(self):
        return "F4"


F4 = MakeFunc(F4Gate()).func
F4d = MakeFunc(get_inverse(F4Gate())).func
