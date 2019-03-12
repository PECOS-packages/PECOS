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

from typing import Tuple
import cmath
import numpy as np
from projectq import ops
from projectq.ops import BasicRotationGate
from .gates_one_qubit import Q, R, Rd, H


def II(state,
       qubits: Tuple[int, int]) -> None:
    pass


def G2(state,
       qubits: Tuple[int, int]) -> None:
    """
    Applies a CZ.H(1).H(2).CZ

    Returns:

    """

    CZ(state, qubits)
    H(state, qubits[0])
    H(state, qubits[1])
    CZ(state, qubits)


def CNOT(state,
         qubits: Tuple[int, int]) -> None:
    q1 = state.qids[qubits[0]]
    q2 = state.qids[qubits[1]]

    ops.CNOT | (q1, q2)


def CZ(state,
         qubits: Tuple[int, int]) -> None:
    q1 = state.qids[qubits[0]]
    q2 = state.qids[qubits[1]]

    ops.C(ops.Z) | (q1, q2)


def CY(state,
         qubits: Tuple[int, int]) -> None:
    q1 = state.qids[qubits[0]]
    q2 = state.qids[qubits[1]]

    ops.C(ops.Z) | (q1, q2)


def SWAP(state,
         qubits: Tuple[int, int]) -> None:
    q1 = state.qids[qubits[0]]
    q2 = state.qids[qubits[1]]

    ops.SWAP | (q1, q2)


def SqrtXX(state,
           qubits: Tuple[int, int]) -> None:
    """
    Applies a square root of XX rotation to generators

    state (SparseSim): Instance representing the stabilizer state.
    qubit (int): Integer that indexes the qubit being acted on.

    Returns: None

    """

    qubit1, qubit2 = qubits

    Q(state, qubit1)  # Sqrt X
    Q(state, qubit2)  # Sqrt X
    Rd(state, qubit1)  # (Sqrt Y)^\dagger
    CNOT(state, qubits)  # CNOT
    R(state, qubit1)  # Sqrt Y


class RXXGate(BasicRotationGate):

    def __init__(self, angle):
        BasicRotationGate.__init__(self, angle)
        self.interchangeable_qubit_indices = [[0, 1]]

    @property
    def matrix(self):

        e1 = -1j*cmath.exp(1j * self.angle)
        e2 = -1j*cmath.exp(-1j * self.angle)
        n = 1. / cmath.sqrt(2.)

        return n * np.array([[1,   0,   0, e1],
                             [0,   1, -1j, 0],
                             [0, -1j,   1, 0],
                             [e2,   0,   0, 1]])

    def __str__(self):
        return "RXX"


def RXX(state, qubits, angle=None):
    q1 = state.qids[qubits[0]]
    q2 = state.qids[qubits[1]]
    RXXGate(angle) | (q1, q2)


class RYYGate(BasicRotationGate):

    def __init__(self, angle):
        BasicRotationGate.__init__(self, angle)
        self.interchangeable_qubit_indices = [[0, 1]]

    @property
    def matrix(self):

        return np.array([[np.cos(self.angle),   0,   0, 1j*np.sin(self.angle)],
                         [0,   np.cos(self.angle), -1j*np.sin(self.angle), 0],
                         [0, -1j*np.sin(self.angle),   np.cos(self.angle), 0],
                         [1j*np.sin(self.angle),   0,   0, np.cos(self.angle)]])

    def __str__(self):
        return "RYY"


def RYY(state, qubits, angle=None):
    q1 = state.qids[qubits[0]]
    q2 = state.qids[qubits[1]]
    RYYGate(angle) | (q1, q2)


class RZZGate(BasicRotationGate):
    """Square-root of X gate class"""

    def __init__(self, angle):
        BasicRotationGate.__init__(self, angle)
        self.interchangeable_qubit_indices = [[0, 1]]

    @property
    def matrix(self):

        p = cmath.exp(0.5j * self.angle)
        n = cmath.exp(-0.5j * self.angle)

        return np.array([[p, 0, 0, 0],
                         [0, n, 0, 0],
                         [0, 0, n, 0],
                         [0, 0, 0, p]])

    def __str__(self):
        return "RZZ"


def RZZ(state, qubits, angle=None):
    q1 = state.qids[qubits[0]]
    q2 = state.qids[qubits[1]]
    RZZGate(angle) | (q1, q2)
