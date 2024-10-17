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

from numpy import pi
from projectq import ops

from pecos.simulators.projectq.gates_one_qubit import H


def II(state, qubits: tuple[int, int], **params: Any) -> None:
    pass


def G2(state, qubits: tuple[int, int], **params: Any) -> None:
    """Applies a CZ.H(1).H(2).CZ.

    Returns:
    -------

    """
    CZ(state, qubits)
    H(state, qubits[0])
    H(state, qubits[1])
    CZ(state, qubits)


def CNOT(state, qubits: tuple[int, int], **params: Any) -> None:
    q1 = state.qids[qubits[0]]
    q2 = state.qids[qubits[1]]

    ops.CNOT | (q1, q2)


def CZ(state, qubits: tuple[int, int], **params: Any) -> None:
    q1 = state.qids[qubits[0]]
    q2 = state.qids[qubits[1]]

    ops.C(ops.Z) | (q1, q2)


def CY(state, qubits: tuple[int, int], **params: Any) -> None:
    q1 = state.qids[qubits[0]]
    q2 = state.qids[qubits[1]]

    ops.C(ops.Y) | (q1, q2)


def SWAP(state, qubits: tuple[int, int], **params: Any) -> None:
    q1 = state.qids[qubits[0]]
    q2 = state.qids[qubits[1]]

    ops.Swap | (q1, q2)


def SXX(state, qubits: tuple[int, int], **params: Any) -> None:
    """Square root of XX rotation to generators.

    state (SparseSim): Instance representing the stabilizer state.
    qubit (int): Integer that indexes the qubit being acted on.

    Returns: None
    """
    q1 = state.qids[qubits[0]]
    q2 = state.qids[qubits[1]]
    ops.Rxx(pi / 2) | (q1, q2)


def SXXdg(state, qubits: tuple[int, int], **params: Any) -> None:
    """Adjoint of square root of XX rotation.

    state: Instance representing the stabilizer state.
    qubit: Integer that indexes the qubit being acted on.

    Returns: None
    """
    q1 = state.qids[qubits[0]]
    q2 = state.qids[qubits[1]]
    ops.Rxx(-pi / 2) | (q1, q2)


def SYY(state, qubits: tuple[int, int], **params: Any) -> None:
    """Square root of YY rotation to generators.

    state: Instance representing the stabilizer state.
    qubit: Integer that indexes the qubit being acted on.

    Returns: None
    """
    q1 = state.qids[qubits[0]]
    q2 = state.qids[qubits[1]]
    ops.Ryy(pi / 2) | (q1, q2)


def SYYdg(state, qubits: tuple[int, int], **params: Any) -> None:
    """Adjoint of square root of YY rotation to generators.

    state: Instance representing the stabilizer state.
    qubit: Integer that indexes the qubit being acted on.

    Returns: None
    """
    q1 = state.qids[qubits[0]]
    q2 = state.qids[qubits[1]]
    ops.Ryy(-pi / 2) | (q1, q2)


def SZZ(state, qubits: tuple[int, int], **params: Any) -> None:
    """Applies a square root of ZZ rotation to generators.

    state: Instance representing the stabilizer state.
    qubit: Integer that indexes the qubit being acted on.

    Returns: None
    """
    q1 = state.qids[qubits[0]]
    q2 = state.qids[qubits[1]]
    ops.Rzz(pi / 2) | (q1, q2)


def SZZdg(state, qubits: tuple[int, int], **params: Any) -> None:
    """Applies an adjoint of square root of ZZ rotation to generators.

    state: Instance representing the stabilizer state.
    qubit: Integer that indexes the qubit being acted on.

    Returns: None
    """
    q1 = state.qids[qubits[0]]
    q2 = state.qids[qubits[1]]
    ops.Rzz(-pi / 2) | (q1, q2)


def RXX(state, qubits, angle=None, **params: Any) -> None:
    q1 = state.qids[qubits[0]]
    q2 = state.qids[qubits[1]]
    ops.Rxx(angle) | (q1, q2)


def RYY(state, qubits, angle=None, **params: Any) -> None:
    q1 = state.qids[qubits[0]]
    q2 = state.qids[qubits[1]]
    ops.Ryy(angle) | (q1, q2)


def RZZ(state, qubits, angle=None, **params: Any) -> None:
    q1 = state.qids[qubits[0]]
    q2 = state.qids[qubits[1]]
    ops.Rzz(angle) | (q1, q2)


def R2XXYYZZ(state, qubits, angles=None, **params: Any) -> None:
    q1 = state.qids[qubits[0]]
    q2 = state.qids[qubits[1]]
    ops.Rxx(angles[0]) | (q1, q2)
    ops.Ryy(angles[1]) | (q1, q2)
    ops.Rzz(angles[2]) | (q1, q2)
