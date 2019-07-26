# Copyright 2018 The PECOS Developers
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

from pecos.simulators.paulifaultprop.gates_one_qubit import SX, SY, SZ, H, SYdg, SZdg, X


def CX(state, qubits: tuple[int, int]) -> None:
    """Applies the controlled-X gate.

    state (SparseSim): Instance representing the stabilizer state.
    qubit (int): Integer that indexes the qubit being acted on.

    Returns: None

    XI -> XX
    XX -> XI
    XZ -> -YY
    XY -> YZ

    ZI -> ZI
    ZX -> ZX
    ZZ -> IZ
    ZY -> IY

    YI -> YX
    YX -> YI
    YZ -> XY
    YY -> -XZ

    II -> II
    IX -> IX
    IZ -> ZZ
    IY -> ZY

    """
    q1, q2 = qubits

    if state.track_sign and (q1 in state.faults["Y"] and q2 in state.faults["Y"]):
        state.flip_sign()

    if q1 in state.faults["X"]:
        if q2 in state.faults["X"]:  # XX
            state.faults["X"].remove(q2)
            # XX -> XI

        elif q2 in state.faults["Z"]:  # XZ
            state.faults["X"].remove(q1)
            state.faults["Z"].remove(q2)
            state.faults["Y"].add(q1)
            state.faults["Y"].add(q2)
            # XZ -> -YY

        elif q2 in state.faults["Y"]:  # XY
            state.faults["X"].remove(q1)
            state.faults["Y"].remove(q2)
            state.faults["Y"].add(q1)
            state.faults["Z"].add(q2)
            # XY -> YZ

        else:  # XI
            state.faults["X"].add(q2)
            # XI -> XX

    elif q1 in state.faults["Z"]:
        # ZZ -> IZ
        # ZY -> IY
        if q2 in state.faults["Z"] or q2 in state.faults["Y"]:
            state.faults["Z"].remove(q1)

    elif q1 in state.faults["Y"]:
        if q2 in state.faults["X"]:  # YX
            state.faults["X"].remove(q2)
            # YX -> YI

        elif q2 in state.faults["Z"]:  # YZ
            state.faults["Y"].remove(q1)
            state.faults["Z"].remove(q2)
            state.faults["X"].add(q1)
            state.faults["Y"].add(q2)
            # YZ -> XY

        elif q2 in state.faults["Y"]:  # YY
            state.faults["Y"].remove(q1)
            state.faults["Y"].remove(q2)
            state.faults["X"].add(q1)
            state.faults["Z"].add(q2)
            # YY -> -XZ

        else:  # YI
            state.faults["X"].add(q2)
            # YI -> YX

    else:
        # IZ -> ZZ
        # IY -> ZY
        if q2 in state.faults["Z"] or q2 in state.faults["Y"]:  # IY
            state.faults["Z"].add(q1)

    if state.track_sign and (q1 in state.faults["Y"] and q2 in state.faults["Y"]):
        state.flip_sign()


def CZ(state, qubits: tuple[int, int]) -> None:
    """Applies the controlled-Z gate.

    II -> II
    XI -> XZ
    ZI -> ZI
    YI -> YZ
    IX -> ZX
    IZ -> IZ
    IY -> ZY
    XX -> YY
    XZ -> XI
    XY -> -YX
    ZX -> IX
    ZZ -> ZZ
    ZY -> IY
    YX -> -XY
    YZ -> YI
    YY -> XX

    state (SparseSim): Instance representing the stabilizer state.
    qubit (int): Integer that indexes the qubit being acted on.

    Returns: None

    """
    H(state, qubits[1])
    CX(state, qubits)
    H(state, qubits[1])


def CY(state, qubits: tuple[int, int]) -> None:
    """Applies the controlled-Y gate.

    II -> II
    XI -> XZ
    ZI -> ZI
    YI -> YZ
    IX -> ZX
    IZ -> IZ
    IY -> ZY
    XX -> YY
    XZ -> XI
    XY -> -YX
    ZX -> IX
    ZZ -> ZZ
    ZY -> IY
    YX -> -XY
    YZ -> YI
    YY -> XX

    state (SparseSim): Instance representing the stabilizer state.
    qubit (int): Integer that indexes the qubit being acted on.

    Returns: None

    """
    SZ(state, qubits[1])
    CX(state, qubits)
    SZ(state, qubits[1])


def SWAP(state, qubits: tuple[int, int]) -> None:
    """Applies a SWAP gate.

    state (SparseSim): Instance representing the stabilizer state.
    qubit (int): Integer that indexes the qubit being acted on.

    Returns: None

    """
    q1, q2 = qubits

    CX(state, (q1, q2))
    CX(state, (q2, q1))
    CX(state, (q1, q2))


def G2(state, qubits: tuple[int, int]) -> None:
    """Applies a CZ.H(1).H(2).CZ.

    state (SparseSim): Instance representing the stabilizer state.
    qubit (int): Integer that indexes the qubit being acted on.

    Returns: None

    """
    CZ(state, qubits)
    H(state, qubits[0])
    H(state, qubits[1])
    CZ(state, qubits)


def II(state, qubits: tuple[int, int], **params) -> None:
    """Two qubit identity.

    state (SparseSim): Instance representing the stabilizer state.
    qubit (int): Integer that indexes the qubit being acted on.

    Returns: None

    """


def SXX(state, qubits: tuple[int, int], **params) -> None:
    """Applies a square root of XX rotation to generators.

    state (SparseSim): Instance representing the stabilizer state.
    qubit (int): Integer that indexes the qubit being acted on.

    Returns: None

    """
    qubit1, qubit2 = qubits
    SX(state, qubit1)  # Sqrt X
    SX(state, qubit2)  # Sqrt X
    SYdg(state, qubit1)  # (Sqrt Y)^\dagger
    CX(state, qubits)  # CNOT
    SY(state, qubit1)  # Sqrt Y


def SXXdg(state, qubits: tuple[int, int], **params) -> None:
    """Applies a square root of XX rotation to generators.

    state (SparseSim): Instance representing the stabilizer state.
    qubit (int): Integer that indexes the qubit being acted on.

    Returns: None

    """
    qubit1, qubit2 = qubits
    X(state, qubit1)
    X(state, qubit2)
    SXX(state, qubits)


def SYY(state, qubits: tuple[int, int], **params: Any) -> None:
    r"""Sqrt of YY == (rZ,rZ).SqrtXX.(rZd,rZd).

    XI -> -ZY
    IX -> -YZ
    ZI -> XY
    IZ -> YX

    TODO: verify implementation!

    Args:
    ----
        state:
        qubits:

    Returns:
    -------

    """
    qubit1, qubit2 = qubits
    SZdg(state, qubit1)  # rZd
    SZdg(state, qubit2)  # rZd
    SXX(state, qubits)
    SZ(state, qubit1)  # rZ
    SZ(state, qubit2)  # rZ


def SYYdg(state, qubits: tuple[int, int], **params: Any) -> None:
    """Adjoint of SYY.

    Args:
    ----
        state:
        qubits:
        **params:

    Returns:
    -------

    """
    qubit1, qubit2 = qubits
    SZdg(state, qubit1)
    SZdg(state, qubit2)
    SXXdg(state, qubits)
    SZ(state, qubit1)
    SZ(state, qubit2)


def SZZ(state, qubits: tuple[int, int], **params) -> None:
    """Applies a square root of ZZ rotation to generators.

    state (SparseSim): Instance representing the stabilizer state.
    qubit (int): Integer that indexes the qubit being acted on.
    """
    qubit1, qubit2 = qubits
    SYdg(state, qubit1)  # rYd
    SYdg(state, qubit2)  # rYd
    SXX(state, qubits)
    SY(state, qubit1)  # rY
    SY(state, qubit2)  # rY


def SZZdg(state, qubits: tuple[int, int], **params) -> None:
    """Applies an adjoint of square root of ZZ rotation to generators.

    state (SparseSim): Instance representing the stabilizer state.
    qubit (int): Integer that indexes the qubit being acted on.
    """
    qubit1, qubit2 = qubits
    SYdg(state, qubit1)  # rYd
    SYdg(state, qubit2)  # rYd
    SXXdg(state, qubits)
    SY(state, qubit1)  # rY
    SY(state, qubit2)  # rY
