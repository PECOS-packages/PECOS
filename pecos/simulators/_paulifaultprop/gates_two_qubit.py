from typing import Tuple
from .gates_one_qubit import H, S, Q, R, Rd


def CNOT(state,
         qubits: Tuple[int, int]) -> None:
    """
    Applies the controlled-X gate.

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

    if q1 in state.faults['X']:

        if q2 in state.faults['X']:  # XX
            state.faults['X'].remove(q2)
            # XX -> XI

        elif q2 in state.faults['Z']:  # XZ
            state.faults['X'].remove(q1)
            state.faults['Z'].remove(q2)
            state.faults['Y'].add(q1)
            state.faults['Y'].add(q2)
            # XZ -> -YY

        elif q2 in state.faults['Y']:  # XY
            state.faults['X'].remove(q1)
            state.faults['Y'].remove(q2)
            state.faults['Y'].add(q1)
            state.faults['Z'].add(q2)
            # XY -> YZ

        else:  # XI
            state.faults['X'].add(q2)
            # XI -> XX

    elif q1 in state.faults['Z']:

        if q2 in state.faults['Z']:  # ZZ
            state.faults['Z'].remove(q1)
            # ZZ -> IZ

        elif q2 in state.faults['Y']:  # ZY
            state.faults['Z'].remove(q1)
            # ZY -> IY

    elif q1 in state.faults['Y']:

        if q2 in state.faults['X']:  # YX
            state.faults['X'].remove(q2)
            # YX -> YI

        elif q2 in state.faults['Z']:  # YZ
            state.faults['Y'].remove(q1)
            state.faults['Z'].remove(q2)
            state.faults['X'].add(q1)
            state.faults['Y'].add(q2)
            # YZ -> XY

        elif q2 in state.faults['Y']:  # YY
            state.faults['Y'].remove(q1)
            state.faults['Y'].remove(q2)
            state.faults['X'].add(q1)
            state.faults['Z'].add(q2)
            # YY -> -XZ

        else:  # YI
            state.faults['X'].add(q2)
            # YI -> YX

    else:

        if q2 in state.faults['Z']:  # IZ
            state.faults['Z'].add(q1)
            # IZ -> ZZ

        elif q2 in state.faults['Y']:  # IY
            state.faults['Z'].add(q1)
            # IY -> ZY


def CZ(state,
       qubits: Tuple[int, int]) -> None:
    """
    Applies the controlled-Z gate.

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
    CNOT(state, qubits)
    H(state, qubits[1])


def CY(state,
       qubits: Tuple[int, int]) -> None:
    """
    Applies the controlled-Y gate.

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

    S(state, qubits[1])
    CNOT(state, qubits)
    S(state, qubits[1])


def SWAP(state,
         qubits: Tuple[int, int]) -> None:
    """
    Applies a SWAP gate.

    state (SparseSim): Instance representing the stabilizer state.
    qubit (int): Integer that indexes the qubit being acted on.

    Returns: None

    """

    q1, q2 = qubits

    CNOT(state, (q1, q2))
    CNOT(state, (q2, q1))
    CNOT(state, (q1, q2))


def G2(state,
       qubits: Tuple[int, int]) -> None:
    """
    Applies a CZ.H(1).H(2).CZ

    state (SparseSim): Instance representing the stabilizer state.
    qubit (int): Integer that indexes the qubit being acted on.

    Returns: None

    """

    CZ(state, qubits)
    H(state, qubits[0])
    H(state, qubits[1])
    CZ(state, qubits)


def II(state,
       qubits: Tuple[int, int]) -> None:
    """
    Two qubit identity.

    state (SparseSim): Instance representing the stabilizer state.
    qubit (int): Integer that indexes the qubit being acted on.

    Returns: None

    """
    pass


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
