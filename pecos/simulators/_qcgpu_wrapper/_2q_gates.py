import numpy as np
import qcgpu
from qcgpu.gate import x, y, z, s, t, h
from ._1q_gates import H, Q, R, Rd


class ControlQubit:

    def __init__(self, gate):

        if isinstance(gate, np.ndarray):
            self.gate = qcgpu.Gate(gate)
        else:
            self.gate = gate

    def func(self, state, qubits):
        control, target = qubits
        state.apply_controlled_gate(self.gate, control, target)


def II(state, qubits):
    pass


CNOT = ControlQubit(x()).func
CY = ControlQubit(y()).func
CZ = ControlQubit(z()).func
CS = ControlQubit(s()).func
CT = ControlQubit(t()).func
CH = ControlQubit(h()).func


def SWAP(state, qubits):
    CNOT(state, qubits)
    CNOT(state, (qubits[1], qubits[0]))
    CNOT(state, qubits)


def G2(state, qubits):
    """
    Applies a CZ.H(1).H(2).CZ

    Returns:

    """
    q1, q2 = qubits

    CZ(state, qubits)
    H(state, q1)
    H(state, q2)
    CZ(state, qubits)


def SqrtXX(state, qubits):

    qubit1, qubit2 = qubits
    Q(state, qubit1)  # Sqrt X
    Q(state, qubit2)  # Sqrt X
    Rd(state, qubit1)  # (Sqrt Y)^\dagger
    CNOT(state, qubits)  # CNOT (why didn't I capitalized this?)
    R(state, qubit1)  # Sqrt Y

