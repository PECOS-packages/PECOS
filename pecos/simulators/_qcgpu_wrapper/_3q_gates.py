import numpy as np
import qcgpu
from qcgpu.gate import x, y, z


class ControlControlQubit:

    def __init__(self, gate):

        if isinstance(gate, np.ndarray):
            self.gate = qcgpu.Gate(gate)
        else:
            self.gate = gate

    def func(self, state, qubits):
        control1, control2, target = qubits
        state.apply_control_controlled_gate(self.gate, control1, control2, target)


def II(state, qubits):
    pass


TOFFOLI = ControlControlQubit(x()).func
CCY = ControlControlQubit(y()).func
CCZ = ControlControlQubit(z()).func
