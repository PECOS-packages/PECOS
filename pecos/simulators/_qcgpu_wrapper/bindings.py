from qcgpu.gate import h, x, y, z, s, t, sqrt_x


class MakeFunc:
    """
    Converts qcgpu gate to a function.
    """
    def __init__(self, gate):
        """

        Args:
            gate:
        """

        self.gate = gate

    def func(self, state, qubits):

        if isinstance(qubits, int):
            state.apply_gate(self.gate, qubits)
        else:
            state.apply_gate(self.gate, *qubits)


gate_dict = {
    'H': MakeFunc(h).func,
    'S': MakeFunc(sqrt_x).func,
    'T': MakeFunc(t).func,
    'X': MakeFunc(x).func,
    'Y': MakeFunc(y).func,
    'Z': MakeFunc(z).func,

    'CNOT': MakeFunc(h).func,
    'SWAP': MakeFunc(h).func,
    'Toffoli': MakeFunc(h).func,
}