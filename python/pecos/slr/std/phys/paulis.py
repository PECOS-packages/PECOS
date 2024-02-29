from numpy import array

from pecos.slr.std.phys.metaclasses import SQPauliGate


class XGate(SQPauliGate):
    """The Pauli X unitary."""

    def __init__(self):
        super().__init__("X", qasm_sym="x")

        self.matrix = array(
            [
                [0, 1],
                [1, 0],
            ],
            dtype=complex,
        )

        self.pauli_rules = {
            "X": "+X",
            "Z": "-Z",
        }


X = XGate()


class YGate(SQPauliGate):
    """The Pauli Y unitary."""

    def __init__(self):
        super().__init__("Y", qasm_sym="y")

        self.matrix = array(
            [
                [0, -1j],
                [1j, 0],
            ],
            dtype=complex,
        )

        self.pauli_rules = {
            "X": "-X",
            "Z": "-Z",
        }


Y = YGate()


class ZGate(SQPauliGate):
    """The Pauli Z unitary."""

    def __init__(self):
        super().__init__("Z", qasm_sym="z")

        self.matrix = array(
            [
                [1, 0],
                [0, -1],
            ],
            dtype=complex,
        )

        self.pauli_rules = {
            "X": "-X",
            "Z": "+Z",
        }


Z = ZGate()
