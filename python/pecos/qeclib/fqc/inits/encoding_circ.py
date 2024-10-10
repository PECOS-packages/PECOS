from pecos.qeclib import phys
from pecos.slr import Block, QReg


class EncodingCircuit(Block):
    """The encoding circuit for the five-qubit code. q[4] is the input qubit.

    Taken from Fig. 9 of arXiv:1707.09951.
    """

    # TODO: Double check that the order of these qubits match everything else

    def __init__(self, q: QReg):
        # fmt: off
        super().__init__(
            phys.H(q[0], q[1], q[2], q[3]),
            phys.Z(q[4]),

            phys.CX(q[0], q[4]),
            phys.CX(q[1], q[4]),
            phys.CX(q[2], q[4]),
            phys.CX(q[3], q[4]),

            phys.CZ(q[0], q[1]),
            phys.CZ(q[1], q[2]),
            phys.CZ(q[2], q[3]),
            phys.CZ(q[3], q[4]),
            phys.CZ(q[4], q[0]),

        )
        # fmt: on
