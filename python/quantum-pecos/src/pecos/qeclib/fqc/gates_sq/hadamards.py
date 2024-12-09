from pecos.qeclib import phys
from pecos.slr import Block, Comment, Permute, QReg


class H(Block):
    """
    Hadamard

    X -> Z
    Z -> X

    Y -> -Y
    """

    def __init__(self, q: QReg):
        if len(q.elems) != 7:
            msg = f"Size of register {len(q.elems)} != 5"
            raise Exception(msg)

        super().__init__(
            Comment("Logical H"),
            phys.H(q),
            Permute(
                [q[0], q[1], q[3], q[4]],
                [q[3], q[0], q[4], q[1]],
            ),
            # TODO: Double check this permutation is correct
        )
